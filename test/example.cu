#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "block_sparse_matrix.h"
#include "triplet_loader.h"

struct Options {
    std::string format = "triplets";
    std::string matrix_path;
    std::string rhs_path;
    std::string output_path;
    int iterations = 1;
    bool verbose = false;
};

static bool read_vector_file(const std::string& path, std::vector<float>& out) {
    std::ifstream in(path);
    if (!in)
        return false;
    float v;
    while (in >> v)
        out.push_back(v);
    return true;
}

static bool write_vector_file(const std::string& path, const std::vector<float>& v) {
    std::ofstream out(path);
    if (!out)
        return false;
    for (size_t i = 0; i < v.size(); ++i) {
        out << v[i] << '\n';
    }
    return true;
}

static std::string derive_rhs_path(const std::string& matrix_path) {
    return std::string("rhs_") + matrix_path;
}

static void print_usage() {
    std::cout << "Usage:\n"
              << "  example --matrix <matrix_triplets> [--rhs <rhs_vector>] [--format triplets|matrix-market]\n"
              << "          [--mults <iterations>] [--output <file>] [--verbose]\n\n"
              << "Options:\n"
              << "  --matrix   Path to sparse matrix file in triplet or Matrix Market format.\n"
              << "  --rhs      Path to RHS vector (defaults to rhs_<matrix>).\n"
              << "  --format   Input format (triplets|matrix-market). Default: triplets.\n"
              << "  --mults    Number of sequential matrix-vector multiplications. Default: 1.\n"
              << "  --output   Optional path to save resulting vector from final iteration.\n"
              << "  --verbose  Print detailed statistics per iteration.\n"
              << std::endl;
}

static bool parse_cli(int argc, char** argv, Options& opt) {
    if (argc < 2)
        return false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--matrix" && i + 1 < argc) {
            opt.matrix_path = argv[++i];
        } else if (arg == "--rhs" && i + 1 < argc) {
            opt.rhs_path = argv[++i];
        } else if (arg == "--format" && i + 1 < argc) {
            opt.format = argv[++i];
        } else if ((arg == "--mults" || arg == "--iterations") && i + 1 < argc) {
            opt.iterations = std::max(1, std::atoi(argv[++i]));
        } else if (arg == "--output" && i + 1 < argc) {
            opt.output_path = argv[++i];
        } else if (arg == "--verbose") {
            opt.verbose = true;
        } else if (arg == "-h" || arg == "--help") {
            return false;
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            return false;
        }
    }

    if (opt.matrix_path.empty()) {
        std::cerr << "--matrix is required" << std::endl;
        return false;
    }
    if (opt.rhs_path.empty()) {
        opt.rhs_path = derive_rhs_path(opt.matrix_path);
    }
    return true;
}

static bool load_matrix_by_format(
    const std::string& fmt,
    const std::string& path,
    BlockSparseMatrixConfig& cfg,
    std::vector<int>& br,
    std::vector<int>& bc,
    std::vector<float>& bd) {
    std::string f = fmt;
    for (char& c : f)
        c = (char)std::tolower((unsigned char)c);
    if (f == "triplets" || f == "triplet" || f == "tripletss") {
        return load_triplet_file_as_block_sparse(path, bsmp::kBlockSize, cfg, br, bc, bd);
    } else if (f == "matrix-market" || f == "mm" || f == "mtx") {
        return load_matrix_market_as_block_sparse(path, bsmp::kBlockSize, cfg, br, bc, bd);
    } else {
        std::cerr << "Unknown format: " << fmt << std::endl;
        return false;
    }
}

int main(int argc, char** argv) {
    Options opt;
    if (!parse_cli(argc, argv, opt)) {
        print_usage();
        return 1;
    }

    BlockSparseMatrixConfig cfg;
    std::vector<int> block_rows, block_cols;
    std::vector<float> block_data;
    if (!load_matrix_by_format(opt.format, opt.matrix_path, cfg, block_rows, block_cols, block_data)) {
        std::cerr << "Failed to load matrix from " << opt.matrix_path << std::endl;
        return 2;
    }

    if (opt.verbose) {
        std::cout << "Matrix: " << opt.matrix_path << std::endl;
        std::cout << "Format: " << opt.format << std::endl;
        std::cout << "Dimensions: " << cfg.num_rows << " x " << cfg.num_cols << std::endl;
        std::cout << "Non-zero blocks: " << cfg.num_nonzero_blocks << std::endl;
        std::cout << "Iterations: " << opt.iterations << std::endl;
    }

    BlockSparseMatrix A(cfg);
    A.initialize(block_rows, block_cols, block_data);

    std::vector<float> hx;
    if (!read_vector_file(opt.rhs_path, hx)) {
        std::cerr << "Failed to read RHS from " << opt.rhs_path << std::endl;
        return 3;
    }
    if ((int)hx.size() != cfg.num_cols) {
        std::cerr << "RHS size (" << hx.size() << ") does not match matrix columns (" << cfg.num_cols << ")" << std::endl;
        return 3;
    }

    float* d_x = nullptr;
    float* d_y = nullptr;
    cudaMalloc(&d_x, cfg.num_cols * sizeof(float));
    cudaMalloc(&d_y, cfg.num_rows * sizeof(float));
    cudaMemcpy(d_x, hx.data(), cfg.num_cols * sizeof(float), cudaMemcpyHostToDevice);

    std::vector<float> hy(cfg.num_rows);
    double total_ms = 0.0;

    for (int iter = 0; iter < opt.iterations; ++iter) {
        auto iter_start = std::chrono::steady_clock::now();
        A.multiply(d_x, d_y, 0);
        cudaDeviceSynchronize();
        auto iter_end = std::chrono::steady_clock::now();
        double iter_ms = std::chrono::duration_cast<std::chrono::microseconds>(iter_end - iter_start).count() / 1000.0;
        total_ms += iter_ms;
        if (opt.verbose) {
            std::cout << "Iteration " << (iter + 1) << "/" << opt.iterations << ": " << iter_ms << " ms" << std::endl;
        }
    }

    cudaMemcpy(hy.data(), d_y, cfg.num_rows * sizeof(float), cudaMemcpyDeviceToHost);

    if (!opt.output_path.empty()) {
        write_vector_file(opt.output_path, hy);
    }

    double avg_ms = total_ms / static_cast<double>(opt.iterations);
    std::cout << "Total time: " << total_ms << " ms for " << opt.iterations << " multiplies" << std::endl;
    std::cout << "Average per multiply: " << avg_ms << " ms" << std::endl;

    if (opt.verbose) {
        double max_abs = 0.0;
        for (float v : hy) {
            max_abs = std::max(max_abs, std::abs(static_cast<double>(v)));
        }
        std::cout << "Result preview (first 10 entries):";
        int preview = std::min<int>(10, hy.size());
        for (int i = 0; i < preview; ++i) {
            std::cout << " " << hy[i];
        }
        std::cout << std::endl;
        std::cout << "Max |y|: " << max_abs << std::endl;
    }

    cudaFree(d_x);
    cudaFree(d_y);
    return 0;
}
