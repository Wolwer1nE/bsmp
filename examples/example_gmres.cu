#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "block_sparse_matrix.h"
#include "gmres.h"
#include "triplet_loader.h"

static PreconditionerOptions parse_preconditioner_option(const char* value) {
    PreconditionerOptions options;
    if (value != nullptr && *value != '\0') {
        if (!parsePreconditionerType(value, options.type)) {
            std::cerr << "Unsupported preconditioner: " << value
                      << ". Supported values: amg, scalar-jacobi, block-jacobi, none" << std::endl;
            std::exit(2);
        }
    }
    return options;
}

static bool read_vector_file(const std::string& path, std::vector<float>& out) {
    std::ifstream in(path);
    if (!in)
        return false;

    std::string line;
    bool first_noncomment = true;
    while (std::getline(in, line)) {
        size_t p = line.find_first_not_of(" \t\r\n");
        if (p == std::string::npos)
            continue;
        if (line[p] == '%' || line[p] == '#')
            continue;

        std::istringstream iss(line);
        if (first_noncomment) {
            int m, n;
            if ((iss >> m >> n) && iss.eof()) {
                first_noncomment = false;
                continue;
            }
            iss.clear();
            iss.str(line);
            first_noncomment = false;
        }

        float v;
        if (iss >> v) {
            out.push_back(v);
        }
    }
    return !out.empty();
}

static bool write_vector_file(const std::string& path, const std::vector<float>& vec) {
    std::ofstream out(path);
    if (!out)
        return false;
    for (size_t i = 0; i < vec.size(); i++) {
        out << vec[i] << "\n";
    }
    return true;
}

int main(int argc, char** argv) {
    std::cout << "=== GMRES Solver Test ===" << std::endl;

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <matrix_file> <rhs_file> [output_file] [restart] [max_iters] [tol] [preconditioner]" << std::endl
                  << "  preconditioner: amg, scalar-jacobi, block-jacobi, none" << std::endl;
        return 2;
    }

    std::string matrix_file = argv[1];
    std::string rhs_file = argv[2];
    std::string output_file = argc >= 4 ? argv[3] : std::string("");
    int restart = argc >= 5 ? std::max(1, std::atoi(argv[4])) : 30;
    int max_iters = argc >= 6 ? std::max(1, std::atoi(argv[5])) : 1000;
    float tol = argc >= 7 ? std::max(1e-12f, std::strtof(argv[6], nullptr)) : 1e-6f;
    PreconditionerOptions preconditioner_options = argc >= 8
                                                       ? parse_preconditioner_option(argv[7])
                                                       : PreconditionerOptions{};

    std::cout << "Matrix file: " << matrix_file << std::endl;
    std::cout << "RHS file: " << rhs_file << std::endl;
    if (!output_file.empty()) {
        std::cout << "Output file: " << output_file << std::endl;
    }

    BlockSparseMatrixConfig config;
    std::vector<int> block_rows, block_cols;
    std::vector<float> block_data;

    bool loaded = false;
    {
        std::ifstream fin(matrix_file);
        if (!fin) {
            std::cerr << "Failed to open matrix file: " << matrix_file << std::endl;
            return 1;
        }
        std::string line;
        std::string first_nonempty;
        while (std::getline(fin, line)) {
            size_t p = line.find_first_not_of(" \t\r\n");
            if (p == std::string::npos)
                continue;
            first_nonempty = line.substr(p);
            break;
        }

        if (!first_nonempty.empty() && first_nonempty.rfind("%%MatrixMarket", 0) == 0) {
            loaded = load_matrix_market_as_block_sparse(matrix_file, BSMP_BLOCK_SIZE,
                                                        config, block_rows, block_cols, block_data);
        } else {
            loaded = load_triplet_file_as_block_sparse(matrix_file, BSMP_BLOCK_SIZE,
                                                       config, block_rows, block_cols, block_data);
        }
    }

    if (!loaded) {
        std::cerr << "Failed to load matrix from " << matrix_file << std::endl;
        return 1;
    }

    std::cout << "Loaded matrix: " << config.num_rows << " x " << config.num_cols
              << " with " << config.num_nonzero_blocks << " nonzero blocks (block size "
              << BSMP_BLOCK_SIZE << ")" << std::endl;

    BlockSparseMatrix A(config);
    A.initialize(block_rows, block_cols, block_data);

    std::vector<float> b_host;
    if (!read_vector_file(rhs_file, b_host)) {
        std::cerr << "Failed to load RHS from " << rhs_file << std::endl;
        return 1;
    }

    if ((int)b_host.size() != config.num_rows) {
        std::cerr << "RHS size mismatch: expected " << config.num_rows
                  << ", got " << b_host.size() << std::endl;
        return 1;
    }

    float *d_x, *d_b;
    cudaMalloc(&d_x, config.num_rows * sizeof(float));
    cudaMalloc(&d_b, config.num_rows * sizeof(float));

    cudaMemset(d_x, 0, config.num_rows * sizeof(float));
    cudaMemcpy(d_b, b_host.data(), config.num_rows * sizeof(float), cudaMemcpyHostToDevice);

    int iters_out;
    float resid_out;

    std::cout << "\nSolving Ax = b with GMRES..." << std::endl;
    std::cout << "Preconditioner: " << preconditionerTypeName(preconditioner_options.type) << std::endl;
    std::cout << "Restart: " << restart << std::endl;
    std::cout << "Max iterations: " << max_iters << ", tolerance: " << tol << std::endl;

    auto solve_start = std::chrono::steady_clock::now();
    bool converged = gmres(A, d_b, d_x, max_iters, restart, tol,
                           iters_out, resid_out, preconditioner_options);
    cudaDeviceSynchronize();
    auto solve_end = std::chrono::steady_clock::now();
    double solve_ms = std::chrono::duration_cast<std::chrono::microseconds>(solve_end - solve_start).count() / 1000.0;

    std::cout << "\nResults:" << std::endl;
    std::cout << "Converged: " << (converged ? "YES" : "NO") << std::endl;
    std::cout << "Iterations: " << iters_out << std::endl;
    std::cout << "Final relative residual: " << resid_out << std::endl;
    std::cout << "Solve time (ms): " << solve_ms << std::endl;

    std::vector<float> x_host(config.num_rows);
    cudaMemcpy(x_host.data(), d_x, config.num_rows * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "\nFirst 10 elements of solution:" << std::endl;
    for (int i = 0; i < std::min(10, config.num_rows); i++) {
        std::cout << "x[" << i << "] = " << x_host[i] << std::endl;
    }

    if (output_file.empty()) {
        output_file = "data/gmres_cpp_solution.txt";
    }

    if (write_vector_file(output_file, x_host)) {
        std::cout << "\nSolution saved to " << output_file << std::endl;
    } else {
        std::cerr << "Failed to save solution to " << output_file << std::endl;
    }

    cudaFree(d_x);
    cudaFree(d_b);

    return converged ? 0 : 1;
}