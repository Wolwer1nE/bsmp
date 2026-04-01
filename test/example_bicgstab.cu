#include <cuda_runtime.h>

#include <cmath>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "bicgstab.h"
#include "block_sparse_matrix.h"
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
        // Skip comments and empty lines (handle '%' Matrix Market comments and '#')
        size_t p = line.find_first_not_of(" \t\r\n");
        if (p == std::string::npos)
            continue;
        if (line[p] == '%' || line[p] == '#')
            continue;

        std::istringstream iss(line);
        if (first_noncomment) {
            // Matrix Market array format may have a size line: "m n". If this line contains
            // two integers, treat it as the size header and skip it.
            int m, n;
            if ((iss >> m >> n) && iss.eof()) {
                first_noncomment = false;
                continue; // skip size line
            }
            // not a size line; fall through to parse as a value
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
    std::cout << "=== BiCGStab Solver Test ===" << std::endl;

    // Parse command line arguments (no defaults; require input paths)
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <matrix_file> <rhs_file> [output_file] [max_iters] [tol] [preconditioner]" << std::endl
                  << "  preconditioner: amg, scalar-jacobi, block-jacobi, none" << std::endl;
        return 2;
    }
    std::string matrix_file = argv[1];
    std::string rhs_file = argv[2];
    std::string output_file = argc >= 4 ? argv[3] : std::string("");
    int max_iters = argc >= 5 ? std::max(1, std::atoi(argv[4])) : 1000;
    float tol = argc >= 6 ? std::max(1e-12f, std::strtof(argv[5], nullptr)) : 1e-6f;
    PreconditionerOptions preconditioner_options = argc >= 7
                                                       ? parse_preconditioner_option(argv[6])
                                                       : PreconditionerOptions{};

    std::cout << "Matrix file: " << matrix_file << std::endl;
    std::cout << "RHS file: " << rhs_file << std::endl;
    if (!output_file.empty()) {
        std::cout << "Output file: " << output_file << std::endl;
    }

    // Load matrix: detect format and call appropriate loader
    BlockSparseMatrixConfig config;
    std::vector<int> block_rows, block_cols;
    std::vector<float> block_data;

    // Peek first non-empty line to detect Matrix Market header
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
            // trim leading spaces
            size_t p = line.find_first_not_of(" \t\r\n");
            if (p == std::string::npos) continue;
            first_nonempty = line.substr(p);
            break;
        }

        // If file begins with MatrixMarket header (%%MatrixMarket) use the dedicated loader.
        // Note: do not treat a single leading '%' as the header; MatrixMarket header starts with '%%'.
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

    // Create BlockSparseMatrix
    BlockSparseMatrix A(config);
    A.initialize(block_rows, block_cols, block_data);

    // Load RHS vector
    std::vector<float> b_host;
    if (!read_vector_file(rhs_file, b_host)) {
        std::cerr << "Failed to load RHS from " << rhs_file << std::endl;
        // Create synthetic RHS: b = A * ones
        std::cout << "Creating synthetic RHS: b = A * ones" << std::endl;
        b_host.resize(config.num_rows, 0.0f);
        std::vector<float> ones(config.num_cols, 1.0f);

        float *d_ones, *d_b;
        cudaMalloc(&d_ones, config.num_cols * sizeof(float));
        cudaMalloc(&d_b, config.num_rows * sizeof(float));
        cudaMemcpy(d_ones, ones.data(), config.num_cols * sizeof(float), cudaMemcpyHostToDevice);

        A.multiply(d_ones, d_b);
        cudaMemcpy(b_host.data(), d_b, config.num_rows * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_ones);
        cudaFree(d_b);
    }

    if ((int)b_host.size() != config.num_rows) {
        std::cerr << "RHS size mismatch: expected " << config.num_rows
                  << ", got " << b_host.size() << std::endl;
        return 1;
    }

    // Allocate device memory for x and b
    float *d_x, *d_b;
    cudaMalloc(&d_x, config.num_rows * sizeof(float));
    cudaMalloc(&d_b, config.num_rows * sizeof(float));

    // Initialize x = 0
    cudaMemset(d_x, 0, config.num_rows * sizeof(float));

    // Copy b to device
    cudaMemcpy(d_b, b_host.data(), config.num_rows * sizeof(float), cudaMemcpyHostToDevice);

    // Solve using BiCGStab
    int iters_out;
    float resid_out;

    std::cout << "\nSolving Ax = b with BiCGStab..." << std::endl;
    std::cout << "Preconditioner: " << preconditionerTypeName(preconditioner_options.type) << std::endl;
    std::cout << "Max iterations: " << max_iters << ", tolerance: " << tol << std::endl;

    auto solve_start = std::chrono::steady_clock::now();
    bool converged = bicgstab(A, d_b, d_x, max_iters, tol,
                              iters_out, resid_out, preconditioner_options);
    cudaDeviceSynchronize();
    auto solve_end = std::chrono::steady_clock::now();
    double solve_ms = std::chrono::duration_cast<std::chrono::microseconds>(solve_end - solve_start).count() / 1000.0;

    std::cout << "\nResults:" << std::endl;
    std::cout << "Converged: " << (converged ? "YES" : "NO") << std::endl;
    std::cout << "Iterations: " << iters_out << std::endl;
    std::cout << "Final relative residual: " << resid_out << std::endl;
    std::cout << "Solve time (ms): " << solve_ms << std::endl;

    // Copy solution back to host
    std::vector<float> x_host(config.num_rows);
    cudaMemcpy(x_host.data(), d_x, config.num_rows * sizeof(float), cudaMemcpyDeviceToHost);

    // Print first few elements of solution
    std::cout << "\nFirst 10 elements of solution:" << std::endl;
    for (int i = 0; i < std::min(10, config.num_rows); i++) {
        std::cout << "x[" << i << "] = " << x_host[i] << std::endl;
    }

    // Save solution to file
    if (output_file.empty()) {
        output_file = "data/bicgstab_cpp_solution.txt";
    }

    if (write_vector_file(output_file, x_host)) {
        std::cout << "\nSolution saved to " << output_file << std::endl;
        std::cout << "Verify in MATLAB with: test_bicgstab" << std::endl;
    } else {
        std::cerr << "Failed to save solution to " << output_file << std::endl;
    }

    // Cleanup
    cudaFree(d_x);
    cudaFree(d_b);

    return converged ? 0 : 1;
}
