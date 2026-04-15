#include <algorithm>
#include <cmath>
#include <iostream>

#include "libbsmp/bicgstab.h"
#include "libbsmp/generalized_eigen.h"

namespace bsmp {

namespace {

/// @brief Vector norm
float vectorNorm(const std::vector<float>& v) {
    float sum = 0.0f;
    for (float val : v) {
        sum += val * val;
    }
    return std::sqrt(sum);
}

/// @brief Normalize vector
void normalizeVector(std::vector<float>& v) {
    float norm = vectorNorm(v);
    if (norm > 1e-15f) {
        for (float& val : v) {
            val /= norm;
        }
    }
}

/// @brief Scalar dot product
float dotProduct(const std::vector<float>& a, const std::vector<float>& b) {
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

/// @brief Gram-Schmidt orthogonalization
/// @note `v = v - sum((v^T * b_i) * b_i)` for all `b_i` in basis
void orthogonalize(std::vector<float>& v, const std::vector<std::vector<float>>& basis) {
    for (const auto& b : basis) {
        float proj = dotProduct(v, b);
        for (size_t i = 0; i < v.size(); ++i) {
            v[i] -= proj * b[i];
        }
    }
    normalizeVector(v);
}

/// @brief Rayleigh quotient: `λ = (v^T * A * v) / (v^T * B * v)`
/// @note Overwrites `numerator` and `denominator` parameters
void rayleighQuotient(
    BlockSparseMatrix& A, BlockSparseMatrix& B,
    const std::vector<float>& v, float* d_v, float* d_Av, float* d_Bv,
    float& numerator, float& denominator) {
    int n = v.size();

    cudaMemcpy(d_v, v.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    A.multiply(d_v, d_Av);
    B.multiply(d_v, d_Bv);

    std::vector<float> Av(n);
    std::vector<float> Bv(n);
    cudaMemcpy(Av.data(), d_Av, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Bv.data(), d_Bv, n * sizeof(float), cudaMemcpyDeviceToHost);

    numerator = dotProduct(v, Av);
    denominator = dotProduct(v, Bv);
}

/// @brief BiCGStab-based method of inverse iterations with shift
/// `A*v = λ*B*v -> (A - shift*B)*v = 0`
/// @note Solves A*x = B*v_prev, normalizes x to get v_new
/// `λ = (v^T * A * v) / (v^T * B * v)`
bool inverseIterationWithShift(BlockSparseMatrix& A, BlockSparseMatrix& B,
                               float shift,
                               float& eigenvalue, std::vector<float>& eigenvector,
                               const std::vector<std::vector<float>>& previous_eigenvectors,
                               int max_iter, float tol) {
    int n = A.getConfig().num_rows;

    float *d_v, *d_Av, *d_Bv, *d_rhs, *d_solution;
    cudaMalloc(&d_v, n * sizeof(float));
    cudaMalloc(&d_Av, n * sizeof(float));
    cudaMalloc(&d_Bv, n * sizeof(float));
    cudaMalloc(&d_rhs, n * sizeof(float));
    cudaMalloc(&d_solution, n * sizeof(float));

    /// @brief Frees `d_v`, `d_Av`, `d_Bv`, `d_rhs` and `d_solution` device buffers
    auto cleanup = [=] {
        cudaFree(d_v);
        cudaFree(d_Av);
        cudaFree(d_Bv);
        cudaFree(d_rhs);
        cudaFree(d_solution);
    };

    // Initial guess (random vector)
    // FIXME: use better initialization? P.A.
    eigenvector.resize(n);
    for (int i = 0; i < n; ++i) {
        eigenvector[i] = (float)rand() / RAND_MAX - 0.5f;
    }

    orthogonalize(eigenvector, previous_eigenvectors);

    float prev_lambda = shift;
    int stagnation_count = 0;

    for (int iter = 0; iter < max_iter; ++iter) {
        cudaMemcpy(d_v, eigenvector.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        B.multiply(d_v, d_rhs);

        // BiCGStab
        cudaMemset(d_solution, 0, n * sizeof(float));

        int bicg_iters;
        float bicg_resid;
        bool solved = bicgstab(A, d_rhs, d_solution, 200, 1e-7f, bicg_iters, bicg_resid);
        // FIXME 5 is a magic number here P.A.
        if (!solved && iter > 5) {
            // BiCGStab did not converge, but we will use the current solution anyway
            std::cerr << "BiCGStab failed to converge at iter= " << iter
                      << " (residual is " << bicg_resid << ")" << std::endl;
        }

        cudaMemcpy(eigenvector.data(), d_solution, n * sizeof(float), cudaMemcpyDeviceToHost);

        orthogonalize(eigenvector, previous_eigenvectors);

        float numerator, denominator;
        rayleighQuotient(A, B, eigenvector, d_v, d_Av, d_Bv, numerator, denominator);

        if (std::abs(denominator) < 1e-15f) {
            std::cerr << "Division by zero in Rayleigh quotient" << std::endl;
            break;
        }

        float lambda = numerator / denominator;

        // Check convergence
        float lambda_change = std::abs(lambda - prev_lambda);
        float rel_change = lambda_change / (std::abs(lambda) + 1e-10f);

        if (iter > 5 && rel_change < tol) {
            eigenvalue = lambda;
            cleanup();
            return true;
        }

        // Are we stuck?
        if (rel_change < tol * 10) {
            stagnation_count++;
            if (stagnation_count > 20) {
                // We are all stuck, return current estimate
                eigenvalue = lambda;
                cleanup();
                return false;
            }
        } else {
            stagnation_count = 0;
        }

        prev_lambda = lambda;
    }

    eigenvalue = prev_lambda;
    cleanup();
    return false;
}

}  // namespace

EigenResult solveGeneralizedEigen(
    BlockSparseMatrix& A,
    BlockSparseMatrix& B,
    int num_eigenvalues,
    int max_iter,
    float tol) {
    EigenResult result;
    result.eigenvalues.reserve(num_eigenvalues);
    result.eigenvectors.reserve(num_eigenvalues);
    result.converged = true;
    result.iterations = 0;

    auto configA = A.getConfig();
    auto configB = B.getConfig();

    if (configA.num_rows != configB.num_rows || configA.num_cols != configB.num_cols) {
        std::cerr << "A and B should have same size" << std::endl;
        result.converged = false;
        return result;
    }

    std::cout << "Solving A*v = λ*B*v" << std::endl;
    std::cout << "Size: " << configA.num_rows << " x " << configA.num_cols << std::endl;
    std::cout << "Searching for " << num_eigenvalues << " eigen values..." << std::endl;
    // FIXME: We need to find eigens at arbitrary locations, for now we just do shifts P.A.

    for (int k = 0; k < num_eigenvalues; ++k) {
        float eigenvalue;
        std::vector<float> eigenvector;

        float shift = 0.0f;
        if (k > 0) {
            shift = result.eigenvalues[k - 1] + 0.01f;
        }

        std::cout << "  Вычисление λ[" << k << "]... " << std::flush;
        bool converged = inverseIterationWithShift(A, B, shift, eigenvalue, eigenvector,
                                                   result.eigenvectors, max_iter, tol);

        if (converged) {
            std::cout << eigenvalue << " ✓" << std::endl;
        } else {
            std::cout << eigenvalue << " ✗" << std::endl;
            result.converged = false;
        }

        result.eigenvalues.push_back(eigenvalue);
        result.eigenvectors.push_back(eigenvector);
    }

    std::vector<std::pair<float, int>> sorted_indices;
    for (size_t i = 0; i < result.eigenvalues.size(); ++i) {
        sorted_indices.push_back({result.eigenvalues[i], i});
    }
    std::sort(sorted_indices.begin(), sorted_indices.end());

    std::vector<float> sorted_eigenvalues;
    std::vector<std::vector<float>> sorted_eigenvectors;

    for (const auto& p : sorted_indices) {
        sorted_eigenvalues.push_back(result.eigenvalues[p.second]);
        sorted_eigenvectors.push_back(result.eigenvectors[p.second]);
    }

    result.eigenvalues = sorted_eigenvalues;
    result.eigenvectors = sorted_eigenvectors;

    return result;
}

EigenResult solveEigen(
    BlockSparseMatrix& A,
    int num_eigenvalues,
    int max_iter,
    float tol) {
    auto config = A.getConfig();
    BlockSparseMatrixConfig identity_config = config;
    identity_config.num_nonzero_blocks = (config.num_rows + config.block_size - 1) / config.block_size;

    BlockSparseMatrix I(identity_config);

    int num_blocks = identity_config.num_nonzero_blocks;
    std::vector<int> block_rows(num_blocks);
    std::vector<int> block_cols(num_blocks);
    std::vector<float> block_data(num_blocks * config.block_size * config.block_size, 0.0f);

    for (int i = 0; i < num_blocks; ++i) {
        block_rows[i] = i;
        block_cols[i] = i;
        for (int j = 0; j < config.block_size; ++j) {
            int idx = i * config.block_size * config.block_size + j * config.block_size + j;
            if (idx < (int)block_data.size()) {
                block_data[idx] = 1.0f;
            }
        }
    }

    I.initialize(block_rows, block_cols, block_data);

    return solveGeneralizedEigen(A, I, num_eigenvalues, max_iter, tol);
}

}  // namespace bsmp
