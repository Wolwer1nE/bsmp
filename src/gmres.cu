#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <vector>

#include "gmres.h"

#define CUBLAS_CHECK(call)                                                              \
    do {                                                                                \
        cublasStatus_t status = call;                                                   \
        if (status != CUBLAS_STATUS_SUCCESS) {                                          \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status); \
            return false;                                                               \
        }                                                                               \
    } while (0)

bool gmres(BlockSparseMatrix& A,
           const float* d_b,
           float* d_x,
           int max_iters,
           int restart,
           float tol,
           int& iters_out,
           float& resid_out) {
    PreconditionerOptions options;
    return gmres(A, d_b, d_x, max_iters, restart, tol, iters_out, resid_out, options);
}

bool gmres(BlockSparseMatrix& A,
           const float* d_b,
           float* d_x,
           int max_iters,
           int restart,
           float tol,
           int& iters_out,
           float& resid_out,
           const PreconditionerOptions& preconditioner_options) {
    auto config = A.getConfig();
    const int n = config.num_rows;
    restart = std::max(1, std::min(restart, max_iters));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    float* d_r = nullptr;
    float* d_Av = nullptr;
    float* d_w = nullptr;
    cudaMalloc(&d_r, n * sizeof(float));
    cudaMalloc(&d_Av, n * sizeof(float));
    cudaMalloc(&d_w, n * sizeof(float));

    std::vector<float*> d_basis(restart + 1, nullptr);
    for (float*& ptr : d_basis) {
        cudaMalloc(&ptr, n * sizeof(float));
    }

    auto cleanup = [&]() {
        for (float* ptr : d_basis) {
            if (ptr != nullptr) {
                cudaFree(ptr);
            }
        }
        if (d_r != nullptr)
            cudaFree(d_r);
        if (d_Av != nullptr)
            cudaFree(d_Av);
        if (d_w != nullptr)
            cudaFree(d_w);
        cublasDestroy(handle);
    };

    std::unique_ptr<LinearPreconditioner> preconditioner =
        createPreconditioner(A, preconditioner_options);
    if (!preconditioner) {
        cleanup();
        return false;
    }

    float norm_b = 0.0f;
    CUBLAS_CHECK(cublasSnrm2(handle, n, d_b, 1, &norm_b));
    if (norm_b == 0.0f)
        norm_b = 1.0f;

    auto compute_true_residual = [&]() -> float {
        cublasStatus_t status = cublasScopy(handle, n, d_b, 1, d_r, 1);
        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status);
            resid_out = INFINITY;
            return INFINITY;
        }
        A.multiply(d_x, d_Av);
        float minus_one = -1.0f;
        status = cublasSaxpy(handle, n, &minus_one, d_Av, 1, d_r, 1);
        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status);
            resid_out = INFINITY;
            return INFINITY;
        }
        float norm_r = 0.0f;
        status = cublasSnrm2(handle, n, d_r, 1, &norm_r);
        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status);
            resid_out = INFINITY;
            return INFINITY;
        }
        resid_out = norm_r / norm_b;
        return norm_r;
    };

    float norm_r = compute_true_residual();
    if (resid_out < tol) {
        iters_out = 0;
        cleanup();
        return true;
    }

    if (!preconditioner->apply(d_r, d_basis[0])) {
        cleanup();
        return false;
    }

    float beta = 0.0f;
    CUBLAS_CHECK(cublasSnrm2(handle, n, d_basis[0], 1, &beta));
    if (beta == 0.0f)
        beta = norm_r;
    if (beta > 0.0f) {
        float inv_beta = 1.0f / beta;
        CUBLAS_CHECK(cublasSscal(handle, n, &inv_beta, d_basis[0], 1));
    }

    bool converged = false;
    int total_iters = 0;

    while (total_iters < max_iters) {
        std::vector<std::vector<float>> H(restart + 1, std::vector<float>(restart, 0.0f));
        std::vector<float> cs(restart, 0.0f);
        std::vector<float> sn(restart, 0.0f);
        std::vector<float> g(restart + 1, 0.0f);
        g[0] = beta;

        int inner_iters = 0;
        for (int j = 0; j < restart && total_iters < max_iters; ++j) {
            A.multiply(d_basis[j], d_Av);
            if (!preconditioner->apply(d_Av, d_w)) {
                cleanup();
                return false;
            }

            for (int i = 0; i <= j; ++i) {
                CUBLAS_CHECK(cublasSdot(handle, n, d_basis[i], 1, d_w, 1, &H[i][j]));
                float neg_h = -H[i][j];
                CUBLAS_CHECK(cublasSaxpy(handle, n, &neg_h, d_basis[i], 1, d_w, 1));
            }

            CUBLAS_CHECK(cublasSnrm2(handle, n, d_w, 1, &H[j + 1][j]));
            if (H[j + 1][j] > 1e-30f) {
                CUBLAS_CHECK(cublasScopy(handle, n, d_w, 1, d_basis[j + 1], 1));
                float inv_norm = 1.0f / H[j + 1][j];
                CUBLAS_CHECK(cublasSscal(handle, n, &inv_norm, d_basis[j + 1], 1));
            } else {
                cudaMemset(d_basis[j + 1], 0, n * sizeof(float));
            }

            for (int i = 0; i < j; ++i) {
                float temp = cs[i] * H[i][j] + sn[i] * H[i + 1][j];
                H[i + 1][j] = -sn[i] * H[i][j] + cs[i] * H[i + 1][j];
                H[i][j] = temp;
            }

            float denom = std::sqrt(H[j][j] * H[j][j] + H[j + 1][j] * H[j + 1][j]);
            if (denom > 1e-30f) {
                cs[j] = H[j][j] / denom;
                sn[j] = H[j + 1][j] / denom;
            } else {
                cs[j] = 1.0f;
                sn[j] = 0.0f;
            }

            H[j][j] = cs[j] * H[j][j] + sn[j] * H[j + 1][j];
            H[j + 1][j] = 0.0f;

            float g_next = -sn[j] * g[j];
            g[j] = cs[j] * g[j];
            g[j + 1] = g_next;

            ++total_iters;
            inner_iters = j + 1;

            if (std::fabs(g[j + 1]) <= tol * beta) {
                break;
            }
        }

        if (inner_iters == 0)
            break;

        std::vector<float> y(inner_iters, 0.0f);
        for (int i = inner_iters - 1; i >= 0; --i) {
            float sum = g[i];
            for (int k = i + 1; k < inner_iters; ++k) {
                sum -= H[i][k] * y[k];
            }
            y[i] = std::fabs(H[i][i]) > 1e-30f ? (sum / H[i][i]) : 0.0f;
        }

        for (int i = 0; i < inner_iters; ++i) {
            if (std::fabs(y[i]) > 0.0f) {
                CUBLAS_CHECK(cublasSaxpy(handle, n, &y[i], d_basis[i], 1, d_x, 1));
            }
        }

        norm_r = compute_true_residual();
        if (resid_out < tol) {
            converged = true;
            break;
        }

        if (!preconditioner->apply(d_r, d_basis[0])) {
            cleanup();
            return false;
        }
        CUBLAS_CHECK(cublasSnrm2(handle, n, d_basis[0], 1, &beta));
        if (beta <= 1e-30f) {
            converged = true;
            resid_out = 0.0f;
            break;
        }
        float inv_beta = 1.0f / beta;
        CUBLAS_CHECK(cublasSscal(handle, n, &inv_beta, d_basis[0], 1));
    }

    iters_out = total_iters;
    if (!converged) {
        norm_r = compute_true_residual();
        (void)norm_r;
    }

    cleanup();
    return converged;
}
