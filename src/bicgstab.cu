#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <memory>

#include "bicgstab.h"

#define CUBLAS_CHECK(call)                                                              \
    do {                                                                                \
        cublasStatus_t status = call;                                                   \
        if (status != CUBLAS_STATUS_SUCCESS) {                                          \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status); \
            return false;                                                               \
        }                                                                               \
    } while (0)

bool bicgstab(BlockSparseMatrix& A,
              const float* d_b,
              float* d_x,
              int max_iters,
              float tol,
              int& iters_out,
              float& resid_out) {
    PreconditionerOptions options;
    return bicgstab(A, d_b, d_x, max_iters, tol, iters_out, resid_out, options);
}

bool bicgstab(BlockSparseMatrix& A,
              const float* d_b,
              float* d_x,
              int max_iters,
              float tol,
              int& iters_out,
              float& resid_out,
              const PreconditionerOptions& preconditioner_options) {
    auto config = A.getConfig();
    const int n = config.num_rows;

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    float* d_r = nullptr;
    float* d_r0 = nullptr;
    float* d_p = nullptr;
    float* d_v = nullptr;
    float* d_s = nullptr;
    float* d_t = nullptr;
    float* d_y = nullptr;
    float* d_z = nullptr;
    cudaMalloc(&d_r, n * sizeof(float));
    cudaMalloc(&d_r0, n * sizeof(float));
    cudaMalloc(&d_p, n * sizeof(float));
    cudaMalloc(&d_v, n * sizeof(float));
    cudaMalloc(&d_s, n * sizeof(float));
    cudaMalloc(&d_t, n * sizeof(float));
    cudaMalloc(&d_y, n * sizeof(float));
    cudaMalloc(&d_z, n * sizeof(float));

    auto cleanup = [&]() {
        if (d_r != nullptr)
            cudaFree(d_r);
        if (d_r0 != nullptr)
            cudaFree(d_r0);
        if (d_p != nullptr)
            cudaFree(d_p);
        if (d_v != nullptr)
            cudaFree(d_v);
        if (d_s != nullptr)
            cudaFree(d_s);
        if (d_t != nullptr)
            cudaFree(d_t);
        if (d_y != nullptr)
            cudaFree(d_y);
        if (d_z != nullptr)
            cudaFree(d_z);
        cublasDestroy(handle);
    };

    std::unique_ptr<LinearPreconditioner> preconditioner =
        createPreconditioner(A, preconditioner_options);
    if (!preconditioner) {
        cleanup();
        return false;
    }

    CUBLAS_CHECK(cublasScopy(handle, n, d_b, 1, d_r, 1));
    A.multiply(d_x, d_v);
    float minus_one = -1.0f;
    CUBLAS_CHECK(cublasSaxpy(handle, n, &minus_one, d_v, 1, d_r, 1));
    CUBLAS_CHECK(cublasScopy(handle, n, d_r, 1, d_r0, 1));

    float norm_b = 0.0f;
    CUBLAS_CHECK(cublasSnrm2(handle, n, d_b, 1, &norm_b));
    if (norm_b == 0.0f)
        norm_b = 1.0f;

    float norm_r = 0.0f;
    CUBLAS_CHECK(cublasSnrm2(handle, n, d_r, 1, &norm_r));

    float rho = 1.0f;
    float alpha = 1.0f;
    float omega = 1.0f;

    cudaMemset(d_p, 0, n * sizeof(float));
    cudaMemset(d_v, 0, n * sizeof(float));

    bool converged = false;
    int iter = 0;

    for (iter = 0; iter < max_iters; ++iter) {
        const float rel_resid = norm_r / norm_b;
        if (rel_resid < tol) {
            converged = true;
            break;
        }

        float rho_new = 0.0f;
        CUBLAS_CHECK(cublasSdot(handle, n, d_r0, 1, d_r, 1, &rho_new));
        if (std::fabs(rho_new) < 1e-30f) {
            fprintf(stderr, "BiCGStab breakdown: rho = %e at iteration %d\n", rho_new, iter);
            break;
        }

        const float beta = (rho_new / rho) * (alpha / omega);
        rho = rho_new;

        const float neg_omega = -omega;
        CUBLAS_CHECK(cublasSaxpy(handle, n, &neg_omega, d_v, 1, d_p, 1));
        CUBLAS_CHECK(cublasSscal(handle, n, &beta, d_p, 1));
        const float one = 1.0f;
        CUBLAS_CHECK(cublasSaxpy(handle, n, &one, d_r, 1, d_p, 1));

        if (!preconditioner->apply(d_p, d_y)) {
            cleanup();
            return false;
        }
        A.multiply(d_y, d_v);

        float r0_dot_v = 0.0f;
        CUBLAS_CHECK(cublasSdot(handle, n, d_r0, 1, d_v, 1, &r0_dot_v));
        if (std::fabs(r0_dot_v) < 1e-30f) {
            fprintf(stderr, "BiCGStab breakdown: <r0,v> = %e at iteration %d\n", r0_dot_v, iter);
            break;
        }

        alpha = rho / r0_dot_v;

        CUBLAS_CHECK(cublasScopy(handle, n, d_r, 1, d_s, 1));
        const float neg_alpha = -alpha;
        CUBLAS_CHECK(cublasSaxpy(handle, n, &neg_alpha, d_v, 1, d_s, 1));

        float norm_s = 0.0f;
        CUBLAS_CHECK(cublasSnrm2(handle, n, d_s, 1, &norm_s));
        if (norm_s / norm_b < tol) {
            CUBLAS_CHECK(cublasSaxpy(handle, n, &alpha, d_y, 1, d_x, 1));
            norm_r = norm_s;
            converged = true;
            ++iter;
            break;
        }

        if (!preconditioner->apply(d_s, d_z)) {
            cleanup();
            return false;
        }
        A.multiply(d_z, d_t);

        float t_dot_s = 0.0f;
        float t_dot_t = 0.0f;
        CUBLAS_CHECK(cublasSdot(handle, n, d_t, 1, d_s, 1, &t_dot_s));
        CUBLAS_CHECK(cublasSdot(handle, n, d_t, 1, d_t, 1, &t_dot_t));
        if (std::fabs(t_dot_t) < 1e-30f) {
            fprintf(stderr, "BiCGStab breakdown: <t,t> = %e at iteration %d\n", t_dot_t, iter);
            break;
        }

        omega = t_dot_s / t_dot_t;
        CUBLAS_CHECK(cublasSaxpy(handle, n, &alpha, d_y, 1, d_x, 1));
        CUBLAS_CHECK(cublasSaxpy(handle, n, &omega, d_z, 1, d_x, 1));

        CUBLAS_CHECK(cublasScopy(handle, n, d_s, 1, d_r, 1));
        const float neg_omega_t = -omega;
        CUBLAS_CHECK(cublasSaxpy(handle, n, &neg_omega_t, d_t, 1, d_r, 1));
        CUBLAS_CHECK(cublasSnrm2(handle, n, d_r, 1, &norm_r));

        if (std::fabs(omega) < 1e-30f) {
            fprintf(stderr, "BiCGStab breakdown: omega = %e at iteration %d\n", omega, iter);
            break;
        }
    }

    iters_out = iter;
    resid_out = norm_r / norm_b;

    cleanup();
    return converged;
}
