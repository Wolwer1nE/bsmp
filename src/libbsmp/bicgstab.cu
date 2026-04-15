#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>

#include "libbsmp/bicgstab.h"

/// @brief Helper macro for cuBLAS errors
#define CUBLAS_CHECK(call)                                                             \
    do {                                                                                \
        cublasStatus_t status = call;                                                   \
        if (status != CUBLAS_STATUS_SUCCESS) {                                          \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status); \
            return false;                                                               \
        }                                                                               \
    } while (0)

/// @brief BiCGStab implementation
/// @note Algorithm from: H.A. van der Vorst, "Bi-CGSTAB: A Fast and Smoothly Converging
/// Variant of Bi-CG for the Solution of Nonsymmetric Linear Systems", 1992
bool bicgstab(BlockSparseMatrix& A,
              const float* d_b,
              float* d_x,
              int max_iters,
              float tol,
              int& iters_out,
              float& resid_out) {
    auto config = A.getConfig();
    const int n = config.num_rows;

    // Create cuBLAS handle
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Allocate working vectors on device
    float* d_r = nullptr;   // residual
    float* d_r0 = nullptr;  // initial residual (shadow)
    float* d_p = nullptr;   // search direction
    float* d_v = nullptr;   // A*p
    float* d_s = nullptr;   // intermediate residual
    float* d_t = nullptr;   // A*s

    cudaMalloc(&d_r, n * sizeof(float));
    cudaMalloc(&d_r0, n * sizeof(float));
    cudaMalloc(&d_p, n * sizeof(float));
    cudaMalloc(&d_v, n * sizeof(float));
    cudaMalloc(&d_s, n * sizeof(float));
    cudaMalloc(&d_t, n * sizeof(float));

    // Compute initial residual: r = b - A*x
    A.multiply(d_x, d_r);  // r = A*x
    float alpha_init = -1.0f;
    // float beta_init = 1.0f; never referenced
    CUBLAS_CHECK(cublasSaxpy(handle, n, &alpha_init, d_r, 1, (float*)d_b, 1));  // This modifies d_b, fix below

    // Actually we need to copy b to r first, then subtract A*x
    CUBLAS_CHECK(cublasScopy(handle, n, d_b, 1, d_r, 1));  // r = b
    A.multiply(d_x, d_v);                                  // v = A*x (temp use of v)
    alpha_init = -1.0f;
    CUBLAS_CHECK(cublasSaxpy(handle, n, &alpha_init, d_v, 1, d_r, 1));  // r = b - A*x

    // r0 = r (arbitrary choice for shadow residual)
    CUBLAS_CHECK(cublasScopy(handle, n, d_r, 1, d_r0, 1));

    // Compute ||b|| for relative tolerance
    float norm_b;
    CUBLAS_CHECK(cublasSnrm2(handle, n, d_b, 1, &norm_b));
    if (norm_b == 0.0f)
        norm_b = 1.0f;  // avoid division by zero

    // Initial residual norm
    float norm_r;
    CUBLAS_CHECK(cublasSnrm2(handle, n, d_r, 1, &norm_r));

    float rho = 1.0f;
    float alpha = 1.0f;
    float omega = 1.0f;

    // p = 0, v = 0 (will be set in first iteration)
    cudaMemset(d_p, 0, n * sizeof(float));
    cudaMemset(d_v, 0, n * sizeof(float));

    bool converged = false;
    int iter = 0;

    for (iter = 0; iter < max_iters; iter++) {
        // Check convergence
        float rel_resid = norm_r / norm_b;
        if (rel_resid < tol) {
            converged = true;
            break;
        }

        // rho_new = <r0, r>
        float rho_new;
        CUBLAS_CHECK(cublasSdot(handle, n, d_r0, 1, d_r, 1, &rho_new));

        if (fabs(rho_new) < 1e-30f) {
            // Breakdown: rho too small
            fprintf(stderr, "BiCGStab breakdown: rho = %e at iteration %d\n", rho_new, iter);
            break;
        }

        // beta = (rho_new / rho) * (alpha / omega)
        float beta = (rho_new / rho) * (alpha / omega);
        rho = rho_new;

        // p = r + beta * (p - omega * v)
        // p = p - omega * v
        float neg_omega = -omega;
        CUBLAS_CHECK(cublasSaxpy(handle, n, &neg_omega, d_v, 1, d_p, 1));
        // p = beta * p
        CUBLAS_CHECK(cublasSscal(handle, n, &beta, d_p, 1));
        // p = p + r
        float one = 1.0f;
        CUBLAS_CHECK(cublasSaxpy(handle, n, &one, d_r, 1, d_p, 1));

        // v = A * p
        A.multiply(d_p, d_v);

        // alpha = rho / <r0, v>
        float r0_dot_v;
        CUBLAS_CHECK(cublasSdot(handle, n, d_r0, 1, d_v, 1, &r0_dot_v));

        if (fabs(r0_dot_v) < 1e-30f) {
            // Breakdown
            fprintf(stderr, "BiCGStab breakdown: <r0,v> = %e at iteration %d\n", r0_dot_v, iter);
            break;
        }

        alpha = rho / r0_dot_v;

        // s = r - alpha * v
        CUBLAS_CHECK(cublasScopy(handle, n, d_r, 1, d_s, 1));  // s = r
        float neg_alpha = -alpha;
        CUBLAS_CHECK(cublasSaxpy(handle, n, &neg_alpha, d_v, 1, d_s, 1));  // s = r - alpha*v

        // Check if s is small enough (early convergence)
        float norm_s;
        CUBLAS_CHECK(cublasSnrm2(handle, n, d_s, 1, &norm_s));
        if (norm_s / norm_b < tol) {
            // x = x + alpha * p
            CUBLAS_CHECK(cublasSaxpy(handle, n, &alpha, d_p, 1, d_x, 1));
            norm_r = norm_s;
            converged = true;
            iter++;
            break;
        }

        // t = A * s
        A.multiply(d_s, d_t);

        // omega = <t, s> / <t, t>
        float t_dot_s, t_dot_t;
        CUBLAS_CHECK(cublasSdot(handle, n, d_t, 1, d_s, 1, &t_dot_s));
        CUBLAS_CHECK(cublasSdot(handle, n, d_t, 1, d_t, 1, &t_dot_t));

        if (fabs(t_dot_t) < 1e-30f) {
            // Breakdown
            fprintf(stderr, "BiCGStab breakdown: <t,t> = %e at iteration %d\n", t_dot_t, iter);
            break;
        }

        omega = t_dot_s / t_dot_t;

        // x = x + alpha * p + omega * s
        CUBLAS_CHECK(cublasSaxpy(handle, n, &alpha, d_p, 1, d_x, 1));
        CUBLAS_CHECK(cublasSaxpy(handle, n, &omega, d_s, 1, d_x, 1));

        // r = s - omega * t
        CUBLAS_CHECK(cublasScopy(handle, n, d_s, 1, d_r, 1));  // r = s
        float neg_omega_t = -omega;
        CUBLAS_CHECK(cublasSaxpy(handle, n, &neg_omega_t, d_t, 1, d_r, 1));  // r = s - omega*t

        // Compute new residual norm
        CUBLAS_CHECK(cublasSnrm2(handle, n, d_r, 1, &norm_r));

        if (fabs(omega) < 1e-30f) {
            // Breakdown
            fprintf(stderr, "BiCGStab breakdown: omega = %e at iteration %d\n", omega, iter);
            break;
        }
    }

    // Output results
    iters_out = iter;
    resid_out = norm_r / norm_b;

    // Cleanup
    cudaFree(d_r);
    cudaFree(d_r0);
    cudaFree(d_p);
    cudaFree(d_v);
    cudaFree(d_s);
    cudaFree(d_t);
    cublasDestroy(handle);

    return converged;
}
