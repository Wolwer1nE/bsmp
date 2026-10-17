#include <cuda_runtime.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "chebyshev_smoother.h"
#include "elastic_nullspace.h"
#include "piezo_block_system.h"
#include "sa_amg_preconditioner.h"

namespace {

std::vector<float> multiply_dense(const std::vector<float>& matrix,
                                  int rows,
                                  int cols,
                                  const std::vector<float>& x) {
    std::vector<float> y(rows, 0.0f);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            y[row] += matrix[row * cols + col] * x[col];
        }
    }
    return y;
}

float vector_norm(const std::vector<float>& v) {
    float sum = 0.0f;
    for (float value : v) {
        sum += value * value;
    }
    return std::sqrt(sum);
}

std::vector<float> build_inverse_diagonal(const std::vector<float>& dense, int n) {
    std::vector<float> inverse_diagonal(n, 1.0f);
    for (int i = 0; i < n; ++i) {
        inverse_diagonal[i] = 1.0f / dense[i * n + i];
    }
    return inverse_diagonal;
}

std::vector<float> residual_of(const std::vector<float>& matrix,
                               int n,
                               const std::vector<float>& rhs,
                               const std::vector<float>& x) {
    const std::vector<float> ax = multiply_dense(matrix, n, n, x);
    std::vector<float> residual(n, 0.0f);
    for (int i = 0; i < n; ++i) {
        residual[i] = rhs[i] - ax[i];
    }
    return residual;
}

}  // namespace

int main() {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "=== SA-AMG Chebyshev smoother smoke test ===" << std::endl;

    const std::vector<float> coordinates = {
        0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        1.0f, 1.0f, 0.0f,
    };
    const bsmp::NodeLayout layout = bsmp::NodeLayout::fromNodeCoordinates(coordinates, false);

    const int n = 12;
    const std::vector<float> c_u_dense = {
        6,0,0,-2,0,0,-2,0,0,0,0,0,
        0,6,0,0,-2,0,0,-2,0,0,0,0,
        0,0,6,0,0,-2,0,0,-2,0,0,0,
        -2,0,0,6,0,0,0,0,0,-2,0,0,
        0,-2,0,0,6,0,0,0,0,0,-2,0,
        0,0,-2,0,0,6,0,0,0,0,0,-2,
        -2,0,0,0,0,0,6,0,0,-2,0,0,
        0,-2,0,0,0,0,0,6,0,0,-2,0,
        0,0,-2,0,0,0,0,0,6,0,0,-2,
        0,0,0,-2,0,0,-2,0,0,6,0,0,
        0,0,0,0,-2,0,0,-2,0,0,6,0,
        0,0,0,0,0,-2,0,0,-2,0,0,6,
    };

    const auto inverse_diagonal = build_inverse_diagonal(c_u_dense, n);
    const std::vector<float> x_true = {1.0f, -1.0f, 0.5f, 0.5f, 0.75f, -0.25f,
                                       -0.5f, 1.25f, -0.75f, 0.25f, -0.5f, 1.0f};
    const std::vector<float> rhs = multiply_dense(c_u_dense, n, n, x_true);

    bsmp::ChebyshevSmootherParameters chebyshev;
    chebyshev.steps = 3;
    chebyshev.power_iterations = 10;
    chebyshev.lower_bound_ratio = 0.25f;
    chebyshev.eigenvalue_safety = 1.05f;

    const float lambda_max = bsmp::estimate_max_eigenvalue(c_u_dense, inverse_diagonal, n, chebyshev);
    if (!(lambda_max > 0.0f) || !std::isfinite(lambda_max)) {
        std::cerr << "Chebyshev lambda_max estimate is invalid" << std::endl;
        return 1;
    }

    std::vector<float> x_smoothed(n, 0.0f);
    const float initial_residual = vector_norm(rhs);
    bsmp::chebyshev_smooth(c_u_dense, inverse_diagonal, rhs, x_smoothed, lambda_max, chebyshev);
    const float smoothed_residual = vector_norm(residual_of(c_u_dense, n, rhs, x_smoothed));

    std::cout << "estimated lambda_max(D^-1 A) = " << lambda_max << std::endl;
    std::cout << "relative residual after standalone smoothing = "
              << smoothed_residual / (initial_residual > 0.0f ? initial_residual : 1.0f)
              << std::endl;
    if (!(smoothed_residual < initial_residual)) {
        std::cerr << "Standalone Chebyshev smoothing did not reduce residual" << std::endl;
        return 1;
    }

    auto c_u = bsmp::HostBlockMatrix::fromDense(n, n, c_u_dense).toDeviceMatrix();
    const auto rbm = bsmp::build_rigid_body_modes(layout);

    bsmp::SAAMGParameters params;
    params.regularization_epsilon = 1e-6f;
    params.hierarchy.pre_sweeps = 1;
    params.hierarchy.post_sweeps = 1;
    params.hierarchy.prolongation_damping = 0.25f;
    params.hierarchy.use_chebyshev = true;
    params.hierarchy.chebyshev = chebyshev;

    auto preconditioner = bsmp::createSAAMGPreconditioner(*c_u, layout, rbm, params);
    if (!preconditioner) {
        std::cerr << "Failed to construct Chebyshev-backed SA-AMG preconditioner" << std::endl;
        return 1;
    }

    float *d_rhs = nullptr, *d_out = nullptr;
    cudaMalloc(&d_rhs, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));
    cudaMemcpy(d_rhs, rhs.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    if (!preconditioner->apply(d_rhs, d_out)) {
        std::cerr << "SA-AMG Chebyshev V-cycle apply() failed" << std::endl;
        cudaFree(d_rhs);
        cudaFree(d_out);
        return 1;
    }

    std::vector<float> out(n, 0.0f);
    cudaMemcpy(out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_rhs);
    cudaFree(d_out);

    for (float value : out) {
        if (!std::isfinite(value)) {
            std::cerr << "Non-finite value produced by Chebyshev V-cycle" << std::endl;
            return 1;
        }
    }

    const float vcycle_residual = vector_norm(residual_of(c_u_dense, n, rhs, out));
    const float relative_vcycle_residual = vcycle_residual / (initial_residual > 0.0f ? initial_residual : 1.0f);
    std::cout << "relative residual after Chebyshev V-cycle = " << relative_vcycle_residual << std::endl;
    if (relative_vcycle_residual > 0.45f) {
        std::cerr << "Chebyshev-backed two-level V-cycle did not reduce the residual enough" << std::endl;
        return 1;
    }

    std::cout << "Smoke test passed: Chebyshev smoothing is wired into the reference SA-AMG path." << std::endl;
    return 0;
}