#include <cuda_runtime.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

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

float dot_product(const std::vector<float>& lhs, const std::vector<float>& rhs) {
    float sum = 0.0f;
    for (size_t i = 0; i < lhs.size(); ++i) {
        sum += lhs[i] * rhs[i];
    }
    return sum;
}

std::vector<float> apply_preconditioner_host(bsmp::SAAMGPreconditioner& preconditioner,
                                             const std::vector<float>& rhs) {
    float *d_rhs = nullptr, *d_out = nullptr;
    cudaMalloc(&d_rhs, rhs.size() * sizeof(float));
    cudaMalloc(&d_out, rhs.size() * sizeof(float));
    cudaMemcpy(d_rhs, rhs.data(), rhs.size() * sizeof(float), cudaMemcpyHostToDevice);

    if (!preconditioner.apply(d_rhs, d_out)) {
        cudaFree(d_rhs);
        cudaFree(d_out);
        throw std::runtime_error("SA-AMG V-cycle apply() failed");
    }

    std::vector<float> out(rhs.size(), 0.0f);
    cudaMemcpy(out.data(), d_out, rhs.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_rhs);
    cudaFree(d_out);
    return out;
}

}  // namespace

int main() {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "=== SA-AMG two-level V-cycle smoke test ===" << std::endl;

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

    auto c_u = bsmp::HostBlockMatrix::fromDense(n, n, c_u_dense).toDeviceMatrix();
    const auto rbm = bsmp::build_rigid_body_modes(layout);

    bsmp::SAAMGParameters params;
    params.regularization_epsilon = 1e-6f;
    params.hierarchy.pre_sweeps = 2;
    params.hierarchy.post_sweeps = 2;
    params.hierarchy.jacobi_damping = 0.8f;
    params.hierarchy.prolongation_damping = 0.25f;

    auto preconditioner = bsmp::createSAAMGPreconditioner(*c_u, layout, rbm, params);
    if (!preconditioner) {
        std::cerr << "Failed to construct two-level SA-AMG preconditioner" << std::endl;
        return 1;
    }

    const std::vector<float> x_true = {1.0f, -1.0f, 0.5f, 0.5f, 0.75f, -0.25f,
                                       -0.5f, 1.25f, -0.75f, 0.25f, -0.5f, 1.0f};
    const std::vector<float> rhs = multiply_dense(c_u_dense, n, n, x_true);

    const std::vector<float> out = apply_preconditioner_host(*preconditioner, rhs);

    for (float value : out) {
        if (!std::isfinite(value)) {
            std::cerr << "Non-finite value produced by V-cycle" << std::endl;
            return 1;
        }
    }

    const std::vector<float> residual_vec = [&]() {
        std::vector<float> ax = multiply_dense(c_u_dense, n, n, out);
        std::vector<float> residual(n, 0.0f);
        for (int i = 0; i < n; ++i) {
            residual[i] = rhs[i] - ax[i];
        }
        return residual;
    }();

    const float rhs_norm = vector_norm(rhs);
    const float residual_norm = vector_norm(residual_vec);
    const float relative_residual = residual_norm / (rhs_norm > 0.0f ? rhs_norm : 1.0f);
    std::cout << "relative residual after one V-cycle = " << relative_residual << std::endl;
    if (relative_residual > 0.35f) {
        std::cerr << "Two-level V-cycle did not reduce the residual enough" << std::endl;
        return 1;
    }

    const std::vector<std::vector<float>> probes = {
        rhs,
        x_true,
        {1.0f, 0.25f, -0.5f, -1.0f, 0.75f, 0.5f, -0.75f, 1.5f, -1.25f, 0.2f, -0.3f, 0.8f},
    };
    for (size_t probe_index = 0; probe_index < probes.size(); ++probe_index) {
        const std::vector<float> image = apply_preconditioner_host(*preconditioner, probes[probe_index]);
        const float energy = dot_product(probes[probe_index], image);
        std::cout << "probe " << probe_index << " energy = " << energy << std::endl;
        if (!(energy > 0.0f) || !std::isfinite(energy)) {
            std::cerr << "SA-AMG preconditioner produced a non-positive probe energy" << std::endl;
            return 1;
        }
    }

    const std::vector<float> u = {1.0f, -0.75f, 0.25f, 0.5f, -0.6f, 0.8f, -1.1f, 0.7f, -0.4f, 0.3f, -0.2f, 0.9f};
    const std::vector<float> v = {-0.4f, 1.2f, -0.7f, 0.6f, 0.1f, -0.9f, 0.5f, -0.8f, 1.1f, -0.3f, 0.4f, -0.2f};
    const std::vector<float> mu = apply_preconditioner_host(*preconditioner, u);
    const std::vector<float> mv = apply_preconditioner_host(*preconditioner, v);
    const float uv = dot_product(u, mv);
    const float vu = dot_product(v, mu);
    const float symmetry_scale = std::max(1.0f, std::max(std::fabs(uv), std::fabs(vu)));
    const float symmetry_error = std::fabs(uv - vu) / symmetry_scale;
    std::cout << "symmetry relative error = " << symmetry_error << std::endl;
    if (symmetry_error > 0.25f) {
        std::cerr << "SA-AMG preconditioner is too non-symmetric for PCG-style use on the toy problem" << std::endl;
        return 1;
    }

    std::cout << "Smoke test passed: two-level SA-AMG V-cycle acts as a usable reference preconditioner and preserves basic PCG-friendly positivity/symmetry checks." << std::endl;
    return 0;
}