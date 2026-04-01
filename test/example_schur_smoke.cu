#include <cuda_runtime.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "piezo_block_system.h"
#include "piezo_scaling.h"
#include "schur_operator.h"

namespace {

float max_abs_difference(const std::vector<float>& a, const std::vector<float>& b) {
    float result = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        result = std::max(result, std::fabs(a[i] - b[i]));
    }
    return result;
}

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

void print_diag_summary(const std::vector<float>& dense, int n, const char* label) {
    std::cout << label << " diagonal:";
    for (int i = 0; i < n; ++i) {
        std::cout << ' ' << dense[i * n + i];
    }
    std::cout << std::endl;
}

}  // namespace

int main() {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "=== Schur complement smoke test ===" << std::endl;

    const int mechanical_dofs = 6;
    const int electrical_dofs = 2;

    const std::vector<float> c_u = {
        4.0f, 1.0f, 0.0f, 0.2f, 0.0f, 0.0f,
        1.0f, 5.0f, 0.1f, 0.0f, 0.3f, 0.0f,
        0.0f, 0.1f, 6.0f, 0.0f, 0.0f, 0.2f,
        0.2f, 0.0f, 0.0f, 7.0f, 1.0f, 0.0f,
        0.0f, 0.3f, 0.0f, 1.0f, 8.0f, 0.4f,
        0.0f, 0.0f, 0.2f, 0.0f, 0.4f, 9.0f,
    };

    const std::vector<float> c_uphi = {
        0.50f, 0.20f,
        0.10f, 0.00f,
        0.00f, 0.15f,
        0.25f, 0.05f,
        0.00f, 0.30f,
        0.20f, 0.10f,
    };

    const std::vector<float> c_phi = {
        2.0f, -2.0f,
        -2.0f, 2.0f,
    };

    const std::vector<float> m = {
        2.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 2.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 2.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 3.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 3.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 3.0f,
    };

    bsmp::PiezoBlockSystem system = bsmp::PiezoBlockSystem::fromDense(
        c_u, c_uphi, c_phi, m, mechanical_dofs, electrical_dofs);

    if (!bsmp::apply_grounding_constraint(system.dielectric(), 0)) {
        std::cerr << "Failed to apply grounding constraint" << std::endl;
        return 1;
    }

    const std::vector<float> grounded_phi = bsmp::HostBlockMatrix::fromMatrix(system.dielectric()).toDense();
    if (std::fabs(grounded_phi[0] - 1.0f) > 1e-6f ||
        std::fabs(grounded_phi[1]) > 1e-6f ||
        std::fabs(grounded_phi[2]) > 1e-6f) {
        std::cerr << "Grounding did not zero row/column as expected" << std::endl;
        return 1;
    }

    const bsmp::PiezoEquilibrationScaling scaling = bsmp::build_equilibration_scaling(system);
    bsmp::PiezoBlockSystem scaled = bsmp::scale_piezo_system(system, scaling);

    const std::vector<float> scaled_c_u = bsmp::HostBlockMatrix::fromMatrix(scaled.mechanicalStiffness()).toDense();
    const std::vector<float> scaled_c_phi = bsmp::HostBlockMatrix::fromMatrix(scaled.dielectric()).toDense();

    print_diag_summary(scaled_c_u, mechanical_dofs, "Scaled C_u");
    print_diag_summary(scaled_c_phi, electrical_dofs, "Scaled C_phi");

    for (int i = 0; i < mechanical_dofs; ++i) {
        if (std::fabs(scaled_c_u[i * mechanical_dofs + i] - 1.0f) > 1e-5f) {
            std::cerr << "Mechanical scaling failed at diagonal entry " << i << std::endl;
            return 1;
        }
    }

    for (int i = 0; i < electrical_dofs; ++i) {
        if (std::fabs(scaled_c_phi[i * electrical_dofs + i] - 1.0f) > 1e-5f) {
            std::cerr << "Electrical scaling failed at diagonal entry " << i << std::endl;
            return 1;
        }
    }

    bsmp::SchurOperator schur(scaled);
    const std::vector<float> schur_dense = schur.explicitDenseSchur();

    const std::vector<float> x = {1.0f, -0.5f, 0.25f, 0.75f, -1.0f, 0.5f};
    const std::vector<float> expected = multiply_dense(schur_dense, mechanical_dofs, mechanical_dofs, x);

    float* d_x = nullptr;
    float* d_y = nullptr;
    cudaMalloc(&d_x, mechanical_dofs * sizeof(float));
    cudaMalloc(&d_y, mechanical_dofs * sizeof(float));
    cudaMemcpy(d_x, x.data(), mechanical_dofs * sizeof(float), cudaMemcpyHostToDevice);

    schur.apply(d_x, d_y);
    cudaDeviceSynchronize();

    std::vector<float> actual(mechanical_dofs, 0.0f);
    cudaMemcpy(actual.data(), d_y, mechanical_dofs * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);

    const float error = max_abs_difference(expected, actual);
    std::cout << "Schur operator max |expected - actual| = " << error << std::endl;
    if (error > 5e-5f) {
        std::cerr << "Schur operator mismatch is too large" << std::endl;
        return 1;
    }

    std::cout << "Smoke test passed: grounding, balancing, and Schur apply are consistent." << std::endl;
    return 0;
}