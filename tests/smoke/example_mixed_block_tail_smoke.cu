#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "piezo_block_system.h"

namespace {

float max_abs_difference(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector sizes must match");
    }
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

std::vector<float> multiply_dense_transpose(const std::vector<float>& matrix,
                                            int rows,
                                            int cols,
                                            const std::vector<float>& x) {
    std::vector<float> y(cols, 0.0f);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            y[col] += matrix[row * cols + col] * x[row];
        }
    }
    return y;
}

std::vector<float> apply_block_matrix(BlockSparseMatrix& matrix,
                                      const std::vector<float>& x,
                                      bool transpose) {
    float* d_x = nullptr;
    float* d_y = nullptr;
    const auto cfg = matrix.getConfig();
    const int in_size = transpose ? cfg.num_rows : cfg.num_cols;
    const int out_size = transpose ? cfg.num_cols : cfg.num_rows;

    cudaMalloc(&d_x, in_size * sizeof(float));
    cudaMalloc(&d_y, out_size * sizeof(float));
    cudaMemcpy(d_x, x.data(), in_size * sizeof(float), cudaMemcpyHostToDevice);

    if (transpose) {
        matrix.multiplyTranspose(d_x, d_y);
    } else {
        matrix.multiply(d_x, d_y);
    }
    cudaDeviceSynchronize();

    std::vector<float> y(out_size, 0.0f);
    cudaMemcpy(y.data(), d_y, out_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
    return y;
}

}  // namespace

int main() {
    try {
        std::cout << "=== Mixed block tail smoke test ===" << std::endl;

        constexpr int mechanical_dofs = 6;
        constexpr int electrical_dofs = 4;  // deliberately not a multiple of 3

        const std::vector<float> c_u = {
            6.0f, 0.2f, 0.0f, -1.0f, 0.0f, 0.0f,
            0.2f, 7.0f, 0.3f, 0.0f, -1.1f, 0.0f,
            0.0f, 0.3f, 8.0f, 0.0f, 0.0f, -0.9f,
            -1.0f, 0.0f, 0.0f, 6.5f, 0.1f, 0.0f,
            0.0f, -1.1f, 0.0f, 0.1f, 7.5f, 0.4f,
            0.0f, 0.0f, -0.9f, 0.0f, 0.4f, 8.5f,
        };

        const std::vector<float> c_uphi = {
            0.50f, 0.10f, 0.00f, 0.20f,
            0.00f, 0.15f, 0.05f, 0.00f,
            0.10f, 0.00f, 0.25f, 0.05f,
            0.20f, 0.00f, 0.10f, 0.30f,
            0.05f, 0.35f, 0.00f, 0.10f,
            0.00f, 0.20f, 0.15f, 0.00f,
        };

        const std::vector<float> c_phi = {
            3.0f, -0.4f, 0.0f, -0.2f,
            -0.4f, 2.5f, -0.3f, 0.0f,
            0.0f, -0.3f, 2.8f, -0.5f,
            -0.2f, 0.0f, -0.5f, 3.2f,
        };

        const std::vector<float> mass = {
            2.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 2.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 2.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 3.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 3.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 3.0f,
        };

        bsmp::PiezoBlockSystem system = bsmp::PiezoBlockSystem::fromDense(
            c_u, c_uphi, c_phi, mass, mechanical_dofs, electrical_dofs);

        const std::vector<float> electrical_x = {1.0f, -0.5f, 0.25f, 0.75f};
        const std::vector<float> mechanical_x = {0.5f, -1.0f, 0.75f, -0.25f, 1.25f, -0.5f};

        const std::vector<float> cuphi_expected = multiply_dense(c_uphi, mechanical_dofs, electrical_dofs, electrical_x);
        const std::vector<float> cuphi_actual = apply_block_matrix(system.coupling(), electrical_x, false);
        const float cuphi_error = max_abs_difference(cuphi_expected, cuphi_actual);
        std::cout << "C_uphi * x max error = " << cuphi_error << std::endl;
        if (cuphi_error > 1e-6f) {
            throw std::runtime_error("Block C_uphi matvec does not match dense reference for non-multiple-of-3 electrical DOFs");
        }

        const std::vector<float> cuphi_t_expected = multiply_dense_transpose(c_uphi, mechanical_dofs, electrical_dofs, mechanical_x);
        const std::vector<float> cuphi_t_actual = apply_block_matrix(system.coupling(), mechanical_x, true);
        const float cuphi_t_error = max_abs_difference(cuphi_t_expected, cuphi_t_actual);
        std::cout << "C_uphi^T * x max error = " << cuphi_t_error << std::endl;
        if (cuphi_t_error > 1e-6f) {
            throw std::runtime_error("Block C_uphi transpose matvec does not match dense reference for non-multiple-of-3 electrical DOFs");
        }

        const std::vector<float> cphi_expected = multiply_dense(c_phi, electrical_dofs, electrical_dofs, electrical_x);
        const std::vector<float> cphi_actual = apply_block_matrix(system.dielectric(), electrical_x, false);
        const float cphi_error = max_abs_difference(cphi_expected, cphi_actual);
        std::cout << "C_phi * x max error = " << cphi_error << std::endl;
        if (cphi_error > 1e-6f) {
            throw std::runtime_error("Block C_phi matvec does not match dense reference for non-multiple-of-3 electrical DOFs");
        }

        const std::vector<float> cphi_roundtrip = bsmp::HostBlockMatrix::fromMatrix(system.dielectric()).toDense();
        const float roundtrip_error = max_abs_difference(c_phi, cphi_roundtrip);
        std::cout << "C_phi dense roundtrip max error = " << roundtrip_error << std::endl;
        if (roundtrip_error > 1e-6f) {
            throw std::runtime_error("Host/device block roundtrip corrupted C_phi with non-multiple-of-3 electrical DOFs");
        }

        std::cout << "Smoke test passed: artificial 3x3 packing of electrical tails preserves exact algebra for C_uphi and C_phi." << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Mixed block tail smoke test failed: " << error.what() << std::endl;
        return 1;
    }
}