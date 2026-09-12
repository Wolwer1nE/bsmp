#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "elastic_nullspace.h"
#include "piezo_block_system.h"
#include "sa_amg_hierarchy.h"

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

float max_abs_difference(const std::vector<float>& a, const std::vector<float>& b) {
    float result = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        result = std::max(result, std::fabs(a[i] - b[i]));
    }
    return result;
}

}  // namespace

int main() {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "=== SA-AMG smoothed prolongator smoke test ===" << std::endl;

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
    const auto aggregation = bsmp::aggregateNodes(*c_u, layout);
    const auto tentative = bsmp::build_tentative_prolongator(aggregation, rbm);
    const auto smoothed = bsmp::build_smoothed_prolongator(*c_u, tentative, 0.25f);

    std::cout << "Tentative nnz: " << tentative.nnz() << ", smoothed nnz: " << smoothed.nnz() << std::endl;
    if (smoothed.num_rows != tentative.num_rows || smoothed.num_cols != tentative.num_cols) {
        std::cerr << "Smoothed prolongator dimensions changed unexpectedly" << std::endl;
        return 1;
    }

    const std::vector<float> p_tentative = tentative.toDense();
    const std::vector<float> p_smoothed = smoothed.toDense();
    const float delta = max_abs_difference(p_tentative, p_smoothed);
    std::cout << "max |P_tentative - P_smoothed| = " << delta << std::endl;
    if (delta <= 1e-6f) {
        std::cerr << "Smoothing did not change the prolongator" << std::endl;
        return 1;
    }

    for (size_t mode_index = 0; mode_index < rbm.size(); ++mode_index) {
        std::vector<float> coarse(smoothed.num_cols, 0.0f);
        for (int col = 0; col < smoothed.num_cols; ++col) {
            float coeff = 0.0f;
            for (int row = 0; row < smoothed.num_rows; ++row) {
                coeff += p_smoothed[row * smoothed.num_cols + col] * rbm[mode_index][row];
            }
            coarse[col] = coeff;
        }

        const std::vector<float> reconstructed = multiply_dense(p_smoothed,
                                                                smoothed.num_rows,
                                                                smoothed.num_cols,
                                                                coarse);
        const float error = max_abs_difference(reconstructed, rbm[mode_index]);
        std::cout << "RBM[" << mode_index << "] smoothed reconstruction error = " << error << std::endl;
        if (error > 3e-1f) {
            std::cerr << "Smoothed prolongator degraded RBM representation too much for mode "
                      << mode_index << std::endl;
            return 1;
        }
    }

    std::cout << "Smoke test passed: damped Jacobi smoothing modifies P while keeping RBM information." << std::endl;
    return 0;
}