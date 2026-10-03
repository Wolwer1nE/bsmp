#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "elastic_regularization.h"

int main() {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "=== Regularized elastic operator smoke test ===" << std::endl;

    const int n = 6;
    const std::vector<float> c_u_dense = {
        4.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        1.0f, 7.0f, 0.5f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.5f, 5.0f, 0.2f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.2f, 9.0f, 0.3f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.3f, 6.0f, 0.4f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.4f, 8.0f,
    };

    auto c_u = bsmp::HostBlockMatrix::fromDense(n, n, c_u_dense).toDeviceMatrix();
    const float epsilon = 1e-6f;
    const float expected_alpha = 9.0f * epsilon;

    const float alpha = bsmp::compute_regularization_alpha(*c_u, epsilon);
    std::cout << "Computed alpha = " << alpha << std::endl;
    if (std::fabs(alpha - expected_alpha) > 1e-7f) {
        std::cerr << "Unexpected alpha value" << std::endl;
        return 1;
    }

    const bsmp::RegularizedElasticOperator regularized =
        bsmp::build_regularized_elastic_operator(*c_u, epsilon);

    const std::vector<float> regularized_dense =
        bsmp::HostBlockMatrix::fromMatrix(*regularized.matrix).toDense();
    const std::vector<float> original_dense =
        bsmp::HostBlockMatrix::fromMatrix(*c_u).toDense();

    for (int i = 0; i < n; ++i) {
        const float diagonal_delta = regularized_dense[i * n + i] - original_dense[i * n + i];
        if (std::fabs(diagonal_delta - expected_alpha) > 1e-5f) {
            std::cerr << "Diagonal regularization mismatch at row " << i << std::endl;
            return 1;
        }
    }

    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            if (row == col) {
                continue;
            }
            if (std::fabs(regularized_dense[row * n + col] - original_dense[row * n + col]) > 1e-6f) {
                std::cerr << "Off-diagonal entry changed unexpectedly at (" << row << ", " << col << ")" << std::endl;
                return 1;
            }
        }
    }

    std::cout << "Smoke test passed: alpha is correct and only the diagonal is shifted." << std::endl;
    return 0;
}