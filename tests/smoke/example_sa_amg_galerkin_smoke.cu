#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "elastic_nullspace.h"
#include "piezo_block_system.h"
#include "sa_amg_hierarchy.h"

namespace {

std::vector<float> multiply_dense(const std::vector<float>& lhs,
                                  int lhs_rows,
                                  int lhs_cols,
                                  const std::vector<float>& rhs,
                                  int rhs_cols) {
    std::vector<float> out(lhs_rows * rhs_cols, 0.0f);
    for (int row = 0; row < lhs_rows; ++row) {
        for (int k = 0; k < lhs_cols; ++k) {
            const float lhs_value = lhs[row * lhs_cols + k];
            if (std::fabs(lhs_value) <= 1e-12f) {
                continue;
            }
            for (int col = 0; col < rhs_cols; ++col) {
                out[row * rhs_cols + col] += lhs_value * rhs[k * rhs_cols + col];
            }
        }
    }
    return out;
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
    std::cout << "=== SA-AMG Galerkin smoke test ===" << std::endl;

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

    const std::vector<float> p_dense = tentative.toDense();
    const std::vector<float> r_dense = bsmp::build_restriction_transpose_dense(tentative);
    const std::vector<float> ap_dense = multiply_dense(c_u_dense, n, n, p_dense, tentative.num_cols);
    const std::vector<float> expected_coarse = multiply_dense(r_dense,
                                                              tentative.num_cols,
                                                              n,
                                                              ap_dense,
                                                              tentative.num_cols);

    const bsmp::HostBlockMatrix coarse_host = bsmp::build_galerkin_coarse_operator(*c_u, tentative);
    const std::vector<float> coarse_dense = coarse_host.toDense();

    std::cout << "Coarse matrix size: " << coarse_host.config.num_rows << " x "
              << coarse_host.config.num_cols << ", nonzero blocks: "
              << coarse_host.config.num_nonzero_blocks << std::endl;

    if (coarse_host.config.num_rows != tentative.num_cols ||
        coarse_host.config.num_cols != tentative.num_cols) {
        std::cerr << "Unexpected coarse matrix dimensions" << std::endl;
        return 1;
    }

    const float error = max_abs_difference(coarse_dense, expected_coarse);
    std::cout << "Galerkin max |expected - actual| = " << error << std::endl;
    if (error > 1e-4f) {
        std::cerr << "Galerkin coarse operator mismatch is too large" << std::endl;
        return 1;
    }

    for (int row = 0; row < coarse_host.config.num_rows; ++row) {
        for (int col = 0; col < coarse_host.config.num_cols; ++col) {
            const float symmetry_error = std::fabs(coarse_dense[row * coarse_host.config.num_cols + col] -
                                                   coarse_dense[col * coarse_host.config.num_cols + row]);
            if (symmetry_error > 1e-4f) {
                std::cerr << "Coarse operator lost symmetry at (" << row << ", " << col << ")" << std::endl;
                return 1;
            }
        }
    }

    std::cout << "Smoke test passed: host-reference Galerkin coarse operator is correct and symmetric." << std::endl;
    return 0;
}