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
    std::cout << "=== SA-AMG aggregation/prolongator smoke test ===" << std::endl;

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

    std::cout << "Aggregates: " << aggregation.numAggregates() << std::endl;
    std::cout << "Tentative P size: " << tentative.num_rows << " x " << tentative.num_cols
              << ", nnz=" << tentative.nnz() << std::endl;

    if (aggregation.numAggregates() <= 0 || tentative.num_cols <= 0 || tentative.nnz() <= 0) {
        std::cerr << "Aggregation/prolongator construction failed" << std::endl;
        return 1;
    }

    const std::vector<float> p_dense = tentative.toDense();
    for (size_t mode_index = 0; mode_index < rbm.size(); ++mode_index) {
        std::vector<float> coarse(tentative.num_cols, 0.0f);
        for (int col = 0; col < tentative.num_cols; ++col) {
            float coeff = 0.0f;
            for (int row = 0; row < tentative.num_rows; ++row) {
                coeff += p_dense[row * tentative.num_cols + col] * rbm[mode_index][row];
            }
            coarse[col] = coeff;
        }

        const std::vector<float> reconstructed = multiply_dense(p_dense,
                                                                tentative.num_rows,
                                                                tentative.num_cols,
                                                                coarse);
        const float error = max_abs_difference(reconstructed, rbm[mode_index]);
        std::cout << "RBM[" << mode_index << "] reconstruction error = " << error << std::endl;
        if (error > 1e-4f) {
            std::cerr << "Tentative prolongator failed to preserve RBM " << mode_index << std::endl;
            return 1;
        }
    }

    std::cout << "Smoke test passed: connectivity aggregation and tentative prolongator preserve RBM." << std::endl;
    return 0;
}