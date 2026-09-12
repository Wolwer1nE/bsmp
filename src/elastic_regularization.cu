#include "elastic_regularization.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace bsmp {

float compute_regularization_alpha(const BlockSparseMatrix& c_u, float epsilon) {
    if (epsilon < 0.0f) {
        throw std::invalid_argument("Regularization epsilon must be non-negative");
    }

    const HostBlockMatrix host = HostBlockMatrix::fromMatrix(c_u);
    const std::vector<float> dense = host.toDense();

    float max_diagonal = 0.0f;
    for (int i = 0; i < host.config.num_rows; ++i) {
        max_diagonal = std::max(max_diagonal, std::fabs(dense[i * host.config.num_cols + i]));
    }
    return epsilon * max_diagonal;
}

RegularizedElasticOperator build_regularized_elastic_operator(const BlockSparseMatrix& c_u,
                                                              float epsilon) {
    const HostBlockMatrix host = HostBlockMatrix::fromMatrix(c_u);
    if (host.config.num_rows != host.config.num_cols) {
        throw std::invalid_argument("Regularized elastic operator requires a square C_u");
    }

    std::vector<float> dense = host.toDense();
    const float alpha = compute_regularization_alpha(c_u, epsilon);
    for (int i = 0; i < host.config.num_rows; ++i) {
        dense[i * host.config.num_cols + i] += alpha;
    }

    RegularizedElasticOperator result;
    result.alpha = alpha;
    result.matrix = HostBlockMatrix::fromDense(host.config.num_rows,
                                               host.config.num_cols,
                                               dense)
                        .toDeviceMatrix();
    return result;
}

}  // namespace bsmp