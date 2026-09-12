#ifndef BSMP_ELASTIC_REGULARIZATION_H
#define BSMP_ELASTIC_REGULARIZATION_H

#include <memory>

#include "piezo_block_system.h"

namespace bsmp {

struct RegularizedElasticOperator {
    std::unique_ptr<BlockSparseMatrix> matrix;
    float alpha = 0.0f;
};

float compute_regularization_alpha(const BlockSparseMatrix& c_u, float epsilon = 1e-6f);

RegularizedElasticOperator build_regularized_elastic_operator(const BlockSparseMatrix& c_u,
                                                              float epsilon = 1e-6f);

}  // namespace bsmp

#endif  // BSMP_ELASTIC_REGULARIZATION_H