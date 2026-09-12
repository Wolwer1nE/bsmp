#ifndef BSMP_MASS_ORTHOGONALIZATION_H
#define BSMP_MASS_ORTHOGONALIZATION_H

#include <vector>

#include "block_sparse_matrix.h"

namespace bsmp {

struct MassDeflationSubspace {
    std::vector<std::vector<float>> basis;
    std::vector<std::vector<float>> mass_basis;

    bool empty() const { return basis.empty(); }
    int size() const { return static_cast<int>(basis.size()); }
};

std::vector<float> apply_mass_matrix(BlockSparseMatrix& mass,
                                     const std::vector<float>& vector);

float mass_inner_product(BlockSparseMatrix& mass,
                         const std::vector<float>& lhs,
                         const std::vector<float>& rhs);

float mass_norm(BlockSparseMatrix& mass,
                const std::vector<float>& vector);

void normalize_in_mass_metric(BlockSparseMatrix& mass,
                              std::vector<float>& vector,
                              float tolerance = 1e-7f);

MassDeflationSubspace build_mass_deflation_subspace(
    BlockSparseMatrix& mass,
    const std::vector<std::vector<float>>& candidate_basis,
    float tolerance = 1e-6f);

void project_mass_orthogonal_complement(const MassDeflationSubspace& subspace,
                                        std::vector<float>& vector,
                                        float tolerance = 1e-7f);

void deflate_gradient(const MassDeflationSubspace& subspace,
                      std::vector<float>& gradient,
                      float tolerance = 1e-7f);

void deflate_preconditioned_vector(const MassDeflationSubspace& subspace,
                                   std::vector<float>& vector,
                                   float tolerance = 1e-7f);

}  // namespace bsmp

#endif  // BSMP_MASS_ORTHOGONALIZATION_H