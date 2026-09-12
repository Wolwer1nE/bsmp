#ifndef BSMP_SA_AMG_HIERARCHY_H
#define BSMP_SA_AMG_HIERARCHY_H

#include <vector>

#include "block_sparse_matrix.h"
#include "chebyshev_smoother.h"
#include "node_layout.h"
#include "piezo_block_system.h"

namespace bsmp {

struct SAAMGHierarchyParameters {
    int pre_sweeps = 2;
    int post_sweeps = 2;
    float jacobi_damping = 0.8f;
    float prolongation_damping = 0.6666667f;  // standard SA-AMG: (I - omega*D^{-1}*A), omega ~ 2/3
    bool use_chebyshev = false;
    ChebyshevSmootherParameters chebyshev;
};

struct NodeAggregation {
    std::vector<int> aggregate_ids;
    std::vector<std::vector<int>> aggregate_nodes;

    int numAggregates() const { return static_cast<int>(aggregate_nodes.size()); }
};

struct SparseTentativeProlongator {
    int num_rows = 0;
    int num_cols = 0;
    std::vector<int> row_indices;
    std::vector<int> col_indices;
    std::vector<float> values;
    std::vector<int> aggregate_col_offsets;
    std::vector<int> aggregate_basis_sizes;

    int nnz() const { return static_cast<int>(values.size()); }
    std::vector<float> toDense() const;
};

NodeAggregation aggregateNodes(const BlockSparseMatrix& current_a,
                               const NodeLayout& layout);

SparseTentativeProlongator build_tentative_prolongator(
    const NodeAggregation& aggregation,
    const std::vector<std::vector<float>>& nullspace);

std::vector<float> build_restriction_transpose_dense(
    const SparseTentativeProlongator& prolongator);

SparseTentativeProlongator build_smoothed_prolongator(
    const BlockSparseMatrix& current_a,
    const SparseTentativeProlongator& tentative,
    float damping = 0.6666667f,
    float drop_tolerance = 1e-7f);

HostBlockMatrix build_galerkin_coarse_operator(
    const BlockSparseMatrix& current_a,
    const SparseTentativeProlongator& prolongator,
    float drop_tolerance = 1e-7f);

struct SAAMGLevel {
    HostBlockMatrix a_host;
    std::vector<float> a_dense;
    std::vector<float> inverse_diagonal;
    float chebyshev_lambda_max = 0.0f;
    NodeAggregation aggregation;
    SparseTentativeProlongator tentative;
    SparseTentativeProlongator prolongator;
    std::vector<float> prolongator_dense;
    std::vector<float> restriction_dense;
    HostBlockMatrix coarse_host;
    std::vector<float> coarse_dense;
    // Precomputed LU factorization of coarse_dense; filled at setup time so
    // applyVcycle does not re-factorize on every call.
    std::vector<float> coarse_lu;
    std::vector<int> coarse_pivots;
};

struct SAAMGHierarchy {
    SAAMGHierarchyParameters parameters;
    SAAMGLevel fine_level;

    bool isValid() const;
    std::vector<float> applyVcycle(const std::vector<float>& rhs) const;
};

SAAMGHierarchy build_sa_amg_two_level_hierarchy(
    const BlockSparseMatrix& current_a,
    const NodeLayout& layout,
    const std::vector<std::vector<float>>& nullspace,
    const SAAMGHierarchyParameters& parameters = SAAMGHierarchyParameters{});

}  // namespace bsmp

#endif  // BSMP_SA_AMG_HIERARCHY_H