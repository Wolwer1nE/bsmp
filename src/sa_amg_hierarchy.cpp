#include "sa_amg_hierarchy.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "chebyshev_smoother.h"
#include "piezo_block_system.h"

namespace bsmp {

namespace {

float dot_product(const std::vector<float>& a, const std::vector<float>& b) {
    float value = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        value += a[i] * b[i];
    }
    return value;
}

float vector_norm(const std::vector<float>& v) {
    return std::sqrt(std::max(dot_product(v, v), 0.0f));
}

void normalize(std::vector<float>& v) {
    const float norm = vector_norm(v);
    if (norm <= 1e-12f) {
        return;
    }
    for (float& value : v) {
        value /= norm;
    }
}

float distance_between_nodes(const NodeLayout& layout, int lhs, int rhs) {
    const float dx = layout.coordinates[lhs * 3 + 0] - layout.coordinates[rhs * 3 + 0];
    const float dy = layout.coordinates[lhs * 3 + 1] - layout.coordinates[rhs * 3 + 1];
    const float dz = layout.coordinates[lhs * 3 + 2] - layout.coordinates[rhs * 3 + 2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

struct PairHash {
    size_t operator()(const std::pair<int, int>& value) const noexcept {
        return (static_cast<size_t>(value.first) << 32) ^ static_cast<size_t>(value.second);
    }
};

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

std::vector<float> matvec_dense(const std::vector<float>& matrix,
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

std::vector<float> build_inverse_diagonal(const std::vector<float>& matrix, int n) {
    std::vector<float> inv_diag(n, 1.0f);
    for (int i = 0; i < n; ++i) {
        const float diagonal = matrix[i * n + i];
        if (std::fabs(diagonal) > 1e-12f) {
            inv_diag[i] = 1.0f / diagonal;
        }
    }
    return inv_diag;
}

void jacobi_smooth(const std::vector<float>& a_dense,
                   const std::vector<float>& inverse_diagonal,
                   const std::vector<float>& rhs,
                   std::vector<float>& x,
                   int sweeps,
                   float damping) {
    const int n = static_cast<int>(rhs.size());
    for (int sweep = 0; sweep < sweeps; ++sweep) {
        const std::vector<float> ax = matvec_dense(a_dense, n, n, x);
        for (int i = 0; i < n; ++i) {
            const float residual = rhs[i] - ax[i];
            x[i] += damping * inverse_diagonal[i] * residual;
        }
    }
}

void apply_reference_smoother(const SAAMGHierarchyParameters& parameters,
                              const SAAMGLevel& level,
                              const std::vector<float>& rhs,
                              std::vector<float>& x,
                              int sweeps) {
    if (sweeps <= 0) {
        return;
    }

    if (parameters.use_chebyshev) {
        ChebyshevSmootherParameters chebyshev_parameters = parameters.chebyshev;
        chebyshev_parameters.steps *= sweeps;
        chebyshev_smooth(level.a_dense,
                         level.inverse_diagonal,
                         rhs,
                         x,
                         level.chebyshev_lambda_max,
                         chebyshev_parameters);
        return;
    }

    jacobi_smooth(level.a_dense,
                  level.inverse_diagonal,
                  rhs,
                  x,
                  sweeps,
                  parameters.jacobi_damping);
}

std::vector<float> compute_residual(const std::vector<float>& a_dense,
                                    const std::vector<float>& rhs,
                                    const std::vector<float>& x) {
    const int n = static_cast<int>(rhs.size());
    const std::vector<float> ax = matvec_dense(a_dense, n, n, x);
    std::vector<float> residual(n, 0.0f);
    for (int i = 0; i < n; ++i) {
        residual[i] = rhs[i] - ax[i];
    }
    return residual;
}

bool lu_factorize(std::vector<float>& lu, std::vector<int>& pivots, int n) {
    pivots.resize(n);
    for (int i = 0; i < n; ++i) {
        pivots[i] = i;
    }

    for (int col = 0; col < n; ++col) {
        int pivot_row = col;
        float pivot_abs = std::fabs(lu[col * n + col]);
        for (int row = col + 1; row < n; ++row) {
            const float candidate = std::fabs(lu[row * n + col]);
            if (candidate > pivot_abs) {
                pivot_abs = candidate;
                pivot_row = row;
            }
        }
        if (pivot_abs <= 1e-12f) {
            return false;
        }
        if (pivot_row != col) {
            for (int j = 0; j < n; ++j) {
                std::swap(lu[col * n + j], lu[pivot_row * n + j]);
            }
            std::swap(pivots[col], pivots[pivot_row]);
        }

        const float pivot = lu[col * n + col];
        for (int row = col + 1; row < n; ++row) {
            lu[row * n + col] /= pivot;
            const float factor = lu[row * n + col];
            for (int j = col + 1; j < n; ++j) {
                lu[row * n + j] -= factor * lu[col * n + j];
            }
        }
    }
    return true;
}

std::vector<float> lu_solve(const std::vector<float>& lu,
                            const std::vector<int>& pivots,
                            const std::vector<float>& rhs,
                            int n) {
    std::vector<float> y(n, 0.0f);
    std::vector<float> x(n, 0.0f);

    for (int i = 0; i < n; ++i) {
        float sum = rhs[pivots[i]];
        for (int j = 0; j < i; ++j) {
            sum -= lu[i * n + j] * y[j];
        }
        y[i] = sum;
    }

    for (int i = n - 1; i >= 0; --i) {
        float sum = y[i];
        for (int j = i + 1; j < n; ++j) {
            sum -= lu[i * n + j] * x[j];
        }
        x[i] = sum / lu[i * n + i];
    }

    return x;
}

}  // namespace

std::vector<float> SparseTentativeProlongator::toDense() const {
    std::vector<float> dense(num_rows * num_cols, 0.0f);
    for (size_t i = 0; i < values.size(); ++i) {
        dense[row_indices[i] * num_cols + col_indices[i]] += values[i];
    }
    return dense;
}

NodeAggregation aggregateNodes(const BlockSparseMatrix& current_a,
                               const NodeLayout& layout) {
    if (!layout.isValid()) {
        throw std::invalid_argument("NodeLayout is invalid");
    }

    const HostBlockMatrix host = HostBlockMatrix::fromMatrix(current_a);
    const int block_size = host.config.block_size;
    const int num_block_rows = (host.config.num_rows + block_size - 1) / block_size;
    if (num_block_rows != layout.numNodes()) {
        throw std::invalid_argument("aggregateNodes expects one mechanical block row per node");
    }

    const int block_area = block_size * block_size;
    std::vector<std::unordered_map<int, float>> adjacency(num_block_rows);
    for (int idx = 0; idx < host.config.num_nonzero_blocks; ++idx) {
        const int row = host.block_rows[idx];
        const int col = host.block_cols[idx];
        if (row == col) {
            continue;
        }

        float block_norm_sq = 0.0f;
        const float* block = host.block_data.data() + idx * block_area;
        for (int i = 0; i < block_area; ++i) {
            block_norm_sq += block[i] * block[i];
        }
        const float connectivity = std::sqrt(block_norm_sq);
        const float distance = distance_between_nodes(layout, row, col);
        const float score = connectivity / (1.0f + distance);

        adjacency[row][col] = std::max(adjacency[row][col], score);
        adjacency[col][row] = std::max(adjacency[col][row], score);
    }

    NodeAggregation aggregation;
    aggregation.aggregate_ids.assign(num_block_rows, -1);

    int next_aggregate = 0;
    for (int node = 0; node < num_block_rows; ++node) {
        if (aggregation.aggregate_ids[node] != -1) {
            continue;
        }

        int best_neighbor = -1;
        float best_score = -std::numeric_limits<float>::infinity();
        for (const auto& edge : adjacency[node]) {
            if (aggregation.aggregate_ids[edge.first] != -1) {
                continue;
            }
            if (edge.second > best_score) {
                best_score = edge.second;
                best_neighbor = edge.first;
            }
        }

        aggregation.aggregate_ids[node] = next_aggregate;
        aggregation.aggregate_nodes.push_back({node});
        if (best_neighbor >= 0 && best_neighbor != node) {
            aggregation.aggregate_ids[best_neighbor] = next_aggregate;
            aggregation.aggregate_nodes.back().push_back(best_neighbor);
        }
        ++next_aggregate;
    }

    for (auto& nodes : aggregation.aggregate_nodes) {
        std::sort(nodes.begin(), nodes.end());
    }

    return aggregation;
}

SparseTentativeProlongator build_tentative_prolongator(
    const NodeAggregation& aggregation,
    const std::vector<std::vector<float>>& nullspace) {
    if (nullspace.empty()) {
        throw std::invalid_argument("Nullspace basis must not be empty");
    }
    const int fine_dofs = static_cast<int>(nullspace.front().size());
    if (fine_dofs % 3 != 0) {
        throw std::invalid_argument("Tentative prolongator currently expects 3 DOFs per node");
    }
    for (const auto& mode : nullspace) {
        if (static_cast<int>(mode.size()) != fine_dofs) {
            throw std::invalid_argument("All nullspace vectors must have the same size");
        }
    }

    const int num_nodes = fine_dofs / 3;
    if (static_cast<int>(aggregation.aggregate_ids.size()) != num_nodes) {
        throw std::invalid_argument("Aggregation size must match the number of nodes in the nullspace");
    }

    SparseTentativeProlongator prolongator;
    prolongator.num_rows = fine_dofs;
    prolongator.aggregate_col_offsets.resize(aggregation.numAggregates() + 1, 0);
    prolongator.aggregate_basis_sizes.resize(aggregation.numAggregates(), 0);

    int coarse_cols = 0;
    for (int aggregate_id = 0; aggregate_id < aggregation.numAggregates(); ++aggregate_id) {
        const std::vector<int>& nodes = aggregation.aggregate_nodes[aggregate_id];
        const int local_dofs = static_cast<int>(nodes.size()) * 3;

        std::vector<std::vector<float>> local_basis;
        local_basis.reserve(nullspace.size());
        for (const auto& global_mode : nullspace) {
            std::vector<float> local_vector(local_dofs, 0.0f);
            for (size_t local_node = 0; local_node < nodes.size(); ++local_node) {
                const int global_node = nodes[local_node];
                for (int component = 0; component < 3; ++component) {
                    local_vector[local_node * 3 + component] = global_mode[global_node * 3 + component];
                }
            }

            for (const auto& basis_vector : local_basis) {
                const float projection = dot_product(local_vector, basis_vector);
                for (int i = 0; i < local_dofs; ++i) {
                    local_vector[i] -= projection * basis_vector[i];
                }
            }

            if (vector_norm(local_vector) > 1e-5f) {
                normalize(local_vector);
                local_basis.push_back(std::move(local_vector));
            }
        }

        prolongator.aggregate_col_offsets[aggregate_id] = coarse_cols;
        prolongator.aggregate_basis_sizes[aggregate_id] = static_cast<int>(local_basis.size());

        for (size_t local_col = 0; local_col < local_basis.size(); ++local_col) {
            const int global_col = coarse_cols + static_cast<int>(local_col);
            for (size_t local_node = 0; local_node < nodes.size(); ++local_node) {
                const int global_node = nodes[local_node];
                for (int component = 0; component < 3; ++component) {
                    const float value = local_basis[local_col][local_node * 3 + component];
                    if (std::fabs(value) <= 1e-7f) {
                        continue;
                    }
                    prolongator.row_indices.push_back(global_node * 3 + component);
                    prolongator.col_indices.push_back(global_col);
                    prolongator.values.push_back(value);
                }
            }
        }

        coarse_cols += static_cast<int>(local_basis.size());
    }

    prolongator.aggregate_col_offsets[aggregation.numAggregates()] = coarse_cols;
    prolongator.num_cols = coarse_cols;
    return prolongator;
}

std::vector<float> build_restriction_transpose_dense(
    const SparseTentativeProlongator& prolongator) {
    const std::vector<float> p_dense = prolongator.toDense();
    std::vector<float> r_dense(prolongator.num_cols * prolongator.num_rows, 0.0f);
    for (int row = 0; row < prolongator.num_rows; ++row) {
        for (int col = 0; col < prolongator.num_cols; ++col) {
            r_dense[col * prolongator.num_rows + row] = p_dense[row * prolongator.num_cols + col];
        }
    }
    return r_dense;
}

SparseTentativeProlongator build_smoothed_prolongator(
    const BlockSparseMatrix& current_a,
    const SparseTentativeProlongator& tentative,
    float damping,
    float drop_tolerance) {
    if (damping <= 0.0f || damping > 1.0f) {
        throw std::invalid_argument("Damping must be in the interval (0, 1]");
    }

    const HostBlockMatrix fine_host = HostBlockMatrix::fromMatrix(current_a);
    if (fine_host.config.num_rows != fine_host.config.num_cols) {
        throw std::invalid_argument("Smoothed prolongator requires a square fine operator");
    }
    if (tentative.num_rows != fine_host.config.num_rows) {
        throw std::invalid_argument("Tentative prolongator row count must match fine operator size");
    }

    const std::vector<float> a_dense = fine_host.toDense();
    const std::vector<float> p_tentative = tentative.toDense();
    const std::vector<float> ap = multiply_dense(a_dense,
                                                 fine_host.config.num_rows,
                                                 fine_host.config.num_cols,
                                                 p_tentative,
                                                 tentative.num_cols);

    std::vector<float> p_smoothed = p_tentative;
    for (int row = 0; row < fine_host.config.num_rows; ++row) {
        const float diagonal = a_dense[row * fine_host.config.num_cols + row];
        if (std::fabs(diagonal) <= 1e-12f) {
            continue;
        }
        const float scale = damping / diagonal;
        for (int col = 0; col < tentative.num_cols; ++col) {
            p_smoothed[row * tentative.num_cols + col] -= scale * ap[row * tentative.num_cols + col];
        }
    }

    SparseTentativeProlongator smoothed;
    smoothed.num_rows = tentative.num_rows;
    smoothed.num_cols = tentative.num_cols;
    smoothed.aggregate_col_offsets = tentative.aggregate_col_offsets;
    smoothed.aggregate_basis_sizes = tentative.aggregate_basis_sizes;

    for (int row = 0; row < smoothed.num_rows; ++row) {
        for (int col = 0; col < smoothed.num_cols; ++col) {
            const float value = p_smoothed[row * smoothed.num_cols + col];
            if (std::fabs(value) <= drop_tolerance) {
                continue;
            }
            smoothed.row_indices.push_back(row);
            smoothed.col_indices.push_back(col);
            smoothed.values.push_back(value);
        }
    }

    return smoothed;
}

HostBlockMatrix build_galerkin_coarse_operator(
    const BlockSparseMatrix& current_a,
    const SparseTentativeProlongator& prolongator,
    float drop_tolerance) {
    const HostBlockMatrix fine_host = HostBlockMatrix::fromMatrix(current_a);
    if (fine_host.config.num_rows != fine_host.config.num_cols) {
        throw std::invalid_argument("Galerkin coarse operator requires a square fine operator");
    }
    if (prolongator.num_rows != fine_host.config.num_rows) {
        throw std::invalid_argument("Prolongator row count must match fine operator size");
    }

    const std::vector<float> a_dense = fine_host.toDense();
    const std::vector<float> p_dense = prolongator.toDense();
    const std::vector<float> r_dense = build_restriction_transpose_dense(prolongator);
    const std::vector<float> ap_dense = multiply_dense(a_dense,
                                                       fine_host.config.num_rows,
                                                       fine_host.config.num_cols,
                                                       p_dense,
                                                       prolongator.num_cols);
    const std::vector<float> coarse_dense = multiply_dense(r_dense,
                                                           prolongator.num_cols,
                                                           prolongator.num_rows,
                                                           ap_dense,
                                                           prolongator.num_cols);

    return HostBlockMatrix::fromDense(prolongator.num_cols,
                                      prolongator.num_cols,
                                      coarse_dense,
                                      drop_tolerance);
}

bool SAAMGHierarchy::isValid() const {
    return !fine_level.a_dense.empty() && !fine_level.prolongator_dense.empty() &&
           !fine_level.coarse_dense.empty() && !fine_level.coarse_lu.empty();
}

std::vector<float> SAAMGHierarchy::applyVcycle(const std::vector<float>& rhs) const {
    if (!isValid()) {
        throw std::runtime_error("SA-AMG hierarchy is not initialized");
    }
    const int n = fine_level.a_host.config.num_rows;
    if (static_cast<int>(rhs.size()) != n) {
        throw std::invalid_argument("V-cycle rhs size does not match fine level");
    }
    if (fine_level.coarse_lu.empty() || fine_level.coarse_pivots.empty()) {
        throw std::runtime_error("Coarse-level LU factorization was not precomputed during hierarchy setup");
    }

    std::vector<float> x(n, 0.0f);
    apply_reference_smoother(parameters, fine_level, rhs, x, parameters.pre_sweeps);

    const std::vector<float> residual = compute_residual(fine_level.a_dense, rhs, x);
    const std::vector<float> coarse_rhs = matvec_dense(fine_level.restriction_dense,
                                                       fine_level.prolongator.num_cols,
                                                       fine_level.prolongator.num_rows,
                                                       residual);

    const std::vector<float> coarse_error = lu_solve(fine_level.coarse_lu,
                                                     fine_level.coarse_pivots,
                                                     coarse_rhs,
                                                     fine_level.coarse_host.config.num_rows);
    const std::vector<float> fine_correction = matvec_dense(fine_level.prolongator_dense,
                                                            fine_level.prolongator.num_rows,
                                                            fine_level.prolongator.num_cols,
                                                            coarse_error);
    for (int i = 0; i < n; ++i) {
        x[i] += fine_correction[i];
    }

    apply_reference_smoother(parameters, fine_level, rhs, x, parameters.post_sweeps);
    return x;
}

SAAMGHierarchy build_sa_amg_two_level_hierarchy(
    const BlockSparseMatrix& current_a,
    const NodeLayout& layout,
    const std::vector<std::vector<float>>& nullspace,
    const SAAMGHierarchyParameters& parameters) {
    SAAMGHierarchy hierarchy;
    hierarchy.parameters = parameters;

    hierarchy.fine_level.a_host = HostBlockMatrix::fromMatrix(current_a);
    hierarchy.fine_level.a_dense = hierarchy.fine_level.a_host.toDense();
    hierarchy.fine_level.inverse_diagonal = build_inverse_diagonal(hierarchy.fine_level.a_dense,
                                                                   hierarchy.fine_level.a_host.config.num_rows);
    if (parameters.use_chebyshev) {
        hierarchy.fine_level.chebyshev_lambda_max = estimate_max_eigenvalue(
            hierarchy.fine_level.a_dense,
            hierarchy.fine_level.inverse_diagonal,
            hierarchy.fine_level.a_host.config.num_rows,
            parameters.chebyshev);
    }
    hierarchy.fine_level.aggregation = aggregateNodes(current_a, layout);
    hierarchy.fine_level.tentative = build_tentative_prolongator(hierarchy.fine_level.aggregation,
                                                                 nullspace);
    hierarchy.fine_level.prolongator = build_smoothed_prolongator(current_a,
                                                                  hierarchy.fine_level.tentative,
                                                                  parameters.prolongation_damping);
    hierarchy.fine_level.prolongator_dense = hierarchy.fine_level.prolongator.toDense();
    hierarchy.fine_level.restriction_dense = build_restriction_transpose_dense(hierarchy.fine_level.prolongator);
    hierarchy.fine_level.coarse_host = build_galerkin_coarse_operator(current_a,
                                                                      hierarchy.fine_level.prolongator);
    hierarchy.fine_level.coarse_dense = hierarchy.fine_level.coarse_host.toDense();
    // Precompute the LU factorization of the coarse operator so applyVcycle
    // can reuse it without re-factorizing on every call.
    hierarchy.fine_level.coarse_lu = hierarchy.fine_level.coarse_dense;
    if (!lu_factorize(hierarchy.fine_level.coarse_lu,
                      hierarchy.fine_level.coarse_pivots,
                      hierarchy.fine_level.coarse_host.config.num_rows)) {
        throw std::runtime_error("Coarse operator LU factorization failed during SA-AMG setup");
    }
    return hierarchy;
}

}  // namespace bsmp