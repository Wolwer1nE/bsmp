#include "amg_preconditioner.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

#include "bsmp_config.h"

#define CHECK_CUDA_BOOL(call)                                                            \
    do {                                                                                 \
        cudaError_t err__ = (call);                                                      \
        if (err__ != cudaSuccess) {                                                      \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
                         cudaGetErrorString(err__));                                     \
            return false;                                                                \
        }                                                                                \
    } while (0)

#define CUBLAS_CHECK_BOOL(call)                                                          \
    do {                                                                                 \
        cublasStatus_t status__ = (call);                                                \
        if (status__ != CUBLAS_STATUS_SUCCESS) {                                         \
            std::fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__,   \
                         static_cast<int>(status__));                                    \
            return false;                                                                \
        }                                                                                \
    } while (0)

namespace {

constexpr int kThreads = 128;
constexpr float kPivotTolerance = 1e-12f;

struct PairHash {
    size_t operator()(const std::pair<int, int>& value) const noexcept {
        return (static_cast<size_t>(value.first) << 32) ^ static_cast<size_t>(value.second);
    }
};

struct HostMatrixData {
    BlockSparseMatrixConfig config{};
    int num_block_rows = 0;
    std::vector<int> block_rows;
    std::vector<int> block_cols;
    std::vector<float> block_data;
};

struct AggregateData {
    std::vector<int> ids;
    std::vector<float> weights;
    int coarse_block_rows = 0;
};

static __global__ void apply_block_jacobi_kernel(const float* inv_diag_blocks,
                                                 const float* r,
                                                 float* z,
                                                 int n,
                                                 int block_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;

    int block_row = idx / block_size;
    int local_row = idx % block_size;
    int base = block_row * block_size;
    const float* inv_block = inv_diag_blocks + block_row * block_size * block_size;

    float value = 0.0f;
    for (int j = 0; j < block_size; ++j) {
        int global_col = base + j;
        if (global_col < n) {
            value += inv_block[local_row * block_size + j] * r[global_col];
        }
    }
    z[idx] = value;
}

static __global__ void apply_scalar_jacobi_kernel(const float* inv_diag,
                                                  const float* r,
                                                  float* z,
                                                  int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;

    z[idx] = inv_diag[idx] * r[idx];
}

static __global__ void residual_kernel(const float* rhs,
                                       const float* Ax,
                                       float* residual,
                                       int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        residual[idx] = rhs[idx] - Ax[idx];
    }
}

static __global__ void axpy_kernel(float* y, const float* x, float alpha, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] += alpha * x[idx];
    }
}

static __global__ void restrict_blocks_kernel(const int* aggregate_ids,
                                              const float* aggregate_weights,
                                              const float* fine,
                                              float* coarse,
                                              int n,
                                              int block_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;

    int block_row = idx / block_size;
    int local_row = idx % block_size;
    int coarse_block = aggregate_ids[block_row];
    int coarse_idx = coarse_block * block_size + local_row;
    atomicAdd(&coarse[coarse_idx], aggregate_weights[block_row] * fine[idx]);
}

static __global__ void prolongate_blocks_kernel(const int* aggregate_ids,
                                                const float* aggregate_weights,
                                                const float* coarse,
                                                float* fine,
                                                int n,
                                                int block_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;

    int block_row = idx / block_size;
    int local_row = idx % block_size;
    int coarse_block = aggregate_ids[block_row];
    int coarse_idx = coarse_block * block_size + local_row;
    fine[idx] += aggregate_weights[block_row] * coarse[coarse_idx];
}

bool invert_small_block_host(const float* block, float* inv_block, int block_size) {
    constexpr int kMaxBlockSize = 16;
    if (block_size <= 0 || block_size > kMaxBlockSize)
        return false;

    float work[kMaxBlockSize * kMaxBlockSize];
    float inv[kMaxBlockSize * kMaxBlockSize];

    for (int i = 0; i < block_size; ++i) {
        for (int j = 0; j < block_size; ++j) {
            work[i * block_size + j] = block[i * block_size + j];
            inv[i * block_size + j] = (i == j) ? 1.0f : 0.0f;
        }
    }

    for (int col = 0; col < block_size; ++col) {
        int pivot = col;
        float pivot_abs = std::fabs(work[col * block_size + col]);
        for (int row = col + 1; row < block_size; ++row) {
            const float candidate_abs = std::fabs(work[row * block_size + col]);
            if (candidate_abs > pivot_abs) {
                pivot = row;
                pivot_abs = candidate_abs;
            }
        }

        if (pivot_abs <= kPivotTolerance)
            return false;

        if (pivot != col) {
            for (int j = 0; j < block_size; ++j) {
                std::swap(work[col * block_size + j], work[pivot * block_size + j]);
                std::swap(inv[col * block_size + j], inv[pivot * block_size + j]);
            }
        }

        const float diag = work[col * block_size + col];
        const float inv_diag = 1.0f / diag;
        for (int j = 0; j < block_size; ++j) {
            work[col * block_size + j] *= inv_diag;
            inv[col * block_size + j] *= inv_diag;
        }

        for (int row = 0; row < block_size; ++row) {
            if (row == col)
                continue;
            const float factor = work[row * block_size + col];
            if (std::fabs(factor) <= kPivotTolerance)
                continue;
            for (int j = 0; j < block_size; ++j) {
                work[row * block_size + j] -= factor * work[col * block_size + j];
                inv[row * block_size + j] -= factor * inv[col * block_size + j];
            }
        }
    }

    for (int i = 0; i < block_size * block_size; ++i) {
        inv_block[i] = inv[i];
    }
    return true;
}

float block_frobenius_norm(const float* block, int block_size) {
    float sum = 0.0f;
    for (int i = 0; i < block_size * block_size; ++i) {
        sum += block[i] * block[i];
    }
    return std::sqrt(sum);
}

bool build_inverse_diagonal_blocks(const HostMatrixData& matrix,
                                   std::vector<float>& inv_diag_blocks) {
    const int block_size = matrix.config.block_size;
    inv_diag_blocks.assign(matrix.num_block_rows * block_size * block_size, 0.0f);

    for (int block_row = 0; block_row < matrix.num_block_rows; ++block_row) {
        float* target = inv_diag_blocks.data() + block_row * block_size * block_size;
        for (int i = 0; i < block_size; ++i) {
            target[i * block_size + i] = 1.0f;
        }
    }

    std::vector<float> local_inverse(block_size * block_size, 0.0f);
    for (int idx = 0; idx < matrix.config.num_nonzero_blocks; ++idx) {
        if (matrix.block_rows[idx] != matrix.block_cols[idx])
            continue;

        const float* block = matrix.block_data.data() + idx * block_size * block_size;
        if (!invert_small_block_host(block, local_inverse.data(), block_size)) {
            continue;
        }

        float* target = inv_diag_blocks.data() + matrix.block_rows[idx] * block_size * block_size;
        std::copy(local_inverse.begin(), local_inverse.end(), target);
    }

    return true;
}

bool build_inverse_diagonal(const HostMatrixData& matrix,
                            std::vector<float>& inverse_diagonal) {
    const int block_size = matrix.config.block_size;
    inverse_diagonal.assign(matrix.config.num_rows, 1.0f);
    std::vector<float> diagonal(matrix.config.num_rows, 0.0f);

    for (int idx = 0; idx < matrix.config.num_nonzero_blocks; ++idx) {
        if (matrix.block_rows[idx] != matrix.block_cols[idx])
            continue;

        const int base_row = matrix.block_rows[idx] * block_size;
        const float* block = matrix.block_data.data() + idx * block_size * block_size;
        for (int i = 0; i < block_size; ++i) {
            const int global_row = base_row + i;
            if (global_row >= matrix.config.num_rows)
                continue;
            diagonal[global_row] += block[i * block_size + i];
        }
    }

    for (int i = 0; i < matrix.config.num_rows; ++i) {
        if (std::fabs(diagonal[i]) > 1e-20f) {
            inverse_diagonal[i] = 1.0f / diagonal[i];
        }
    }

    return true;
}

std::vector<int> build_pairwise_aggregates(const HostMatrixData& matrix) {
    std::vector<std::unordered_map<int, float>> adjacency(matrix.num_block_rows);
    const int block_area = matrix.config.block_size * matrix.config.block_size;

    for (int idx = 0; idx < matrix.config.num_nonzero_blocks; ++idx) {
        const int row = matrix.block_rows[idx];
        const int col = matrix.block_cols[idx];
        if (row == col)
            continue;

        const float weight = block_frobenius_norm(matrix.block_data.data() + idx * block_area,
                                                  matrix.config.block_size);
        auto update = [&](int from, int to) {
            auto it = adjacency[from].find(to);
            if (it == adjacency[from].end()) {
                adjacency[from].emplace(to, weight);
            } else if (weight > it->second) {
                it->second = weight;
            }
        };
        update(row, col);
        update(col, row);
    }

    std::vector<int> aggregate_ids(matrix.num_block_rows, -1);
    int next_aggregate = 0;

    for (int i = 0; i < matrix.num_block_rows; ++i) {
        if (aggregate_ids[i] != -1)
            continue;

        int best_neighbor = -1;
        float best_weight = -1.0f;
        for (const auto& edge : adjacency[i]) {
            if (edge.first == i || aggregate_ids[edge.first] != -1)
                continue;
            if (edge.second > best_weight) {
                best_weight = edge.second;
                best_neighbor = edge.first;
            }
        }

        aggregate_ids[i] = next_aggregate;
        if (best_neighbor >= 0) {
            aggregate_ids[best_neighbor] = next_aggregate;
        }
        ++next_aggregate;
    }

    return aggregate_ids;
}

AggregateData build_weighted_aggregates(const HostMatrixData& matrix) {
    AggregateData aggregate_data;
    aggregate_data.ids = build_pairwise_aggregates(matrix);
    aggregate_data.coarse_block_rows = aggregate_data.ids.empty()
                                           ? 0
                                           : (*std::max_element(aggregate_data.ids.begin(),
                                                                aggregate_data.ids.end()) + 1);
    aggregate_data.weights.assign(matrix.num_block_rows, 1.0f);

    if (aggregate_data.coarse_block_rows <= 0) {
        return aggregate_data;
    }

    std::vector<int> aggregate_sizes(aggregate_data.coarse_block_rows, 0);
    for (int aggregate_id : aggregate_data.ids) {
        if (aggregate_id >= 0) {
            ++aggregate_sizes[aggregate_id];
        }
    }

    for (int block_row = 0; block_row < matrix.num_block_rows; ++block_row) {
        const int aggregate_id = aggregate_data.ids[block_row];
        if (aggregate_id < 0 || aggregate_sizes[aggregate_id] <= 0) {
            continue;
        }
        aggregate_data.weights[block_row] = 1.0f / std::sqrt(static_cast<float>(aggregate_sizes[aggregate_id]));
    }

    return aggregate_data;
}

HostMatrixData build_coarse_matrix(const HostMatrixData& fine,
                                   const AggregateData& aggregate_data,
                                   int coarse_block_rows) {
    const int block_size = fine.config.block_size;
    const int block_area = block_size * block_size;
    std::unordered_map<std::pair<int, int>, std::vector<float>, PairHash> coarse_blocks;

    for (int idx = 0; idx < fine.config.num_nonzero_blocks; ++idx) {
        const int fine_row = fine.block_rows[idx];
        const int fine_col = fine.block_cols[idx];
        const int coarse_row = aggregate_data.ids[fine_row];
        const int coarse_col = aggregate_data.ids[fine_col];
        const float row_weight = aggregate_data.weights[fine_row];
        const float col_weight = aggregate_data.weights[fine_col];
        auto key = std::make_pair(coarse_row, coarse_col);
        auto it = coarse_blocks.find(key);
        if (it == coarse_blocks.end()) {
            std::vector<float> values(block_area, 0.0f);
            const float* block = fine.block_data.data() + idx * block_area;
            for (int j = 0; j < block_area; ++j) {
                values[j] = row_weight * block[j] * col_weight;
            }
            coarse_blocks.emplace(key, std::move(values));
        } else {
            float* target = it->second.data();
            const float* block = fine.block_data.data() + idx * block_area;
            for (int j = 0; j < block_area; ++j) {
                target[j] += row_weight * block[j] * col_weight;
            }
        }
    }

    HostMatrixData coarse;
    coarse.num_block_rows = coarse_block_rows;
    coarse.config.num_rows = coarse_block_rows * block_size;
    coarse.config.num_cols = coarse_block_rows * block_size;
    coarse.config.block_size = block_size;
    coarse.config.num_nonzero_blocks = static_cast<int>(coarse_blocks.size());

    coarse.block_rows.reserve(coarse_blocks.size());
    coarse.block_cols.reserve(coarse_blocks.size());
    coarse.block_data.reserve(coarse_blocks.size() * block_area);

    for (const auto& item : coarse_blocks) {
        coarse.block_rows.push_back(item.first.first);
        coarse.block_cols.push_back(item.first.second);
        coarse.block_data.insert(coarse.block_data.end(), item.second.begin(), item.second.end());
    }

    return coarse;
}

std::vector<float> build_dense_matrix(const HostMatrixData& matrix) {
    std::vector<float> dense(matrix.config.num_rows * matrix.config.num_cols, 0.0f);
    const int block_size = matrix.config.block_size;
    const int block_area = block_size * block_size;

    for (int idx = 0; idx < matrix.config.num_nonzero_blocks; ++idx) {
        const int row_base = matrix.block_rows[idx] * block_size;
        const int col_base = matrix.block_cols[idx] * block_size;
        const float* block = matrix.block_data.data() + idx * block_area;
        for (int i = 0; i < block_size; ++i) {
            const int row = row_base + i;
            if (row >= matrix.config.num_rows)
                continue;
            for (int j = 0; j < block_size; ++j) {
                const int col = col_base + j;
                if (col >= matrix.config.num_cols)
                    continue;
                dense[row * matrix.config.num_cols + col] += block[i * block_size + j];
            }
        }
    }

    return dense;
}

bool lu_factorize(std::vector<float>& lu, std::vector<int>& pivots, int n) {
    pivots.resize(n);
    std::iota(pivots.begin(), pivots.end(), 0);

    for (int k = 0; k < n; ++k) {
        int pivot_row = k;
        float pivot_abs = std::fabs(lu[k * n + k]);
        for (int row = k + 1; row < n; ++row) {
            const float candidate = std::fabs(lu[row * n + k]);
            if (candidate > pivot_abs) {
                pivot_abs = candidate;
                pivot_row = row;
            }
        }

        if (pivot_abs <= kPivotTolerance)
            return false;

        if (pivot_row != k) {
            for (int col = 0; col < n; ++col) {
                std::swap(lu[k * n + col], lu[pivot_row * n + col]);
            }
            std::swap(pivots[k], pivots[pivot_row]);
        }

        const float pivot = lu[k * n + k];
        for (int row = k + 1; row < n; ++row) {
            lu[row * n + k] /= pivot;
            const float factor = lu[row * n + k];
            for (int col = k + 1; col < n; ++col) {
                lu[row * n + col] -= factor * lu[k * n + col];
            }
        }
    }

    return true;
}

void lu_solve(const std::vector<float>& lu,
              const std::vector<int>& pivots,
              const std::vector<float>& rhs,
              std::vector<float>& x,
              int n) {
    std::vector<float> y(n, 0.0f);
    x.assign(n, 0.0f);

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
}

HostMatrixData exportHostMatrix(BlockSparseMatrix& A) {
    HostMatrixData host;
    host.config = A.getConfig();
    host.num_block_rows = (host.config.num_rows + host.config.block_size - 1) / host.config.block_size;
    A.copyToHost(host.block_rows, host.block_cols, host.block_data);
    return host;
}

class IdentityPreconditioner final : public LinearPreconditioner {
   public:
    explicit IdentityPreconditioner(int n) : n_(n) {}

    bool apply(const float* d_rhs, float* d_out) override {
        return cudaMemcpy(d_out, d_rhs, n_ * sizeof(float), cudaMemcpyDeviceToDevice) == cudaSuccess;
    }

    PreconditionerType type() const override { return PreconditionerType::None; }

   private:
    int n_;
};

class BlockJacobiPreconditioner final : public LinearPreconditioner {
   public:
    BlockJacobiPreconditioner() = default;

    ~BlockJacobiPreconditioner() override {
        if (d_inv_diag_blocks_ != nullptr) {
            cudaFree(d_inv_diag_blocks_);
        }
    }

    bool initialize(const HostMatrixData& matrix) {
        n_ = matrix.config.num_rows;
        block_size_ = matrix.config.block_size;
        num_block_rows_ = matrix.num_block_rows;

        std::vector<float> inv_diag_blocks;
        if (!build_inverse_diagonal_blocks(matrix, inv_diag_blocks)) {
            return false;
        }

        const size_t bytes = inv_diag_blocks.size() * sizeof(float);
        CHECK_CUDA_BOOL(cudaMalloc(&d_inv_diag_blocks_, bytes));
        CHECK_CUDA_BOOL(cudaMemcpy(d_inv_diag_blocks_, inv_diag_blocks.data(), bytes,
                                   cudaMemcpyHostToDevice));
        return true;
    }

    bool apply(const float* d_rhs, float* d_out) override {
        const int blocks = (n_ + kThreads - 1) / kThreads;
        apply_block_jacobi_kernel<<<blocks, kThreads>>>(d_inv_diag_blocks_, d_rhs, d_out, n_,
                                                        block_size_);
        return cudaGetLastError() == cudaSuccess;
    }

    PreconditionerType type() const override { return PreconditionerType::BlockJacobi; }

   private:
    int n_ = 0;
    int block_size_ = 0;
    int num_block_rows_ = 0;
    float* d_inv_diag_blocks_ = nullptr;
};

class ScalarJacobiPreconditioner final : public LinearPreconditioner {
   public:
    ScalarJacobiPreconditioner() = default;

    ~ScalarJacobiPreconditioner() override {
        if (d_inv_diag_ != nullptr) {
            cudaFree(d_inv_diag_);
        }
    }

    bool initialize(const HostMatrixData& matrix) {
        n_ = matrix.config.num_rows;

        std::vector<float> inverse_diagonal;
        if (!build_inverse_diagonal(matrix, inverse_diagonal)) {
            return false;
        }

        const size_t bytes = inverse_diagonal.size() * sizeof(float);
        CHECK_CUDA_BOOL(cudaMalloc(&d_inv_diag_, bytes));
        CHECK_CUDA_BOOL(cudaMemcpy(d_inv_diag_, inverse_diagonal.data(), bytes,
                                   cudaMemcpyHostToDevice));
        return true;
    }

    bool apply(const float* d_rhs, float* d_out) override {
        const int blocks = (n_ + kThreads - 1) / kThreads;
        apply_scalar_jacobi_kernel<<<blocks, kThreads>>>(d_inv_diag_, d_rhs, d_out, n_);
        return cudaGetLastError() == cudaSuccess;
    }

    PreconditionerType type() const override { return PreconditionerType::ScalarJacobi; }

   private:
    int n_ = 0;
    float* d_inv_diag_ = nullptr;
};

class AMGPreconditioner final : public LinearPreconditioner {
   public:
    AMGPreconditioner() = default;

    ~AMGPreconditioner() override {
        for (Level& level : levels_) {
            level.release();
        }
    }

    bool initialize(BlockSparseMatrix& A, const AMGParameters& parameters) {
        params_ = parameters;
        params_.max_levels = std::max(1, params_.max_levels);
        params_.min_coarse_block_rows = std::max(1, params_.min_coarse_block_rows);
        params_.pre_sweeps = std::max(0, params_.pre_sweeps);
        params_.post_sweeps = std::max(0, params_.post_sweeps);
        params_.coarse_sweeps = std::max(1, params_.coarse_sweeps);
        params_.relaxation = std::max(0.05f, std::min(params_.relaxation, 1.0f));

        std::vector<HostMatrixData> host_levels;
        std::vector<AggregateData> aggregates;
        host_levels.push_back(exportHostMatrix(A));

        while (static_cast<int>(host_levels.size()) < params_.max_levels) {
            const HostMatrixData& current = host_levels.back();
            if (current.num_block_rows <= params_.min_coarse_block_rows) {
                break;
            }

            AggregateData aggregate_data = build_weighted_aggregates(current);
            const int coarse_block_rows = aggregate_data.coarse_block_rows;
            if (coarse_block_rows <= 0 || coarse_block_rows >= current.num_block_rows) {
                break;
            }

            HostMatrixData coarse = build_coarse_matrix(current, aggregate_data, coarse_block_rows);
            if (coarse.config.num_nonzero_blocks == 0) {
                break;
            }

            aggregates.push_back(std::move(aggregate_data));
            host_levels.push_back(std::move(coarse));
        }

        levels_.resize(host_levels.size());
        for (size_t level_idx = 0; level_idx < host_levels.size(); ++level_idx) {
            Level& level = levels_[level_idx];
            const HostMatrixData& host = host_levels[level_idx];
            level.num_rows = host.config.num_rows;
            level.block_size = host.config.block_size;
            level.num_block_rows = host.num_block_rows;

            if (level_idx == 0) {
                level.matrix = &A;
            } else {
                level.owned_matrix.reset(new BlockSparseMatrix(host.config));
                level.owned_matrix->initialize(host.block_rows, host.block_cols, host.block_data);
                level.matrix = level.owned_matrix.get();
            }

            std::vector<float> inv_diag_blocks;
            if (!build_inverse_diagonal_blocks(host, inv_diag_blocks)) {
                return false;
            }

            const size_t inv_bytes = inv_diag_blocks.size() * sizeof(float);
            CHECK_CUDA_BOOL(cudaMalloc(&level.d_inv_diag_blocks, inv_bytes));
            CHECK_CUDA_BOOL(cudaMemcpy(level.d_inv_diag_blocks, inv_diag_blocks.data(), inv_bytes,
                                       cudaMemcpyHostToDevice));

            CHECK_CUDA_BOOL(cudaMalloc(&level.d_residual, level.num_rows * sizeof(float)));
            CHECK_CUDA_BOOL(cudaMalloc(&level.d_temp, level.num_rows * sizeof(float)));

            if (level_idx + 1 < host_levels.size()) {
                const size_t aggregate_bytes = aggregates[level_idx].ids.size() * sizeof(int);
                CHECK_CUDA_BOOL(cudaMalloc(&level.d_aggregate_ids, aggregate_bytes));
                CHECK_CUDA_BOOL(cudaMemcpy(level.d_aggregate_ids, aggregates[level_idx].ids.data(),
                                           aggregate_bytes, cudaMemcpyHostToDevice));

                const size_t weight_bytes = aggregates[level_idx].weights.size() * sizeof(float);
                CHECK_CUDA_BOOL(cudaMalloc(&level.d_aggregate_weights, weight_bytes));
                CHECK_CUDA_BOOL(cudaMemcpy(level.d_aggregate_weights,
                                           aggregates[level_idx].weights.data(),
                                           weight_bytes,
                                           cudaMemcpyHostToDevice));

                const int coarse_rows = host_levels[level_idx + 1].config.num_rows;
                CHECK_CUDA_BOOL(cudaMalloc(&level.d_coarse_rhs, coarse_rows * sizeof(float)));
                CHECK_CUDA_BOOL(cudaMalloc(&level.d_coarse_x, coarse_rows * sizeof(float)));
            } else {
                level.has_direct_solver = initializeHostCoarseSolve(level, host);
            }
        }

        return true;
    }

    bool apply(const float* d_rhs, float* d_out) override {
        if (levels_.empty()) {
            return false;
        }
        return applyLevel(0, d_rhs, d_out);
    }

    PreconditionerType type() const override { return PreconditionerType::AMG; }

   private:
    struct Level {
        BlockSparseMatrix* matrix = nullptr;
        std::unique_ptr<BlockSparseMatrix> owned_matrix;
        int num_rows = 0;
        int block_size = 0;
        int num_block_rows = 0;

        float* d_inv_diag_blocks = nullptr;
        float* d_residual = nullptr;
        float* d_temp = nullptr;

        int* d_aggregate_ids = nullptr;
        float* d_aggregate_weights = nullptr;
        float* d_coarse_rhs = nullptr;
        float* d_coarse_x = nullptr;

        bool has_direct_solver = false;
        std::vector<float> dense_lu;
        std::vector<int> dense_pivots;
        std::vector<float> direct_rhs_host;
        std::vector<float> direct_solution_host;

        void release() {
            if (d_inv_diag_blocks != nullptr) {
                cudaFree(d_inv_diag_blocks);
                d_inv_diag_blocks = nullptr;
            }
            if (d_residual != nullptr) {
                cudaFree(d_residual);
                d_residual = nullptr;
            }
            if (d_temp != nullptr) {
                cudaFree(d_temp);
                d_temp = nullptr;
            }
            if (d_aggregate_ids != nullptr) {
                cudaFree(d_aggregate_ids);
                d_aggregate_ids = nullptr;
            }
            if (d_aggregate_weights != nullptr) {
                cudaFree(d_aggregate_weights);
                d_aggregate_weights = nullptr;
            }
            if (d_coarse_rhs != nullptr) {
                cudaFree(d_coarse_rhs);
                d_coarse_rhs = nullptr;
            }
            if (d_coarse_x != nullptr) {
                cudaFree(d_coarse_x);
                d_coarse_x = nullptr;
            }
            dense_lu.clear();
            dense_pivots.clear();
            direct_rhs_host.clear();
            direct_solution_host.clear();
            owned_matrix.reset();
            matrix = nullptr;
        }
    };

    bool initializeHostCoarseSolve(Level& level, const HostMatrixData& host) {
        std::vector<float> dense = build_dense_matrix(host);
        const int n = host.config.num_rows;
        if (n <= 0) {
            return false;
        }

        auto try_factorize = [&](std::vector<float> candidate_lu, float shift) {
            if (shift > 0.0f) {
                float max_diag = 0.0f;
                for (int i = 0; i < n; ++i) {
                    max_diag = std::max(max_diag, std::fabs(candidate_lu[i * n + i]));
                }
                const float diagonal_boost = (max_diag > 0.0f ? max_diag : 1.0f) * shift;
                for (int i = 0; i < n; ++i) {
                    candidate_lu[i * n + i] += diagonal_boost;
                }
            }

            std::vector<int> pivots;
            if (!lu_factorize(candidate_lu, pivots, n)) {
                return false;
            }

            level.dense_lu = std::move(candidate_lu);
            level.dense_pivots = std::move(pivots);
            return true;
        };

        if (!try_factorize(dense, 0.0f) &&
            !try_factorize(dense, 1e-8f) &&
            !try_factorize(dense, 1e-6f) &&
            !try_factorize(dense, 1e-4f)) {
            std::fprintf(stderr,
                         "AMG coarse host LU factorization failed; disabling exact coarse solve.\n");
            return false;
        }

        level.direct_rhs_host.assign(n, 0.0f);
        level.direct_solution_host.assign(n, 0.0f);
        return true;
    }

    bool smooth(Level& level, const float* d_rhs, float* d_x, int sweeps) {
        if (sweeps <= 0) {
            return true;
        }

        const int blocks = (level.num_rows + kThreads - 1) / kThreads;
        for (int sweep = 0; sweep < sweeps; ++sweep) {
            level.matrix->multiply(d_x, level.d_temp);
            residual_kernel<<<blocks, kThreads>>>(d_rhs, level.d_temp, level.d_residual,
                                                  level.num_rows);
            if (cudaGetLastError() != cudaSuccess) {
                return false;
            }
            apply_block_jacobi_kernel<<<blocks, kThreads>>>(level.d_inv_diag_blocks,
                                                            level.d_residual,
                                                            level.d_temp,
                                                            level.num_rows,
                                                            level.block_size);
            if (cudaGetLastError() != cudaSuccess) {
                return false;
            }
            axpy_kernel<<<blocks, kThreads>>>(d_x, level.d_temp, params_.relaxation,
                                              level.num_rows);
            if (cudaGetLastError() != cudaSuccess) {
                return false;
            }
        }

        return true;
    }

    bool computeResidual(Level& level, const float* d_rhs, const float* d_x) {
        level.matrix->multiply(d_x, level.d_temp);
        const int blocks = (level.num_rows + kThreads - 1) / kThreads;
        residual_kernel<<<blocks, kThreads>>>(d_rhs, level.d_temp, level.d_residual,
                                              level.num_rows);
        return cudaGetLastError() == cudaSuccess;
    }

    bool directSolve(const Level& level, const float* d_rhs, float* d_out) {
        Level& mutable_level = const_cast<Level&>(level);
        CHECK_CUDA_BOOL(cudaMemcpy(mutable_level.direct_rhs_host.data(),
                                   d_rhs,
                                   level.num_rows * sizeof(float),
                                   cudaMemcpyDeviceToHost));
        lu_solve(level.dense_lu,
                 level.dense_pivots,
                 mutable_level.direct_rhs_host,
                 mutable_level.direct_solution_host,
                 level.num_rows);
        CHECK_CUDA_BOOL(cudaMemcpy(d_out,
                                   mutable_level.direct_solution_host.data(),
                                   level.num_rows * sizeof(float),
                                   cudaMemcpyHostToDevice));
        return true;
    }

    bool applyLevel(size_t level_idx, const float* d_rhs, float* d_out) {
        Level& level = levels_[level_idx];
        CHECK_CUDA_BOOL(cudaMemset(d_out, 0, level.num_rows * sizeof(float)));

        if (level_idx + 1 == levels_.size()) {
            if (level.has_direct_solver) {
                return directSolve(level, d_rhs, d_out);
            }
            return smooth(level, d_rhs, d_out, params_.coarse_sweeps);
        }

        if (!smooth(level, d_rhs, d_out, params_.pre_sweeps)) {
            return false;
        }
        if (!computeResidual(level, d_rhs, d_out)) {
            return false;
        }

        const Level& coarse_level = levels_[level_idx + 1];
        CHECK_CUDA_BOOL(cudaMemset(level.d_coarse_rhs, 0, coarse_level.num_rows * sizeof(float)));
        const int blocks = (level.num_rows + kThreads - 1) / kThreads;
        restrict_blocks_kernel<<<blocks, kThreads>>>(level.d_aggregate_ids,
                                                     level.d_aggregate_weights,
                                                     level.d_residual,
                                                     level.d_coarse_rhs,
                                                     level.num_rows,
                                                     level.block_size);
        if (cudaGetLastError() != cudaSuccess) {
            return false;
        }

        if (!applyLevel(level_idx + 1, level.d_coarse_rhs, level.d_coarse_x)) {
            return false;
        }

        prolongate_blocks_kernel<<<blocks, kThreads>>>(level.d_aggregate_ids,
                                                       level.d_aggregate_weights,
                                                       level.d_coarse_x,
                                                       d_out,
                                                       level.num_rows,
                                                       level.block_size);
        if (cudaGetLastError() != cudaSuccess) {
            return false;
        }

        return smooth(level, d_rhs, d_out, params_.post_sweeps);
    }

    AMGParameters params_;
    std::vector<Level> levels_;
};

}  // namespace

const char* preconditionerTypeName(PreconditionerType type) {
    switch (type) {
        case PreconditionerType::None:
            return "none";
        case PreconditionerType::ScalarJacobi:
            return "scalar-jacobi";
        case PreconditionerType::BlockJacobi:
            return "block-jacobi";
        case PreconditionerType::AMG:
            return "amg";
        case PreconditionerType::SAAMG:
            return "sa-amg";
    }
    return "unknown";
}

bool parsePreconditionerType(const std::string& value, PreconditionerType& type_out) {
    std::string normalized(value);
    std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    if (normalized.empty() || normalized == "amg") {
        type_out = PreconditionerType::AMG;
        return true;
    }
    if (normalized == "scalar-jacobi" || normalized == "scalar_jacobi" ||
        normalized == "diag" || normalized == "diagonal") {
        type_out = PreconditionerType::ScalarJacobi;
        return true;
    }
    if (normalized == "block-jacobi" || normalized == "block_jacobi" ||
        normalized == "jacobi") {
        type_out = PreconditionerType::BlockJacobi;
        return true;
    }
    if (normalized == "none" || normalized == "identity") {
        type_out = PreconditionerType::None;
        return true;
    }
    return false;
}

std::unique_ptr<LinearPreconditioner> createPreconditioner(BlockSparseMatrix& A,
                                                           const PreconditionerOptions& options) {
    HostMatrixData host_matrix = exportHostMatrix(A);

    if (options.type == PreconditionerType::None) {
        return std::unique_ptr<LinearPreconditioner>(
            new IdentityPreconditioner(host_matrix.config.num_rows));
    }

    if (options.type == PreconditionerType::ScalarJacobi) {
        std::unique_ptr<ScalarJacobiPreconditioner> preconditioner(
            new ScalarJacobiPreconditioner());
        if (!preconditioner->initialize(host_matrix)) {
            return nullptr;
        }
        return std::move(preconditioner);
    }

    if (options.type == PreconditionerType::BlockJacobi) {
        std::unique_ptr<BlockJacobiPreconditioner> preconditioner(new BlockJacobiPreconditioner());
        if (!preconditioner->initialize(host_matrix)) {
            return nullptr;
        }
        return std::move(preconditioner);
    }

    std::unique_ptr<AMGPreconditioner> amg(new AMGPreconditioner());
    if (amg->initialize(A, options.amg)) {
        return std::move(amg);
    }

    std::cerr << "Warning: AMG preconditioner setup failed, falling back to block Jacobi."
              << std::endl;
    std::unique_ptr<BlockJacobiPreconditioner> fallback(new BlockJacobiPreconditioner());
    if (!fallback->initialize(host_matrix)) {
        return nullptr;
    }
    return std::move(fallback);
}