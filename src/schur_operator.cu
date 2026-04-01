#include "schur_operator.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace bsmp {

namespace {

#define CHECK_CUDA_SCHUR(call)                                                           \
    do {                                                                                 \
        cudaError_t err__ = (call);                                                      \
        if (err__ != cudaSuccess) {                                                      \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
                         cudaGetErrorString(err__));                                     \
            throw std::runtime_error("CUDA failure inside SchurOperator");              \
        }                                                                                \
    } while (0)

constexpr int kAxpyThreads = 128;
constexpr float kPivotTolerance = 1e-12f;

__global__ void vector_add_kernel(float* y, const float* x, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] += x[idx];
    }
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
            const float candidate_abs = std::fabs(lu[row * n + col]);
            if (candidate_abs > pivot_abs) {
                pivot_abs = candidate_abs;
                pivot_row = row;
            }
        }

        if (pivot_abs <= kPivotTolerance) {
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

std::vector<float> invert_from_lu(const std::vector<float>& lu,
                                  const std::vector<int>& pivots,
                                  int n) {
    std::vector<float> inverse(n * n, 0.0f);
    std::vector<float> rhs(n, 0.0f);
    std::vector<float> column;

    for (int col = 0; col < n; ++col) {
        std::fill(rhs.begin(), rhs.end(), 0.0f);
        rhs[col] = 1.0f;
        lu_solve(lu, pivots, rhs, column, n);
        for (int row = 0; row < n; ++row) {
            inverse[row * n + col] = column[row];
        }
    }

    return inverse;
}

}  // namespace

SchurOperator::SchurOperator(PiezoBlockSystem& system)
    : system_(system),
      mechanical_dofs_(system.mechanicalDofs()),
      electrical_dofs_(system.electricalDofs()),
      host_rhs_(system.electricalDofs(), 0.0f),
      host_solution_(system.electricalDofs(), 0.0f) {
    const auto c_u_cfg = system_.mechanicalStiffness().getConfig();
    const auto c_uphi_cfg = system_.coupling().getConfig();
    const auto c_phi_cfg = system_.dielectric().getConfig();

    if (c_u_cfg.num_rows != c_u_cfg.num_cols) {
        throw std::invalid_argument("C_u must be square");
    }
    if (c_phi_cfg.num_rows != c_phi_cfg.num_cols) {
        throw std::invalid_argument("C_phi must be square");
    }
    if (c_uphi_cfg.num_rows != c_u_cfg.num_rows || c_uphi_cfg.num_cols != c_phi_cfg.num_rows) {
        throw std::invalid_argument("C_uphi dimensions must match C_u rows and C_phi rows");
    }

    CHECK_CUDA_SCHUR(cudaMalloc(&d_t1_, electrical_dofs_ * sizeof(float)));
    CHECK_CUDA_SCHUR(cudaMalloc(&d_z_, electrical_dofs_ * sizeof(float)));
    CHECK_CUDA_SCHUR(cudaMalloc(&d_t2_, mechanical_dofs_ * sizeof(float)));

    factorizeDielectric();
}

SchurOperator::~SchurOperator() {
    if (d_t1_ != nullptr) {
        cudaFree(d_t1_);
    }
    if (d_z_ != nullptr) {
        cudaFree(d_z_);
    }
    if (d_t2_ != nullptr) {
        cudaFree(d_t2_);
    }
}

void SchurOperator::factorizeDielectric() {
    const HostBlockMatrix host_phi = HostBlockMatrix::fromMatrix(system_.dielectric());
    phi_lu_ = host_phi.toDense();
    if (!lu_factorize(phi_lu_, phi_pivots_, electrical_dofs_)) {
        throw std::runtime_error("C_phi factorization failed; grounding may be missing");
    }
}

void SchurOperator::apply(const float* d_x, float* d_y, cudaStream_t stream) {
    system_.coupling().multiplyTranspose(d_x, d_t1_, stream);
    CHECK_CUDA_SCHUR(cudaStreamSynchronize(stream));
    CHECK_CUDA_SCHUR(cudaMemcpy(host_rhs_.data(),
                                d_t1_,
                                electrical_dofs_ * sizeof(float),
                                cudaMemcpyDeviceToHost));

    lu_solve(phi_lu_, phi_pivots_, host_rhs_, host_solution_, electrical_dofs_);
    CHECK_CUDA_SCHUR(cudaMemcpy(d_z_,
                                host_solution_.data(),
                                electrical_dofs_ * sizeof(float),
                                cudaMemcpyHostToDevice));

    system_.mechanicalStiffness().multiply(d_x, d_y, stream);
    system_.coupling().multiply(d_z_, d_t2_, stream);

    const int blocks = (mechanical_dofs_ + kAxpyThreads - 1) / kAxpyThreads;
    vector_add_kernel<<<blocks, kAxpyThreads, 0, stream>>>(d_y, d_t2_, mechanical_dofs_);
    CHECK_CUDA_SCHUR(cudaGetLastError());
}

std::vector<float> SchurOperator::explicitDenseSchur() const {
    const std::vector<float> c_u = HostBlockMatrix::fromMatrix(system_.mechanicalStiffness()).toDense();
    const std::vector<float> c_uphi = HostBlockMatrix::fromMatrix(system_.coupling()).toDense();
    std::vector<float> phi_inverse = invert_from_lu(phi_lu_, phi_pivots_, electrical_dofs_);

    std::vector<float> temp(mechanical_dofs_ * electrical_dofs_, 0.0f);
    for (int row = 0; row < mechanical_dofs_; ++row) {
        for (int col = 0; col < electrical_dofs_; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < electrical_dofs_; ++k) {
                sum += c_uphi[row * electrical_dofs_ + k] * phi_inverse[k * electrical_dofs_ + col];
            }
            temp[row * electrical_dofs_ + col] = sum;
        }
    }

    std::vector<float> schur = c_u;
    for (int row = 0; row < mechanical_dofs_; ++row) {
        for (int col = 0; col < mechanical_dofs_; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < electrical_dofs_; ++k) {
                sum += temp[row * electrical_dofs_ + k] * c_uphi[col * electrical_dofs_ + k];
            }
            schur[row * mechanical_dofs_ + col] += sum;
        }
    }

    return schur;
}

}  // namespace bsmp