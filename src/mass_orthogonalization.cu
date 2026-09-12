#include "mass_orthogonalization.h"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>

namespace bsmp {

namespace {

float dot_product(const std::vector<float>& lhs, const std::vector<float>& rhs) {
    if (lhs.size() != rhs.size()) {
        throw std::invalid_argument("Dot product requires vectors of the same size");
    }

    float value = 0.0f;
    for (size_t i = 0; i < lhs.size(); ++i) {
        value += lhs[i] * rhs[i];
    }
    return value;
}

}  // namespace

std::vector<float> apply_mass_matrix(BlockSparseMatrix& mass,
                                     const std::vector<float>& vector) {
    const BlockSparseMatrixConfig config = mass.getConfig();
    if (config.num_cols != static_cast<int>(vector.size())) {
        throw std::invalid_argument("Mass matrix application dimension mismatch");
    }

    float* d_x = nullptr;
    float* d_y = nullptr;
    cudaMalloc(&d_x, config.num_cols * sizeof(float));
    cudaMalloc(&d_y, config.num_rows * sizeof(float));

    auto cleanup = [&]() {
        cudaFree(d_x);
        cudaFree(d_y);
    };

    if (cudaMemcpy(d_x, vector.data(), config.num_cols * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        cleanup();
        throw std::runtime_error("Failed to upload vector for mass matrix application");
    }

    mass.multiply(d_x, d_y);

    std::vector<float> out(config.num_rows, 0.0f);
    if (cudaMemcpy(out.data(), d_y, config.num_rows * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        cleanup();
        throw std::runtime_error("Failed to download result of mass matrix application");
    }

    cleanup();
    return out;
}

float mass_inner_product(BlockSparseMatrix& mass,
                         const std::vector<float>& lhs,
                         const std::vector<float>& rhs) {
    if (lhs.size() != rhs.size()) {
        throw std::invalid_argument("Mass inner product requires vectors of the same size");
    }

    const std::vector<float> mass_rhs = apply_mass_matrix(mass, rhs);
    return dot_product(lhs, mass_rhs);
}

float mass_norm(BlockSparseMatrix& mass,
                const std::vector<float>& vector) {
    return std::sqrt(std::max(mass_inner_product(mass, vector, vector), 0.0f));
}

void normalize_in_mass_metric(BlockSparseMatrix& mass,
                              std::vector<float>& vector,
                              float tolerance) {
    const float norm = mass_norm(mass, vector);
    if (norm <= tolerance) {
        throw std::runtime_error("Cannot normalize a near-zero vector in mass metric");
    }

    for (float& value : vector) {
        value /= norm;
    }
}

MassDeflationSubspace build_mass_deflation_subspace(
    BlockSparseMatrix& mass,
    const std::vector<std::vector<float>>& candidate_basis,
    float tolerance) {
    MassDeflationSubspace subspace;
    if (candidate_basis.empty()) {
        return subspace;
    }

    const size_t dimension = candidate_basis.front().size();
    for (const auto& candidate : candidate_basis) {
        if (candidate.size() != dimension) {
            throw std::invalid_argument("All basis vectors must have the same size");
        }

        std::vector<float> q = candidate;
        for (size_t basis_index = 0; basis_index < subspace.basis.size(); ++basis_index) {
            const float coefficient = dot_product(subspace.mass_basis[basis_index], q);
            for (size_t i = 0; i < q.size(); ++i) {
                q[i] -= coefficient * subspace.basis[basis_index][i];
            }
        }

        const float q_norm = mass_norm(mass, q);
        if (q_norm <= tolerance) {
            continue;
        }

        for (float& value : q) {
            value /= q_norm;
        }

        subspace.basis.push_back(q);
        subspace.mass_basis.push_back(apply_mass_matrix(mass, q));
    }

    return subspace;
}

void project_mass_orthogonal_complement(const MassDeflationSubspace& subspace,
                                        std::vector<float>& vector,
                                        float tolerance) {
    if (subspace.empty()) {
        return;
    }
    if (subspace.basis.size() != subspace.mass_basis.size()) {
        throw std::invalid_argument("Mass deflation subspace basis and mass basis are inconsistent");
    }
    if (vector.size() != subspace.basis.front().size()) {
        throw std::invalid_argument("Projection vector size does not match deflation subspace");
    }

    for (size_t basis_index = 0; basis_index < subspace.basis.size(); ++basis_index) {
        const float coefficient = dot_product(subspace.mass_basis[basis_index], vector);
        if (std::fabs(coefficient) <= tolerance) {
            continue;
        }
        for (size_t i = 0; i < vector.size(); ++i) {
            vector[i] -= coefficient * subspace.basis[basis_index][i];
        }
    }
}

void deflate_gradient(const MassDeflationSubspace& subspace,
                      std::vector<float>& gradient,
                      float tolerance) {
    // Paper Algorithm 2 lines 9/22: g := g - M*U*(U^T*M*U)^{-1}*U^T*g
    // With M-orthonormal U this is: g -= sum_i (M*u_i) * (u_i^T * g)
    // i.e. subtract mass_basis[i] * dot(basis[i], g).
    // NOTE: deflate_preconditioned_vector uses the transpose formula
    //       (project_mass_orthogonal_complement), which is correct for y.
    if (subspace.empty()) {
        return;
    }
    if (subspace.basis.size() != subspace.mass_basis.size()) {
        throw std::invalid_argument("Mass deflation subspace basis and mass basis are inconsistent");
    }
    if (gradient.size() != subspace.basis.front().size()) {
        throw std::invalid_argument("Gradient size does not match deflation subspace");
    }
    for (size_t basis_index = 0; basis_index < subspace.basis.size(); ++basis_index) {
        const float coefficient = dot_product(subspace.basis[basis_index], gradient);
        if (std::fabs(coefficient) <= tolerance) {
            continue;
        }
        for (size_t i = 0; i < gradient.size(); ++i) {
            gradient[i] -= coefficient * subspace.mass_basis[basis_index][i];
        }
    }
}

void deflate_preconditioned_vector(const MassDeflationSubspace& subspace,
                                   std::vector<float>& vector,
                                   float tolerance) {
    project_mass_orthogonal_complement(subspace, vector, tolerance);
}

}  // namespace bsmp