#include "deflated_pcg_eigensolver.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>

#include "mass_orthogonalization.h"

namespace bsmp {

namespace {

constexpr float kEpsilon = 1e-8f;
constexpr float kMassMetricToleranceCap = 1e-8f;

struct DeviceTransferWorkspace {
    float* d_in = nullptr;
    float* d_out = nullptr;
    int in_capacity = 0;
    int out_capacity = 0;

    ~DeviceTransferWorkspace() {
        cudaFree(d_in);
        cudaFree(d_out);
    }

    void ensureInputCapacity(int size) {
        if (size <= in_capacity) {
            return;
        }
        cudaFree(d_in);
        d_in = nullptr;
        cudaMalloc(&d_in, size * sizeof(float));
        in_capacity = size;
    }

    void ensureOutputCapacity(int size) {
        if (size <= out_capacity) {
            return;
        }
        cudaFree(d_out);
        d_out = nullptr;
        cudaMalloc(&d_out, size * sizeof(float));
        out_capacity = size;
    }
};

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

// Double-precision accumulator for inner products used in the 2x2 Ritz step.
// With single precision and vectors normalized in the M-metric (entries ~1e5),
// Schur application results can be large enough that pairwise cancellation
// violates Cauchy-Schwarz: |a01|^2 > a00*a11.  Accumulating in double avoids
// this and keeps the Ritz matrix positive semi-definite as required.
float double_dot_product(const std::vector<float>& lhs, const std::vector<float>& rhs) {
    if (lhs.size() != rhs.size()) {
        throw std::invalid_argument("Dot product requires vectors of the same size");
    }

    double value = 0.0;
    for (size_t i = 0; i < lhs.size(); ++i) {
        value += static_cast<double>(lhs[i]) * static_cast<double>(rhs[i]);
    }
    return static_cast<float>(value);
}

float l2_norm(const std::vector<float>& vector) {
    return std::sqrt(std::max(dot_product(vector, vector), 0.0f));
}

float max_abs_difference(const std::vector<float>& lhs, const std::vector<float>& rhs) {
    if (lhs.size() != rhs.size()) {
        throw std::invalid_argument("Vector comparison requires vectors of the same size");
    }

    float result = 0.0f;
    for (size_t i = 0; i < lhs.size(); ++i) {
        result = std::max(result, std::fabs(lhs[i] - rhs[i]));
    }
    return result;
}

float max_relative_difference(const std::vector<float>& lhs, const std::vector<float>& rhs) {
    if (lhs.size() != rhs.size()) {
        throw std::invalid_argument("Vector comparison requires vectors of the same size");
    }

    float result = 0.0f;
    for (size_t i = 0; i < lhs.size(); ++i) {
        const float scale = std::max(std::fabs(rhs[i]), 1.0f);
        result = std::max(result, std::fabs(lhs[i] - rhs[i]) / scale);
    }
    return result;
}

std::vector<float> multiply_dense_square(const std::vector<float>& matrix,
                                         int n,
                                         const std::vector<float>& x) {
    if (static_cast<int>(x.size()) != n) {
        throw std::invalid_argument("Dense square matvec dimension mismatch");
    }

    std::vector<float> y(n, 0.0f);
    for (int row = 0; row < n; ++row) {
        float sum = 0.0f;
        for (int col = 0; col < n; ++col) {
            sum += matrix[row * n + col] * x[col];
        }
        y[row] = sum;
    }
    return y;
}

float generalized_2x2_norm(float alpha,
                           float beta,
                           float b00,
                           float b01,
                           float b11) {
    return alpha * alpha * b00 + 2.0f * alpha * beta * b01 + beta * beta * b11;
}

RayleighRitz2x2Result build_generalized_eigenpair_2x2(float lambda,
                                                      float c00,
                                                      float c11,
                                                      float sym_c01,
                                                      float b00,
                                                      float b01,
                                                      float b11,
                                                      float inv_l00,
                                                      float inv_l10,
                                                      float inv_l11,
                                                      float tolerance) {
    RayleighRitz2x2Result result;

    float y0 = -sym_c01;
    float y1 = c00 - lambda;
    if (y0 * y0 + y1 * y1 <= tolerance * tolerance) {
        y0 = c11 - lambda;
        y1 = -sym_c01;
    }
    const float y_norm = std::sqrt(y0 * y0 + y1 * y1);
    if (y_norm <= tolerance) {
        return result;
    }
    y0 /= y_norm;
    y1 /= y_norm;

    const float alpha = inv_l00 * y0 + inv_l10 * y1;
    const float beta = inv_l11 * y1;
    const float b_norm = generalized_2x2_norm(alpha, beta, b00, b01, b11);
    if (b_norm <= tolerance) {
        return result;
    }

    const float inv_norm = 1.0f / std::sqrt(b_norm);
    result.eigenvalue = lambda;
    result.alpha = alpha * inv_norm;
    result.beta = beta * inv_norm;
    result.valid = true;
    return result;
}

void axpy(float alpha, const std::vector<float>& x, std::vector<float>& y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("axpy requires vectors of the same size");
    }

    for (size_t i = 0; i < x.size(); ++i) {
        y[i] += alpha * x[i];
    }
}

std::vector<float> combine(const std::vector<float>& lhs,
                           float alpha,
                           const std::vector<float>& rhs) {
    if (lhs.size() != rhs.size()) {
        throw std::invalid_argument("Vector combination requires vectors of the same size");
    }

    std::vector<float> out(lhs.size(), 0.0f);
    for (size_t i = 0; i < lhs.size(); ++i) {
        out[i] = lhs[i] + alpha * rhs[i];
    }
    return out;
}

std::vector<float> host_apply_block_matrix_with_workspace(BlockSparseMatrix& matrix,
                                                          const std::vector<float>& x,
                                                          DeviceTransferWorkspace& workspace) {
    const BlockSparseMatrixConfig config = matrix.getConfig();
    if (config.num_cols != static_cast<int>(x.size())) {
        throw std::invalid_argument("Matrix application dimension mismatch");
    }

    workspace.ensureInputCapacity(config.num_cols);
    workspace.ensureOutputCapacity(config.num_rows);

    if (cudaMemcpy(workspace.d_in, x.data(), config.num_cols * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        throw std::runtime_error("Failed to upload vector for matrix application");
    }

    matrix.multiply(workspace.d_in, workspace.d_out);

    std::vector<float> y(config.num_rows, 0.0f);
    if (cudaMemcpy(y.data(), workspace.d_out, config.num_rows * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        throw std::runtime_error("Failed to download matrix application result");
    }
    return y;
}

std::vector<float> host_apply_block_matrix(BlockSparseMatrix& matrix,
                                           const std::vector<float>& x) {
    DeviceTransferWorkspace workspace;
    return host_apply_block_matrix_with_workspace(matrix, x, workspace);
}

float mass_inner_product_with_workspace(BlockSparseMatrix& mass,
                                        const std::vector<float>& lhs,
                                        const std::vector<float>& rhs,
                                        DeviceTransferWorkspace& workspace) {
    const std::vector<float> mass_rhs = host_apply_block_matrix_with_workspace(mass, rhs, workspace);
    return dot_product(lhs, mass_rhs);
}

float mass_norm_with_workspace(BlockSparseMatrix& mass,
                               const std::vector<float>& vector,
                               DeviceTransferWorkspace& workspace) {
    return std::sqrt(std::max(mass_inner_product_with_workspace(mass, vector, vector, workspace), 0.0f));
}

void normalize_in_mass_metric_with_workspace(BlockSparseMatrix& mass,
                                             std::vector<float>& vector,
                                             float tolerance,
                                             DeviceTransferWorkspace& workspace) {
    const float norm = mass_norm_with_workspace(mass, vector, workspace);
    if (norm <= tolerance) {
        throw std::runtime_error("Cannot normalize a near-zero vector in mass metric");
    }
    for (float& value : vector) {
        value /= norm;
    }
}

MassDeflationSubspace build_mass_deflation_subspace_with_workspace(
    BlockSparseMatrix& mass,
    const std::vector<std::vector<float>>& candidate_basis,
    float tolerance,
    DeviceTransferWorkspace& workspace) {
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

        const float q_norm = mass_norm_with_workspace(mass, q, workspace);
        if (q_norm <= tolerance) {
            continue;
        }

        for (float& value : q) {
            value /= q_norm;
        }

        subspace.basis.push_back(q);
        subspace.mass_basis.push_back(host_apply_block_matrix_with_workspace(mass, q, workspace));
    }

    return subspace;
}

std::vector<float> host_apply_schur_with_workspace(SchurOperator& schur,
                                                   const std::vector<float>& x,
                                                   DeviceTransferWorkspace& workspace) {
    if (schur.mechanicalDofs() != static_cast<int>(x.size())) {
        throw std::invalid_argument("Schur operator application dimension mismatch");
    }

    workspace.ensureInputCapacity(static_cast<int>(x.size()));
    workspace.ensureOutputCapacity(static_cast<int>(x.size()));

    if (cudaMemcpy(workspace.d_in, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        throw std::runtime_error("Failed to upload vector for Schur application");
    }

    schur.apply(workspace.d_in, workspace.d_out);

    std::vector<float> y(x.size(), 0.0f);
    if (cudaMemcpy(y.data(), workspace.d_out, x.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        throw std::runtime_error("Failed to download Schur application result");
    }
    return y;
}

std::vector<float> host_apply_schur(SchurOperator& schur,
                                    const std::vector<float>& x) {
    DeviceTransferWorkspace workspace;
    return host_apply_schur_with_workspace(schur, x, workspace);
}

bool host_apply_preconditioner_with_workspace(LinearPreconditioner& preconditioner,
                                              int dimension,
                                              const std::vector<float>& rhs,
                                              std::vector<float>& out,
                                              DeviceTransferWorkspace& workspace) {
    if (static_cast<int>(rhs.size()) != dimension) {
        throw std::invalid_argument("Preconditioner application dimension mismatch");
    }

    workspace.ensureInputCapacity(dimension);
    workspace.ensureOutputCapacity(dimension);

    if (cudaMemcpy(workspace.d_in, rhs.data(), dimension * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        throw std::runtime_error("Failed to upload RHS for preconditioner application");
    }

    if (!preconditioner.apply(workspace.d_in, workspace.d_out)) {
        return false;
    }

    out.assign(dimension, 0.0f);
    if (cudaMemcpy(out.data(), workspace.d_out, dimension * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        throw std::runtime_error("Failed to download preconditioner application result");
    }
    return true;
}

bool host_apply_preconditioner(LinearPreconditioner& preconditioner,
                               int dimension,
                               const std::vector<float>& rhs,
                               std::vector<float>& out) {
    DeviceTransferWorkspace workspace;
    return host_apply_preconditioner_with_workspace(preconditioner, dimension, rhs, out, workspace);
}

void project_against_current_state(const std::vector<float>& x,
                                   const std::vector<float>& mass_x,
                                   std::vector<float>& vector) {
    const float denominator = std::max(dot_product(mass_x, x), kEpsilon);
    const float coefficient = dot_product(mass_x, vector) / denominator;
    axpy(-coefficient, x, vector);
}

float evaluate_rayleigh_on_step(float lambda,
                                float b,
                                float c,
                                float m,
                                float tau) {
    const float numerator = lambda + 2.0f * b * tau + c * tau * tau;
    const float denominator = 1.0f + m * tau * tau;
    if (std::fabs(denominator) <= kEpsilon) {
        return std::numeric_limits<float>::infinity();
    }
    return numerator / denominator;
}

std::vector<float> build_ritz_vector(float alpha,
                                     const std::vector<float>& x,
                                     float beta,
                                     const std::vector<float>& p) {
    std::vector<float> out(x.size(), 0.0f);
    for (size_t i = 0; i < x.size(); ++i) {
        out[i] = alpha * x[i] + beta * p[i];
    }
    return out;
}

float choose_step(float lambda,
                  float b,
                  float c,
                  float m) {
    std::vector<float> candidates = {0.0f};

    const float quadratic_a = -b * m;
    const float quadratic_b = c - lambda * m;
    const float quadratic_c = b;

    if (std::fabs(quadratic_a) <= kEpsilon) {
        if (std::fabs(quadratic_b) > kEpsilon) {
            candidates.push_back(-quadratic_c / quadratic_b);
        }
    } else {
        const float discriminant = quadratic_b * quadratic_b - 4.0f * quadratic_a * quadratic_c;
        if (discriminant >= 0.0f) {
            const float sqrt_discriminant = std::sqrt(discriminant);
            candidates.push_back((-quadratic_b + sqrt_discriminant) / (2.0f * quadratic_a));
            candidates.push_back((-quadratic_b - sqrt_discriminant) / (2.0f * quadratic_a));
        }
    }

    if (std::fabs(c) > kEpsilon) {
        candidates.push_back(-b / c);
    }
    if (std::fabs(quadratic_b) > kEpsilon) {
        candidates.push_back(-quadratic_c / quadratic_b);
    }

    float best_tau = 0.0f;
    float best_value = std::numeric_limits<float>::infinity();
    for (float tau : candidates) {
        if (!std::isfinite(tau) || std::fabs(tau) > 1e6f) {
            continue;
        }
        const float value = evaluate_rayleigh_on_step(lambda, b, c, m, tau);
        if (value < best_value) {
            best_value = value;
            best_tau = tau;
        }
    }

    return best_tau;
}

std::vector<float> build_initial_guess(int dimension,
                                       std::mt19937& generator) {
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    std::vector<float> guess(dimension, 0.0f);
    for (float& value : guess) {
        value = distribution(generator);
    }
    return guess;
}

bool fallback_preconditioned_iteration(const HostPreconditionerApply& preconditioner_apply,
                                       BlockSparseMatrix& mass,
                                       const MassDeflationSubspace& deflation,
                                       const std::vector<float>& restart_rhs,
                                       float tolerance,
                                       DeviceTransferWorkspace& workspace,
                                       std::vector<float>& x) {
    std::vector<float> updated = restart_rhs;
    if (preconditioner_apply) {
        if (!preconditioner_apply(restart_rhs, updated)) {
            throw std::runtime_error("Fallback preconditioner application failed in deflated PCG eigensolver");
        }
    }

    project_mass_orthogonal_complement(deflation, updated, tolerance * 0.1f);
    const float updated_norm = mass_norm_with_workspace(mass, updated, workspace);
    if (updated_norm <= tolerance * 0.1f) {
        return false;
    }

    normalize_in_mass_metric_with_workspace(mass, updated, tolerance * 0.1f, workspace);
    x = std::move(updated);
    return true;
}

}  // namespace

RayleighRitz2x2Result solve_smallest_generalized_eigen_2x2(float a00,
                                                           float a01,
                                                           float a11,
                                                           float b00,
                                                           float b01,
                                                           float b11,
                                                           float tolerance) {
    RayleighRitz2x2Result result;

    if (b00 <= tolerance) {
        return result;
    }

    const float l00 = std::sqrt(b00);
    const float l10 = b01 / l00;
    const float trailing = b11 - l10 * l10;
    if (trailing <= tolerance) {
        return result;
    }
    const float l11 = std::sqrt(trailing);

    const float inv_l00 = 1.0f / l00;
    const float inv_l11 = 1.0f / l11;
    const float inv_l10 = -l10 / (l00 * l11);

    const float m00 = inv_l00 * a00;
    const float m01 = inv_l00 * a01;
    const float m10 = inv_l10 * a00 + inv_l11 * a01;
    const float m11 = inv_l10 * a01 + inv_l11 * a11;

    const float c00 = m00 * inv_l00;
    const float c01 = m00 * inv_l10 + m01 * inv_l11;
    const float c10 = m10 * inv_l00;
    const float c11 = m10 * inv_l10 + m11 * inv_l11;
    const float sym_c01 = 0.5f * (c01 + c10);

    const float trace = c00 + c11;
    const float diff = c00 - c11;
    const float discriminant = std::max(diff * diff + 4.0f * sym_c01 * sym_c01, 0.0f);
    const float sqrt_discriminant = std::sqrt(discriminant);
    const float lambda_small = 0.5f * (trace - sqrt_discriminant);
    const float lambda_large = 0.5f * (trace + sqrt_discriminant);

    auto try_candidate = [&](float lambda) {
        return build_generalized_eigenpair_2x2(lambda,
                                               c00,
                                               c11,
                                               sym_c01,
                                               b00,
                                               b01,
                                               b11,
                                               inv_l00,
                                               inv_l10,
                                               inv_l11,
                                               tolerance);
    };

    result = try_candidate(lambda_small);
    if (result.valid) {
        return result;
    }
    result = try_candidate(lambda_large);
    if (result.valid) {
        return result;
    }
    return result;
}

HostLinearOperator make_block_matrix_operator(BlockSparseMatrix& matrix) {
    return [&matrix](const std::vector<float>& x, std::vector<float>& y) {
        y = host_apply_block_matrix(matrix, x);
    };
}

HostLinearOperator make_schur_operator(SchurOperator& schur) {
    return [&schur](const std::vector<float>& x, std::vector<float>& y) {
        y = host_apply_schur(schur, x);
    };
}

HostPreconditionerApply make_host_preconditioner_apply(LinearPreconditioner& preconditioner,
                                                       int dimension) {
    return [&preconditioner, dimension](const std::vector<float>& rhs, std::vector<float>& out) {
        return host_apply_preconditioner(preconditioner, dimension, rhs, out);
    };
}

DeflatedPCGEigenResult solve_deflated_pcg_eigenproblem(
    const HostLinearOperator& stiffness_apply,
    BlockSparseMatrix& mass,
    int dimension,
    const DeflatedPCGEigenSolverParameters& parameters,
    const HostPreconditionerApply& preconditioner_apply) {
    if (!stiffness_apply) {
        throw std::invalid_argument("Deflated PCG eigensolver requires a stiffness operator");
    }
    if (dimension <= 0) {
        throw std::invalid_argument("Deflated PCG eigensolver requires positive dimension");
    }
    if (parameters.num_eigenpairs <= 0) {
        throw std::invalid_argument("Deflated PCG eigensolver requires at least one eigenpair");
    }

    DeflatedPCGEigenResult result;
    result.eigenpairs.reserve(parameters.num_eigenpairs);

    std::mt19937 generator(parameters.random_seed);
    std::vector<std::vector<float>> converged_basis;
    DeviceTransferWorkspace mass_workspace;
    const float metric_tolerance = std::min(parameters.tolerance * 0.1f, kMassMetricToleranceCap);

    for (int eigen_index = 0; eigen_index < parameters.num_eigenpairs; ++eigen_index) {
        MassDeflationSubspace deflation = build_mass_deflation_subspace_with_workspace(mass,
                                                                                       converged_basis,
                                                                                       metric_tolerance,
                                                                                       mass_workspace);

        std::vector<float> x;
        bool initialized = false;
        for (int attempt = 0; attempt < 8 && !initialized; ++attempt) {
            x = build_initial_guess(dimension, generator);
            project_mass_orthogonal_complement(deflation, x, metric_tolerance);
            if (mass_norm_with_workspace(mass, x, mass_workspace) > metric_tolerance) {
                normalize_in_mass_metric_with_workspace(mass, x, metric_tolerance, mass_workspace);
                initialized = true;
            }
        }
        if (!initialized) {
            throw std::runtime_error("Failed to initialize an M-independent starting vector");
        }

        DeflatedPCGEigenpair eigenpair;
        std::vector<float> previous_search;
        float previous_rz = 0.0f;

        for (int iteration = 0; iteration < parameters.max_iterations; ++iteration) {
            std::vector<float> ax;
            stiffness_apply(x, ax);
            std::vector<float> mx = host_apply_block_matrix_with_workspace(mass, x, mass_workspace);

            // Use double-precision accumulators for mass metric and Rayleigh quotient;
            // single-precision causes Cauchy-Schwarz violations in the 2x2 Ritz matrix
            // when the Schur correction term dominates C_u (e.g. strong piezo coupling).
            const float denominator = double_dot_product(x, mx);
            if (denominator <= kEpsilon) {
                throw std::runtime_error("Encountered non-positive M-norm during deflated PCG iteration");
            }

            const float lambda = double_dot_product(x, ax) / denominator;
            std::vector<float> residual = combine(ax, -lambda, mx);
            deflate_gradient(deflation, residual, metric_tolerance);
            project_against_current_state(x, mx, residual);

            const float residual_norm = l2_norm(residual);
            const float scale = std::max({l2_norm(ax), std::fabs(lambda) * l2_norm(mx), 1.0f});
            const float relative_residual = residual_norm / scale;

            if (parameters.verbose) {
                std::cout << "[deflated-pcg] mode " << eigen_index
                          << ", iter " << iteration
                          << ": lambda=" << lambda
                          << ", relative residual=" << relative_residual
                          << std::endl;
            }

            eigenpair.eigenvalue = lambda;
            eigenpair.eigenvector = x;
            eigenpair.iterations = iteration + 1;
            eigenpair.residual_norm = residual_norm;

            if (relative_residual <= parameters.tolerance) {
                eigenpair.converged = true;
                break;
            }

            std::vector<float> z = residual;
            if (preconditioner_apply) {
                if (!preconditioner_apply(residual, z)) {
                    throw std::runtime_error("Deflated PCG preconditioner application failed");
                }
            }

            deflate_preconditioned_vector(deflation, z, parameters.tolerance * 0.1f);
            project_against_current_state(x, mx, z);
            const float raw_rz = dot_product(residual, z);
            const float residual_energy = std::max(dot_product(residual, residual), 0.0f);
            if (!std::isfinite(raw_rz) || raw_rz <= std::max(parameters.tolerance * residual_energy, kEpsilon)) {
                z = residual;
                deflate_preconditioned_vector(deflation, z, metric_tolerance);
                project_against_current_state(x, mx, z);
                previous_search.clear();
                previous_rz = 0.0f;
            }
            if (l2_norm(z) <= parameters.tolerance * 0.1f) {
                if (!fallback_preconditioned_iteration(preconditioner_apply,
                                                       mass,
                                                       deflation,
                                                       residual,
                                                       parameters.tolerance,
                                                       mass_workspace,
                                                       x)) {
                    break;
                }
                previous_search.clear();
                previous_rz = 0.0f;
                continue;
            }

            float beta = 0.0f;
            float current_rz = std::max(dot_product(residual, z), 0.0f);
            if (!std::isfinite(current_rz) || current_rz <= kEpsilon) {
                z = residual;
                project_against_current_state(x, mx, z);
                previous_search.clear();
                previous_rz = 0.0f;
                current_rz = std::max(dot_product(residual, z), 0.0f);
            }
            // In the SA-AMG-preconditioned eigen solve, the linear-CG style
            // recurrence can become unstable on strongly coupled real piezo
            // systems. Use the safer preconditioned steepest-descent direction
            // (with the same local 2x2 Ritz acceleration) whenever an explicit
            // preconditioner is active.
            if (!preconditioner_apply && !previous_search.empty() && previous_rz > kEpsilon) {
                beta = current_rz / previous_rz;
                if (!std::isfinite(beta) || beta < 0.0f || beta > 5.0f) {
                    beta = 0.0f;
                }
            }

            std::vector<float> search(z.size(), 0.0f);
            for (size_t i = 0; i < z.size(); ++i) {
                search[i] = -z[i] + beta * (previous_search.empty() ? 0.0f : previous_search[i]);
            }

            deflate_preconditioned_vector(deflation, search, metric_tolerance);
            project_against_current_state(x, mx, search);
            if (l2_norm(search) <= parameters.tolerance * 0.1f) {
                search = z;
                for (float& value : search) {
                    value = -value;
                }
                project_against_current_state(x, mx, search);
            }
            if (l2_norm(search) <= parameters.tolerance * 0.1f) {
                if (!fallback_preconditioned_iteration(preconditioner_apply,
                                                       mass,
                                                       deflation,
                                                       residual,
                                                       parameters.tolerance,
                                                       mass_workspace,
                                                       x)) {
                    break;
                }
                previous_search.clear();
                previous_rz = 0.0f;
                continue;
            }

            normalize_in_mass_metric_with_workspace(mass, search, metric_tolerance, mass_workspace);

            std::vector<float> ap;
            stiffness_apply(search, ap);
            std::vector<float> mp = host_apply_block_matrix_with_workspace(mass, search, mass_workspace);
            const float m_search = double_dot_product(search, mp);
            if (m_search <= kEpsilon) {
                if (!fallback_preconditioned_iteration(preconditioner_apply,
                                                       mass,
                                                       deflation,
                                                       residual,
                                                       parameters.tolerance,
                                                       mass_workspace,
                                                       x)) {
                    break;
                }
                previous_search.clear();
                previous_rz = 0.0f;
                continue;
            }

            const float a00 = double_dot_product(x, ax);
            const float a01 = double_dot_product(x, ap);
            const float a11 = double_dot_product(search, ap);
            const float b00 = denominator;
            const float b01 = double_dot_product(x, mp);
            const float b11 = m_search;

            auto try_accept_candidate = [&](std::vector<float> candidate,
                                            const char* source_label,
                                            float candidate_step_value) {
                project_mass_orthogonal_complement(deflation, candidate, metric_tolerance);
                const float candidate_mass_norm = mass_norm_with_workspace(mass, candidate, mass_workspace);
                if (!std::isfinite(candidate_mass_norm) || candidate_mass_norm <= metric_tolerance) {
                    return false;
                }

                normalize_in_mass_metric_with_workspace(mass, candidate, metric_tolerance, mass_workspace);

                std::vector<float> candidate_ax;
                stiffness_apply(candidate, candidate_ax);
                std::vector<float> candidate_mx = host_apply_block_matrix_with_workspace(mass, candidate, mass_workspace);
                const float candidate_denominator = double_dot_product(candidate, candidate_mx);
                if (!std::isfinite(candidate_denominator) || candidate_denominator <= kEpsilon) {
                    return false;
                }

                const float candidate_lambda = double_dot_product(candidate, candidate_ax) / candidate_denominator;
                if (!std::isfinite(candidate_lambda) || candidate_lambda <= 0.0f || candidate_lambda > lambda) {
                    if (parameters.verbose) {
                        std::cout << "[deflated-pcg] mode " << eigen_index
                                  << ", iter " << iteration
                                  << ": rejected " << source_label
                                  << " candidate after explicit Rayleigh check; lambda_candidate="
                                  << candidate_lambda
                                  << ", lambda_current=" << lambda
                                  << ", step_value=" << candidate_step_value
                                  << std::endl;
                    }
                    return false;
                }

                x = std::move(candidate);
                return true;
            };

            const RayleighRitz2x2Result ritz = solve_smallest_generalized_eigen_2x2(a00,
                                                                                     a01,
                                                                                     a11,
                                                                                     b00,
                                                                                     b01,
                                                                                     b11,
                                                                                     parameters.tolerance * 0.1f);
            const bool accept_ritz = ritz.valid && std::isfinite(ritz.eigenvalue) &&
                                     ritz.eigenvalue > 0.0f &&
                                     ritz.eigenvalue <= lambda;
            bool accepted_update = false;
            if (accept_ritz) {
                if (parameters.verbose) {
                    std::cout << "[deflated-pcg] mode " << eigen_index
                              << ", iter " << iteration
                              << ": accepted 2x2 Ritz step with lambda=" << ritz.eigenvalue
                              << ", A=[[" << a00 << ", " << a01 << "], [" << a01 << ", " << a11
                              << "]], B=[[" << b00 << ", " << b01 << "], [" << b01 << ", " << b11
                              << "]], alpha=" << ritz.alpha
                              << ", beta=" << ritz.beta
                              << std::endl;
                }
                accepted_update = try_accept_candidate(build_ritz_vector(ritz.alpha, x, ritz.beta, search),
                                                       "Ritz",
                                                       ritz.eigenvalue);
            }

            if (!accepted_update) {
                const float b = a01;
                const float c = a11;
                float tau = choose_step(lambda, b, c, m_search);
                if (parameters.verbose) {
                    std::cout << "[deflated-pcg] mode " << eigen_index
                              << ", iter " << iteration
                              << ": rejected unhealthy 2x2 Ritz step; "
                              << "A=[[" << a00 << ", " << a01 << "], [" << a01 << ", " << a11
                              << "]], B=[[" << b00 << ", " << b01 << "], [" << b01 << ", " << b11
                              << "]], lambda_rr=" << ritz.eigenvalue
                              << ", fallback tau=" << tau
                              << std::endl;
                }
                if (!std::isfinite(tau) || std::fabs(tau) <= kEpsilon) {
                    const float fallback_denominator = c - lambda * m_search;
                    if (std::fabs(fallback_denominator) > kEpsilon) {
                        tau = -b / fallback_denominator;
                    }
                }
                if (!std::isfinite(tau) || std::fabs(tau) <= kEpsilon) {
                    if (!fallback_preconditioned_iteration(preconditioner_apply,
                                                           mass,
                                                           deflation,
                                                           residual,
                                                           parameters.tolerance,
                                                           mass_workspace,
                                                           x)) {
                        break;
                    }
                    previous_search.clear();
                    previous_rz = 0.0f;
                    continue;
                }

                float trial_tau = tau;
                for (int backtrack = 0; backtrack < 8; ++backtrack) {
                    if (!std::isfinite(trial_tau) || std::fabs(trial_tau) <= kEpsilon) {
                        break;
                    }
                    if (try_accept_candidate(combine(x, trial_tau, search), "line-search", trial_tau)) {
                        accepted_update = true;
                        break;
                    }
                    trial_tau *= 0.5f;
                }

                if (!accepted_update) {
                    std::vector<float> opposite_search = search;
                    for (float& value : opposite_search) {
                        value = -value;
                    }
                    trial_tau = std::fabs(tau);
                    for (int backtrack = 0; backtrack < 8; ++backtrack) {
                        if (!std::isfinite(trial_tau) || std::fabs(trial_tau) <= kEpsilon) {
                            break;
                        }
                        if (try_accept_candidate(combine(x, trial_tau, opposite_search),
                                                 "opposite-line-search",
                                                 trial_tau)) {
                            accepted_update = true;
                            search = std::move(opposite_search);
                            break;
                        }
                        trial_tau *= 0.5f;
                    }
                }

                if (!accepted_update) {
                    std::vector<float> residual_restart = residual;
                    for (float& value : residual_restart) {
                        value = -value;
                    }
                    deflate_preconditioned_vector(deflation, residual_restart, metric_tolerance);
                    project_against_current_state(x, mx, residual_restart);
                    const float residual_restart_norm = l2_norm(residual_restart);
                    if (std::isfinite(residual_restart_norm) && residual_restart_norm > parameters.tolerance * 0.1f) {
                        float residual_tau = 1.0f / std::max(residual_restart_norm, 1.0f);
                        for (int backtrack = 0; backtrack < 12; ++backtrack) {
                            if (!std::isfinite(residual_tau) || std::fabs(residual_tau) <= kEpsilon) {
                                break;
                            }
                            if (try_accept_candidate(combine(x, residual_tau, residual_restart),
                                                     "residual-restart",
                                                     residual_tau)) {
                                accepted_update = true;
                                search = std::move(residual_restart);
                                break;
                            }
                            residual_tau *= 0.5f;
                        }
                    }
                }

                if (!accepted_update) {
                    if (!fallback_preconditioned_iteration(preconditioner_apply,
                                                           mass,
                                                           deflation,
                                                           residual,
                                                           parameters.tolerance,
                                                           mass_workspace,
                                                           x)) {
                        break;
                    }
                    previous_search.clear();
                    previous_rz = 0.0f;
                    continue;
                }
            }

            previous_search = search;
            previous_rz = current_rz;
        }

        result.total_iterations += eigenpair.iterations;
        result.converged = result.converged && eigenpair.converged;
        result.eigenpairs.push_back(std::move(eigenpair));

        if (!result.eigenpairs.back().converged) {
            break;
        }
        converged_basis.push_back(result.eigenpairs.back().eigenvector);
    }

    return result;
}

DeflatedPCGEigenResult solve_deflated_pcg_eigenproblem(
    BlockSparseMatrix& stiffness,
    BlockSparseMatrix& mass,
    const DeflatedPCGEigenSolverParameters& parameters,
    LinearPreconditioner* preconditioner) {
    std::shared_ptr<DeviceTransferWorkspace> workspace = std::make_shared<DeviceTransferWorkspace>();
    HostPreconditionerApply host_preconditioner;
    if (preconditioner) {
        host_preconditioner = [preconditioner, workspace, dimension = stiffness.getConfig().num_rows](const std::vector<float>& rhs,
                                                                                                      std::vector<float>& out) {
            return host_apply_preconditioner_with_workspace(*preconditioner, dimension, rhs, out, *workspace);
        };
    }

    HostLinearOperator stiffness_apply = [&stiffness, workspace](const std::vector<float>& x, std::vector<float>& y) {
        y = host_apply_block_matrix_with_workspace(stiffness, x, *workspace);
    };

    return solve_deflated_pcg_eigenproblem(stiffness_apply,
                                           mass,
                                           stiffness.getConfig().num_rows,
                                           parameters,
                                           host_preconditioner);
}

DeflatedPCGEigenResult solve_deflated_pcg_eigenproblem(
    SchurOperator& stiffness,
    BlockSparseMatrix& mass,
    const DeflatedPCGEigenSolverParameters& parameters,
    LinearPreconditioner* preconditioner) {
    std::shared_ptr<DeviceTransferWorkspace> workspace = std::make_shared<DeviceTransferWorkspace>();
    struct SchurApplyDiagnostics {
        int call_count = 0;
        float worst_abs_error = 0.0f;
        float worst_rel_error = 0.0f;
        int worst_call_index = -1;
    };

    std::shared_ptr<SchurApplyDiagnostics> diagnostics = std::make_shared<SchurApplyDiagnostics>();
    const bool compare_dense_schur = parameters.verbose;
    std::shared_ptr<std::vector<float>> dense_schur;
    if (compare_dense_schur) {
        dense_schur = std::make_shared<std::vector<float>>(stiffness.explicitDenseSchur());
    }

    HostPreconditionerApply host_preconditioner;
    if (preconditioner) {
        host_preconditioner = [preconditioner, workspace, dimension = stiffness.mechanicalDofs()](const std::vector<float>& rhs,
                                                                                                  std::vector<float>& out) {
            return host_apply_preconditioner_with_workspace(*preconditioner, dimension, rhs, out, *workspace);
        };
    }

    HostLinearOperator stiffness_apply = [&stiffness, workspace, diagnostics, dense_schur, compare_dense_schur](const std::vector<float>& x, std::vector<float>& y) {
        y = host_apply_schur_with_workspace(stiffness, x, *workspace);

        if (!compare_dense_schur || !dense_schur) {
            return;
        }

        const std::vector<float> dense_y = multiply_dense_square(*dense_schur, stiffness.mechanicalDofs(), x);
        const float abs_error = max_abs_difference(y, dense_y);
        const float rel_error = max_relative_difference(y, dense_y);

        ++diagnostics->call_count;
        if (rel_error > diagnostics->worst_rel_error || abs_error > diagnostics->worst_abs_error) {
            diagnostics->worst_abs_error = std::max(diagnostics->worst_abs_error, abs_error);
            diagnostics->worst_rel_error = std::max(diagnostics->worst_rel_error, rel_error);
            diagnostics->worst_call_index = diagnostics->call_count;
        }

        if (diagnostics->call_count <= 12 || rel_error > 1e-3f) {
            std::cout << "[schur-compare] call " << diagnostics->call_count
                      << ": max_abs_error=" << abs_error
                      << ", max_rel_error=" << rel_error
                      << std::endl;
        }
    };

    DeflatedPCGEigenResult result = solve_deflated_pcg_eigenproblem(stiffness_apply,
                                                                    mass,
                                                                    stiffness.mechanicalDofs(),
                                                                    parameters,
                                                                    host_preconditioner);

    if (compare_dense_schur && diagnostics->call_count > 0) {
        std::cout << "[schur-compare] summary: calls=" << diagnostics->call_count
                  << ", worst_abs_error=" << diagnostics->worst_abs_error
                  << ", worst_rel_error=" << diagnostics->worst_rel_error
                  << ", worst_call=" << diagnostics->worst_call_index
                  << std::endl;
    }

    return result;
}

}  // namespace bsmp