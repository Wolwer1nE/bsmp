#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "elastic_nullspace.h"
#include "deflated_pcg_eigensolver.h"
#include "mass_orthogonalization.h"
#include "piezo_scaling.h"
#include "piezo_block_system.h"
#include "sa_amg_preconditioner.h"
#include "schur_operator.h"

namespace {

bool nearly_equal(float lhs, float rhs, float tolerance = 1e-3f) {
    return std::fabs(lhs - rhs) <= tolerance;
}

std::vector<float> build_diagonal_dense(const std::vector<float>& diagonal) {
    const int n = static_cast<int>(diagonal.size());
    std::vector<float> dense(n * n, 0.0f);
    for (int i = 0; i < n; ++i) {
        dense[i * n + i] = diagonal[i];
    }
    return dense;
}

std::vector<float> sorted_ratios(const std::vector<float>& stiffness_diagonal,
                                 const std::vector<float>& mass_diagonal) {
    if (stiffness_diagonal.size() != mass_diagonal.size()) {
        throw std::invalid_argument("Diagonal vectors must have the same size");
    }

    std::vector<float> eigenvalues(stiffness_diagonal.size(), 0.0f);
    for (size_t i = 0; i < stiffness_diagonal.size(); ++i) {
        eigenvalues[i] = stiffness_diagonal[i] / mass_diagonal[i];
    }
    std::sort(eigenvalues.begin(), eigenvalues.end());
    return eigenvalues;
}

void verify_ritz_helper() {
    const bsmp::RayleighRitz2x2Result ritz = bsmp::solve_smallest_generalized_eigen_2x2(
        2.0f, 0.3f, 5.0f,
        1.0f, 0.0f, 2.0f,
        1e-7f);
    if (!ritz.valid) {
        throw std::runtime_error("2x2 Rayleigh-Ritz helper returned an invalid result on an SPD toy problem");
    }

    const float generalized_norm = ritz.alpha * ritz.alpha * 1.0f +
                                   2.0f * ritz.alpha * ritz.beta * 0.0f +
                                   ritz.beta * ritz.beta * 2.0f;
    if (!nearly_equal(generalized_norm, 1.0f, 1e-4f)) {
        throw std::runtime_error("2x2 Rayleigh-Ritz helper did not return a B-normalized eigenvector");
    }
    if (ritz.eigenvalue < 1.9f || ritz.eigenvalue > 2.1f) {
        throw std::runtime_error("2x2 Rayleigh-Ritz helper returned an unexpected smallest eigenvalue");
    }
}

void verify_case(const std::vector<float>& stiffness_diagonal,
                 const std::vector<float>& mass_diagonal,
                 int num_eigenpairs,
                 int max_iterations,
                 float solver_tolerance,
                 float eigen_tolerance,
                 float residual_tolerance,
                 const std::string& label) {
    const int n = static_cast<int>(stiffness_diagonal.size());

    auto stiffness = bsmp::HostBlockMatrix::fromDense(n, n, build_diagonal_dense(stiffness_diagonal)).toDeviceMatrix();
    auto mass = bsmp::HostBlockMatrix::fromDense(n, n, build_diagonal_dense(mass_diagonal)).toDeviceMatrix();

    bsmp::DeflatedPCGEigenSolverParameters params;
    params.num_eigenpairs = num_eigenpairs;
    params.max_iterations = max_iterations;
    params.tolerance = solver_tolerance;
    params.random_seed = 7u;

    const bsmp::HostPreconditionerApply exact_diagonal_preconditioner =
        [&stiffness_diagonal](const std::vector<float>& rhs, std::vector<float>& out) {
            out.assign(rhs.size(), 0.0f);
            for (size_t i = 0; i < rhs.size(); ++i) {
                out[i] = rhs[i] / stiffness_diagonal[i];
            }
            return true;
        };

    const bsmp::DeflatedPCGEigenResult result = bsmp::solve_deflated_pcg_eigenproblem(bsmp::make_block_matrix_operator(*stiffness),
                                                                                       *mass,
                                                                                       n,
                                                                                       params,
                                                                                       exact_diagonal_preconditioner);
    if (!result.converged) {
        throw std::runtime_error("Deflated PCG eigensolver did not converge on case: " + label);
    }
    if (static_cast<int>(result.eigenpairs.size()) != num_eigenpairs) {
        throw std::runtime_error("Unexpected number of eigenpairs on case: " + label);
    }

    const std::vector<float> reference = sorted_ratios(stiffness_diagonal, mass_diagonal);
    for (int i = 0; i < num_eigenpairs; ++i) {
        const float eigenvalue_error = std::fabs(result.eigenpairs[i].eigenvalue - reference[i]);
        if (eigenvalue_error > eigen_tolerance) {
            throw std::runtime_error("Computed eigenvalue does not match the dense diagonal reference on case: " + label);
        }
        if (result.eigenpairs[i].residual_norm > residual_tolerance) {
            throw std::runtime_error("Computed eigenpair residual is too large on case: " + label);
        }
    }

    for (size_t i = 0; i < result.eigenpairs.size(); ++i) {
        const float norm = bsmp::mass_inner_product(*mass,
                                                    result.eigenpairs[i].eigenvector,
                                                    result.eigenpairs[i].eigenvector);
        if (!nearly_equal(norm, 1.0f, 1e-3f)) {
            throw std::runtime_error("Returned eigenvector is not normalized in the M metric on case: " + label);
        }

        for (size_t j = i + 1; j < result.eigenpairs.size(); ++j) {
            const float coupling = bsmp::mass_inner_product(*mass,
                                                            result.eigenpairs[i].eigenvector,
                                                            result.eigenpairs[j].eigenvector);
            if (!nearly_equal(coupling, 0.0f, 2e-3f)) {
                throw std::runtime_error("Returned eigenvectors are not M-orthogonal on case: " + label);
            }
        }
    }

    std::cout << label << " eigenvalues:";
    for (const auto& eigenpair : result.eigenpairs) {
        std::cout << " " << eigenpair.eigenvalue;
    }
    std::cout << std::endl;
}

bsmp::PiezoBlockSystem build_synthetic_piezo_system(std::vector<float>& coordinates_out) {
    coordinates_out = {
        0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        1.0f, 1.0f, 0.0f,
    };

    const int mechanical_dofs = 12;
    const int electrical_dofs = 4;

    const std::vector<float> c_u = {
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

    const std::vector<float> c_uphi = {
        0.30f, 0.00f, 0.00f, 0.10f,
        0.10f, 0.05f, 0.00f, 0.00f,
        0.05f, 0.00f, 0.10f, 0.00f,
        0.25f, 0.10f, 0.00f, 0.00f,
        0.00f, 0.20f, 0.10f, 0.00f,
        0.00f, 0.00f, 0.20f, 0.10f,
        0.00f, 0.15f, 0.00f, 0.25f,
        0.05f, 0.10f, 0.15f, 0.00f,
        0.10f, 0.00f, 0.25f, 0.05f,
        0.20f, 0.00f, 0.05f, 0.20f,
        0.00f, 0.25f, 0.00f, 0.15f,
        0.15f, 0.10f, 0.10f, 0.00f,
    };

    const std::vector<float> c_phi = {
        3.0f, -1.0f,  0.0f, -1.0f,
       -1.0f,  3.0f, -1.0f, -0.5f,
        0.0f, -1.0f,  2.5f, -0.5f,
       -1.0f, -0.5f, -0.5f,  3.0f,
    };

    const std::vector<float> mass = {
        2,0,0,0,0,0,0,0,0,0,0,0,
        0,2,0,0,0,0,0,0,0,0,0,0,
        0,0,2,0,0,0,0,0,0,0,0,0,
        0,0,0,2,0,0,0,0,0,0,0,0,
        0,0,0,0,2,0,0,0,0,0,0,0,
        0,0,0,0,0,2,0,0,0,0,0,0,
        0,0,0,0,0,0,3,0,0,0,0,0,
        0,0,0,0,0,0,0,3,0,0,0,0,
        0,0,0,0,0,0,0,0,3,0,0,0,
        0,0,0,0,0,0,0,0,0,3,0,0,
        0,0,0,0,0,0,0,0,0,0,3,0,
        0,0,0,0,0,0,0,0,0,0,0,3,
    };

    return bsmp::PiezoBlockSystem::fromDense(c_u, c_uphi, c_phi, mass, mechanical_dofs, electrical_dofs);
}

void verify_schur_case_with_sa_amg() {
    std::vector<float> coordinates;
    bsmp::PiezoBlockSystem system = build_synthetic_piezo_system(coordinates);
    if (!bsmp::apply_grounding_constraint(system.dielectric(), 0)) {
        throw std::runtime_error("Failed to ground synthetic dielectric block in Schur smoke case");
    }

    bsmp::SchurOperator schur(system);
    const bsmp::NodeLayout layout = bsmp::NodeLayout::fromNodeCoordinates(coordinates, false);
    const auto rbm = bsmp::build_rigid_body_modes(layout);
    auto preconditioner = bsmp::createSAAMGPreconditioner(system.mechanicalStiffness(), layout, rbm);
    if (!preconditioner) {
        throw std::runtime_error("Failed to construct SA-AMG preconditioner in Schur smoke case");
    }

    bsmp::DeflatedPCGEigenSolverParameters params;
    params.num_eigenpairs = 1;
    params.max_iterations = 400;
    params.tolerance = 1e-3f;
    params.random_seed = 7u;

    const bsmp::DeflatedPCGEigenResult unpreconditioned = bsmp::solve_deflated_pcg_eigenproblem(schur,
                                                                                                 system.mass(),
                                                                                                 params,
                                                                                                 nullptr);
    const bsmp::DeflatedPCGEigenResult sa_amg = bsmp::solve_deflated_pcg_eigenproblem(schur,
                                                                                       system.mass(),
                                                                                       params,
                                                                                       preconditioner.get());
    if (!unpreconditioned.converged) {
        throw std::runtime_error("Unpreconditioned Schur smoke case failed to converge");
    }
    if (!sa_amg.converged) {
        throw std::runtime_error("SA-AMG Schur smoke case failed to converge on the synthetic toy problem");
    }

    const float lambda_ref = unpreconditioned.eigenpairs.front().eigenvalue;
    const float lambda_amg = sa_amg.eigenpairs.front().eigenvalue;
    if (!(lambda_ref > 0.0f) || !(lambda_amg > 0.0f)) {
        throw std::runtime_error("Synthetic Schur smoke case produced a non-positive eigenvalue");
    }
    const float rel_error = std::fabs(lambda_amg - lambda_ref) / std::max(1.0f, std::fabs(lambda_ref));
    if (rel_error > 5e-2f) {
        throw std::runtime_error("SA-AMG Schur smoke case drifted too far from the unpreconditioned reference eigenvalue");
    }

    std::cout << "synthetic Schur eigenvalue (unpreconditioned vs sa-amg): "
              << lambda_ref << " vs " << lambda_amg << std::endl;
}

}  // namespace

int main() {
    try {
        std::cout << "=== Deflated PCG eigensolver smoke test ===" << std::endl;

        verify_ritz_helper();

        const std::vector<float> baseline_stiffness = {
            1.0f, 2.4f, 4.8f,
            5.6f, 7.2f, 8.4f,
            10.8f, 11.6f, 13.2f,
            15.0f, 17.5f, 19.0f,
        };
        const std::vector<float> baseline_mass = {
            2.0f, 2.0f, 2.0f,
            3.0f, 3.0f, 3.0f,
            4.0f, 4.0f, 4.0f,
            5.0f, 5.0f, 5.0f,
        };

        verify_case(baseline_stiffness,
                    baseline_mass,
                    3,
                    250,
                    1e-5f,
                    2e-3f,
                    5e-4f,
                    "baseline");

        const std::vector<float> clustered_stiffness = {
            1.0000f, 1.0002f, 1.0004f,
            1.0020f, 1.0030f, 1.0100f,
            1.0500f, 1.1000f, 1.3000f,
            1.6000f, 2.0000f, 2.5000f,
        };
        const std::vector<float> clustered_mass(12, 1.0f);

        verify_case(clustered_stiffness,
                    clustered_mass,
                    3,
                    400,
                    1e-4f,
                    3e-3f,
                    1e-3f,
                    "clustered");

        verify_schur_case_with_sa_amg();

        std::cout << "Smoke test passed: deflated PCG eigensolver handles 2x2 Ritz updates, clustered low modes, and a synthetic Schur+SA-AMG case." << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Deflated PCG eigensolver smoke test failed: " << error.what() << std::endl;
        return 1;
    }
}