#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "mass_orthogonalization.h"
#include "piezo_block_system.h"

namespace {

bool nearly_equal(float lhs, float rhs, float tolerance = 1e-4f) {
    return std::fabs(lhs - rhs) <= tolerance;
}

}  // namespace

int main() {
    try {
        std::cout << "=== Mass orthogonalization smoke test ===" << std::endl;

        const int n = 6;
        const std::vector<float> mass_dense = {
            2.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 3.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 5.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 7.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 11.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 13.0f,
        };

        std::unique_ptr<BlockSparseMatrix> mass = bsmp::HostBlockMatrix::fromDense(n, n, mass_dense).toDeviceMatrix();

        std::vector<std::vector<float>> candidate_basis = {
            {1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f},
            {0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f},
            {1.0f, -1.0f, 0.5f, 0.0f, 0.0f, 0.0f},
        };

        const bsmp::MassDeflationSubspace subspace = bsmp::build_mass_deflation_subspace(*mass,
                                                                                           candidate_basis,
                                                                                           1e-6f);
        if (subspace.size() != 3) {
            throw std::runtime_error("Expected three independent M-orthonormal vectors in the deflation subspace");
        }

        for (int i = 0; i < subspace.size(); ++i) {
            const float norm = bsmp::mass_inner_product(*mass, subspace.basis[i], subspace.basis[i]);
            if (!nearly_equal(norm, 1.0f, 5e-4f)) {
                throw std::runtime_error("Subspace vector is not normalized in the M metric");
            }

            for (int j = i + 1; j < subspace.size(); ++j) {
                const float coupling = bsmp::mass_inner_product(*mass, subspace.basis[i], subspace.basis[j]);
                if (!nearly_equal(coupling, 0.0f, 5e-4f)) {
                    throw std::runtime_error("Subspace vectors are not M-orthogonal");
                }
            }
        }

        std::vector<float> gradient = {1.0f, 2.0f, -3.0f, 4.0f, 5.0f, -6.0f};
        bsmp::deflate_gradient(subspace, gradient);
        for (int i = 0; i < subspace.size(); ++i) {
            const float coupling = bsmp::mass_inner_product(*mass, subspace.basis[i], gradient);
            if (!nearly_equal(coupling, 0.0f, 5e-4f)) {
                throw std::runtime_error("Deflated gradient is not M-orthogonal to the converged subspace");
            }
        }

        std::vector<float> preconditioned = {0.5f, -1.0f, 1.5f, -2.0f, 2.5f, -3.0f};
        bsmp::deflate_preconditioned_vector(subspace, preconditioned);
        for (int i = 0; i < subspace.size(); ++i) {
            const float coupling = bsmp::mass_inner_product(*mass, subspace.basis[i], preconditioned);
            if (!nearly_equal(coupling, 0.0f, 5e-4f)) {
                throw std::runtime_error("Deflated preconditioned vector is not M-orthogonal to the converged subspace");
            }
        }

        std::vector<float> normalized = {1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f};
        bsmp::normalize_in_mass_metric(*mass, normalized);
        const float normalized_norm = bsmp::mass_inner_product(*mass, normalized, normalized);
        if (!nearly_equal(normalized_norm, 1.0f, 5e-4f)) {
            throw std::runtime_error("normalize_in_mass_metric did not produce unit M-norm");
        }

        std::cout << "Smoke test passed: M-orthogonalization and deflation layer behaves as expected." << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Mass orthogonalization smoke test failed: " << error.what() << std::endl;
        return 1;
    }
}