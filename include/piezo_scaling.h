#ifndef BSMP_PIEZO_SCALING_H
#define BSMP_PIEZO_SCALING_H

#include <vector>

#include "piezo_block_system.h"

namespace bsmp {

struct PiezoEquilibrationScaling {
    std::vector<float> mechanical_inv_sqrt_diag;
    std::vector<float> electrical_inv_sqrt_diag;
};

bool apply_grounding_constraint(BlockSparseMatrix& c_phi,
                                int grounded_dof,
                                BlockSparseMatrix* auxiliary_mass = nullptr);

PiezoEquilibrationScaling build_identity_scaling(const PiezoBlockSystem& system);

PiezoEquilibrationScaling build_field_scaling(const PiezoBlockSystem& system);

PiezoEquilibrationScaling build_equilibration_scaling(const PiezoBlockSystem& system);

std::unique_ptr<BlockSparseMatrix> scale_matrix(const BlockSparseMatrix& matrix,
                                                const std::vector<float>& left_scaling,
                                                const std::vector<float>& right_scaling);

PiezoBlockSystem scale_piezo_system(const PiezoBlockSystem& system,
                                    const PiezoEquilibrationScaling& scaling);

}  // namespace bsmp

#endif  // BSMP_PIEZO_SCALING_H