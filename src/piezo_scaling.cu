#include "piezo_scaling.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace bsmp {

namespace {

std::vector<float> extract_absolute_diagonal(const BlockSparseMatrix& matrix) {
    const HostBlockMatrix host = HostBlockMatrix::fromMatrix(matrix);
    const std::vector<float> dense = host.toDense();
    std::vector<float> diagonal(host.config.num_rows, 0.0f);
    const int stride = host.config.num_cols;

    for (int i = 0; i < host.config.num_rows; ++i) {
        diagonal[i] = std::fabs(dense[i * stride + i]);
    }
    return diagonal;
}

std::vector<float> build_inv_sqrt(const std::vector<float>& diagonal) {
    std::vector<float> inv_sqrt(diagonal.size(), 1.0f);
    for (size_t i = 0; i < diagonal.size(); ++i) {
        if (diagonal[i] > std::numeric_limits<float>::epsilon()) {
            inv_sqrt[i] = 1.0f / std::sqrt(diagonal[i]);
        }
    }
    return inv_sqrt;
}

std::vector<float> build_constant_scaling(int size, float diagonal_value) {
    std::vector<float> scaling(std::max(size, 0), 1.0f);
    if (diagonal_value > std::numeric_limits<float>::epsilon()) {
        const float factor = 1.0f / std::sqrt(diagonal_value);
        std::fill(scaling.begin(), scaling.end(), factor);
    }
    return scaling;
}

float mean_positive_diagonal(const std::vector<float>& diagonal) {
    double sum = 0.0;
    int count = 0;
    for (float value : diagonal) {
        if (value > std::numeric_limits<float>::epsilon()) {
            sum += static_cast<double>(value);
            ++count;
        }
    }
    if (count <= 0) {
        return 1.0f;
    }
    return static_cast<float>(sum / static_cast<double>(count));
}

void zero_row_and_column(std::vector<float>& dense,
                         int rows,
                         int cols,
                         int pivot,
                         float diagonal_value) {
    for (int col = 0; col < cols; ++col) {
        dense[pivot * cols + col] = 0.0f;
    }
    for (int row = 0; row < rows; ++row) {
        dense[row * cols + pivot] = 0.0f;
    }
    dense[pivot * cols + pivot] = diagonal_value;
}

void overwrite_existing_block_values(const std::vector<float>& dense,
                                     HostBlockMatrix& host) {
    const int block_size = host.config.block_size;
    const int block_area = block_size * block_size;
    if (static_cast<int>(host.block_data.size()) != host.config.num_nonzero_blocks * block_area) {
        throw std::invalid_argument("Host block data size does not match stored block structure");
    }

    for (int idx = 0; idx < host.config.num_nonzero_blocks; ++idx) {
        const int row_base = host.block_rows[idx] * block_size;
        const int col_base = host.block_cols[idx] * block_size;
        float* block = host.block_data.data() + idx * block_area;
        for (int local_row = 0; local_row < block_size; ++local_row) {
            const int row = row_base + local_row;
            for (int local_col = 0; local_col < block_size; ++local_col) {
                const int col = col_base + local_col;
                const int offset = local_row * block_size + local_col;
                if (row < host.config.num_rows && col < host.config.num_cols) {
                    block[offset] = dense[row * host.config.num_cols + col];
                } else {
                    block[offset] = 0.0f;
                }
            }
        }
    }
}

}  // namespace

bool apply_grounding_constraint(BlockSparseMatrix& c_phi,
                                int grounded_dof,
                                BlockSparseMatrix* auxiliary_mass) {
    HostBlockMatrix c_phi_host = HostBlockMatrix::fromMatrix(c_phi);
    if (grounded_dof < 0 || grounded_dof >= c_phi_host.config.num_rows ||
        grounded_dof >= c_phi_host.config.num_cols) {
        return false;
    }

    std::vector<float> c_phi_dense = c_phi_host.toDense();
    zero_row_and_column(c_phi_dense,
                        c_phi_host.config.num_rows,
                        c_phi_host.config.num_cols,
                        grounded_dof,
                        1.0f);

    overwrite_existing_block_values(c_phi_dense, c_phi_host);
    c_phi.initialize(c_phi_host.block_rows, c_phi_host.block_cols, c_phi_host.block_data);

    if (auxiliary_mass != nullptr) {
        HostBlockMatrix mass_host = HostBlockMatrix::fromMatrix(*auxiliary_mass);
        if (grounded_dof < mass_host.config.num_rows && grounded_dof < mass_host.config.num_cols) {
            std::vector<float> mass_dense = mass_host.toDense();
            zero_row_and_column(mass_dense,
                                mass_host.config.num_rows,
                                mass_host.config.num_cols,
                                grounded_dof,
                                0.0f);
            overwrite_existing_block_values(mass_dense, mass_host);
            auxiliary_mass->initialize(mass_host.block_rows,
                                       mass_host.block_cols,
                                       mass_host.block_data);
        }
    }

    return true;
}

PiezoEquilibrationScaling build_identity_scaling(const PiezoBlockSystem& system) {
    PiezoEquilibrationScaling scaling;
    scaling.mechanical_inv_sqrt_diag.assign(system.mechanicalDofs(), 1.0f);
    scaling.electrical_inv_sqrt_diag.assign(system.electricalDofs(), 1.0f);
    return scaling;
}

PiezoEquilibrationScaling build_field_scaling(const PiezoBlockSystem& system) {
    PiezoEquilibrationScaling scaling;
    scaling.mechanical_inv_sqrt_diag = build_constant_scaling(
        system.mechanicalDofs(),
        mean_positive_diagonal(extract_absolute_diagonal(system.mechanicalStiffness())));
    scaling.electrical_inv_sqrt_diag = build_constant_scaling(
        system.electricalDofs(),
        mean_positive_diagonal(extract_absolute_diagonal(system.dielectric())));
    return scaling;
}

PiezoEquilibrationScaling build_equilibration_scaling(const PiezoBlockSystem& system) {
    PiezoEquilibrationScaling scaling;
    scaling.mechanical_inv_sqrt_diag = build_inv_sqrt(extract_absolute_diagonal(system.mechanicalStiffness()));
    scaling.electrical_inv_sqrt_diag = build_inv_sqrt(extract_absolute_diagonal(system.dielectric()));
    return scaling;
}

std::unique_ptr<BlockSparseMatrix> scale_matrix(const BlockSparseMatrix& matrix,
                                                const std::vector<float>& left_scaling,
                                                const std::vector<float>& right_scaling) {
    HostBlockMatrix host = HostBlockMatrix::fromMatrix(matrix);
    if (static_cast<int>(left_scaling.size()) != host.config.num_rows ||
        static_cast<int>(right_scaling.size()) != host.config.num_cols) {
        throw std::invalid_argument("Scaling vectors do not match matrix dimensions");
    }

    std::vector<float> dense = host.toDense();
    for (int row = 0; row < host.config.num_rows; ++row) {
        for (int col = 0; col < host.config.num_cols; ++col) {
            dense[row * host.config.num_cols + col] *= left_scaling[row] * right_scaling[col];
        }
    }

    return HostBlockMatrix::fromDense(host.config.num_rows, host.config.num_cols, dense).toDeviceMatrix();
}

PiezoBlockSystem scale_piezo_system(const PiezoBlockSystem& system,
                                    const PiezoEquilibrationScaling& scaling) {
    return PiezoBlockSystem(
        scale_matrix(system.mechanicalStiffness(),
                     scaling.mechanical_inv_sqrt_diag,
                     scaling.mechanical_inv_sqrt_diag),
        scale_matrix(system.coupling(),
                     scaling.mechanical_inv_sqrt_diag,
                     scaling.electrical_inv_sqrt_diag),
        scale_matrix(system.dielectric(),
                     scaling.electrical_inv_sqrt_diag,
                     scaling.electrical_inv_sqrt_diag),
        scale_matrix(system.mass(),
                     scaling.mechanical_inv_sqrt_diag,
                     scaling.mechanical_inv_sqrt_diag),
        system.mechanicalDofs(),
        system.electricalDofs());
}

}  // namespace bsmp