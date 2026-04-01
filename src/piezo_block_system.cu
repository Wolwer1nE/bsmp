#include "piezo_block_system.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>

#include "triplet_loader.h"

namespace bsmp {

namespace {

std::unique_ptr<BlockSparseMatrix> makeMatrixFromDense(int num_rows,
                                                       int num_cols,
                                                       const std::vector<float>& dense,
                                                       float drop_tolerance) {
    HostBlockMatrix host = HostBlockMatrix::fromDense(num_rows, num_cols, dense, drop_tolerance);
    return host.toDeviceMatrix();
}

}  // namespace

HostBlockMatrix HostBlockMatrix::fromMatrix(const BlockSparseMatrix& matrix) {
    HostBlockMatrix host;
    host.config = matrix.getConfig();
    matrix.copyToHost(host.block_rows, host.block_cols, host.block_data);
    return host;
}

HostBlockMatrix HostBlockMatrix::fromDense(int num_rows,
                                           int num_cols,
                                           const std::vector<float>& dense,
                                           float drop_tolerance) {
    if (num_rows < 0 || num_cols < 0) {
        throw std::invalid_argument("Matrix dimensions must be non-negative");
    }
    if (dense.size() != static_cast<size_t>(num_rows * num_cols)) {
        throw std::invalid_argument("Dense matrix size does not match dimensions");
    }

    HostBlockMatrix host;
    host.config.num_rows = num_rows;
    host.config.num_cols = num_cols;
    host.config.block_size = kBlockSize;

    std::map<std::pair<int, int>, std::vector<float>> blocks;
    for (int row = 0; row < num_rows; ++row) {
        for (int col = 0; col < num_cols; ++col) {
            const float value = dense[row * num_cols + col];
            if (std::fabs(value) <= drop_tolerance) {
                continue;
            }

            const int block_row = row / kBlockSize;
            const int block_col = col / kBlockSize;
            const int local_row = row % kBlockSize;
            const int local_col = col % kBlockSize;
            auto key = std::make_pair(block_row, block_col);

            auto it = blocks.find(key);
            if (it == blocks.end()) {
                std::vector<float> block(kBlockSize * kBlockSize, 0.0f);
                block[local_row * kBlockSize + local_col] = value;
                blocks.emplace(key, std::move(block));
            } else {
                it->second[local_row * kBlockSize + local_col] = value;
            }
        }
    }

    host.config.num_nonzero_blocks = static_cast<int>(blocks.size());
    host.block_rows.reserve(blocks.size());
    host.block_cols.reserve(blocks.size());
    host.block_data.reserve(blocks.size() * kBlockSize * kBlockSize);

    for (const auto& item : blocks) {
        host.block_rows.push_back(item.first.first);
        host.block_cols.push_back(item.first.second);
        host.block_data.insert(host.block_data.end(), item.second.begin(), item.second.end());
    }

    return host;
}

std::vector<float> HostBlockMatrix::toDense() const {
    std::vector<float> dense(config.num_rows * config.num_cols, 0.0f);
    const int block_area = config.block_size * config.block_size;

    for (int idx = 0; idx < config.num_nonzero_blocks; ++idx) {
        const int row_base = block_rows[idx] * config.block_size;
        const int col_base = block_cols[idx] * config.block_size;
        const float* block = block_data.data() + idx * block_area;
        for (int i = 0; i < config.block_size; ++i) {
            const int row = row_base + i;
            if (row >= config.num_rows) {
                continue;
            }
            for (int j = 0; j < config.block_size; ++j) {
                const int col = col_base + j;
                if (col >= config.num_cols) {
                    continue;
                }
                dense[row * config.num_cols + col] += block[i * config.block_size + j];
            }
        }
    }

    return dense;
}

std::unique_ptr<BlockSparseMatrix> HostBlockMatrix::toDeviceMatrix() const {
    std::unique_ptr<BlockSparseMatrix> matrix(new BlockSparseMatrix(config));
    matrix->initialize(block_rows, block_cols, block_data);
    return matrix;
}

PiezoBlockSystem::PiezoBlockSystem(std::unique_ptr<BlockSparseMatrix> c_u,
                                   std::unique_ptr<BlockSparseMatrix> c_uphi,
                                   std::unique_ptr<BlockSparseMatrix> c_phi,
                                   std::unique_ptr<BlockSparseMatrix> m,
                                   int mechanical_dofs,
                                   int electrical_dofs)
    : c_u_(std::move(c_u)),
      c_uphi_(std::move(c_uphi)),
      c_phi_(std::move(c_phi)),
      m_(std::move(m)),
      mechanical_dofs_(mechanical_dofs),
      electrical_dofs_(electrical_dofs) {
    if (!c_u_ || !c_uphi_ || !c_phi_ || !m_) {
        throw std::invalid_argument("PiezoBlockSystem requires all four block matrices");
    }
}

PiezoBlockSystem PiezoBlockSystem::fromDense(const std::vector<float>& c_u_dense,
                                             const std::vector<float>& c_uphi_dense,
                                             const std::vector<float>& c_phi_dense,
                                             const std::vector<float>& m_dense,
                                             int mechanical_dofs,
                                             int electrical_dofs,
                                             float drop_tolerance) {
    return PiezoBlockSystem(
        makeMatrixFromDense(mechanical_dofs, mechanical_dofs, c_u_dense, drop_tolerance),
        makeMatrixFromDense(mechanical_dofs, electrical_dofs, c_uphi_dense, drop_tolerance),
        makeMatrixFromDense(electrical_dofs, electrical_dofs, c_phi_dense, drop_tolerance),
        makeMatrixFromDense(mechanical_dofs, mechanical_dofs, m_dense, drop_tolerance),
        mechanical_dofs,
        electrical_dofs);
}

BlockSparseMatrix& PiezoBlockSystem::mechanicalStiffness() {
    return *c_u_;
}

const BlockSparseMatrix& PiezoBlockSystem::mechanicalStiffness() const {
    return *c_u_;
}

BlockSparseMatrix& PiezoBlockSystem::coupling() {
    return *c_uphi_;
}

const BlockSparseMatrix& PiezoBlockSystem::coupling() const {
    return *c_uphi_;
}

BlockSparseMatrix& PiezoBlockSystem::dielectric() {
    return *c_phi_;
}

const BlockSparseMatrix& PiezoBlockSystem::dielectric() const {
    return *c_phi_;
}

BlockSparseMatrix& PiezoBlockSystem::mass() {
    return *m_;
}

const BlockSparseMatrix& PiezoBlockSystem::mass() const {
    return *m_;
}

bool load_block_sparse_matrix_auto(const std::string& path,
                                   int block_size,
                                   BlockSparseMatrixConfig& config_out,
                                   std::vector<int>& block_rows_out,
                                   std::vector<int>& block_cols_out,
                                   std::vector<float>& block_data_out) {
    std::ifstream in(path);
    if (!in) {
        return false;
    }

    std::string line;
    std::string first_nonempty;
    while (std::getline(in, line)) {
        const size_t pos = line.find_first_not_of(" \t\r\n");
        if (pos == std::string::npos) {
            continue;
        }
        first_nonempty = line.substr(pos);
        break;
    }

    if (!first_nonempty.empty() && first_nonempty.rfind("%%MatrixMarket", 0) == 0) {
        return load_matrix_market_as_block_sparse(path,
                                                  block_size,
                                                  config_out,
                                                  block_rows_out,
                                                  block_cols_out,
                                                  block_data_out);
    }

    return load_triplet_file_as_block_sparse(path,
                                             block_size,
                                             config_out,
                                             block_rows_out,
                                             block_cols_out,
                                             block_data_out);
}

bool load_piezo_block_system_from_files(const std::string& c_u_path,
                                        const std::string& c_uphi_path,
                                        const std::string& c_phi_path,
                                        const std::string& m_path,
                                        int block_size,
                                        PiezoBlockSystem& system_out) {
    BlockSparseMatrixConfig c_u_cfg;
    BlockSparseMatrixConfig c_uphi_cfg;
    BlockSparseMatrixConfig c_phi_cfg;
    BlockSparseMatrixConfig m_cfg;

    std::vector<int> c_u_rows, c_u_cols, c_uphi_rows, c_uphi_cols, c_phi_rows, c_phi_cols, m_rows, m_cols;
    std::vector<float> c_u_data, c_uphi_data, c_phi_data, m_data;

    if (!load_block_sparse_matrix_auto(c_u_path, block_size, c_u_cfg, c_u_rows, c_u_cols, c_u_data) ||
        !load_block_sparse_matrix_auto(c_uphi_path,
                                       block_size,
                                       c_uphi_cfg,
                                       c_uphi_rows,
                                       c_uphi_cols,
                                       c_uphi_data) ||
        !load_block_sparse_matrix_auto(c_phi_path,
                                       block_size,
                                       c_phi_cfg,
                                       c_phi_rows,
                                       c_phi_cols,
                                       c_phi_data) ||
        !load_block_sparse_matrix_auto(m_path, block_size, m_cfg, m_rows, m_cols, m_data)) {
        return false;
    }

    PiezoBlockSystem loaded(
        std::unique_ptr<BlockSparseMatrix>(new BlockSparseMatrix(c_u_cfg)),
        std::unique_ptr<BlockSparseMatrix>(new BlockSparseMatrix(c_uphi_cfg)),
        std::unique_ptr<BlockSparseMatrix>(new BlockSparseMatrix(c_phi_cfg)),
        std::unique_ptr<BlockSparseMatrix>(new BlockSparseMatrix(m_cfg)),
        c_u_cfg.num_rows,
        c_phi_cfg.num_rows);

    loaded.mechanicalStiffness().initialize(c_u_rows, c_u_cols, c_u_data);
    loaded.coupling().initialize(c_uphi_rows, c_uphi_cols, c_uphi_data);
    loaded.dielectric().initialize(c_phi_rows, c_phi_cols, c_phi_data);
    loaded.mass().initialize(m_rows, m_cols, m_data);

    system_out = std::move(loaded);
    return true;
}

}  // namespace bsmp