#ifndef BSMP_PIEZO_BLOCK_SYSTEM_H
#define BSMP_PIEZO_BLOCK_SYSTEM_H

#include <memory>
#include <string>
#include <vector>

#include "block_sparse_matrix.h"

namespace bsmp {

struct HostBlockMatrix {
    BlockSparseMatrixConfig config{};
    std::vector<int> block_rows;
    std::vector<int> block_cols;
    std::vector<float> block_data;

    static HostBlockMatrix fromMatrix(const BlockSparseMatrix& matrix);
    static HostBlockMatrix fromDense(int num_rows,
                                     int num_cols,
                                     const std::vector<float>& dense,
                                     float drop_tolerance = 0.0f);

    std::vector<float> toDense() const;
    std::unique_ptr<BlockSparseMatrix> toDeviceMatrix() const;
};

class PiezoBlockSystem {
   public:
    PiezoBlockSystem(std::unique_ptr<BlockSparseMatrix> c_u,
                     std::unique_ptr<BlockSparseMatrix> c_uphi,
                     std::unique_ptr<BlockSparseMatrix> c_phi,
                     std::unique_ptr<BlockSparseMatrix> m,
                     int mechanical_dofs,
                     int electrical_dofs);

    static PiezoBlockSystem fromDense(const std::vector<float>& c_u_dense,
                                      const std::vector<float>& c_uphi_dense,
                                      const std::vector<float>& c_phi_dense,
                                      const std::vector<float>& m_dense,
                                      int mechanical_dofs,
                                      int electrical_dofs,
                                      float drop_tolerance = 0.0f);

    BlockSparseMatrix& mechanicalStiffness();
    const BlockSparseMatrix& mechanicalStiffness() const;

    BlockSparseMatrix& coupling();
    const BlockSparseMatrix& coupling() const;

    BlockSparseMatrix& dielectric();
    const BlockSparseMatrix& dielectric() const;

    BlockSparseMatrix& mass();
    const BlockSparseMatrix& mass() const;

    int mechanicalDofs() const { return mechanical_dofs_; }
    int electricalDofs() const { return electrical_dofs_; }

   private:
    std::unique_ptr<BlockSparseMatrix> c_u_;
    std::unique_ptr<BlockSparseMatrix> c_uphi_;
    std::unique_ptr<BlockSparseMatrix> c_phi_;
    std::unique_ptr<BlockSparseMatrix> m_;
    int mechanical_dofs_ = 0;
    int electrical_dofs_ = 0;
};

bool load_block_sparse_matrix_auto(const std::string& path,
                                   int block_size,
                                   BlockSparseMatrixConfig& config_out,
                                   std::vector<int>& block_rows_out,
                                   std::vector<int>& block_cols_out,
                                   std::vector<float>& block_data_out);

bool load_piezo_block_system_from_files(const std::string& c_u_path,
                                        const std::string& c_uphi_path,
                                        const std::string& c_phi_path,
                                        const std::string& m_path,
                                        int block_size,
                                        PiezoBlockSystem& system_out);

}  // namespace bsmp

#endif  // BSMP_PIEZO_BLOCK_SYSTEM_H