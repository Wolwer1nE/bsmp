#ifndef BSMP_SCHUR_OPERATOR_H
#define BSMP_SCHUR_OPERATOR_H

#include <vector>

#include <cuda_runtime.h>

#include "piezo_block_system.h"

namespace bsmp {

class SchurOperator {
   public:
    explicit SchurOperator(PiezoBlockSystem& system);
    ~SchurOperator();

    SchurOperator(const SchurOperator&) = delete;
    SchurOperator& operator=(const SchurOperator&) = delete;

    void apply(const float* d_x, float* d_y, cudaStream_t stream = 0);

    std::vector<float> explicitDenseSchur() const;
    int mechanicalDofs() const { return mechanical_dofs_; }
    int electricalDofs() const { return electrical_dofs_; }

   private:
    PiezoBlockSystem& system_;
    int mechanical_dofs_ = 0;
    int electrical_dofs_ = 0;

    float* d_t1_ = nullptr;
    float* d_z_ = nullptr;
    float* d_t2_ = nullptr;

    std::vector<float> phi_lu_;
    std::vector<int> phi_pivots_;
    std::vector<float> host_rhs_;
    std::vector<float> host_solution_;

    void factorizeDielectric();
};

}  // namespace bsmp

#endif  // BSMP_SCHUR_OPERATOR_H