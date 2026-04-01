#ifndef BSMP_SA_AMG_PRECONDITIONER_H
#define BSMP_SA_AMG_PRECONDITIONER_H

#include <memory>

#include "amg_preconditioner.h"
#include "elastic_regularization.h"
#include "sa_amg_hierarchy.h"

namespace bsmp {

struct SAAMGParameters {
    float regularization_epsilon = 1e-6f;
    AMGParameters setup_amg;
    SAAMGHierarchyParameters hierarchy;
};

class SAAMGPreconditioner final : public LinearPreconditioner {
   public:
    SAAMGPreconditioner() = default;
    ~SAAMGPreconditioner() override = default;

    bool initialize(const BlockSparseMatrix& c_u, const SAAMGParameters& parameters = SAAMGParameters{});
    bool initialize(const BlockSparseMatrix& c_u,
                    const NodeLayout& layout,
                    const std::vector<std::vector<float>>& nullspace,
                    const SAAMGParameters& parameters = SAAMGParameters{});

    bool apply(const float* d_rhs, float* d_out) override;
    PreconditionerType type() const override { return PreconditionerType::SAAMG; }

    float alpha() const { return regularized_operator_.alpha; }
    const BlockSparseMatrix& setupMatrix() const { return *regularized_operator_.matrix; }

   private:
    SAAMGParameters parameters_{};
    RegularizedElasticOperator regularized_operator_{};
    std::unique_ptr<SAAMGHierarchy> hierarchy_;
    std::unique_ptr<LinearPreconditioner> backend_;
};

std::unique_ptr<SAAMGPreconditioner> createSAAMGPreconditioner(
    const BlockSparseMatrix& c_u,
    const SAAMGParameters& parameters = SAAMGParameters{});

std::unique_ptr<SAAMGPreconditioner> createSAAMGPreconditioner(
    const BlockSparseMatrix& c_u,
    const NodeLayout& layout,
    const std::vector<std::vector<float>>& nullspace,
    const SAAMGParameters& parameters = SAAMGParameters{});

}  // namespace bsmp

#endif  // BSMP_SA_AMG_PRECONDITIONER_H