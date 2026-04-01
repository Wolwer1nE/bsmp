#include "sa_amg_preconditioner.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <vector>

namespace bsmp {

bool SAAMGPreconditioner::initialize(const BlockSparseMatrix& c_u,
                                    const SAAMGParameters& parameters) {
    parameters_ = parameters;
    regularized_operator_ = build_regularized_elastic_operator(c_u, parameters.regularization_epsilon);

    PreconditionerOptions options;
    options.type = PreconditionerType::AMG;
    options.amg = parameters.setup_amg;

    backend_ = createPreconditioner(*regularized_operator_.matrix, options);
    return static_cast<bool>(backend_);
}

bool SAAMGPreconditioner::initialize(const BlockSparseMatrix& c_u,
                                    const NodeLayout& layout,
                                    const std::vector<std::vector<float>>& nullspace,
                                    const SAAMGParameters& parameters) {
    parameters_ = parameters;
    regularized_operator_ = build_regularized_elastic_operator(c_u, parameters.regularization_epsilon);
    hierarchy_.reset(new SAAMGHierarchy(build_sa_amg_two_level_hierarchy(*regularized_operator_.matrix,
                                                                         layout,
                                                                         nullspace,
                                                                         parameters.hierarchy)));
    backend_.reset();
    return hierarchy_->isValid();
}

bool SAAMGPreconditioner::apply(const float* d_rhs, float* d_out) {
    if (hierarchy_) {
        const int n = regularized_operator_.matrix->getConfig().num_rows;
        std::vector<float> rhs_host(n, 0.0f);
        std::vector<float> out_host;
        if (cudaMemcpy(rhs_host.data(), d_rhs, n * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
            return false;
        }
        out_host = hierarchy_->applyVcycle(rhs_host);
        return cudaMemcpy(d_out, out_host.data(), n * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess;
    }
    if (!backend_) {
        return false;
    }
    return backend_->apply(d_rhs, d_out);
}

std::unique_ptr<SAAMGPreconditioner> createSAAMGPreconditioner(
    const BlockSparseMatrix& c_u,
    const SAAMGParameters& parameters) {
    std::unique_ptr<SAAMGPreconditioner> preconditioner(new SAAMGPreconditioner());
    if (!preconditioner->initialize(c_u, parameters)) {
        return nullptr;
    }
    return preconditioner;
}

std::unique_ptr<SAAMGPreconditioner> createSAAMGPreconditioner(
    const BlockSparseMatrix& c_u,
    const NodeLayout& layout,
    const std::vector<std::vector<float>>& nullspace,
    const SAAMGParameters& parameters) {
    std::unique_ptr<SAAMGPreconditioner> preconditioner(new SAAMGPreconditioner());
    if (!preconditioner->initialize(c_u, layout, nullspace, parameters)) {
        return nullptr;
    }
    return preconditioner;
}

}  // namespace bsmp