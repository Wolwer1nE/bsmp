#ifndef BSMP_DEFLATED_PCG_EIGENSOLVER_H
#define BSMP_DEFLATED_PCG_EIGENSOLVER_H

#include <functional>
#include <vector>

#include "amg_preconditioner.h"
#include "block_sparse_matrix.h"
#include "schur_operator.h"

namespace bsmp {

using HostLinearOperator = std::function<void(const std::vector<float>&, std::vector<float>&)>;
using HostPreconditionerApply = std::function<bool(const std::vector<float>&, std::vector<float>&)>;

struct DeflatedPCGEigenSolverParameters {
    int num_eigenpairs = 6;
    int max_iterations = 200;
    float tolerance = 1e-6f;
    unsigned int random_seed = 1234u;
    bool verbose = false;
};

struct DeflatedPCGEigenpair {
    float eigenvalue = 0.0f;
    std::vector<float> eigenvector;
    int iterations = 0;
    float residual_norm = 0.0f;
    bool converged = false;
};

struct DeflatedPCGEigenResult {
    std::vector<DeflatedPCGEigenpair> eigenpairs;
    int total_iterations = 0;
    bool converged = true;
};

struct RayleighRitz2x2Result {
    float eigenvalue = 0.0f;
    float alpha = 1.0f;
    float beta = 0.0f;
    bool valid = false;
};

RayleighRitz2x2Result solve_smallest_generalized_eigen_2x2(float a00,
                                                           float a01,
                                                           float a11,
                                                           float b00,
                                                           float b01,
                                                           float b11,
                                                           float tolerance = 1e-7f);

HostLinearOperator make_block_matrix_operator(BlockSparseMatrix& matrix);
HostLinearOperator make_schur_operator(SchurOperator& schur);
HostPreconditionerApply make_host_preconditioner_apply(LinearPreconditioner& preconditioner,
                                                       int dimension);

DeflatedPCGEigenResult solve_deflated_pcg_eigenproblem(
    const HostLinearOperator& stiffness_apply,
    BlockSparseMatrix& mass,
    int dimension,
    const DeflatedPCGEigenSolverParameters& parameters = DeflatedPCGEigenSolverParameters{},
    const HostPreconditionerApply& preconditioner_apply = HostPreconditionerApply{});

DeflatedPCGEigenResult solve_deflated_pcg_eigenproblem(
    BlockSparseMatrix& stiffness,
    BlockSparseMatrix& mass,
    const DeflatedPCGEigenSolverParameters& parameters = DeflatedPCGEigenSolverParameters{},
    LinearPreconditioner* preconditioner = nullptr);

DeflatedPCGEigenResult solve_deflated_pcg_eigenproblem(
    SchurOperator& stiffness,
    BlockSparseMatrix& mass,
    const DeflatedPCGEigenSolverParameters& parameters = DeflatedPCGEigenSolverParameters{},
    LinearPreconditioner* preconditioner = nullptr);

}  // namespace bsmp

#endif  // BSMP_DEFLATED_PCG_EIGENSOLVER_H