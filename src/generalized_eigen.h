#ifndef GENERALIZED_EIGEN_H
#define GENERALIZED_EIGEN_H

#include <vector>

#include "block_sparse_matrix.h"

namespace bsmp {

struct EigenResult {
    std::vector<float> eigenvalues;
    std::vector<std::vector<float>> eigenvectors;
    int iterations;
    bool converged;
};

/// @brief Solves the generalized eigenvalue problem: `A*v = λ*B*v`
/// @param A left matrix, stiffness matrix
/// @param B right matrix, mass matrix, must be positive definite (for now!!)
/// @param num_eigenvalues number of eigenvalues to find
/// @param max_iter iteration limit
/// @param tol tolerance for convergence
/// @return `EigenResult`
EigenResult solveGeneralizedEigen(
    BlockSparseMatrix& A,
    BlockSparseMatrix& B,
    int num_eigenvalues = 6,
    int max_iter = 1000,
    float tol = 1e-6f);

/// @brief Solves the standard eigenvalue problem: `A*v = λ*v`
/// @param A matrix to solve eigenvalues for
/// @param num_eigenvalues number of eigenvalues to find
/// @param max_iter iteration limit
/// @param tol tolerance for convergence
/// @return `EigenResult`
EigenResult solveEigen(
    BlockSparseMatrix& A,
    int num_eigenvalues = 6,
    int max_iter = 1000,
    float tol = 1e-6f);

}  // namespace bsmp

#endif  // GENERALIZED_EIGEN_H
