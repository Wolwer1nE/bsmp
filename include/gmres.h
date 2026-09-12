#ifndef BSMP_GMRES_H
#define BSMP_GMRES_H

#include <cuda_runtime.h>

#include "amg_preconditioner.h"
#include "block_sparse_matrix.h"

/// @brief Restarted GMRES solver for `Ax = b` with block-Jacobi left preconditioning.
/// @param A `BlockSparseMatrix` in BSMP block format (uses device block matvec)
/// @param d_b device pointer to RHS vector of length `A.num_rows`
/// @param d_x device pointer to initial guess and output solution of length `A.num_rows`
/// @param max_iters iteration cap across all restart cycles
/// @param restart Krylov subspace size before restart
/// @param tol relative tolerance on true residual norm ||b - Ax|| / ||b||
/// @param iters_out number of iterations performed
/// @param resid_out final true relative residual norm
/// @return true on convergence, false otherwise
bool gmres(BlockSparseMatrix& A,
           const float* d_b,
           float* d_x,
           int max_iters,
           int restart,
           float tol,
           int& iters_out,
           float& resid_out);

bool gmres(BlockSparseMatrix& A,
           const float* d_b,
           float* d_x,
           int max_iters,
           int restart,
           float tol,
           int& iters_out,
           float& resid_out,
           LinearPreconditioner& preconditioner);

bool gmres(BlockSparseMatrix& A,
           const float* d_b,
           float* d_x,
           int max_iters,
           int restart,
           float tol,
           int& iters_out,
           float& resid_out,
           const PreconditionerOptions& preconditioner_options);

#endif  // BSMP_GMRES_H