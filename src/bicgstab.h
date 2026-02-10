#ifndef BSMP_BICGSTAB_H
#define BSMP_BICGSTAB_H

#include <cuda_runtime.h>

#include "block_sparse_matrix.h"

/// @brief BiCGStab (Bi-Conjugate Gradient Stabilized) solver for `Ax = b`.
/// @note Suitable for non-symmetric, non-positive-definite matrices
/// @param A `BlockSparseMatrix` (uses multiply on device)
/// @param d_b device pointer to RHS vector of length `A.num_rows`
/// @param d_x device pointer to initial guess and output solution of length `A.num_rows`
/// @param max_iters iteration cap
/// @param tol relative tolerance on residual norm ||r|| / ||b||
/// @param iters_out number of iterations performed
/// @param resid_out final relative residual norm
/// @return true on convergence (within tol) or false if max iters reached or breakdown
bool bicgstab(BlockSparseMatrix& A,
              const float* d_b,
              float* d_x,
              int max_iters,
              float tol,
              int& iters_out,
              float& resid_out);

#endif  // BSMP_BICGSTAB_H
