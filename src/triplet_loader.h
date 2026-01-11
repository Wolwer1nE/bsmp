#ifndef TRIPLET_LOADER_H
#define TRIPLET_LOADER_H

#include <string>
#include <vector>

#include "block_sparse_matrix.h"

// Read a file with triplets (i j v) where i and j are zero-based integer
// row/col indices and v is a floating point value. Lines starting with '#'
// or empty lines are ignored.
//
// This function will coalesce values that fall into the same block (by summing
// their values inside the block) and produce vectors suitable for
// BlockSparseMatrix::initialize: block_rows, block_cols and block_data.
//
// Parameters:
//  - path: path to triplet file
//  - block_size: block size in elements (blocks are square)
//  - config_out: populated BlockSparseMatrixConfig (num_rows/num_cols inferred
//                as max index+1)
//  - block_rows_out, block_cols_out, block_data_out: outputs for initialize
//
// Returns true on success, false on parse / I/O errors.
bool load_triplet_file_as_block_sparse(const std::string& path,
                                       int block_size,
                                       BlockSparseMatrixConfig& config_out,
                                       std::vector<int>& block_rows_out,
                                       std::vector<int>& block_cols_out,
                                       std::vector<float>& block_data_out);

// Convenience: build a BlockSparseMatrix instance (on host) from a triplet file.
// Caller owns the returned pointer. Returns nullptr on failure.
BlockSparseMatrix* load_triplet_file_to_matrix(const std::string& path,
                                               int block_size);

// Load a Matrix Market file (coordinate, real). Supports 'general' and will
// expand 'symmetric' by mirroring off-diagonal entries.
bool load_matrix_market_as_block_sparse(const std::string& path,
                                        int block_size,
                                        BlockSparseMatrixConfig& config_out,
                                        std::vector<int>& block_rows_out,
                                        std::vector<int>& block_cols_out,
                                        std::vector<float>& block_data_out);

// Convenience wrapper for Matrix Market
BlockSparseMatrix* load_matrix_market_to_matrix(const std::string& path,
                                                int block_size);

#endif  // TRIPLET_LOADER_H
