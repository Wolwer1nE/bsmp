#ifndef BLOCK_SPARSE_MATRIX_H
#define BLOCK_SPARSE_MATRIX_H

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "bsmp_config.h"

struct BlockSparseMatrixConfig {
    int num_rows;
    int num_cols;
    int block_size;  // blocks are squares, no overlaps
    int num_nonzero_blocks;
};

class BlockSparseMatrix {
   public:
    BlockSparseMatrix(const BlockSparseMatrixConfig& config);
    ~BlockSparseMatrix();

    // No copying allowed
    BlockSparseMatrix(const BlockSparseMatrix&) = delete;
    BlockSparseMatrix& operator=(const BlockSparseMatrix&) = delete;

    void initialize(const std::vector<int>& block_rows,
                    const std::vector<int>& block_cols,
                    const std::vector<float>& block_data);

    // Matrix-vector multiplication: y = A * x
    // y will be overwritten
    void multiply(const float* d_x, float* d_y, cudaStream_t stream = 0);

    // Matrix-vector multiplication with addition: y += A * x
    void multiplyAdd(const float* d_x, float* d_y, cudaStream_t stream = 0);

    // Matrix-vector multiplication with transposed matrix: y = A^T * x
    void multiplyTranspose(const float* d_x, float* d_y, cudaStream_t stream = 0);

    // Get matrix configuration
    BlockSparseMatrixConfig getConfig() const { return config_; }

    // Get device pointers to internal data
    const int* getBlockRowsDevice() const { return d_block_rows_; }
    const int* getBlockColsDevice() const { return d_block_cols_; }
    const float* getBlockDataDevice() const { return d_block_data_; }

   private:
    BlockSparseMatrixConfig config_;

    int* d_block_rows_ = nullptr;
    int* d_block_cols_ = nullptr;
    float* d_block_data_ = nullptr;

    float* d_temp_ = nullptr;

    void allocateMemory();
    void freeMemory();
};

#endif  // BLOCK_SPARSE_MATRIX_H
