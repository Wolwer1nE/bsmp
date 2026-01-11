#include <cuda_runtime.h>

#include <cassert>
#include <iostream>

#include "block_sparse_matrix.h"

int THREADS_COUNT = 128;

#define CHECK_CUDA(call)                                                 \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << ": " << cudaGetErrorString(err) << std::endl;   \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

__global__ void block_sparse_matvec_kernel(
    const int* block_rows,
    const int* block_cols,
    const float* block_data,
    const float* x,
    float* y,
    int num_nonzero_blocks,
    int block_size,
    int num_rows,
    int num_cols) {
    // Our blocks are small, we place multiple blocks per CUDA block
    int blocks_per_cuda_block = blockDim.x / block_size;
    int local_block_id = threadIdx.x / block_size;
    int local_row = threadIdx.x % block_size;

    int matrix_block_idx = blockIdx.x * blocks_per_cuda_block + local_block_id;
    if (matrix_block_idx >= num_nonzero_blocks)
        return;

    int row_block = block_rows[matrix_block_idx];
    int col_block = block_cols[matrix_block_idx];
    const float* block = block_data + matrix_block_idx * block_size * block_size;

    int global_row = row_block * block_size + local_row;
    if (global_row >= num_rows)
        return;

    // scalar product of the row of the block with vector x
    float sum = 0.0f;
#pragma unroll
    for (int j = 0; j < block_size; j++) {
        int global_col = col_block * block_size + j;
        if (global_col < num_cols) {
            sum += block[local_row * block_size + j] * x[global_col];
        }
    }

    // Blocks have no overlap in rows or columns
    atomicAdd(&y[global_row], sum);
}

// Transpose multiplication kernel
__global__ void block_sparse_matvec_transpose_kernel(
    const int* block_rows,
    const int* block_cols,
    const float* block_data,
    const float* x,
    float* y,
    int num_nonzero_blocks,
    int block_size,
    int num_rows,
    int num_cols) {
    int matrix_block_idx = blockIdx.x;
    if (matrix_block_idx >= num_nonzero_blocks)
        return;

    int row_block = block_rows[matrix_block_idx];
    int col_block = block_cols[matrix_block_idx];
    const float* block = block_data + matrix_block_idx * block_size * block_size;

    // Each thread computes one column of the block
    int local_col = threadIdx.x;
    if (local_col >= block_size)
        return;

    int global_col = col_block * block_size + local_col;
    if (global_col >= num_cols)
        return;

    float sum = 0.0f;
    for (int i = 0; i < block_size; i++) {
        int global_row = row_block * block_size + i;
        if (global_row < num_rows) {
            float matrix_val = block[i * block_size + local_col];  // Транспонирование
            float vector_val = x[global_row];
            sum += matrix_val * vector_val;
        }
    }

    atomicAdd(&y[global_col], sum);
}

__global__ void initialize_vector_kernel(float* vec, int size, float value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        vec[idx] = value;
    }
}

BlockSparseMatrix::BlockSparseMatrix(const BlockSparseMatrixConfig& config)
    : config_(config) {
    // Enforce compile-time block size
    if (config_.block_size != bsmp::kBlockSize) {
        std::cerr << "Error: Block size mismatch. Provided " << config_.block_size
                  << ", but compiled with BSMP_BLOCK_SIZE=" << bsmp::kBlockSize << std::endl;
        std::cerr << "Recompile or adjust input to match compile-time block size." << std::endl;
        std::abort();
    }
    allocateMemory();
}

BlockSparseMatrix::~BlockSparseMatrix() {
    freeMemory();
}

void BlockSparseMatrix::allocateMemory() {
    int num_blocks = config_.num_nonzero_blocks;
    int block_area = config_.block_size * config_.block_size;

    CHECK_CUDA(cudaMalloc(&d_block_rows_, num_blocks * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_block_cols_, num_blocks * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_block_data_, num_blocks * block_area * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_temp_, config_.num_rows * sizeof(float)));
}

void BlockSparseMatrix::freeMemory() {
    if (d_block_rows_)
        cudaFree(d_block_rows_);
    if (d_block_cols_)
        cudaFree(d_block_cols_);
    if (d_block_data_)
        cudaFree(d_block_data_);
    if (d_temp_)
        cudaFree(d_temp_);
}

void BlockSparseMatrix::initialize(const std::vector<int>& block_rows,
                                   const std::vector<int>& block_cols,
                                   const std::vector<float>& block_data) {
    assert(block_rows.size() == config_.num_nonzero_blocks);
    assert(block_cols.size() == config_.num_nonzero_blocks);
    assert(block_data.size() == config_.num_nonzero_blocks *
                                    config_.block_size * config_.block_size);

    CHECK_CUDA(cudaMemcpy(d_block_rows_, block_rows.data(),
                          block_rows.size() * sizeof(int),
                          cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(d_block_cols_, block_cols.data(),
                          block_cols.size() * sizeof(int),
                          cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(d_block_data_, block_data.data(),
                          block_data.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
}

void BlockSparseMatrix::multiply(const float* d_x, float* d_y, cudaStream_t stream) {
    // Обнуляем выходной вектор перед вычислением
    cudaMemsetAsync(d_y, 0, config_.num_rows * sizeof(float), stream);

    int threads = 128;  // FIXME: I am not sure about this value. P.A.
    int matrix_blocks_per_cuda_block = threads / config_.block_size;
    if (matrix_blocks_per_cuda_block < 1)
        matrix_blocks_per_cuda_block = 1;
    int cuda_blocks = (config_.num_nonzero_blocks + matrix_blocks_per_cuda_block - 1) / matrix_blocks_per_cuda_block;

    block_sparse_matvec_kernel<<<cuda_blocks, threads, 0, stream>>>(
        d_block_rows_,
        d_block_cols_,
        d_block_data_,
        d_x,
        d_y,
        config_.num_nonzero_blocks,
        config_.block_size,
        config_.num_rows,
        config_.num_cols);
    CHECK_CUDA(cudaGetLastError());
}

void BlockSparseMatrix::multiplyAdd(const float* d_x, float* d_y, cudaStream_t stream) {
    // Accumulate result without zeroing: y += A * x
    int threads = THREADS_COUNT;
    int matrix_blocks_per_cuda_block = threads / config_.block_size;
    if (matrix_blocks_per_cuda_block < 1)
        matrix_blocks_per_cuda_block = 1;
    int cuda_blocks = (config_.num_nonzero_blocks + matrix_blocks_per_cuda_block - 1) / matrix_blocks_per_cuda_block;
    block_sparse_matvec_kernel<<<cuda_blocks, threads, 0, stream>>>(
        d_block_rows_,
        d_block_cols_,
        d_block_data_,
        d_x,
        d_y,
        config_.num_nonzero_blocks,
        config_.block_size,
        config_.num_rows,
        config_.num_cols);
    CHECK_CUDA(cudaGetLastError());
}

void BlockSparseMatrix::multiplyTranspose(const float* d_x, float* d_y, cudaStream_t stream) {
    cudaMemsetAsync(d_y, 0, config_.num_cols * sizeof(float), stream);

    int threads = config_.block_size;
    threads = THREADS_COUNT;

    block_sparse_matvec_transpose_kernel<<<config_.num_nonzero_blocks, threads, 0, stream>>>(
        d_block_rows_,
        d_block_cols_,
        d_block_data_,
        d_x,
        d_y,
        config_.num_nonzero_blocks,
        config_.block_size,
        config_.num_rows,
        config_.num_cols);

    CHECK_CUDA(cudaGetLastError());
}
