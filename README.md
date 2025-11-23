# BSMP - Block Sparse Matrix Package

# Requirements and software versions
- C++14 or higher
- CMake 3.18 or higher
- CUDA 12.5 
- Ruby 3+ (for matrix generation scripts)
- MATLAB (optional, for performance comparison scripts)

# Installation Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/Wolwer1nE/bsmp.git
   ```
2. We have `bin/*` scripts for building and running examples.
   To build the project, run:
   ```bash
   bin/g matrix data/100.txt 100 4
   ```
   This will create a block sparse matrix with 100 blocks of size 4x4 and save it to `data/100.txt` with right-hand side vector in `data/rhs_100.txt`.

    After generating the matrix, you can run the multiplication example:
    ```bash
    bin/r mult data/100.txt --mults 10 --verbose
    ``` 

    This will build and run the multiplication example 10 times, printing verbose output.
    It should also work on Ubuntu 22.04 and WSL by hitting F5 in VSCode, but you will have to provide inputs.

## Available Modules

### Computation
### src/triplet_loader
Loads a sparse matrix from a file in triplet format or Matrix Market format into BlockSparseMatrix structure.
### src/block_sparse_matrix
Implements Block Sparse Matrix structure and operations such as matrix-vector multiplication.
To test multiplication, run:
```bashbin/r mult data/100.txt --mults 10 --verbose
```
Data should be generated using the generator script in `generators/matrix.rb`.

### src/bicgstab (IN PROGRESS)
Implements the BiCGSTAB iterative method for solving systems of linear equations with block sparse matrices
### src/generalized_eigen (IN PROGRESS)
Implements methods for solving generalized eigenvalue problems with block sparse matrices.

### Generators 
Scripts for generating test sparse matrices in triplets.
Recomennded convention is to store generated matrices in `data/` directory.
#### matrix.rb
Generates a block sparse matrix in triplet format and saves it to a file.
```bash
ruby generators/matrix.rb output_file.txt [n_blocks] [block_size]
```
- `output_file.txt`: Path to the output file.
- `n_blocks`: (Optional) Number of blocks to generate. Default is 1000.
- `block_size`: (Optional) Size of each block. Default is 4.

Will also generate right-hand side vector with name `rhs_output_file.txt`.

### Matlab
Scripts for comparing BSMP performance with Matlab's built-in functions and visualizing results.

#### spymatrix.m
Generates a sparse matrix in MATLAB format and makes nice plot. Can compute bandwidth.

#### multiplication.m
Multiplies a sparse matrix by a vector

#### slae.m

Solves a system of linear equations with a sparse matrix in MATLAB.
