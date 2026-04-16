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
    bin/r mult data/100.txt --rhs data/rhs_100.txt --mults 10 --verbose
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
```bash
bin/r mult data/100.txt --rhs data/rhs_100.txt --mults 10 --verbose
```
Data should be generated using the generator script in `generators/matrix.rb`.

### src/bicgstab 
Implements the BiCGSTAB iterative method for solving systems of linear equations with block sparse matrices

`BiCGStab` now uses AMG as the default left preconditioner. For explicit comparison you can choose `amg`, `scalar-jacobi`, `block-jacobi`, or `none`.

```bash
bin/r solve -m bicgstab --matrix data/bsmp1.txt --rhs data/bsmp1_rhs.txt --precond amg --output data/bsmp1_x.txt
```

### src/gmres
Implements a restarted GMRES iterative solver for general non-symmetric sparse systems.

`GMRES` also uses AMG as the default left preconditioner and supports `scalar-jacobi`, `block-jacobi`, and `none` for comparison.

Detailed notes on the current AMG/multigrid implementation are available in `docs/amg_multigrid.md`.

```bash
bin/r solve -m gmres --matrix data/bsmp1.txt --rhs data/bsmp1_rhs.txt --precond amg --output data/bsmp1_x_gmres.txt --restart 30 --max-iters 1000 --tol 1e-6
```

### Solver comparison
To compare `BiCGStab` and `GMRES` on the same matrix and right-hand side, run:

```bash
bin/compare_solvers.sh --matrix data/bsmp1.txt --rhs data/bsmp1_rhs.txt --precond amg --restart 30 --max-iters 1000 --tol 1e-6
```

To compare several preconditioners for one solver, run:

```bash
bin/compare_preconditioners.sh --method gmres --matrix data/bsmp1.txt --rhs data/bsmp1_rhs.txt
```

To run the full solver/preconditioner comparison matrix, run:

```bash
bin/compare_all.sh --matrix data/bsmp1.txt --rhs data/bsmp1_rhs.txt
```

### PETSc/SLEPc reference for the full mixed piezo pencil

For comparison against the BSMP block/Schur pipeline, the repository now also includes
`vendor/petsc_mixed_eigen.py`.  Unlike `bin/r eigen`, this script assembles and solves the
*full* mixed generalized eigenproblem

$$
K = \begin{bmatrix} C_u & C_{u\varphi} \\
                    C_{u\varphi}^T & -C_\varphi \end{bmatrix},
\qquad
B = \begin{bmatrix} M & 0 \\
                    0 & 0 \end{bmatrix},
$$

with zero mass on the electrical DOFs.  It can solve either:

- `block-wise`: `ux_1 uy_1 uz_1 ... ux_n uy_n uz_n phi_1 ... phi_n`
- `node-based`: `ux_1 uy_1 uz_1 phi_1 ... ux_n uy_n uz_n phi_n`
- `both`: assemble and solve both orderings in one run.

Example usage:

```bash
python3 vendor/petsc_mixed_eigen.py \
   --stiffness <C_file> \
   --mass <M_file> \
   --coords <xyz_file> \
   --input-ordering block-wise \
   --ordering both \
   --num-eigs 5 \
   --grounded-dof 0
```

The script accepts Matrix Market coordinate files and zero-based plain triplet files, infers the mechanical/electrical split from `coords`, applies the same single-DOF grounding to the electrical block inside the full stiffness matrix by default, and then hands the mixed pencil to PETSc/SLEPc.
This is intended as a reference / comparison path.


## Smoke tests
A small verification executable checks the first sprint pipeline on a synthetic piezo system:

```bash
./build/example_schur_smoke
```

And for the rigid-body basis check:

```bash
./build/example_rbm_smoke
```

And for the regularized elastic setup operator:

```bash
./build/example_regularized_elastic_smoke
```

And for the separate SA-AMG scaffold wiring:

```bash
./build/example_sa_amg_scaffold_smoke
```

And for the first hierarchy pieces (aggregation + tentative prolongator):

```bash
./build/example_sa_amg_aggregation_smoke
```

And for the host-reference Galerkin coarse operator:

```bash
./build/example_sa_amg_galerkin_smoke
```

And for the first smoothed prolongator check:

```bash
./build/example_sa_amg_smoothing_smoke
```

And for the first two-level SA-AMG V-cycle:

```bash
./build/example_sa_amg_vcycle_smoke
```

And for the Chebyshev smoother plus Chebyshev-backed SA-AMG V-cycle path:

```bash
./build/example_sa_amg_chebyshev_smoke
```

And for the `M`-orthogonalization / deflation layer needed by the article eigensolver:

```bash
./build/example_mass_orthogonalization_smoke
```

And for the sequential deflated PCG eigensolver smoke test (including a `2x2` Ritz helper check and a clustered-low-modes case):

```bash
./build/example_deflated_pcg_eigensolver_smoke
```

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
Reads a sparse matrix in MATLAB format and makes nice plot. Can compute bandwidth.

#### sparse_spy.py
Python utility for MATLAB-like sparsity visualization from plain-text triplets.
It understands files with entries like `row, col, value`, including files that
contain both omitted zeros and explicitly written zero values.

Explicit zeros are ignored automatically when drawing the sparsity pattern.

Examples:
```bash
python3 tools/sparse_spy.py data/bsmp1.txt
python3 tools/sparse_spy.py data/martynova/big4/big4_O_phi_Ct.txt --output big4.png
```

#### multiplication.m
Multiplies a sparse matrix by a vector

#### slae.m

Solves a system of linear equations with a sparse matrix in MATLAB, loads external solution for comparison.
