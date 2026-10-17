# BSMP - Block Sparse Matrix Package

## Requirements

- C++17 or higher
- CMake 3.18 or higher
- CUDA 12.5
- Python 3.12+ (for CLI tooling)
- MATLAB (optional, for performance comparison scripts)

## Installation

Install the Python CLI (develop/install mode):

```bash
cd python
pip install -e .[dev]
```

This makes the `bsmp` command available inside your python environment.
You might need to wait, as the whole project is compiled.

## Quick Start

### Generate a test matrix

```bash
bsmp generate matrix data/100.txt -n 100 -b 4
```

This creates a block-sparse matrix with 100 blocks of size 4×4 at `data/100.txt` and
the associated RHS vector at `data/rhs_100.txt`.

### Matrix-vector multiplication

```bash
bsmp launch mult data/100.txt --rhs data/rhs_100.txt --mults 10 --verbose
```

Builds the project (unless `--no-build` is passed), then runs the SpMV kernel 10 times with verbose stats.

### Solve a linear system

```bash
# BiCGStab with AMG preconditioner
bsmp launch solve -m bicgstab --matrix data/bsmp1.txt --rhs data/bsmp1_rhs.txt --precond amg --output data/bsmp1_x.txt

# GMRES with block-Jacobi, custom restart / tolerance
bsmp launch solve -m gmres --matrix data/bsmp1.txt --rhs data/bsmp1_rhs.txt \
    --precond block-jacobi --output data/bsmp1_x_gmres.txt --restart 30 --max-iters 1000 --tol 1e-6
```

### Compare solvers and preconditioners

```bash
# BiCGStab vs GMRES on the same system
bsmp launch compare-solvers --matrix data/bsmp1.txt --rhs data/bsmp1_rhs.txt --precond amg --restart 30 --max-iters 1000 --tol 1e-6

# Compare multiple preconditioners for one solver
bsmp launch compare-preconditioners --method gmres --matrix data/bsmp1.txt --rhs data/bsmp1_rhs.txt

# Full comparison sweep (all solvers × all preconditioners)
bsmp launch compare-all --matrix data/bsmp1.txt --rhs data/bsmp1_rhs.txt
```

### Benchmark

Run one or more examples over several (matrix, rhs) pairs, then aggregate
the JSON records they produce into a single table and plots:

```bash
# 2 cases × (mult + bicgstab)
bsmp benchmark run \
    --matrix data/a.txt --rhs data/a_rhs.txt \
    --matrix data/b.txt --rhs data/b_rhs.txt \
    --method mult --method bicgstab \
    --table results.csv --plot results.png
```

Each (case, method) runs in its own working directory; the records are
collected into `<work-dir>/records/` (default working directory is
temporary — use `--work-dir` / `--records-dir` to keep them). Records can be
re-processed later without re-running:

```bash
bsmp benchmark table records_dir
bsmp benchmark plot records_dir -o results.png
```

Plots require `matplotlib` (`pip install -e .[plot]`).

### Eigenvalue problems

```bash
# Generate test matrices for a generalized eigenvalue problem
bsmp generate eigenvalue data/A.txt data/B.txt --size 200

# Run the SA-AMG PCG eigensolver
bsmp launch eigen --stiffness data/stiffness.txt --mass data/mass.txt --coords data/coords.txt \
    --ordering block-wise --modes 5 --sa-amg-use-chebyshev 1
```

## CLI Reference

The top-level command is `bsmp`. Aliases `gen`/`g` and `run`/`r` are available.

```
bsmp [--version]
bsmp {generate, launch, benchmark} [--help]
```

### `bsmp generate` (aliases: `gen`, `g`)

#### `bsmp generate matrix <output_file>`

Generates a block-diagonal triplet matrix and associated RHS vector.

| Flag | Default | Description |
|------|---------|-------------|
| `-n, --n_blocks` | `1000` | Number of blocks |
| `-b, --block_size` | `4` | Size of each block |
| `-r, --random` | `false` | Use random block values instead of sequential |

#### `bsmp generate eigenvalue <output_file_a> <output_file_b>`

Generates two symmetric sparse matrices A and B as triplets for a generalized eigenvalue problem.

| Flag | Default | Description |
|------|---------|-------------|
| `-n, --size` | `100` | Matrix size |
| `--density-a` | `0.01` | Sparsity density for A |
| `--density-b` | `0.005` | Sparsity density for B |

### `bsmp launch` (aliases: `run`, `r`)

All launchers call CMake configure → build → install automatically unless `--no-build` is passed.

#### `bsmp launch mult <matrix_file>`

Runs the SpMV multiplication example.

| Flag | Default | Description |
|------|---------|-------------|
| `--rhs` | *(required)* | RHS vector file |
| `--format` | `triplets` | Input format (`triplets` / `matrix-market`) |
| `--mults` | `10` | Number of multiply iterations |
| `--output` | — | Save result vector to file |
| `--verbose` | — | Print kernel statistics |
| `--no-build` | — | Skip CMake build |

#### `bsmp launch solve`

Runs BiCGStab or GMRES on the given system.

| Flag | Default | Description |
|------|---------|-------------|
| `-m, --method` | *(required)* | `bicgstab` / `gmres` |
| `--matrix` | *(required)* | Matrix file |
| `--rhs` | *(required)* | RHS vector file |
| `--precond` | `""` (defaults to amg in the executable) | `amg` / `scalar-jacobi` / `block-jacobi` / `none` |
| `--output` | — | Save solution vector to file |
| `--max-iters` | — | Maximum iterations |
| `--tol` | — | Convergence tolerance |
| `--restart` | — | GMRES restart parameter |
| `--no-build` | — | Skip CMake build |

#### `bsmp launch eigen`

Runs the SA-AMG PCG eigensolver. Two mutually exclusive modes:

- **Full-matrix mode**: `--stiffness` + `--mass` + `--coords`
- **Legacy block mode**: `--cu` + `--cuphi` + `--cphi` + `--mass` + `--coords`

| Flag | Default | Description |
|------|---------|-------------|
| `--stiffness` | — | Stiffness matrix file (full-matrix mode) |
| `--mass` | — | Mass matrix file |
| `--coords` | — | Node coordinates file |
| `--cu, --cuphi, --cphi` | — | Legacy block files |
| `--ordering` | `node-based` | `node-based` / `block-wise` |
| `--eigensolve-path` | `auto` | `auto` / `sa-amg` / `unpreconditioned` / `explicit-schur-cusolver` |
| `--grounded-dof` | — | Zero-based DOF to ground |
| `--modes` | `3` | Number of eigenvalues |
| `--tol` | `1e-3` | Eigen solver tolerance |
| `--max-iters` | `400` | Max iterations |
| `--sa-amg-regularization-epsilon` | — | AMG regularization parameter |
| `--sa-amg-pre-sweeps` | — | Pre-smoothing sweeps |
| `--sa-amg-post-sweeps` | — | Post-smoothing sweeps |
| `--sa-amg-jacobi-damping` | — | Jacobi damping factor |
| `--sa-amg-prolongation-damping` | — | Prolongation damping factor |
| `--sa-amg-use-chebyshev` | `0` | Enable Chebyshev smoothing (0 / 1) |
| `--verbose` | — | Verbose output |
| `--no-build` | — | Skip CMake build |

#### `bsmp launch compare-all`

Runs all solvers (BiCGStab + GMRES) against all specified preconditioners.

| Flag | Default | Description |
|------|---------|-------------|
| `--matrix` | *(required)* | Matrix file |
| `--rhs` | *(required)* | RHS vector file |
| `--preconds` | `amg,scalar-jacobi,block-jacobi,none` | Comma-separated preconditioners |
| `--restart` | `30` | GMRES restart |
| `--max-iters` | `1000` | Max iterations |
| `--tol` | `1e-6` | Tolerance |
| `--no-build` | — | Skip CMake build |

#### `bsmp launch compare-preconditioners`

Compares multiple preconditioners for a single solver method.

| Flag | Default | Description |
|------|---------|-------------|
| `--method` | *(required)* | `bicgstab` / `gmres` |
| `--matrix` | *(required)* | Matrix file |
| `--rhs` | *(required)* | RHS vector file |
| `--preconds` | `amg,scalar-jacobi,block-jacobi,none` | Comma-separated preconditioners |
| `--restart` | `30` | GMRES restart |
| `--max-iters` | `1000` | Max iterations |
| `--tol` | `1e-6` | Tolerance |
| `--no-build` | — | Skip CMake build |

#### `bsmp launch compare-solvers`

Compares BiCGStab vs GMRES on the same system.

| Flag | Default | Description |
|------|---------|-------------|
| `--matrix` | *(required)* | Matrix file |
| `--rhs` | *(required)* | RHS vector file |
| `--precond` | `amg` | Preconditioner |
| `--restart` | `30` | GMRES restart |
| `--max-iters` | `1000` | Max iterations |
| `--tol` | `1e-6` | Tolerance |
| `--no-build` | — | Skip CMake build |

### `bsmp benchmark`

Benchmarks the examples over multiple (matrix, rhs) pairs. Each (case, method)
runs in its own working directory; the JSON records the examples write
(`log*.json`) are collected into one records directory, aggregated into a
single table and plotted.

#### `bsmp benchmark run`

| Flag | Default | Description |
|------|---------|-------------|
| `--matrix` | *(repeatable, required)* | Matrix file — paired with `--rhs` by index |
| `--rhs` | *(repeatable, required)* | RHS vector file |
| `--method` | `bicgstab, gmres` | Examples to run: `mult` / `bicgstab` / `gmres` (repeatable) |
| `--mults` | `10` | Chained multiplies for the `mult` example |
| `--precond` | `""` | Preconditioner for the solvers |
| `--max-iters` | — | Solver iteration limit |
| `--tol` | — | Solver relative residual tolerance |
| `--restart` | — | GMRES restart size |
| `--work-dir` | — | Directory for per-case run outputs (default: temporary) |
| `--records-dir` | — | Collected JSON records directory (default: `<work-dir>/records`) |
| `--table` | — | Write the aggregated table to this CSV path |
| `--plot` | — | Write plots to this PNG path |
| `--no-build` | — | Skip CMake build |

#### `bsmp benchmark table <records...>`

Parses collected JSON records (files or directories) into one table and
prints it; `--output <csv>` also writes a CSV. Failed runs are skipped with a
WARNING.

#### `bsmp benchmark plot <records...>`

Parses collected JSON records and writes a plot PNG (`-o`/`--output`,
default `benchmark_plot.png`): mean time and final residual per case, plus a
time-vs-matrix-size scatter. Requires `matplotlib` — `pip install -e .[plot]`.

## Working with presets

It is almost guaranteed that you will need to create a custom preset for CMake. It can help you specifying CUDA version, as well as enabling or disabling static checks. For custom presets, create `CMakeUserPresets.json` file. An example of CMake preset:
```json
{
    "version": 6,
    "configurePresets": [
        {
            "name": "msvc-custom",
            "inherits": "msvc",
            "generator": "Visual Studio 18 2026",
            "cacheVariables": {
                "CMAKE_CUDA_ARCHITECTURES": "70",
                "BSMP_ENABLE_CLANG_TIDY": "OFF",
                "BSMP_ENABLE_COMPUTE_SANITIZER": "OFF"
            }
        }
    ],
    "buildPresets": [
        {
            "name": "msvc-custom-debug",
            "configurePreset": "msvc-custom",
            "configuration": "Debug"
        },
        {
            "name": "msvc-custom-release",
            "configurePreset": "msvc-custom",
            "configuration": "Release"
        }
    ]
}
```

## Available Modules

### C++ Core

### src/triplet_loader
Loads a sparse matrix from a file in triplet format or Matrix Market format into the `BlockSparseMatrix` structure.

### src/block_sparse_matrix
Implements the block-sparse matrix structure and GPU operations such as matrix-vector multiplication.

### src/bicgstab
Implements the BiCGStab iterative method for solving linear systems with block-sparse matrices. Uses SA-AMG as the default left preconditioner. Supports `amg`, `scalar-jacobi`, `block-jacobi`, and `none`.

### src/gmres
Implements a restarted GMRES iterative solver for general non-symmetric sparse systems. Also uses SA-AMG as the default left preconditioner.

### Solvers

Both solvers output convergence status, iteration count, final relative residual, and solve time in milliseconds. Every example also writes a JSON experiment record (`log*.json`) to its working directory — see the schema in `data/schemas/v1/record.schema.json`; `bsmp benchmark` aggregates these records into tables and plots.

### PETSc/SLEPc Reference

`vendor/petsc_mixed_eigen.py` assembles and solves the full mixed generalized eigenproblem for comparison against the BSMP native pipeline. Unlike the C++ `bsmp launch eigen`, this script handles the *full* mixed system with zero mass on electrical DOFs.

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

### Python Tools

### generators/matrix.py / `bsmp generate matrix`
Generates block-sparse matrices in triplet format for testing.

### benchmark/ — `bsmp benchmark`
Runs the examples over many (matrix, rhs) pairs, collects their JSON records
(`log*.json`) into a shared records directory, and produces a combined table
and plots (matplotlib required for plots — `pip install -e .[plot]`).

### tools/sparse_spy.py
Python utility for MATLAB-like sparsity visualization from plain-text triplets. Ignores explicit zeros automatically.

```bash
python3 tools/sparse_spy.py data/bsmp1.txt
python3 tools/sparse_spy.py data/martynova/big4/big4_O_phi_Ct.txt --output big4.png
```

### Smoke Tests

After building, smoke-test executables are found under `build/`:

```bash
./build/example_schur_smoke                 # Schur-complement pipeline
./build/example_rbm_smoke                   # Rigid-body-basis check
./build/example_regularized_elastic_smoke   # Regularized elasticity operator
./build/example_sa_amg_scaffold_smoke       # SA-AMG scaffold wiring
./build/example_sa_amg_aggregation_smoke    # First hierarchy pieces
./build/example_sa_amg_galerkin_smoke       # Host-reference Galerkin coarse operator
./build/example_sa_amg_smoothing_smoke      # First smoothed prolongator check
./build/example_sa_amg_vcycle_smoke         # First two-level SA-AMG V-cycle
./build/example_sa_amg_chebyshev_smoke      # Chebyshev smoother + V-cycle
./build/example_mass_orthogonalization_smoke # M-orthogonalization / deflation layer
./build/example_deflated_pcg_eigensolver_smoke # Deflated PCG eigensolver
```

## Building Directly

The Python CLI invokes CMake automatically, but you can also build manually:

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
cmake --install . --config Release
```

## Development Conventions

- Prefer working through the `BlockSparseMatrix` abstraction in `src/block_sparse_matrix.{h,cu}`.
- Preserve the BSMP block format end-to-end — never silently fall back to scalar CSR/COO.
- Treat `BSMP_BLOCK_SIZE` as a compile-time contract (default `3` from `CMakeLists.txt`).
- Avoid editing generated files under `build/`.
- Every new solver must mirror the existing integration pattern (CLI dispatch, executable target, README).
- CUDA device-link conflicts occur if helper kernels in multiple `.cu` files share global linkage — use internal linkage.
- Host-only C++ executables linking `bsmp` (built with separable compilation) must include a `.cu` anchor source (`examples/device_link.cu`) so CMake runs the device-link step.
- Every example writes a JSON experiment record to its working directory (schema: `data/schemas/v1/record.schema.json`).
