# QWEN.md — BSMP (Block Sparse Matrix Package)

## Project Overview

**BSMP** is a CUDA/C++ library for block-sparse matrix operations and iterative solvers, focused on piezoelectric finite-element systems and generalized eigenproblems. The C++ core implements a custom block-sparse format (BSMP), iterative linear solvers, an SA-AMG (Smoothed-Aggregation Algebraic Multigrid) preconditioner pipeline, and an M-orthogonalized deflated PCG eigensolver — all targeting NVIDIA GPUs via CUDA 12.5.

All user-facing commands flow through a Python CLI (`bsmp`), built on **click**. The `bin/` directory and shell scripts have been fully replaced. Python also owns build orchestration (CMake configure → build → install) via the launchers.

### Core Architecture

| Layer | Key Files / Modules | Purpose |
|-------|--------------------|---------|
| **Storage & Kernels** | `src/block_sparse_matrix.{h,cu}` | `BlockSparseMatrix` struct; GPU matvec via `multiply(...)` |
| **I/O** | `include/triplet_loader.h`, `src/triplet_loader.cpp` | Triplet & Matrix Market loaders → BSMP block format |
| **Linear Solvers** | `include/bicgstab.h`, `src/bicgstab.cu`, `include/gmres.h`, `src/gmres.cu` | BiCGStab, restarted GMRES (both default to SA-AMG left preconditioner) |
| **AMG Pipeline** | `include/amg_preconditioner.h`, `src/amg_preconditioner.cu`, `include/sa_amg_*.h`, `src/sa_amg_preconditioner.cu` | Aggregation → tentative prolongator → Galerkin coarse op → Chebyshev smoothing → V-cycle |
| **Eigensolver** | `include/deflated_pcg_eigensolver.h`, `src/deflated_pcg_eigensolver.cu` | Mass-orthogonalised PCG with deflation for clustered-low-mode generalized eigenproblems |
| **Specialized Ops** | `src/schur_operator.cu`, `src/piezo_block_system.cu`, `src/elastic_regularization.cu`, `src/mass_orthogonalization.cu` | Mixed-piezo Schur-complement, coupling blocks, elasticity regularization, M-orthogonalization |
| **Python CLI** | `python/src/bsmp/cli.py` | Single public entry point — Click CLI with aliases |
| **Python config** | `python/src/bsmp/config.py` | `Config` dataclass — resolves root, build, exe dirs; `find_exe()` |
| **Python launchers** | `python/src/bsmp/launchers/*.py` | Build orchestration + subprocess wrappers for each C++ executable |
| **Python generators** | `python/src/bsmp/generators/*.py` | Synthetic matrix generators (triplet + eigenvalue) |
| **MATLAB tools** | `tools/sparse_spy.py` | Sparsity visualisation from plain-text triplets |
| **Reference path** | `vendor/petsc_mixed_eigen.py` | Full mixed generalized eigenproblem via PETSc/SLEPc |

### CLI Entry Point

```
bsmp [--version]
bsmp {generate, launch} [--help]
```

Aliases: `gen`/`g` → `generate`, `run`/`r` → `launch`.

After `pip install -e .` from the `python/` directory, `bsmp` is available globally.

### Installed Executable Names (CMake targets)

| Target | CLI command | Role |
|--------|-------------|------|
| `bsmp` | — | Shared library |
| `example` | — | SpMV example |
| `example_bicgstab` | `bsmp launch solve -m bicgstab` | BiCGStab solver |
| `example_gmres` | `bsmp launch solve -m gmres` | GMRES solver |
| `example_sa_amg_pcg_eigen` | `bsmp launch eigen` | SA-AMG PCG eigensolver |
| `example_schur_smoke` | — | Schur-complement smoke test |
| `example_rbm_smoke` | — | Rigid-body-basis smoke test |
| `example_regularized_elastic_smoke` | — | Regularised elasticity smoke test |
| `example_sa_amg_*_smoke` | — | Incremental AMG pipeline smoke tests |
| `example_mass_orthogonalization_smoke` | — | M-orthogonalization smoke test |
| `example_deflated_pcg_eigensolver_smoke` | — | Deflated PCG eigensolver smoke test |

## Building & Running

### Prerequisites

- C++14 or higher
- CMake 3.18+
- CUDA 12.5
- Python 3.12+ (for CLI tooling)
- MATLAB (optional, for comparison scripts)

### Install Python CLI

```bash
cd python
pip install -e .[dev]
```

This makes the `bsmp` command available globally.

### Build (automatic via CLI)

All `bsmp launch` commands invoke CMake configure → build → install automatically unless `--no-build` is passed:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --clean-first -j
cmake --install build --clean-first -j
```

Skip with `--no-build`.

### Build (manual)

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
cmake --install . --config Release
```

### Generate Matrices

```bash
# Block-diagonal triplet matrix + RHS
bsmp generate matrix data/100.txt -n 100 -b 4

# Generalized eigenvalue problem matrices
bsmp generate eigenvalue data/A.txt data/B.txt --size 200
```

### Solve Linear Systems

```bash
# BiCGStab with AMG preconditioner
bsmp launch solve -m bicgstab --matrix data/bsmp1.txt --rhs data/bsmp1_rhs.txt --precond amg --output data/bsmp1_x.txt

# GMRES with block-Jacobi, custom restart / tolerance
bsmp launch solve -m gmres --matrix data/bsmp1.txt --rhs data/bsmp1_rhs.txt \
    --precond block-jacobi --output data/bsmp1_x_gmres.txt --restart 30 --max-iters 1000 --tol 1e-6
```

### Compare Solvers / Preconditioners

```bash
bsmp launch compare-solvers --matrix data/bsmp1.txt --rhs data/bsmp1_rhs.txt --precond amg
bsmp launch compare-preconditioners --method gmres --matrix data/bsmp1.txt --rhs data/bsmp1_rhs.txt
bsmp launch compare-all --matrix data/bsmp1.txt --rhs data/bsmp1_rhs.txt
```

### Eigenvalue Problems

```bash
# Generate test matrices
bsmp generate eigenvalue data/A.txt data/B.txt --size 200

# Run the SA-AMG PCG eigensolver
bsmp launch eigen --stiffness data/stiffness.txt --mass data/mass.txt --coords data/coords.txt \
    --ordering block-wise --num-eigs 5 --sa-amg-use-chebyshev 1
```

### PETSc/SLEPc Reference

`vendor/petsc_mixed_eigen.py` assembles and solves the full mixed generalized eigenproblem for comparison against the BSMP native pipeline.

```bash
python3 vendor/petsc_mixed_eigen.py \
   --stiffness C.txt --mass M.txt --coords xyz.txt \
   --input-ordering block-wise --ordering both --num-eigs 5 --grounded-dof 0
```

### Python Tools

```bash
# Sparsity visualization
python3 tools/sparse_spy.py data/bsmp1.txt --output bsmp1_sparsity.png
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

## Python CLI Reference

The top-level command is `bsmp`. Aliases `gen`/`g` and `run`/`r` are available.

```
bsmp [--version]
bsmp {generate, launch} [--help]
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

## Development Conventions

### BSMP Block Format Contract

- `BSMP_BLOCK_SIZE` is a **compile-time constant** (default `3` per `CMakeLists.txt`). All kernels and data structures must respect this contract.
- Never silently fall back to scalar CSR/COO storage. The entire pipeline operates on block rows, block columns, and dense blocks.
- Always use `BlockSparseMatrix::multiply(...)` for GPU matrix-vector products.

### CUDA Pitfalls

- Helper kernels in multiple `.cu` files with global linkage will cause device-link conflicts. Use **internal linkage** (`__device__ static` or unnamed namespaces) for file-local helper kernels.

### Adding a New Solver

Mirror the existing pattern for `bicgstab` / `gmres`:

1. Implement in `src/<solver>.{h,cu}` using `BlockSparseMatrix::multiply(...)`.
2. Add executable target in `CMakeLists.txt` and `examples/CMakeLists.txt`.
3. Add a launcher in `python/src/bsmp/launchers/<solver>.py`.
4. Register the Click subcommand in `python/src/bsmp/cli.py`.
5. Update `README.md` with usage examples.
6. Add smoke-test matrix data under `data/`.

### Solver Output Contract

Every solver must print:
- Convergence status
- Iteration count
- Final relative residual
- Solve time (milliseconds)

### Python Module Structure

```
python/
├── pyproject.toml          # Package metadata, click dependency, entry point
├── _custom_build/
│   ├── __init__.py
│   └── bsmp_custom_build_backend.py  # CMake orchestration + setuptools delegation
├── src/bsmp/
│   ├── __init__.py         # Exposes __version__ = "0.1.0"
│   ├── config.py           # Config dataclass (root, build, exe dirs)
│   ├── cli.py              # Top-level Click CLI (AliasedGroup)
│   ├── generators/
│   │   ├── matrix.py       # generate_matrix()
│   │   └── eigen.py        # generate_eigen_matrices()
│   └── launchers/
│       ├── _build.py       # ensure_built() — build guard + clean_build()
│       ├── mult.py         # run_mult()
│       ├── solve.py        # run_solve()
│       ├── eigen.py        # run_eigen()
│       └── compare.py      # run_compare_*()
└── tests/
    ├── test_cli.py         # Full pytest suite for every CLI command
    └── conftest.py
```

- `config.CONFIG` is the module-level singleton. `find_exe(name)` locates compiled executables under `python/src/bsmp/bin/`.
- `launchers/_build.py::ensure_built(no_build)` is the central build decision — used by every launcher instead of inline `if` checks.
- Matrix files follow the convention: `data/<name>.txt` + `data/rhs_<name>.txt`.

### Testing

```bash
cd python
pytest
```

## Key File Locations

| Path | Role |
|------|------|
| `include/block_sparse_matrix.h`, `src/block_sparse_matrix.cu` | Block-sparse storage, GPU matvec kernels |
| `include/triplet_loader.h`, `src/triplet_loader.cpp` | Triplet / Matrix Market I/O → BSMP block format |
| `include/bicgstab.h`, `src/bicgstab.cu` | BiCGStab solver + AMG preconditioner |
| `include/gmres.h`, `src/gmres.cu` | Restarted GMRES solver |
| `include/amg_preconditioner.h`, `src/amg_preconditioner.cu` | AMG preconditioner |
| `include/sa_amg_preconditioner.h`, `src/sa_amg_preconditioner.cu` | SA-AMG pipeline |
| `include/deflated_pcg_eigensolver.h`, `src/deflated_pcg_eigensolver.cu` | Deflated PCG eigensolver |
| `include/generalized_eigen.h`, `src/generalized_eigen.cu` | Generalized eigenvalue routines |
| `python/src/bsmp/cli.py` | Top-level Click CLI |
| `python/src/bsmp/config.py` | `Config` dataclass, `find_exe()` |
| `python/src/bsmp/launchers/_build.py` | `clean_build()`, `ensure_built()` |
| `python/_custom_build/bsmp_custom_build_backend.py` | Custom PEP 517 build backend (CMake + setuptools) |
| `python/pyproject.toml` | Package definition, entry point, deps |
| `python/tests/test_cli.py` | CLI test suite |
| `vendor/petsc_mixed_eigen.py` | PETSc/SLEPc reference eigenproblem |
| `cmake/` | CMake helper scripts (Setup, FindCuda, AddTest, StaticAnalysis, Sanitizer) |
| `data/` | Sample matrices |
| `tools/sparse_spy.py` | Sparsity pattern visualiser |
| `include/bsmp_config.h` | `BSMP_BLOCK_SIZE` compile-time config (default 3) |

## Notes

- The `bin/` directory has been fully removed. All commands are available via `bsmp`.
- The Python CLI builds the C++ code automatically before launching executables. Use `--no-build` to skip.
- The AMG pipeline is a complete SA-AMG hierarchy: aggregation → tentative prolongator → Galerkin coarse operator → Chebyshev smoothing → V-cycle.
- `vendor/petsc_mixed_eigen.py` supports `block-wise`, `node-based`, and `both` ordering strategies for mixed piezo systems.
- Avoid editing files under `build/` — they are generated artifacts.
- `bsmp generate eigenvalue` produces symmetric sparse matrices A and B suitable for generalized eigenvalue problems (A - λBx = 0), with B strongly diagonally dominant for numerical stability.
- The Python CLI supports both source-tree installs (`pip install -e .`) and wheel installs. `--no-build` is documented in CLI help for both audiences.