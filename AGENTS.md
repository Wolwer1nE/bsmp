# AGENTS.md

## BSMP project guidance

This repository is a CUDA/C++ block sparse matrix package focused on BSMP block-format operations and iterative solvers.

### Core expectations

- Prefer working through the existing `BlockSparseMatrix` abstraction in `src/block_sparse_matrix.{h,cu}`.
- Preserve the BSMP block format end-to-end. New solvers and kernels should operate on block-sparse data, not silently fall back to scalar sparse storage.
- Treat `BSMP_BLOCK_SIZE` as a compile-time contract from `CMakeLists.txt`. The current default is `3`.
- Avoid editing generated files under `build/`.
- Keep public CLI behavior aligned with code changes: if you add a solver or workflow, update the executable, `bin/r`, `bin/cmd/solve.sh`, and `README.md` together.

### Matrix and vector formats

- Matrices are loaded through `triplet_loader` and may come from:
  - plain triplet files;
  - Matrix Market files.
- RHS vectors may also come from Matrix Market-style files with comment and size headers.
- Solver examples currently expect matrix and RHS files as positional arguments.

### Existing project structure

- `src/block_sparse_matrix.{h,cu}`: block sparse storage and GPU matvec kernels.
- `src/triplet_loader.{h,cpp}`: triplet and Matrix Market loaders into BSMP block format.
- `src/bicgstab.{h,cu}`: BiCGStab solver.
- `src/gmres.{h,cu}`: GMRES solver.
- `example_bicgstab.cu`, `example_gmres.cu`: solver executables.
- `bin/r`: main CLI wrapper.
- `bin/cmd/solve.sh`: solver dispatch and argument handling.
- `bin/compare_solvers.sh`: compares BiCGStab and GMRES on the same system.

### Solver implementation rules

- Reuse `BlockSparseMatrix::multiply(...)` for all matrix-vector products.
- If preconditioning is used, make the relationship to the BSMP block format explicit.
- Prefer block-aware preconditioners over scalar approximations unless the task explicitly asks otherwise.
- Keep solver outputs comparable:
  - print convergence status;
  - print iteration count;
  - print final relative residual;
  - print solve time in milliseconds.
- When adding a new solver, mirror the integration pattern already used for `bicgstab` and `gmres`.

### Build and validation workflow

- Prefer the workspace CMake build integration for compilation.
- After solver-related changes, validate with:
  1. project build;
  2. direct example run or `bin/r solve ...`;
  3. `bin/compare_solvers.sh` when comparing iterative methods.
- Use existing sample data in `data/` for smoke tests, especially `data/bsmp1.txt` and `data/bsmp1_rhs.txt`.

### Common pitfalls in this repository

- CUDA device-link conflicts can happen if helper kernels in multiple `.cu` files share global linkage. Prefer internal linkage for file-local helper kernels.
- Changes to solver internals often require synchronized updates to CLI argument plumbing.
- Do not assume scalar CSR/COO semantics; this project is centered on block rows, block columns, and dense blocks.
- Keep documentation consistent with actual supported methods and executable names.

### Preferred ways to run things

- Build: use the configured CMake build.
- Solve with BiCGStab: `bin/r solve -m bicgstab --matrix <matrix> --rhs <rhs>`
- Solve with GMRES: `bin/r solve -m gmres --matrix <matrix> --rhs <rhs>`
- Compare methods: `bin/compare_solvers.sh --matrix <matrix> --rhs <rhs>`

### When making future changes

If the task touches solvers, check all of the following before finishing:

- `src/` implementation matches BSMP block-format expectations;
- executable target is present in `CMakeLists.txt`;
- CLI dispatch is wired in `bin/cmd/solve.sh`;
- `README.md` documents the new behavior;
- build succeeds;
- at least one sample run succeeds.
