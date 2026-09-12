#!/usr/bin/env python3
"""High-level PETSc-based solver wrapper.

This script solves a linear system Ax = b using PETSc (petsc4py) with
configurable KSP/PC options, and is intended as a CPU reference /
validation path for the CUDA-based BSMP solvers.

Typical usage (PETSc binary .dat files):

  python vendor/petsc_solve.py \
      --matrix A.dat --rhs b.dat --solution x.dat \
      --solver cg-ichol

or with algebraic multigrid:

  python vendor/petsc_solve.py \
      --matrix A.dat --rhs b.dat --solution x.dat \
      --solver cg-amg

You can also override solver details with standard PETSc options, e.g.:

  python vendor/petsc_solve.py --matrix A.dat --rhs b.dat --solution x.dat \
      --solver custom \
      -ksp_type cg -pc_type gamg

Prerequisites:
  - PETSc built with real/float or real/double (matching your data)
  - petsc4py installed and pointing at the same PETSc
"""

import argparse
import sys

from petsc4py import PETSc


def load_matrix(path: str) -> PETSc.Mat:
    """Load sparse matrix from PETSc binary file."""
    viewer = PETSc.Viewer().createBinary(path, 'r')
    A = PETSc.Mat().create()
    A.setFromOptions()
    A = A.load(viewer)
    A.assemble()
    return A


def load_vector(path: str) -> PETSc.Vec:
    """Load vector from PETSc binary file."""
    viewer = PETSc.Viewer().createBinary(path, 'r')
    b = PETSc.Vec().create()
    b.setFromOptions()
    b = b.load(viewer)
    return b


def save_vector(vec: PETSc.Vec, path: str) -> None:
    """Save PETSc vector to binary file."""
    viewer = PETSc.Viewer().createBinary(path, 'w')
    vec.view(viewer)


def setup_ksp(A: PETSc.Mat, solver: str) -> PETSc.KSP:
    """Create and configure a KSP object for the requested high-level solver.

    Supported `solver` values:
      - "cg-ichol"  : Conjugate Gradient + incomplete Cholesky (ICC)
      - "cg-amg"    : Conjugate Gradient + algebraic multigrid (GAMG)
      - "custom"    : Leave KSP/PC type mostly to PETSc options (-ksp_type, -pc_type, ...)
    """
    ksp = PETSc.KSP().create()
    ksp.setOperators(A)

    pc = ksp.getPC()

    if solver == "cg-ichol":
        ksp.setType(PETSc.KSP.Type.CG)
        pc.setType(PETSc.PC.Type.ICC)
        # Allow PETSc to choose fill level / drop tolerance from options
    elif solver == "cg-amg":
        ksp.setType(PETSc.KSP.Type.CG)
        pc.setType(PETSc.PC.Type.GAMG)
        # For non-symmetric problems you might prefer GMRES + GAMG instead
    elif solver == "custom":
        # Defer entirely to runtime PETSc options
        pass
    else:
        raise ValueError(f"Unsupported solver preset: {solver}")

    ksp.setFromOptions()  # allow -ksp_* and -pc_* overrides
    return ksp


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Solve Ax = b using PETSc (petsc4py)")
    parser.add_argument("--matrix", required=True, help="PETSc binary matrix file (Mat)")
    parser.add_argument("--rhs", required=True, help="PETSc binary RHS file (Vec)")
    parser.add_argument("--solution", required=True, help="Output PETSc binary Vec file for x")
    parser.add_argument(
        "--solver",
        choices=["cg-ichol", "cg-amg", "custom"],
        default="cg-ichol",
        help="High-level solver preset (default: cg-ichol)",
    )

    args, unknown = parser.parse_known_args(argv)

    # Ensure PETSc sees any extra -ksp_* / -pc_* options on the command line
    if unknown:
        PETSc.Sys.pushErrorHandler("traceback")

    # Load system
    A = load_matrix(args.matrix)
    b = load_vector(args.rhs)

    if A.getSize()[0] != b.getSize()[0]:
        raise RuntimeError(f"Dimension mismatch: A is {A.getSize()}, b has length {b.getSize()[0]}")

    # Prepare solution vector
    x = b.duplicate()
    x.set(0)

    # Configure solver
    ksp = setup_ksp(A, args.solver)

    # Solve
    ksp.solve(b, x)

    its = ksp.getIterationNumber()
    rnorm = ksp.getResidualNorm()
    reason = ksp.getConvergedReason()

    PETSc.Sys.Print(f"KSP converged reason: {reason}, iterations: {its}, residual norm: {rnorm}")

    # Save solution
    save_vector(x, args.solution)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
