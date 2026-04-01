#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Reference PETSc/SLEPc solver for the full mixed piezoelectric eigenproblem.

This script solves the *full* generalized eigenproblem directly from externally
prepared full stiffness and mass matrices

    C, M

and solves it in one of three global orderings:

  - block-wise  : [u_x1, u_y1, u_z1, ..., u_xN, u_yN, u_zN, phi_1, ..., phi_N]
  - node-based  : [u_x1, u_y1, u_z1, phi_1, ..., u_xN, u_yN, u_zN, phi_N]
  - both        : solve both orderings and print separate summaries

The script assumes that the supplied full matrices are already assembled in the
requested physical formulation. For the mixed nodal formulation, the electrical
DOFs may carry zero rows/columns in the mass matrix.

Input files may be either:
  - Matrix Market coordinate files; or
  - zero-based plain triplet files using whitespace or comma separators.

Typical usage:

  python vendor/petsc_mixed_eigen.py \
      --stiffness C.mtx --mass M.mtx --coords coords.txt \
      --ordering both --num-eigs 5

Optional SLEPc / PETSc runtime flags may be appended, for example:

  python vendor/petsc_mixed_eigen.py ... \
      -eps_type krylovschur -st_type sinvert -pc_type lu

Prerequisites:
  - PETSc + petsc4py
  - SLEPc + slepc4py
  - numpy + scipy
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

try:
    import slepc4py
except Exception as exc:  # pragma: no cover - import error path
    raise SystemExit(
        "Failed to import slepc4py. Install SLEPc + slepc4py first. "
        f"Original error: {exc}"
    )

slepc4py.init(sys.argv)

import numpy as np
import scipy.io
import scipy.sparse as sps
from petsc4py import PETSc
from slepc4py import SLEPc


@dataclass
class EigenpairSummary:
    index: int
    eigenvalue_real: float
    eigenvalue_imag: float
    residual: float


@dataclass
class SolveSummary:
    ordering: str
    dimension: int
    mechanical_dofs: int
    electrical_dofs: int
    converged_pairs: int
    iterations: int
    reason: int
    requested_pairs: int
    eigenpairs: List[EigenpairSummary]


def eigenvalue_to_frequency_hz(eigenvalue_real: float, eigenvalue_imag: float) -> float | None:
    if abs(eigenvalue_imag) > 1e-9:
        return None
    if eigenvalue_real <= 0.0:
        return None
    return float(np.sqrt(eigenvalue_real) / (2.0 * np.pi))


def first_nonempty_line(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                return stripped
    return ""


def load_sparse_auto(path: str) -> sps.csr_matrix:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    first = first_nonempty_line(source)
    if first.startswith("%%MatrixMarket"):
        matrix = scipy.io.mmread(source)
        if sps.issparse(matrix):
            matrix = matrix.tocsr().astype(np.float64)
            return complete_missing_symmetric_pairs(symmetrize_if_triangular(matrix))
        dense = np.asarray(matrix, dtype=np.float64)
        return complete_missing_symmetric_pairs(symmetrize_if_triangular(sps.csr_matrix(dense)))

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    max_row = -1
    max_col = -1

    with source.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith("%"):
                continue
            normalized = line.replace(",", " ")
            parts = normalized.split()
            if len(parts) < 3:
                raise ValueError(f"Failed to parse triplet line in {path}: {raw_line.rstrip()}")

            row = int(parts[0])
            col = int(parts[1])
            value = float(parts[2])
            if row < 0 or col < 0:
                raise ValueError(f"Negative indices are not supported in {path}: {raw_line.rstrip()}")

            rows.append(row)
            cols.append(col)
            data.append(value)
            max_row = max(max_row, row)
            max_col = max(max_col, col)

    if max_row < 0 or max_col < 0:
        return sps.csr_matrix((0, 0), dtype=np.float64)

    matrix = sps.coo_matrix((data, (rows, cols)), shape=(max_row + 1, max_col + 1), dtype=np.float64)
    matrix.sum_duplicates()
    return complete_missing_symmetric_pairs(symmetrize_if_triangular(matrix.tocsr()))


def symmetrize_if_triangular(matrix: sps.csr_matrix) -> sps.csr_matrix:
    if matrix.shape[0] != matrix.shape[1]:
        return matrix.tocsr()

    upper = sps.triu(matrix, k=1).tocsr()
    lower = sps.tril(matrix, k=-1).tocsr()
    upper_abs_sum = float(np.abs(upper.data).sum())
    lower_abs_sum = float(np.abs(lower.data).sum())
    triangle_tolerance = 1e-12

    if upper_abs_sum > triangle_tolerance and lower_abs_sum <= triangle_tolerance * max(1.0, upper_abs_sum):
        return (matrix + upper.transpose()).tocsr()
    if lower_abs_sum > triangle_tolerance and upper_abs_sum <= triangle_tolerance * max(1.0, lower_abs_sum):
        return (matrix + lower.transpose()).tocsr()
    return matrix.tocsr()


def complete_missing_symmetric_pairs(matrix: sps.csr_matrix) -> sps.csr_matrix:
    if matrix.shape[0] != matrix.shape[1]:
        return matrix.tocsr()

    coo = matrix.tocoo()
    values: dict[tuple[int, int], float] = {}
    for row, col, value in zip(coo.row.tolist(), coo.col.tolist(), coo.data.tolist()):
        key = (int(row), int(col))
        values[key] = values.get(key, 0.0) + float(value)

    keys_to_process = [(row, col) for row, col in values.keys() if row < col]
    for row, col in keys_to_process:
        transpose_key = (col, row)
        has_upper = (row, col) in values
        has_lower = transpose_key in values
        upper = values.get((row, col), 0.0)
        lower = values.get(transpose_key, 0.0)

        if has_upper and has_lower:
            symmetric_value = 0.5 * (upper + lower)
            values[(row, col)] = symmetric_value
            values[transpose_key] = symmetric_value
        elif has_upper:
            values[transpose_key] = upper
        elif has_lower:
            values[(row, col)] = lower

    if not values:
        return sps.csr_matrix(matrix.shape, dtype=np.float64)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for (row, col), value in values.items():
        if abs(value) <= 1e-12:
            continue
        rows.append(row)
        cols.append(col)
        data.append(value)

    completed = sps.coo_matrix((data, (rows, cols)), shape=matrix.shape, dtype=np.float64)
    completed.sum_duplicates()
    return completed.tocsr()


def validate_full_dimensions(stiffness: sps.csr_matrix,
                             mass: sps.csr_matrix) -> None:
    if stiffness.shape[0] != stiffness.shape[1]:
        raise ValueError("Full stiffness matrix C must be square")
    if mass.shape[0] != mass.shape[1]:
        raise ValueError("Full mass matrix M must be square")
    if stiffness.shape != mass.shape:
        raise ValueError(
            f"Full stiffness and mass matrices must have the same shape, got {stiffness.shape} vs {mass.shape}"
        )


def load_coordinates(path: str) -> np.ndarray:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Coordinates file does not exist: {path}")

    values: list[float] = []
    with source.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith("%"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 3:
                raise ValueError(f"Failed to parse coordinate line in {path}: {raw_line.rstrip()}")
            if len(parts) >= 4:
                values.extend([float(parts[1]), float(parts[2]), float(parts[3])])
            else:
                values.extend([float(parts[0]), float(parts[1]), float(parts[2])])

    if not values:
        raise ValueError(f"Coordinates file is empty: {path}")
    coordinates = np.asarray(values, dtype=np.float64)
    if coordinates.size % 3 != 0:
        raise ValueError(f"Coordinates file must contain triples (x y z): {path}")
    return coordinates


def infer_dimensions_from_coords(full_dimension: int,
                                 coordinates: np.ndarray) -> tuple[int, int, int]:
    num_nodes = coordinates.size // 3
    mechanical_dofs = num_nodes * 3
    electrical_dofs = full_dimension - mechanical_dofs
    if electrical_dofs <= 0:
        raise ValueError(
            "Full system dimension must exceed the mechanical DOF count inferred from coords; "
            f"got full_dimension={full_dimension}, mechanical_dofs={mechanical_dofs}"
        )
    return num_nodes, mechanical_dofs, electrical_dofs


def apply_grounding_to_full_matrix(stiffness: sps.csr_matrix,
                                   mechanical_dofs: int,
                                   grounded_dof: int,
                                   ordering: str) -> sps.csr_matrix:
    n = stiffness.shape[0]
    electrical_dofs = n - mechanical_dofs
    if grounded_dof < 0 or grounded_dof >= n:
        raise ValueError(f"Grounded DOF {grounded_dof} is out of range for full matrix size {n}")
    if grounded_dof >= electrical_dofs:
        raise ValueError(
            f"Grounded electrical DOF {grounded_dof} is out of range for electrical block size {electrical_dofs}"
        )

    if ordering == "block-wise":
        global_index = mechanical_dofs + grounded_dof
    elif ordering == "node-based":
        num_nodes = mechanical_dofs // 3
        if electrical_dofs != num_nodes:
            raise ValueError(
                "node-based grounding currently requires one electrical DOF per node; "
                f"got electrical_dofs={electrical_dofs}, num_nodes={num_nodes}"
            )
        global_index = grounded_dof * 4 + 3
    else:
        raise ValueError(f"Unsupported ordering for grounding: {ordering}")

    grounded = stiffness.tolil(copy=True)
    grounded[global_index, :] = 0.0
    grounded[:, global_index] = 0.0
    grounded[global_index, global_index] = 1.0
    return grounded.tocsr()


def build_node_based_permutation(mechanical_dofs: int, electrical_dofs: int) -> np.ndarray:
    if mechanical_dofs % 3 != 0:
        raise ValueError(
            "node-based mixed ordering requires mechanical DOFs divisible by 3 "
            f"(got {mechanical_dofs})"
        )
    nodes = mechanical_dofs // 3
    if electrical_dofs != nodes:
        raise ValueError(
            "node-based mixed ordering currently requires one electrical DOF per node; "
            f"got mechanical_dofs={mechanical_dofs} -> nodes={nodes}, electrical_dofs={electrical_dofs}"
        )

    permutation: list[int] = []
    phi_offset = mechanical_dofs
    for node in range(nodes):
        permutation.extend([node * 3 + 0, node * 3 + 1, node * 3 + 2, phi_offset + node])
    return np.asarray(permutation, dtype=np.int64)


def reorder_matrix(matrix: sps.csr_matrix, permutation: np.ndarray) -> sps.csr_matrix:
    return matrix[permutation, :][:, permutation].tocsr()


def build_blockwise_to_node_based_permutation(mechanical_dofs: int, electrical_dofs: int) -> np.ndarray:
    return build_node_based_permutation(mechanical_dofs, electrical_dofs)


def build_node_based_to_blockwise_permutation(mechanical_dofs: int, electrical_dofs: int) -> np.ndarray:
    node_based = build_node_based_permutation(mechanical_dofs, electrical_dofs)
    inverse = np.empty_like(node_based)
    inverse[node_based] = np.arange(node_based.size, dtype=np.int64)
    return inverse


def scipy_to_petsc(matrix: sps.csr_matrix) -> PETSc.Mat:
    csr = matrix.tocsr()
    indptr = np.asarray(csr.indptr, dtype=PETSc.IntType)
    indices = np.asarray(csr.indices, dtype=PETSc.IntType)
    data = np.asarray(csr.data, dtype=PETSc.ScalarType)
    petsc_matrix = PETSc.Mat().createAIJ(size=csr.shape, csr=(indptr, indices, data))
    petsc_matrix.assemble()
    return petsc_matrix


def configure_eps(stiffness: PETSc.Mat,
                  mass: PETSc.Mat,
                  num_eigs: int,
                  tol: float,
                  max_iters: int,
                  target: float,
                  which: str) -> SLEPc.EPS:
    eps = SLEPc.EPS().create()
    eps.setOperators(stiffness, mass)
    eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
    eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    eps.setDimensions(num_eigs, PETSc.DECIDE)
    eps.setTolerances(tol=tol, max_it=max_iters)

    if which == "smallest-real":
        eps.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)
    elif which == "largest-real":
        eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_REAL)
    else:
        eps.setTarget(target)
        eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)
        st = eps.getST()
        st.setType(SLEPc.ST.Type.SINVERT)
        st.setShift(target)

    eps.setFromOptions()
    return eps


def solve_ordering(ordering: str,
                   input_ordering: str,
                   stiffness_full: sps.csr_matrix,
                   mass_full: sps.csr_matrix,
                   mechanical_dofs: int,
                   electrical_dofs: int,
                   args: argparse.Namespace) -> SolveSummary:
    stiffness = stiffness_full
    mass = mass_full

    if input_ordering != ordering:
        if input_ordering == "block-wise" and ordering == "node-based":
            permutation = build_blockwise_to_node_based_permutation(mechanical_dofs, electrical_dofs)
        elif input_ordering == "node-based" and ordering == "block-wise":
            permutation = build_node_based_to_blockwise_permutation(mechanical_dofs, electrical_dofs)
        else:  # pragma: no cover - protected by argparse choices
            raise ValueError(f"Unsupported ordering conversion: {input_ordering} -> {ordering}")
        stiffness = reorder_matrix(stiffness, permutation)
        mass = reorder_matrix(mass, permutation)

    stiffness_petsc = scipy_to_petsc(stiffness)
    mass_petsc = scipy_to_petsc(mass)

    eps = configure_eps(
        stiffness_petsc,
        mass_petsc,
        num_eigs=args.num_eigs,
        tol=args.tol,
        max_iters=args.max_iters,
        target=args.target,
        which=args.which,
    )
    eps.solve()

    nconv = eps.getConverged()
    eigenpairs: list[EigenpairSummary] = []
    vr, vi = stiffness_petsc.getVecs()
    for index in range(min(nconv, args.num_eigs)):
        eigenvalue = eps.getEigenpair(index, vr, vi)
        residual = float(eps.computeError(index))
        if isinstance(eigenvalue, complex):
            eig_real = float(eigenvalue.real)
            eig_imag = float(eigenvalue.imag)
        else:
            eig_real = float(eigenvalue)
            eig_imag = 0.0
        eigenpairs.append(
            EigenpairSummary(
                index=index,
                eigenvalue_real=eig_real,
                eigenvalue_imag=eig_imag,
                residual=residual,
            )
        )

    return SolveSummary(
        ordering=ordering,
        dimension=stiffness.shape[0],
        mechanical_dofs=mechanical_dofs,
        electrical_dofs=electrical_dofs,
        converged_pairs=nconv,
        iterations=eps.getIterationNumber(),
        reason=int(eps.getConvergedReason()),
        requested_pairs=args.num_eigs,
        eigenpairs=eigenpairs,
    )


def print_summary(summary: SolveSummary) -> None:
    PETSc.Sys.Print(f"=== PETSc/SLEPc mixed eigen solve: {summary.ordering} ===")
    PETSc.Sys.Print(
        f"dimension={summary.dimension}, mechanical_dofs={summary.mechanical_dofs}, "
        f"electrical_dofs={summary.electrical_dofs}"
    )
    PETSc.Sys.Print(
        f"iterations={summary.iterations}, converged_pairs={summary.converged_pairs}, "
        f"requested_pairs={summary.requested_pairs}, reason={summary.reason}"
    )
    for eigenpair in summary.eigenpairs:
        frequency_hz = eigenvalue_to_frequency_hz(eigenpair.eigenvalue_real, eigenpair.eigenvalue_imag)
        frequency_suffix = (
            f", frequency_hz={frequency_hz:.6f}" if frequency_hz is not None else ", frequency_hz=n/a"
        )
        PETSc.Sys.Print(
            "  mode {idx}: lambda={real:.12g}{imag:+.12g}i, residual={res:.6e}{freq}".format(
                idx=eigenpair.index,
                real=eigenpair.eigenvalue_real,
                imag=eigenpair.eigenvalue_imag,
                res=eigenpair.residual,
                freq=frequency_suffix,
            )
        )


def save_summary(summary: SolveSummary, output_prefix: str) -> None:
    output_path = Path(f"{output_prefix}_{summary.ordering.replace('-', '_')}.json")
    payload = {
        "ordering": summary.ordering,
        "dimension": summary.dimension,
        "mechanical_dofs": summary.mechanical_dofs,
        "electrical_dofs": summary.electrical_dofs,
        "converged_pairs": summary.converged_pairs,
        "iterations": summary.iterations,
        "reason": summary.reason,
        "requested_pairs": summary.requested_pairs,
        "eigenpairs": [
            {
                "index": eigenpair.index,
                "eigenvalue_real": eigenpair.eigenvalue_real,
                "eigenvalue_imag": eigenpair.eigenvalue_imag,
                "residual": eigenpair.residual,
            }
            for eigenpair in summary.eigenpairs
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    PETSc.Sys.Print(f"Wrote summary: {output_path}")


def parse_args(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Solve the full mixed piezo generalized eigenproblem with PETSc/SLEPc"
    )
    parser.add_argument("--stiffness", required=True, help="Full mixed stiffness matrix C")
    parser.add_argument("--mass", required=True, help="Full mixed mass matrix M")
    parser.add_argument("--coords", required=True, help="Node coordinates file (x y z per line)")
    parser.add_argument(
        "--input-ordering",
        choices=["block-wise", "node-based"],
        default="block-wise",
        help="Ordering of the provided full C/M matrices (default: block-wise)",
    )
    parser.add_argument(
        "--ordering",
        choices=["block-wise", "node-based", "both"],
        default="both",
        help="Mixed ordering to solve (default: both)",
    )
    parser.add_argument("--num-eigs", type=int, default=5, help="Number of eigenpairs to request")
    parser.add_argument("--tol", type=float, default=1e-8, help="EPS tolerance")
    parser.add_argument("--max-iters", type=int, default=1000, help="EPS iteration limit")
    parser.add_argument(
        "--which",
        choices=["target-real", "smallest-real", "largest-real"],
        default="target-real",
        help="Eigenvalue selection strategy (default: target-real)",
    )
    parser.add_argument("--target", type=float, default=0.0, help="Target eigenvalue for target-real mode")
    parser.add_argument("--grounded-dof", type=int, default=0, help="Electrical DOF to ground (default: 0)")
    parser.add_argument(
        "--no-grounding",
        action="store_true",
        help="Disable the default grounding step on C_phi before assembly",
    )
    parser.add_argument(
        "--output-prefix",
        help="Optional prefix for JSON summaries (one file per ordering)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print additional matrix statistics before solving",
    )
    return parser.parse_known_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args, unknown = parse_args(argv)
    if args.num_eigs <= 0:
        raise ValueError("--num-eigs must be positive")
    if args.tol <= 0.0:
        raise ValueError("--tol must be positive")
    if args.max_iters <= 0:
        raise ValueError("--max-iters must be positive")

    stiffness = load_sparse_auto(args.stiffness)
    mass = load_sparse_auto(args.mass)
    validate_full_dimensions(stiffness, mass)
    coordinates = load_coordinates(args.coords)
    _, mechanical_dofs, electrical_dofs = infer_dimensions_from_coords(stiffness.shape[0], coordinates)

    if not args.no_grounding:
        stiffness = apply_grounding_to_full_matrix(stiffness,
                                                   mechanical_dofs,
                                                   args.grounded_dof,
                                                   args.input_ordering)

    if args.verbose:
        PETSc.Sys.Print(
            "Loaded full matrices: "
            f"C={stiffness.shape}, nnz={stiffness.nnz}; "
            f"M={mass.shape}, nnz={mass.nnz}; "
            f"mechanical_dofs={mechanical_dofs}; electrical_dofs={electrical_dofs}; "
            f"input_ordering={args.input_ordering}"
        )
        if unknown:
            PETSc.Sys.Print(f"Forwarding PETSc/SLEPc options via slepc4py.init: {' '.join(unknown)}")

    orderings = ["block-wise", "node-based"] if args.ordering == "both" else [args.ordering]
    total_start = time.perf_counter()
    for ordering in orderings:
        PETSc.Sys.Print(f"\n--- Solving {ordering} ordering ---")
        solve_start = time.perf_counter()
        summary = solve_ordering(ordering,
                                 args.input_ordering,
                                 stiffness,
                                 mass,
                                 mechanical_dofs,
                                 electrical_dofs,
                                 args)
        solve_elapsed = time.perf_counter() - solve_start
        PETSc.Sys.Print(f"Solve time: {solve_elapsed:.3f} s")
        print_summary(summary)
        if args.output_prefix:
            save_summary(summary, args.output_prefix)

    total_elapsed = time.perf_counter() - total_start
    PETSc.Sys.Print(f"\nTotal elapsed time: {total_elapsed:.3f} s")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
