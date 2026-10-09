#!/usr/bin/env python3
"""Comparison launchers for the BSMP solvers (mirror bin/compare_*.sh)."""

import os
import re
import shutil
import subprocess
import sys
import tempfile
from os.path import isfile
from pathlib import Path

from bsmp.config import CONFIG
from bsmp.launchers._build import ensure_built
from bsmp.launchers.solve import run_solve

DEFAULT_PRECONDS = "amg,scalar-jacobi,block-jacobi,none"

_METRIC_LABELS = (
    "Converged",
    "Iterations",
    "Final relative residual",
    "Solve time (ms)",
)

_ALL_COLUMNS = (
    ("Solver", 10),
    ("Preconditioner", 18),
    ("Status", 10),
    ("Converged", 12),
    ("Iterations", 12),
    ("RelResidual", 20),
    ("Time(ms)", 16),
    ("Exit", 8),
)

_PRECOND_COLUMNS = _ALL_COLUMNS[1:]

_SOLVER_COLUMNS = (
    ("Method", 12),
    ("Status", 10),
    ("Converged", 12),
    ("RelResidual", 20),
    ("Time(ms)", 16),
    ("Exit", 8),
)

_ITERATION_COLUMNS = (("Method", 12), ("Iterations", 12))

SUPPORTED_METHODS = ("bicgstab", "gmres")


def _parse_preconds(preconds_csv: str) -> list[str]:
    """Split a comma-separated preconditioner list, skipping empty entries."""
    return [p.strip() for p in preconds_csv.split(",") if p.strip()]


def _extract_metrics(log_text: str) -> dict[str, str]:
    """Extract metric values from solver output, last occurrence wins.

    Values are stripped so that Windows CRLF line endings do not leak a
    trailing ``\\r`` into the printed tables (which would move the
    terminal cursor back to column 0 and scramble the rows).
    """
    metrics = {}
    for label in _METRIC_LABELS:
        matches = re.findall(rf"^{re.escape(label)}: (.*)$", log_text, re.MULTILINE)
        metrics[label] = matches[-1].strip() if matches else ""
    return metrics


def _max_abs_diff(path_a: str, path_b: str) -> float:
    """Return the maximum absolute difference of two solution vectors."""

    def _read_values(path: str) -> list[float]:
        values = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if not fields:
                continue
            try:
                values.append(float(fields[0]))
            except ValueError:
                values.append(0.0)  # awk coerces non-numeric fields to zero
        return values

    values_a = _read_values(path_a)
    values_b = _read_values(path_b)
    return max(
        (abs(a - b) for a, b in zip(values_a, values_b, strict=False)),
        default=0.0,
    )


def _table_row(columns, values) -> str:
    return " ".join(
        f"{value:<{width}}" for (_, width), value in zip(columns, values, strict=False)
    )


def _validate(matrix_file: str, rhs_file: str) -> None:
    if not isfile(matrix_file):
        raise ValueError(f"matrix file '{matrix_file}' not found")
    if not isfile(rhs_file):
        raise ValueError(f"rhs file '{rhs_file}' not found")


def _run_solve_logged(
    method: str,
    matrix_file: str,
    rhs_file: str,
    precond: str,
    output_file: str,
    max_iters: str,
    tol: str,
    restart: str,
) -> tuple[int, str]:
    """Run one solver, capturing its output and exit status.

    The solver subprocess inherits the process-level stdout/stderr, so the
    streams are redirected at the fd level (like `> log 2>&1` in the shell).
    """
    status = 0
    with tempfile.NamedTemporaryFile(
        mode="w+", encoding="utf-8", newline="\n"
    ) as log_file:
        saved_stdout = os.dup(1)
        saved_stderr = os.dup(2)
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(log_file.fileno(), 1)
            os.dup2(log_file.fileno(), 2)
            try:
                run_solve(
                    method,
                    matrix_file,
                    rhs_file,
                    precond=precond,
                    output_file=output_file,
                    max_iters=max_iters,
                    tol=tol,
                    restart=restart,
                    no_build=True,
                )
            except subprocess.CalledProcessError as e:
                status = e.returncode
            except ValueError as e:
                status = 1
                print(f"Invalid inputs error: {e}")
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
            os.close(saved_stdout)
            os.close(saved_stderr)
        log_file.flush()
        log_file.seek(0)
        return status, log_file.read()


def _preserve_logs(tmp_dir: Path) -> None:
    print(f"Logs preserved in: {tmp_dir}", file=sys.stderr)


def run_compare_all(
    matrix_file: str,
    rhs_file: str,
    preconds: str = DEFAULT_PRECONDS,
    restart: str = "30",
    max_iters: str = "1000",
    tol: str = "1e-6",
    no_build: bool = False,
) -> None:
    """Run the full solver/preconditioner comparison matrix."""
    _validate(matrix_file, rhs_file)

    ensure_built(no_build=no_build)

    precond_list = _parse_preconds(preconds)
    tmp_dir = Path(tempfile.mkdtemp(prefix="bsmp_compare_all_"))

    overall_status = 0
    try:
        print(f"Matrix:        {matrix_file}")
        print(f"RHS:           {rhs_file}")
        print(f"Tolerance:     {tol}")
        print(f"Max iters:     {max_iters}")
        print(f"GMRES restart: {restart}")
        print()
        print(_table_row(_ALL_COLUMNS, [name for name, _ in _ALL_COLUMNS]))
        for method in SUPPORTED_METHODS:
            for precond in precond_list:
                log_file = tmp_dir / f"{method}_{precond}.log"
                sol_file = tmp_dir / f"{method}_{precond}.txt"
                status, output = _run_solve_logged(
                    method,
                    matrix_file,
                    rhs_file,
                    precond,
                    str(sol_file),
                    max_iters,
                    tol,
                    restart,
                )
                log_file.write_text(output, encoding="utf-8")

                metrics = _extract_metrics(output)
                status_str = "OK" if status == 0 else "FAIL"
                print(
                    _table_row(
                        _ALL_COLUMNS,
                        [
                            method,
                            precond,
                            status_str,
                            metrics["Converged"] or "N/A",
                            metrics["Iterations"] or "N/A",
                            metrics["Final relative residual"] or "N/A",
                            metrics["Solve time (ms)"] or "N/A",
                            status,
                        ],
                    )
                )
                if status != 0:
                    overall_status = 1
    except BaseException:
        _preserve_logs(tmp_dir)
        raise

    if overall_status != 0:
        _preserve_logs(tmp_dir)
        raise SystemExit(overall_status)
    shutil.rmtree(tmp_dir, ignore_errors=True)


def run_compare_preconditioners(
    method: str,
    matrix_file: str,
    rhs_file: str,
    preconds: str = DEFAULT_PRECONDS,
    restart: str = "30",
    max_iters: str = "1000",
    tol: str = "1e-6",
    no_build: bool = False,
) -> None:
    """Compare several preconditioners for one solver."""
    if method not in SUPPORTED_METHODS:
        raise ValueError("--method must be bicgstab or gmres")
    _validate(matrix_file, rhs_file)

    ensure_built(no_build=no_build)

    precond_list = _parse_preconds(preconds)
    tmp_dir = Path(tempfile.mkdtemp(prefix="bsmp_compare_preconditioners_"))

    overall_status = 0
    try:
        print(f"Method:        {method}")
        print(f"Matrix:        {matrix_file}")
        print(f"RHS:           {rhs_file}")
        print(f"Tolerance:     {tol}")
        print(f"Max iters:     {max_iters}")
        if method == "gmres":
            print(f"GMRES restart: {restart}")
        print()
        print(_table_row(_PRECOND_COLUMNS, [name for name, _ in _PRECOND_COLUMNS]))
        for precond in precond_list:
            log_file = tmp_dir / f"{method}_{precond}.log"
            sol_file = tmp_dir / f"{method}_{precond}.txt"
            status, output = _run_solve_logged(
                method,
                matrix_file,
                rhs_file,
                precond,
                str(sol_file),
                max_iters,
                tol,
                restart,
            )
            log_file.write_text(output, encoding="utf-8")

            metrics = _extract_metrics(output)
            status_str = "OK" if status == 0 else "FAIL"
            print(
                _table_row(
                    _PRECOND_COLUMNS,
                    [
                        precond,
                        status_str,
                        metrics["Converged"] or "N/A",
                        metrics["Iterations"] or "N/A",
                        metrics["Final relative residual"] or "N/A",
                        metrics["Solve time (ms)"] or "N/A",
                        status,
                    ],
                )
            )
            if status != 0:
                overall_status = 1
    except BaseException:
        _preserve_logs(tmp_dir)
        raise

    if overall_status != 0:
        _preserve_logs(tmp_dir)
        raise SystemExit(overall_status)
    shutil.rmtree(tmp_dir, ignore_errors=True)


def run_compare_solvers(
    matrix_file: str,
    rhs_file: str,
    precond: str = "amg",
    restart: str = "30",
    max_iters: str = "1000",
    tol: str = "1e-6",
    no_build: bool = False,
) -> None:
    """Compare BiCGStab and GMRES on the same linear system."""
    _validate(matrix_file, rhs_file)

    ensure_built(no_build=no_build)

    tmp_dir = Path(tempfile.mkdtemp(prefix="bsmp_compare_solvers_"))

    try:
        bicg_log = tmp_dir / "bicgstab.log"
        gmres_log = tmp_dir / "gmres.log"
        bicg_sol = tmp_dir / "bicgstab_x.txt"
        gmres_sol = tmp_dir / "gmres_x.txt"

        bicg_status, bicg_out = _run_solve_logged(
            "bicgstab",
            matrix_file,
            rhs_file,
            precond,
            str(bicg_sol),
            max_iters,
            tol,
            "",
        )
        bicg_log.write_text(bicg_out, encoding="utf-8")

        gmres_status, gmres_out = _run_solve_logged(
            "gmres",
            matrix_file,
            rhs_file,
            precond,
            str(gmres_sol),
            max_iters,
            tol,
            restart,
        )
        gmres_log.write_text(gmres_out, encoding="utf-8")

        bicg_metrics = _extract_metrics(bicg_out)
        gmres_metrics = _extract_metrics(gmres_out)

        print(f"Matrix:        {matrix_file}")
        print(f"RHS:           {rhs_file}")
        print(f"Preconditioner: {precond}")
        print(f"Tolerance:     {tol}")
        print(f"Max iters:     {max_iters}")
        print(f"GMRES restart: {restart}")
        print()
        print(_table_row(_SOLVER_COLUMNS, [name for name, _ in _SOLVER_COLUMNS]))
        print(
            _table_row(
                _SOLVER_COLUMNS,
                [
                    "BiCGStab",
                    "OK" if bicg_status == 0 else "FAIL",
                    bicg_metrics["Converged"] or "N/A",
                    bicg_metrics["Final relative residual"] or "N/A",
                    bicg_metrics["Solve time (ms)"] or "N/A",
                    bicg_status,
                ],
            )
        )
        print(
            _table_row(
                _SOLVER_COLUMNS,
                [
                    "GMRES",
                    "OK" if gmres_status == 0 else "FAIL",
                    gmres_metrics["Converged"] or "N/A",
                    gmres_metrics["Final relative residual"] or "N/A",
                    gmres_metrics["Solve time (ms)"] or "N/A",
                    gmres_status,
                ],
            )
        )
        print()
        print(_table_row(_ITERATION_COLUMNS, [name for name, _ in _ITERATION_COLUMNS]))
        print(
            _table_row(
                _ITERATION_COLUMNS,
                ["BiCGStab", bicg_metrics["Iterations"] or "N/A"],
            )
        )
        print(
            _table_row(
                _ITERATION_COLUMNS,
                ["GMRES", gmres_metrics["Iterations"] or "N/A"],
            )
        )

        if bicg_sol.is_file() and gmres_sol.is_file():
            print()
            print(
                "Max |x_bicgstab - x_gmres|: "
                f"{_max_abs_diff(str(bicg_sol), str(gmres_sol)):.9g}"
            )

        if bicg_status != 0 or gmres_status != 0:
            print()
            print(f"BiCGStab log: {bicg_log}", file=sys.stderr)
            print(f"GMRES log:    {gmres_log}", file=sys.stderr)
            raise SystemExit(1)
    except BaseException:
        _preserve_logs(tmp_dir)
        raise

    shutil.rmtree(tmp_dir, ignore_errors=True)
