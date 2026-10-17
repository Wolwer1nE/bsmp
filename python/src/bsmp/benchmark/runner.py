#!/usr/bin/env python3
"""Benchmark runner: execute bsmp examples over many (matrix, rhs) cases."""

import re
import shutil
import sys
from pathlib import Path

from bsmp.benchmark.plots import make_plots
from bsmp.benchmark.records import (
    collect_records,
    print_table,
    rows_from_records,
    write_csv,
)
from bsmp.launchers.mult import run_mult
from bsmp.launchers.solve import run_solve

METHODS = ("mult", "bicgstab", "gmres")

DEFAULT_MULTS = 10

_MM_HEADER_RE = re.compile(r"^\s*%%MatrixMarket")


def _detect_format(matrix_path: str) -> str:
    """Pick the loader format from the matrix file header."""
    with open(matrix_path, encoding="utf-8", errors="replace") as fh:
        first_line = fh.readline()
    return "matrix-market" if _MM_HEADER_RE.search(first_line) else "triplets"


def _run_case(
    method: str,
    matrix: str,
    rhs: str,
    run_dir: Path,
    *,
    mults: int,
    precond: str,
    max_iters: str,
    tol: str,
    restart: str,
    no_build: bool,
) -> None:
    """Run one example in its own working directory."""
    matrix_path = str(Path(matrix).resolve())
    rhs_path = str(Path(rhs).resolve())
    cwd = str(run_dir)
    # The examples save solution files to data/ under their working dir.
    (run_dir / "data").mkdir(exist_ok=True)
    if method == "mult":
        run_mult(
            matrix_path, rhs_path, _detect_format(matrix_path), mults, "", False,
            no_build, cwd=cwd,
        )
    elif method == "bicgstab":
        run_solve(
            method, matrix_path, rhs_path, precond, "", max_iters, tol, "",
            no_build, cwd=cwd,
        )
    elif method == "gmres":
        run_solve(
            method, matrix_path, rhs_path, precond, "", max_iters, tol, restart,
            no_build, cwd=cwd,
        )
    else:
        raise ValueError(f"unsupported method '{method}'")


def run_cases(
    cases: list[tuple[str, str]],
    methods: tuple[str, ...] = METHODS,
    *,
    work_dir: str | Path,
    records_dir: str | Path,
    mults: int = DEFAULT_MULTS,
    precond: str = "",
    max_iters: str = "",
    tol: str = "",
    restart: str = "",
    no_build: bool = False,
) -> list[Path]:
    """Run every method on every (matrix, rhs) case and collect the records.

    Each (case, method) pair executes in a dedicated subdirectory, so the
    JSON records written by the examples are collected without clashing.
    Failed runs are reported on stderr and skipped. Returns the list of
    collected record file paths.
    """
    work_dir = Path(work_dir)
    records_dir = Path(records_dir)
    records_dir.mkdir(parents=True, exist_ok=True)
    collected: list[Path] = []
    failures: list[tuple[str, str, str]] = []

    for index, (matrix_path, rhs_path) in enumerate(cases):
        stem = Path(matrix_path).stem
        label = stem if len(cases) == 1 else f"{index:02d}-{stem}"
        for method in methods:
            run_dir = work_dir / f"{label}-{method}"
            shutil.rmtree(run_dir, ignore_errors=True)
            run_dir.mkdir(parents=True, exist_ok=True)
            try:
                _run_case(
                    method, matrix_path, rhs_path, run_dir,
                    mults=mults, precond=precond, max_iters=max_iters,
                    tol=tol, restart=restart, no_build=no_build,
                )
            except Exception as exc:
                # A single failing case must not abort the whole benchmark.
                failures.append((label, method, str(exc)))
                continue

            logs = sorted(run_dir.glob("log*.json"))
            if not logs:
                failures.append((label, method, "no log*.json produced"))
                continue
            for log_index, log in enumerate(logs):
                suffix = "" if len(logs) == 1 else f"-{log_index}"
                target = records_dir / f"{label}-{method}{suffix}.json"
                shutil.copyfile(log, target)
                collected.append(target)

    for label, method, error in failures:
        print(
            f"WARNING: case '{label}' method '{method}' failed: {error}",
            file=sys.stderr,
        )
    return collected


def run_benchmark(
    cases: list[tuple[str, str]],
    methods: tuple[str, ...] = METHODS,
    *,
    work_dir: str | Path,
    records_dir: str | Path,
    mults: int = DEFAULT_MULTS,
    precond: str = "",
    max_iters: str = "",
    tol: str = "",
    restart: str = "",
    no_build: bool = False,
    table_path: str | Path = "",
    plot_path: str | Path = "",
) -> list[Path]:
    """Run the benchmark pipeline: execute, collect, tabulate, plot."""
    collected = run_cases(
        cases,
        methods,
        work_dir=work_dir,
        records_dir=records_dir,
        mults=mults,
        precond=precond,
        max_iters=max_iters,
        tol=tol,
        restart=restart,
        no_build=no_build,
    )
    records = collect_records(collected)
    rows = rows_from_records(records)
    print_table(rows)
    if table_path:
        write_csv(rows, table_path)
        print(f"Table written to {table_path}")
    if plot_path and rows:
        make_plots(rows, plot_path)
    return collected
