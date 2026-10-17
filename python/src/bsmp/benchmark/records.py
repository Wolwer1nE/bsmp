#!/usr/bin/env python3
"""Collection and tabulation of bsmp experiment JSON records."""

import csv
import json
from collections.abc import Iterable
from pathlib import Path

# Fixed column order for the aggregated table.
COLUMNS = [
    "case",
    "type",
    "solver",
    "preconditioner",
    "max_iters",
    "tolerance",
    "time_ms",
    "avg_time_ms",
    "residual",
    "matrix",
    "rows",
    "columns",
    "nnz",
    "rhs_size",
    "rhs_from_task",
    "gpu_model",
    "cuda_version",
    "compute_capability",
]


def collect_records(records_arg: Iterable[str | Path]) -> list[tuple[str, dict]]:
    """Collect (label, record) pairs from JSON files and/or directories.

    Directories are scanned for ``*.json`` files; files are read as-is. The
    label of each record is its file stem, disambiguated with the parent
    directory name when two files share a stem.
    """
    records: list[tuple[str, dict]] = []
    seen: set[str] = set()
    for arg in records_arg:
        path = Path(arg)
        if path.is_dir():
            files = sorted(path.glob("*.json"))
        elif path.is_file():
            files = [path]
        else:
            raise FileNotFoundError(f"record path not found: {path}")
        for file in files:
            label = file.stem
            if label in seen:
                label = f"{label} ({file.parent.name})"
            seen.add(label)
            with open(file, encoding="utf-8") as fh:
                records.append((label, json.load(fh)))
    return records


def _get(run: dict, section: str, key: str):
    return run.get(section, {}).get(key, "")


def record_to_rows(label: str, record: dict) -> list[dict]:
    """Flatten one experiment record into one row per run."""
    problem = record.get("problem", {})
    matrix = problem.get("matrix", {})
    rhs = problem.get("rhs", {})
    hardware = record.get("hardware", {})
    rows = []
    for run in record.get("runs", []):
        rows.append(
            {
                "case": label,
                "type": problem.get("type", ""),
                "solver": _get(run, "parameters", "solver"),
                "preconditioner": _get(run, "parameters", "preconditioner"),
                "max_iters": _get(run, "parameters", "max_iters"),
                "tolerance": _get(run, "parameters", "tolerance"),
                "time_ms": _get(run, "metrics", "time"),
                "avg_time_ms": _get(run, "metrics", "avg_time"),
                "residual": _get(run, "metrics", "residual"),
                "matrix": matrix.get("name", ""),
                "rows": matrix.get("rows", ""),
                "columns": matrix.get("columns", ""),
                "nnz": matrix.get("nnz", ""),
                "rhs_size": rhs.get("size", ""),
                "rhs_from_task": rhs.get("from_task", ""),
                "gpu_model": hardware.get("gpu_model", ""),
                "cuda_version": hardware.get("cuda_version", ""),
                "compute_capability": hardware.get("compute_capability", ""),
            }
        )
    return rows


def rows_from_records(records: Iterable[tuple[str, dict]]) -> list[dict]:
    """Flatten many (label, record) pairs into a single list of rows."""
    return [
        row
        for label, record in records
        for row in record_to_rows(label, record)
    ]


def _cell(value) -> str:
    return "" if value is None else str(value)


def print_table(rows: list[dict]) -> None:
    """Print the aggregated rows as an aligned text table."""
    if not rows:
        print("No records collected.")
        return
    columns = [c for c in COLUMNS if any(_cell(row[c]) for row in rows)]
    widths = {c: max(len(c), max(len(_cell(row[c])) for row in rows)) for c in columns}
    print("  ".join(c.ljust(widths[c]) for c in columns))
    print("  ".join("-" * widths[c] for c in columns))
    for row in rows:
        print("  ".join(_cell(row[c]).ljust(widths[c]) for c in columns))


def write_csv(rows: list[dict], output: str | Path) -> None:
    """Write the aggregated rows to a CSV file."""
    with open(output, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
