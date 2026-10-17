"""Benchmark: run examples over many matrices, collect records, plot results."""

from bsmp.benchmark.plots import make_plots
from bsmp.benchmark.records import (
    collect_records,
    print_table,
    record_to_rows,
    rows_from_records,
    write_csv,
)
from bsmp.benchmark.runner import METHODS, run_benchmark, run_cases

__all__ = [
    "METHODS",
    "collect_records",
    "make_plots",
    "print_table",
    "record_to_rows",
    "rows_from_records",
    "run_benchmark",
    "run_cases",
    "write_csv",
]
