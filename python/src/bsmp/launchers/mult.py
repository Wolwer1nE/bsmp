#!/usr/bin/env python
"""Launcher for the BSMP multiplication test (mirrors bin/cmd/mult.sh)."""

import subprocess
from os.path import isfile

from bsmp.config import CONFIG
from bsmp.launchers._build import clean_build


def run_mult(
    matrix_file: str,
    rhs_file: str,
    format: str,
    mults: int,
    output: str,
    verbose: bool,
    no_build: bool,
) -> None:
    """Run the multiplication test on the given matrix and RHS vector."""
    if not matrix_file:
        raise ValueError("mult requires <matrix_file>")
    if not isfile(matrix_file):
        raise ValueError(f"matrix file '{matrix_file}' not found")
    if not rhs_file:
        raise ValueError("--rhs <file> must be provided")
    if not isfile(rhs_file):
        raise ValueError(f"rhs file '{rhs_file}' not found")

    if not no_build and CONFIG.is_source_tree:
        clean_build()

    exe = CONFIG.find_exe("example")

    args = ["--matrix", matrix_file, "--format", format, "--mults", str(mults)]
    if rhs_file:
        args += ["--rhs", rhs_file]
    if output:
        args += ["--output", output]
    if verbose:
        args += ["--verbose"]

    subprocess.run([str(exe), *args], check=True)
