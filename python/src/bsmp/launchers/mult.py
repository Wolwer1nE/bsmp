#!/usr/bin/env python
"""Launcher for the BSMP multiplication test (mirrors bin/cmd/mult.sh)."""

from os.path import isfile
import subprocess
from bsmp.config import CONFIG
from bsmp.launchers._build import is_linux, is_windows, clean_build


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

    if not no_build:
        clean_build()

    # TODO: un-hardcode the preset name for Windows
    exe = ""
    if is_windows():
        exe = CONFIG.build_dir / "msvc-ideapad\\examples\\Release\\example.exe"
    else:
        exe = CONFIG.build_dir / "example"
    if not isfile(exe):
        raise ValueError(f"executable '{exe}' not found")

    args = ["--matrix", matrix_file, "--format", format, "--mults", str(mults)]
    if rhs_file:
        args += ["--rhs", rhs_file]
    if output:
        args += ["--output", output]
    if verbose:
        args += ["--verbose"]

    subprocess.run([str(exe)] + args, check=True)
