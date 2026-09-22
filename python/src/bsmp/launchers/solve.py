#!/usr/bin/env python
"""Launcher for the BSMP solver (mirrors bin/cmd/solve.sh)."""
from os.path import isfile
from pathlib import Path
import re
import subprocess
import tempfile
from bsmp.config import CONFIG
from bsmp.launchers._build import clean_build, is_windows

SUPPORTED_METHODS = ("bicgstab", "gmres")
SUPPORTED_PRECONDS = ("amg", "scalar-jacobi", "block-jacobi", "none")

# Matrix Market vector header: "%%MatrixMarket ..."
_MM_HEADER_RE = re.compile(r"^\s*%%MatrixMarket")
# A size line like "m n" (two integers only)
_SIZE_LINE_RE = re.compile(r"^\s*\d+\s+\d+\s*$")
_COMMENT_LINE_RE = re.compile(r"^\s*[%#]")


def _strip_mm_vector_header(rhs_file: str) -> str:
    """Return a cleaned RHS path, stripping Matrix Market comments and size line.

    Mirrors the awk preprocessing in bin/cmd/solve.sh. If the file is not a
    Matrix Market vector, the original path is returned unchanged.
    """
    with open(rhs_file, "r") as f:
        first_line = f.readline()
    if not _MM_HEADER_RE.search(first_line):
        return rhs_file

    cleaned_lines = []
    first_noncomment = True
    with open(rhs_file, "r") as f:
        for line in f:
            if first_noncomment:
                if _COMMENT_LINE_RE.match(line):
                    continue
                if _SIZE_LINE_RE.match(line):
                    first_noncomment = False
                    continue
                first_noncomment = False
            cleaned_lines.append(line)

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".rhs", delete=False, newline="\n")
    with tmp:
        tmp.writelines(cleaned_lines)
    return tmp.name


def run_solve(method: str,
              matrix_file: str,
              rhs_file: str,
              precond: str = "",
              output_file: str = "",
              max_iters: str = "",
              tol: str = "",
              restart: str = "",
              no_build: bool = False) -> None:
    """Solve Ax=b using the selected method on the given matrix and RHS."""
    if not method:
        raise ValueError("--method is required")
    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f"unsupported method '{method}'. Supported methods: {', '.join(SUPPORTED_METHODS)}")
    if not matrix_file or not rhs_file:
        raise ValueError("--matrix and --rhs are required")
    if not isfile(matrix_file):
        raise ValueError(f"matrix file '{matrix_file}' not found")
    if not isfile(rhs_file):
        raise ValueError(f"rhs file '{rhs_file}' not found")

    if not no_build:
        clean_build()

    # TODO: un-hardcode the preset name for Windows
    exe_filename = "example_gmres" if method == "gmres" else "example_bicgstab"
    exe = ""
    if is_windows():
        exe = CONFIG.build_dir / ("msvc-ideapad\\examples\\Release\\" + 
                                  exe_filename + ".exe")
    else:
        exe = CONFIG.build_dir / ("example_gmres" if method == "gmres" else "example_bicgstab")

    if not isfile(exe):
        raise ValueError(f"executable '{exe}' not found")

    if not isfile(exe):
        raise ValueError(f"executable '{exe}' not found")

    rhs_to_use = _strip_mm_vector_header(rhs_file)

    args = [matrix_file, rhs_to_use]
    if output_file:
        args.append(output_file)

    if method == "gmres":
        if restart:
            if not output_file:
                args.append("")
            args.append(restart)
        if max_iters:
            if not output_file and not restart:
                args.append("")
            if not restart:
                args.append("30")
            args.append(max_iters)
        if tol:
            if not output_file and not restart and not max_iters:
                args.append("")
            if not restart:
                args.append("30")
            if not max_iters:
                args.append("1000")
            args.append(tol)
    else:
        if max_iters:
            if not output_file:
                args.append("")
            args.append(max_iters)
        if tol:
            if not output_file and not max_iters:
                args.append("")
            if not max_iters:
                args.append("1000")
            args.append(tol)

    if precond:
        if method == "gmres":
            if not output_file and not restart and not max_iters and not tol:
                args.append("")
            if not restart:
                args.append("30")
            if not max_iters:
                args.append("1000")
            if not tol:
                args.append("1e-6")
        else:
            if not output_file and not max_iters and not tol:
                args.append("")
            if not max_iters:
                args.append("1000")
            if not tol:
                args.append("1e-6")
        args.append(precond)

    try:
        subprocess.run([str(exe)] + args, check=True)
    finally:
        # Clean up the temporary RHS if one was created.
        if rhs_to_use != rhs_file:
            Path(rhs_to_use).unlink(missing_ok=True)