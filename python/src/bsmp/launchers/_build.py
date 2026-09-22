#!/usr/bin/env python
"""Tools for building project with CMake on any platform."""

import sys
import subprocess
from bsmp.config import CONFIG


def is_windows() -> bool:
    return sys.platform.startswith("win")


def is_linux() -> bool:
    return sys.platform.startswith("linux")


def clean_build() -> None:
    """Configure and build the project with CMake (Release)."""
    subprocess.run(
        [
            "cmake",
            "-S",
            str(CONFIG.root_dir),
            "-B",
            str(CONFIG.build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
        ],  # FIXME: doesn't work as expected with multi-config generators
        check=True,
    )
    subprocess.run(
        ["cmake", "--build", str(CONFIG.build_dir), "--clean-first", "-j"],
        check=True,
    )
