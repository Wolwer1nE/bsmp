#!/usr/bin/env python3
"""Tools for building project with CMake on any platform."""

import subprocess
import sys

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
            "-DCMAKE_CONFIGURATION_TYPES=Release"
        ],
        check=True,
    )
    subprocess.run(
        [
            "cmake",
            "--build",
            str(CONFIG.build_dir),
            "--clean-first",
            "-j",
            "--config",
            "Release"
        ],
        check=True,
    )
    subprocess.run(
            [
                "cmake",
                "--install",
                str(CONFIG.build_dir),
                "--prefix", str(CONFIG.exe_dir.parent),
                "--component",
                "runtime"],
            check=True,
    )


def ensure_built(*, no_build: bool = False) -> None:
    """Build the CMake project if we are in a source tree.

    For wheel-installed packages there is no *CMakeLists.txt*, so the build
    step is skipped silently (or with a short hint when the user did *not*
    pass ``--no-build``).
    """
    if not CONFIG.is_source_tree:
        if not no_build:
            print(
                "INFO: build step skipped — installed package. "
                "Use --no-build explicitly or install from source.",
                file=sys.stderr,
            )
        return

    if no_build:
        return

    clean_build()
