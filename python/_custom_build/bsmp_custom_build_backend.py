#!/usr/bin/env python3
"""Custom PEP 517 build backend: CMake build + setuptools delegation."""

from setuptools import build_meta as _orig
from setuptools.build_meta import *
from pathlib import Path
import subprocess
import sys
import os


_BACKEND_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _BACKEND_DIR.parents[1]
_CMAKE_SOURCE = _PROJECT_ROOT
_CMAKE_BUILD = _PROJECT_ROOT / "build"
_CMAKE_PREFIX = _PROJECT_ROOT / "python" / "src" / "bsmp"


def _detect_platform() -> str:
    """Return the current platform tag for wheel naming.

    Returns a PEP 427 compliant platform tag such as ``win_amd64``
    or ``manylinux2014_x86_64`` so that the resulting wheel is not
    tagged ``py3-none-any``.
    """
    import sysconfig
    return sysconfig.get_platform().replace("-", "_")


def _patch_bdist_wheel():
    """Monkey-patch bdist_wheel.get_tag so pure packages get a platform tag.

    ``bdist_wheel`` always returns ``(pyver, abi, "any")`` for packages
    with no compiled extensions, regardless of ``plat_name``.  Because
    BSMP installs pre-compiled CMake binaries the package *is* platform-
    specific even though it has no Python extension modules.
    """
    import wheel.bdist_wheel as _mod

    _old_get_tag = _mod.bdist_wheel.get_tag

    def _new_get_tag(self):
        pytag, abitag, plat = _old_get_tag(self)
        if plat == "any":
            plat = _detect_platform()
        return (pytag, abitag, plat)

    _mod.bdist_wheel.get_tag = _new_get_tag


def _run_cmake(args: list[str], stage: str) -> None:
    """Run a CMake command and raise a descriptive error on failure."""
    cmd = ["cmake", *args]
    try:
        subprocess.run(
            cmd,
            cwd=str(_CMAKE_SOURCE),
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        header = f"ERROR: {stage} failed"
        footer = (
            f"\n  Command : {' '.join(cmd)}\n"
            f"  Working dir: {_CMAKE_SOURCE}\n"
        )
        if exc.stdout:
            footer += f"\n  stdout  :\n{exc.stdout}\n"
        if exc.stderr:
            footer += f"\n  stderr  :\n{exc.stderr}\n"
        # Strip trailing whitespace for a clean block
        footer = footer.rstrip() + "\n"
        print(header + footer, file=sys.stderr)
        raise SystemExit(1) from None
    except FileNotFoundError:
        print(
            f"ERROR: {stage} failed — 'cmake' executable not found.\n"
            f"  Please ensure CMake (≥ 3.18) is on your PATH.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None


def build_wheel(wheel_directory: str, config_settings=None, metadata_directory=None) -> str:
    _run_cmake(
        [
            "-S", str(_CMAKE_SOURCE),
            "-B", str(_CMAKE_BUILD),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_CONFIGURATION_TYPES=Release",
        ],
        stage="CMake configure",
    )

    _run_cmake(
        [
            "--build", str(_CMAKE_BUILD),
            "--clean-first",
            "-j",
            "--config",
            "Release"
        ],
        stage="CMake build",
    )

    _run_cmake(
        [
            "--install", str(_CMAKE_BUILD),
            "--prefix", str(_CMAKE_PREFIX),
            "--component", "runtime",
        ],
        stage="CMake install",
    )

    # Force platform-specific wheel tag (not py3-none-any) because the
    # wheel contains pre-compiled CMake-installed binaries (.exe / .dll
    # / .so).  bdist_wheel treats any package without extension modules
    # as pure-Python and always emits "any" regardless of plat_name.
    _patch_bdist_wheel()

    return _orig.build_wheel(wheel_directory, config_settings, metadata_directory)
