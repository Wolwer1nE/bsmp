#!/usr/bin/env python
"""Launcher for the BSMP eigen example (mirrors bin/cmd/eigen.sh)."""

from os.path import isfile
import subprocess
from bsmp.config import CONFIG
from bsmp.launchers._build import clean_build, is_windows

# Options that take a value and are passed straight through to the executable.
_VALUE_OPTIONS = (
    "--stiffness",
    "--cu",
    "--cuphi",
    "--cphi",
    "--mass",
    "--coords",
    "--ordering",
    "--dielectric-sign",
    "--scaling",
    "--eigensolve-path",
    "--grounded-dof",
    "--modes",
    "--tol",
    "--max-iters",
    "--sa-amg-regularization-epsilon",
    "--sa-amg-pre-sweeps",
    "--sa-amg-post-sweeps",
    "--sa-amg-jacobi-damping",
    "--sa-amg-prolongation-damping",
    "--sa-amg-use-chebyshev",
)
# Options that take no value and are passed straight through.
_FLAG_OPTIONS = ("--synthetic", "--verbose")


def run_eigen(no_build: bool = False, **kwargs) -> None:
    """Run the SA-AMG + deflated PCG eigen example.

    Keyword arguments map directly to the executable's CLI options. Options
    with a value of None are omitted; boolean flags are included only when
    truthy. File-backed modes are validated the same way as bin/cmd/eigen.sh.
    """
    # Collect the raw option/value pairs in a stable order.
    raw_args = []
    for opt in _VALUE_OPTIONS:
        value = kwargs.get(opt[2:].replace("-", "_"))
        if value is not None:
            raw_args.append((opt, str(value)))
    for opt in _FLAG_OPTIONS:
        if kwargs.get(opt[2:].replace("-", "_")):
            raw_args.append((opt, None))

    # Determine the active mode from the provided file options.
    stiffness = kwargs.get("stiffness")
    cu = kwargs.get("cu")
    cuphi = kwargs.get("cuphi")
    cphi = kwargs.get("cphi")
    mass = kwargs.get("mass")
    coords = kwargs.get("coords")

    full_mode = stiffness is not None
    legacy_mode = cu is not None or cuphi is not None or cphi is not None

    if full_mode and legacy_mode:
        raise ValueError(
            "use either full-matrix mode (--stiffness --mass --coords) or "
            "legacy block mode (--cu --cuphi --cphi --mass --coords), not both"
        )

    if full_mode:
        required = {
            "--stiffness": stiffness,
            "--mass": mass,
            "--coords": coords,
        }
        for opt, path in required.items():
            if not path:
                raise ValueError(f"full-matrix eigen mode requires {opt}")
        for opt, path in required.items():
            if not isfile(path):
                raise ValueError(f"required file '{path}' not found")
    elif legacy_mode:
        required = {
            "--cu": cu,
            "--cuphi": cuphi,
            "--cphi": cphi,
            "--mass": mass,
            "--coords": coords,
        }
        for opt, path in required.items():
            if not path:
                raise ValueError(f"legacy block eigen mode requires {opt}")
        for opt, path in required.items():
            if not isfile(str(path)):
                raise ValueError(f"required file '{path}' not found")

    if not no_build:
        clean_build()

    # TODO: un-hardcode the preset name for Windows
    exe = ""
    if is_windows():
        exe = (
            CONFIG.build_dir
            / "msvc-ideapad\\examples\\Release\\example_sa_amg_pcg_eigen.exe"
        )
    else:
        exe = CONFIG.build_dir / "example_sa_amg_pcg_eigen"
    if not isfile(exe):
        raise ValueError(f"executable '{exe}' not found")

    args = []
    for opt, value in raw_args:
        args.append(opt)
        if value is not None:
            args.append(value)

    subprocess.run([str(exe)] + args, check=True)
