"""Pack a standalone python package wheel from sources"""

import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT_DIR / "build"
PACKAGE_BIN_DIR = ROOT_DIR / "python" / "src" / "bsmp" / "bin"
PYTHON_DIR = ROOT_DIR / "python"
DIST_DIR = ROOT_DIR / "dist"

# ---- Build the binaries with CMake ----
subprocess.run(
    [
        "cmake",
        "-S",
        str(ROOT_DIR),
        "-B",
        str(BUILD_DIR),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_CONFIGURATION_TYPES=Release",
    ],
    check=True,
)
subprocess.run(
    ["cmake", "--build", str(BUILD_DIR), "-j", "--config", "Release"],
    check=True,
)

# ---- Install into a staging dir and copy binaries into the package ----
tmpdir = Path(tempfile.mkdtemp()).resolve()
try:
    subprocess.run(
        ["cmake", "--install", str(BUILD_DIR), "--prefix", str(tmpdir)],
        check=True,
    )

    shutil.rmtree(PACKAGE_BIN_DIR, ignore_errors=True)
    shutil.copytree(tmpdir / "bin", PACKAGE_BIN_DIR)
finally:
    shutil.rmtree(tmpdir, ignore_errors=True)

# ---- Build the pure-Python wheel ----
DIST_DIR.mkdir(exist_ok=True)
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "wheel",
        str(PYTHON_DIR),
        "-w",
        str(DIST_DIR),
        "--no-deps",
    ],
    check=True,
)

# ---- Retag the wheel with the current platform tag ----
platform_tag = (
    sysconfig.get_platform().replace("-", "_").replace(".", "_")
)

# Find the freshly built pure wheel in dist/
pure_wheels = list(DIST_DIR.glob("bsmp-*-py3-none-any.whl"))
if not pure_wheels:
    raise SystemExit(f"No pure wheel found in {DIST_DIR}")

for wheel in pure_wheels:
    print(f"Retagging {wheel.name} → platform tag '{platform_tag}'")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "wheel",
            "tags",
            "--platform-tag",
            platform_tag,
            str(wheel),
        ],
        check=True,
    )

print(f"\nDone. Artifacts in {DIST_DIR}:")
for artifact in sorted(DIST_DIR.iterdir()):
    print(f"  {artifact.name}")