#!/usr/bin/env bash
# Install Python dependencies for the tools folder into a virtual environment.
# Usage: ./install_python_deps.sh [venv_dir]
# Default venv_dir: .venv_tools

set -euo pipefail

VENV_DIR=${1:-.venv_tools}

echo "Creating virtual environment in: ${VENV_DIR}"
python3 -m venv "${VENV_DIR}"
echo "Activating virtualenv and installing requirements..."
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip
if [ -f "$(dirname "$0")/requirements.txt" ]; then
  pip install -r "$(dirname "$0")/requirements.txt"
else
  echo "requirements.txt not found in tools/; nothing to install."
fi

echo "Installation finished. To use the tools run:"
echo "  source ${VENV_DIR}/bin/activate"

cat <<'EOF'
Notes:
- petsc4py often requires a system PETSc installation. If `pip install petsc4py` fails,
  install PETSc first (from package manager or from source) and then install petsc4py.
  See https://petsc.org/release/ for PETSc installation instructions and
  https://petsc4py.readthedocs.io/ for petsc4py-specific notes.
EOF
