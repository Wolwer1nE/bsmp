#!/usr/bin/env bash
set -euo pipefail

# MSPROC is for processing a single matrix set (a folder). It compiles bsmp with params for it
# and launches on every matrix from set
# TODO: add CLI args and functions to generalize matrix set processing 


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR%/bin}"
GEN_DIR="${ROOT_DIR}/generators"
BLOCK_SIZE=$1
MATSET=$2

cmake -B build -DBSMP_BLOCK_SIZE=${BLOCK_SIZE}
cmake --build build --config Release --target example

TARGET_DIR="${ROOT_DIR}/data/${MATSET}"
FILES=("${TARGET_DIR}"/*.mtx)

# Warm up
for i in {1..10}; do
    ${SCRIPT_DIR}/r mult "${ROOT_DIR}/data/warmup-matrix/matrix.txt" --rhs "${ROOT_DIR}/data/warmup-matrix/rhs_matrix.txt" --no-build > /dev/null
done

output_file="${ROOT_DIR}/output/${MATSET}/csr_output.txt"
echo "" > "${output_file}"
for file in "${FILES[@]}"; do
  mtx=$(basename "$file" .mtx)
  echo "Processing $mtx" >> "${output_file}"
  rhs="${TARGET_DIR}/${mtx}.rhs"
  ${SCRIPT_DIR}/r mult "$file" --rhs "$rhs" --format matrix-market --no-build >> "${output_file}"
  echo "" >> "${output_file}"
done
