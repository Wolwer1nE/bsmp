#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------
# MSPROC - Process a single matrix set
# Compiles BSMP with specified block size and runs example on all matrices in set
# ---------------------------------------

log()   { printf '%s\n' "$*" ; }
error() { log "[ERROR] $*" >&2 ; exit 1 ; }

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
ROOT_DIR="$SCRIPT_DIR/.."

BLOCK_SIZE=""
MATSET=""
OUTPUT_FILE=""
VERBOSE=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    -b|--block-size)
      [[ $# -lt 2 ]] && error "-b requires an argument"
      BLOCK_SIZE="$2"
      shift 2
      ;;
    -s|--matrix-set)
      [[ $# -lt 2 ]] && error "-s requires an argument"
      MATSET="$2"
      shift 2
      ;;
    -o|--output-file)
      [[ $# -lt 2 ]] && error "-o requires an argument"
      OUTPUT_FILE="$2"
      shift 2
      ;;
    -v|--verbose)
      VERBOSE=true
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [-h] [-v] -b block_size -s matset_folder_name [-o output_file]"
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      error "Unknown option: $1"
      ;;
    *)
      error "Unexpected argument: $1"
      ;;
  esac
done

[[ -n "$BLOCK_SIZE" ]] || error "Block size must be set (use -b)"
[[ -n "$MATSET" ]]     || error "Matrix set name must be set (use -s)"
[[ -n "$OUTPUT_FILE" ]] || OUTPUT_FILE="$ROOT_DIR/output/$MATSET/output.txt"

mkdir -p "$(dirname "$OUTPUT_FILE")"

# CMake build
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
  CMAKE_FLAGS="-DBSMP_BLOCK_SIZE=${BLOCK_SIZE}"
  CMAKE_BUILD_FLAGS="--config Release --target example -j"
else
  CMAKE_FLAGS="-DBSMP_BLOCK_SIZE=${BLOCK_SIZE} -DCMAKE_BUILD_TYPE=Release"
  CMAKE_BUILD_FLAGS="-j"
fi

cmake -B build $CMAKE_FLAGS
cmake --build build $CMAKE_BUILD_FLAGS

# Warm-up runs
log "Running warm-up..."
for _ in {1..10}; do
  "$SCRIPT_DIR"/r mult "$ROOT_DIR/data/warmup-matrix/matrix.txt" \
    --rhs "$ROOT_DIR/data/warmup-matrix/rhs_matrix.txt" \
    --no-build > /dev/null
done

# Process matrices
log "Processing matrices in $MATSET..."
> "$OUTPUT_FILE"

TARGET_DIR="$ROOT_DIR/data/$MATSET"

echo "matrix_name;num_iter;full_time;avg_time" > "$ROOT_DIR/performance.log"

for mtx_file in "$TARGET_DIR"/*.mtx; do
  [[ ! -f "$mtx_file" ]] && continue
  mtx_name=$(basename "$mtx_file" .mtx)
  rhs_file="$TARGET_DIR/$mtx_name.rhs"
  [[ ! -f "$rhs_file" ]] && error "Missing RHS file: $rhs_file"

  log "Processing $mtx_name..."
  echo "[PERF] $mtx_name" >> "$OUTPUT_FILE"
  "$SCRIPT_DIR"/r mult "$mtx_file" \
    --rhs "$rhs_file" \
    --format matrix-market \
    --no-build >> "$OUTPUT_FILE" 2>&1
  echo "" >> "$OUTPUT_FILE"
done

mv "$ROOT_DIR/performance.log" "$ROOT_DIR/output/$MATSET/$(basename "$OUTPUT_FILE" .txt).csv"

log "Done. Results saved to: $OUTPUT_FILE"