#!/usr/bin/env bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
TARGET_DIR="$SCRIPT_DIR/data/"

cmake -B $SCRIPT_DIR/build -DCMAKE_BUILD_TYPE=Release
cmake --build $SCRIPT_DIR/build --clean-first -j

for mtx_file in "$TARGET_DIR"/*.mtx; do
  [[ ! -f "$mtx_file" ]] && continue
  mtx_name=$(basename "$mtx_file" .mtx)
  rhs_file="$TARGET_DIR/$mtx_name.rhs"
    "$SCRIPT_DIR"/bin/compare_all.sh \
    --matrix "$mtx_file" \
    --rhs "$rhs_file" \
    --no-build
done
