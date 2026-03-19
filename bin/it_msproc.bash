#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MSPROC_CMD="$SCRIPT_DIR/msproc.bash"

#MATSETS=("shuffle" "bfw782b" "cylshell")
MATSETS=("cylshell")
BLOCK_SIZES=(1 2)

while [[ $# -gt 0 ]]; do
  case "$1" in
    -b|--block-sizes)
      shift; BLOCK_SIZES=($1); shift ;;
    -s|--matsets)
      shift; MATSETS=($1); shift ;;
    -h|--help)
      echo "Usage: $0 [-b block_sizes] [-s matsets]"
      echo "Example: $0 -b '1 2 4' -s 'shuffle bfw782b'"
      exit 0 ;;
    *)
      echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

for bs in "${BLOCK_SIZES[@]}"; do
  for matset in "${MATSETS[@]}"; do
    echo "Running: $MSPROC_CMD $bs '$matset'"
    "$MSPROC_CMD" -b "$bs" -s "$matset" -o "$SCRIPT_DIR/../output/$matset/block$bs.txt"
  done
done