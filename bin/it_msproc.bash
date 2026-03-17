#!/usr/bin/env bash

# ITMSPROC is an iterative MSPROC. It should iterate over the sets of matrices
# and call MSPROC on each of them
# TODO: adapt to the clean BSMP. Think about varyings sets and block sizes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

${SCRIPT_DIR}/fg 1 'shuffle'
${SCRIPT_DIR}/fg 1 'bfw782b'
${SCRIPT_DIR}/fg 1 'cylshell'

# for i in {1..3}; do
#     bs=$((3 * i))
#     ${SCRIPT_DIR}/fg $bs 'shuffle'
#     ${SCRIPT_DIR}/fg $bs 'bfw782b'
#     ${SCRIPT_DIR}/fg $bs 'cylshell'
# done

# for i in {1..4}; do
#     bs=$((4 * i))
#     ${SCRIPT_DIR}/fg $bs 'shuffle'
#     ${SCRIPT_DIR}/fg $bs 'bfw782b'
#     ${SCRIPT_DIR}/fg $bs 'cylshell'
# done
