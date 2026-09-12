#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}/.."
BUILD_DIR="${ROOT_DIR}/build"

usage() {
  cat <<'EOF'
Usage: bin/compare_all.sh --matrix <file> --rhs <file> [options]

Run the full solver/preconditioner comparison matrix.

Options:
  --matrix <file>       Input matrix file (required)
  --rhs <file>          RHS vector file (required)
  --preconds <list>     Comma-separated list, default: amg,scalar-jacobi,block-jacobi,none
  --restart <n>         GMRES restart size (default: 30)
  --max-iters <n>       Maximum iterations (default: 1000)
  --tol <value>         Relative residual tolerance (default: 1e-6)
  --no-build            Skip build step and use existing executables
  -h, --help            Show this help
EOF
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: required command '$1' not found" >&2
    exit 127
  fi
}

build_once() {
  if [[ $skip_build -eq 1 ]]; then
    return
  fi
  require_cmd cmake
  cmake -S "$ROOT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
  cmake --build "$BUILD_DIR" --clean-first -j
}

extract_metric() {
  local label="$1"
  local file="$2"
  sed -n "s/^${label}: //p" "$file" | tail -n 1
}

matrix_file=""
rhs_file=""
preconds_csv="amg,scalar-jacobi,block-jacobi,none"
restart="30"
max_iters="1000"
tol="1e-6"
skip_build=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --matrix)
      matrix_file="$2"; shift 2 ;;
    --rhs)
      rhs_file="$2"; shift 2 ;;
    --preconds)
      preconds_csv="$2"; shift 2 ;;
    --restart)
      restart="$2"; shift 2 ;;
    --max-iters)
      max_iters="$2"; shift 2 ;;
    --tol)
      tol="$2"; shift 2 ;;
    --no-build)
      skip_build=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1 ;;
  esac
done

if [[ -z "$matrix_file" || -z "$rhs_file" ]]; then
  echo "Error: --matrix and --rhs are required" >&2
  usage
  exit 1
fi

if [[ ! -f "$matrix_file" ]]; then
  echo "Error: matrix file '$matrix_file' not found" >&2
  exit 1
fi

if [[ ! -f "$rhs_file" ]]; then
  echo "Error: rhs file '$rhs_file' not found" >&2
  exit 1
fi

build_once

IFS=',' read -r -a preconds <<< "$preconds_csv"
methods=(bicgstab gmres)

tmp_dir="$(mktemp -d)"
cleanup() {
  local exit_code=$?
  if [[ $exit_code -eq 0 ]]; then
    rm -rf "$tmp_dir"
  else
    echo "Logs preserved in: $tmp_dir" >&2
  fi
}
trap cleanup EXIT

echo "Matrix:        $matrix_file"
echo "RHS:           $rhs_file"
echo "Tolerance:     $tol"
echo "Max iters:     $max_iters"
echo "GMRES restart: $restart"
echo
printf '%-10s %-18s %-10s %-12s %-12s %-20s %-16s %-8s\n' "Solver" "Preconditioner" "Status" "Converged" "Iterations" "RelResidual" "Time(ms)" "Exit"

overall_status=0
for method in "${methods[@]}"; do
  for precond in "${preconds[@]}"; do
    precond="${precond//[[:space:]]/}"
    [[ -z "$precond" ]] && continue

    log_file="$tmp_dir/${method}_${precond}.log"
    sol_file="$tmp_dir/${method}_${precond}.txt"
    cmd=("$ROOT_DIR/bin/r" solve --method "$method" --matrix "$matrix_file" --rhs "$rhs_file" --precond "$precond" --max-iters "$max_iters" --tol "$tol" --output "$sol_file" --no-build)
    if [[ "$method" == "gmres" ]]; then
      cmd+=(--restart "$restart")
    fi

    set +e
    "${cmd[@]}" >"$log_file" 2>&1
    status=$?
    set -e

    converged="$(extract_metric "Converged" "$log_file")"
    iterations="$(extract_metric "Iterations" "$log_file")"
    residual="$(extract_metric "Final relative residual" "$log_file")"
    solve_time="$(extract_metric "Solve time (ms)" "$log_file")"

    printf '%-10s %-18s %-10s %-12s %-12s %-20s %-16s %-8s\n' \
      "$method" \
      "$precond" \
      "$([[ $status -eq 0 ]] && echo OK || echo FAIL)" \
      "${converged:-N/A}" \
      "${iterations:-N/A}" \
      "${residual:-N/A}" \
      "${solve_time:-N/A}" \
      "$status"

    if [[ $status -ne 0 ]]; then
      overall_status=1
    fi
  done
done

exit $overall_status
