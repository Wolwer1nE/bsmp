#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}/.."

usage() {
  cat <<'EOF'
Usage: bin/compare_solvers.sh --matrix <file> --rhs <file> [options]

Compare BiCGStab and GMRES on the same linear system.

Options:
  --matrix <file>       Input matrix file (required)
  --rhs <file>          RHS vector file (required)
  --precond <name>      Preconditioner for both solvers: amg, scalar-jacobi,
                        block-jacobi, none (default: amg)
  --restart <n>         GMRES restart size (default: 30)
  --max-iters <n>       Maximum iterations for both solvers (default: 1000)
  --tol <value>         Relative residual tolerance (default: 1e-6)
  --no-build            Skip build step and use existing executables
  -h, --help            Show this help
EOF
}

extract_metric() {
  local label="$1"
  local file="$2"
  sed -n "s/^${label}: //p" "$file" | tail -n 1
}

max_abs_diff() {
  local lhs="$1"
  local rhs="$2"
  paste "$lhs" "$rhs" | awk '
    BEGIN { max = 0 }
    NF >= 2 {
      diff = $1 - $2
      if (diff < 0) diff = -diff
      if (diff > max) max = diff
    }
    END { printf "%.9g\n", max }
  '
}

matrix_file=""
rhs_file=""
precond="amg"
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
    --precond)
      precond="$2"; shift 2 ;;
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

bicg_log="$tmp_dir/bicgstab.log"
gmres_log="$tmp_dir/gmres.log"
bicg_sol="$tmp_dir/bicgstab_x.txt"
gmres_sol="$tmp_dir/gmres_x.txt"

common_args=(--matrix "$matrix_file" --rhs "$rhs_file" --precond "$precond" --max-iters "$max_iters" --tol "$tol")
if [[ $skip_build -eq 1 ]]; then
  common_args+=(--no-build)
fi

set +e
"$ROOT_DIR/bin/r" solve --method bicgstab "${common_args[@]}" --output "$bicg_sol" >"$bicg_log" 2>&1
bicg_status=$?
"$ROOT_DIR/bin/r" solve --method gmres "${common_args[@]}" --restart "$restart" --output "$gmres_sol" >"$gmres_log" 2>&1
gmres_status=$?
set -e

bicg_converged="$(extract_metric "Converged" "$bicg_log")"
bicg_iters="$(extract_metric "Iterations" "$bicg_log")"
bicg_resid="$(extract_metric "Final relative residual" "$bicg_log")"
bicg_time="$(extract_metric "Solve time (ms)" "$bicg_log")"

gmres_converged="$(extract_metric "Converged" "$gmres_log")"
gmres_iters="$(extract_metric "Iterations" "$gmres_log")"
gmres_resid="$(extract_metric "Final relative residual" "$gmres_log")"
gmres_time="$(extract_metric "Solve time (ms)" "$gmres_log")"

echo "Matrix:        $matrix_file"
echo "RHS:           $rhs_file"
echo "Preconditioner: $precond"
echo "Tolerance:     $tol"
echo "Max iters:     $max_iters"
echo "GMRES restart: $restart"
echo
printf '%-12s %-10s %-12s %-20s %-16s %-8s\n' "Method" "Status" "Converged" "RelResidual" "Time(ms)" "Exit"
printf '%-12s %-10s %-12s %-20s %-16s %-8s\n' "BiCGStab" "$([[ $bicg_status -eq 0 ]] && echo OK || echo FAIL)" "${bicg_converged:-N/A}" "${bicg_resid:-N/A}" "${bicg_time:-N/A}" "$bicg_status"
printf '%-12s %-10s %-12s %-20s %-16s %-8s\n' "GMRES" "$([[ $gmres_status -eq 0 ]] && echo OK || echo FAIL)" "${gmres_converged:-N/A}" "${gmres_resid:-N/A}" "${gmres_time:-N/A}" "$gmres_status"
echo
printf '%-12s %-12s\n' "Method" "Iterations"
printf '%-12s %-12s\n' "BiCGStab" "${bicg_iters:-N/A}"
printf '%-12s %-12s\n' "GMRES" "${gmres_iters:-N/A}"

if [[ -f "$bicg_sol" && -f "$gmres_sol" ]]; then
  echo
  echo "Max |x_bicgstab - x_gmres|: $(max_abs_diff "$bicg_sol" "$gmres_sol")"
fi

if [[ $bicg_status -ne 0 || $gmres_status -ne 0 ]]; then
  echo
  echo "BiCGStab log: $bicg_log" >&2
  echo "GMRES log:    $gmres_log" >&2
  exit 1
fi