# shellcheck shell=bash
solve_usage() {
  cat <<'EOF'
Usage: bin/r solve -m <method> [options]

Solve a system Ax=b using the selected method.
Currently supported methods: bicgstab

Options:
  -m, --method <name>   Solver method (required)
  --matrix <file>       Input matrix in triplet format (required)
  --rhs <file>          RHS vector path (required)
  --output <file>       Save solution vector to path
  --no-build            Skip cmake configure/build step (use existing build)
  -h, --help            Show this message
EOF
}

cmd_solve() {
  local method=""
  local matrix_file=""
  local rhs_file=""
  local output_file=""
  local skip_build=0

  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|--help)
        solve_usage; exit 0 ;;
      -m|--method)
        method="$2"; shift 2 ;;
      --matrix)
        matrix_file="$2"; shift 2 ;;
      --rhs)
        rhs_file="$2"; shift 2 ;;
      --output)
        output_file="$2"; shift 2 ;;
      --no-build)
        skip_build=1; shift ;;
      --)
        shift; break ;;
      *)
        echo "Unknown option: $1" >&2; exit 1 ;;
    esac
  done

  if [[ -z "$method" ]]; then
    echo "Error: --method is required" >&2; exit 1
  fi

  case "$method" in
    bicgstab) ;;
    *)
      echo "Error: unsupported method '$method'" >&2
      echo "Supported methods: bicgstab" >&2
      exit 1 ;;
  esac

  if [[ -z "$matrix_file" || -z "$rhs_file" ]]; then
    echo "Error: --matrix and --rhs are required" >&2
    exit 1
  fi
  if [[ ! -f "$matrix_file" ]]; then
    echo "Error: matrix file '$matrix_file' not found" >&2; exit 1
  fi
  if [[ ! -f "$rhs_file" ]]; then
    echo "Error: rhs file '$rhs_file' not found" >&2; exit 1
  fi

  if [[ $skip_build -eq 0 ]]; then
    clean_build
  fi

  local exe="$BUILD_DIR/example_bicgstab"
  if [[ ! -x "$exe" ]]; then
    echo "Error: executable '$exe' not found" >&2; exit 1
  fi

  if [[ -n "$output_file" ]]; then
    "$exe" "$matrix_file" "$rhs_file" "$output_file"
  else
    "$exe" "$matrix_file" "$rhs_file"
  fi
}
