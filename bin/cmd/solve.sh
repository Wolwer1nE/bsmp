# shellcheck shell=bash
solve_usage() {
  cat <<'EOF'
Usage: bin/r solve -m <method> [options]

Solve a system Ax=b using the selected method.
Currently supported methods: bicgstab, gmres

Options:
  -m, --method <name>   Solver method (required)
  --precond <name>      Preconditioner: amg, scalar-jacobi, block-jacobi, none (default: amg)
  --matrix <file>       Input matrix in triplet format (required)
  --rhs <file>          RHS vector path (required)
  --output <file>       Save solution vector to path
  --max-iters <n>       Maximum number of iterations (default: solver-specific)
  --tol <value>         Relative residual tolerance (default: 1e-6)
  --restart <n>         Restart parameter for GMRES (default: 30)
  --no-build            Skip cmake configure/build step (use existing build)
  -h, --help            Show this message
EOF
}

cmd_solve() {
  local method=""
  local precond=""
  local matrix_file=""
  local rhs_file=""
  local output_file=""
  local max_iters=""
  local tol=""
  local restart=""
  local skip_build=0

  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|--help)
        solve_usage; exit 0 ;;
      -m|--method)
        method="$2"; shift 2 ;;
      --precond)
        precond="$2"; shift 2 ;;
      --matrix)
        matrix_file="$2"; shift 2 ;;
      --rhs)
        rhs_file="$2"; shift 2 ;;
      --output)
        output_file="$2"; shift 2 ;;
      --max-iters)
        max_iters="$2"; shift 2 ;;
      --tol)
        tol="$2"; shift 2 ;;
      --restart)
        restart="$2"; shift 2 ;;
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
    gmres) ;;
    *)
      echo "Error: unsupported method '$method'" >&2
      echo "Supported methods: bicgstab, gmres" >&2
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
  if [[ "$method" == "gmres" ]]; then
    exe="$BUILD_DIR/example_gmres"
  fi
  if [[ ! -x "$exe" ]]; then
    echo "Error: executable '$exe' not found" >&2; exit 1
  fi

  # Preprocess RHS if it's a Matrix Market vector (skip '%' comment lines and the size line)
  local rhs_to_use="$rhs_file"
  local tmp_rhs=""
  if grep -q '^%%MatrixMarket' "$rhs_file" 2>/dev/null; then
    tmp_rhs=$(mktemp)
    # Skip leading '%' comment lines. If the first non-comment line contains two integers (size line), skip it.
    awk '
    BEGIN { first_noncomment = 1 }
    {
      if (first_noncomment) {
        if ($0 ~ /^[[:space:]]*%/) next;
        if ($0 ~ /^[[:space:]]*[0-9]+[[:space:]]+[0-9]+[[:space:]]*$/) { first_noncomment=0; next }
        first_noncomment = 0
      }
      print
    }' "$rhs_file" > "$tmp_rhs"
    rhs_to_use="$tmp_rhs"
  fi

  local args=("$matrix_file" "$rhs_to_use")
  if [[ -n "$output_file" ]]; then
    args+=("$output_file")
  fi
  if [[ "$method" == "gmres" ]]; then
    if [[ -n "$restart" ]]; then
      if [[ -z "$output_file" ]]; then
        args+=("")
      fi
      args+=("$restart")
    fi
    if [[ -n "$max_iters" ]]; then
      if [[ -z "$output_file" && -z "$restart" ]]; then
        args+=("")
      fi
      if [[ -z "$restart" ]]; then
        args+=("30")
      fi
      args+=("$max_iters")
    fi
    if [[ -n "$tol" ]]; then
      if [[ -z "$output_file" && -z "$restart" && -z "$max_iters" ]]; then
        args+=("")
      fi
      if [[ -z "$restart" ]]; then
        args+=("30")
      fi
      if [[ -z "$max_iters" ]]; then
        args+=("1000")
      fi
      args+=("$tol")
    fi
  else
    if [[ -n "$max_iters" ]]; then
      if [[ -z "$output_file" ]]; then
        args+=("")
      fi
      args+=("$max_iters")
    fi
    if [[ -n "$tol" ]]; then
      if [[ -z "$output_file" && -z "$max_iters" ]]; then
        args+=("")
      fi
      if [[ -z "$max_iters" ]]; then
        args+=("1000")
      fi
      args+=("$tol")
    fi
  fi

  if [[ -n "$precond" ]]; then
    if [[ "$method" == "gmres" ]]; then
      if [[ -z "$output_file" && -z "$restart" && -z "$max_iters" && -z "$tol" ]]; then
        args+=("")
      fi
      if [[ -z "$restart" ]]; then
        args+=("30")
      fi
      if [[ -z "$max_iters" ]]; then
        args+=("1000")
      fi
      if [[ -z "$tol" ]]; then
        args+=("1e-6")
      fi
    else
      if [[ -z "$output_file" && -z "$max_iters" && -z "$tol" ]]; then
        args+=("")
      fi
      if [[ -z "$max_iters" ]]; then
        args+=("1000")
      fi
      if [[ -z "$tol" ]]; then
        args+=("1e-6")
      fi
    fi
    args+=("$precond")
  fi

  "$exe" "${args[@]}"

  # cleanup temp rhs
  if [[ -n "$tmp_rhs" && -f "$tmp_rhs" ]]; then
    rm -f "$tmp_rhs"
  fi
}
