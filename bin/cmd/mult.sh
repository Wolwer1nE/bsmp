# shellcheck shell=bash
# This won't work on windows because of file extensions. TODO: Think about that.
mult_usage() {
  cat <<'EOF'
Usage: bin/r mult <matrix_file> [options]

Clean builds the project and runs the multiplication test.

Options:
  --rhs <file>          RHS vector path (required)
  --format <fmt>        Input format: triplets|matrix-market (default: triplets)
  --mults <n>           Number of chained multiplies (default: 10)
  --output <file>       Save resulting vector to path
  --verbose             Verbose kernel statistics
  --no-build            Skip cmake configure/build step (use existing build)
  -h, --help            Show this message
EOF
}

cmd_mult() {
  local matrix_file=""
  local rhs_file=""
  local format="triplets"
  local mults=10
  local output=""
  local verbose=0
  local skip_build=0

  # Parse args (support -h without requiring positional)
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|--help)
        mult_usage; exit 0 ;;
      --rhs)
        rhs_file="$2"; shift 2 ;;
      --format)
        format="$2"; shift 2 ;;
      --mults|--iterations)
        mults="$2"; shift 2 ;;
      --output)
        output="$2"; shift 2 ;;
      --verbose)
        verbose=1; shift ;;
      --no-build)
        skip_build=1; shift ;;
      --)
        shift; break ;;
      -*)
        echo "Unknown option: $1" >&2; exit 1 ;;
      *)
        if [[ -z "$matrix_file" ]]; then
          matrix_file="$1"; shift
        else
          echo "Unexpected argument: $1" >&2; exit 1
        fi
        ;;
    esac
  done

  if [[ -z "$matrix_file" ]]; then
    echo "Error: mult requires <matrix_file>" >&2
    echo "Try 'bin/r mult -h' for more information." >&2
    exit 1
  fi

  if [[ ! -f "$matrix_file" ]]; then
    echo "Error: matrix file '$matrix_file' not found" >&2
    exit 1
  fi

  if [[ -z "$rhs_file" ]]; then
    echo "Error: --rhs <file> must be provided" >&2
    exit 1
  fi

  if [[ ! -f "$rhs_file" ]]; then
    echo "Error: rhs file '$rhs_file' not found" >&2
    exit 1
  fi

  if [[ $skip_build -eq 0 ]]; then
    clean_build
  fi

  local exe="${BUILD_DIR}/example"
  if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
	  exe="${BUILD_DIR}/Release/example.exe"
  else
	  exe="${BUILD_DIR}/example"
  fi
  
  if [[ ! -x "$exe" ]]; then
    echo "Error: executable '$exe' not found" >&2
    exit 1
  fi

  args=("--matrix" "$matrix_file" "--format" "$format" "--mults" "$mults")
  if [[ -n "$rhs_file" ]]; then
    args+=("--rhs" "$rhs_file")
  fi
  if [[ -n "$output" ]]; then
    args+=("--output" "$output")
  fi
  if [[ $verbose -eq 1 ]]; then
    args+=("--verbose")
  fi

  "$exe" "${args[@]}"
}
