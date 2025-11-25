# shellcheck shell=bash
cmd_mult() {
  if [[ $# -lt 1 ]]; then
    echo "Error: mult requires <matrix_file>" >&2
    exit 1
  fi
  local matrix_file="$1"; shift
  local rhs_file=""
  local format="triplets"
  local mults=10
  local output=""
  local verbose=0
  local skip_build=0

  while [[ $# -gt 0 ]]; do
    case "$1" in
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
      --help|-h)
        print_usage; exit 0 ;;
      --)
        shift; break ;;
      *)
        echo "Unknown option: $1" >&2
        exit 1 ;;
    esac
  done

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
