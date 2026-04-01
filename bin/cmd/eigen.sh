# shellcheck shell=bash
eigen_usage() {
  cat <<'EOF'
Usage: bin/r eigen [options]

Run the article-style SA-AMG + deflated PCG eigen example.

Modes:
  Synthetic demo mode is used by default.
  Full-matrix file-backed mode is enabled automatically when --stiffness is provided.
  Legacy block mode is still available via --cu/--cuphi/--cphi.

Options:
  --synthetic                      Force synthetic demo mode
  --stiffness <file>               Full mixed stiffness matrix C
  --cu <file>                      Mechanical stiffness block C_u
  --cuphi <file>                   Coupling block C_uphi
  --cphi <file>                    Dielectric block C_phi
  --mass <file>                    Full mixed mass matrix M (or mechanical M in legacy block mode)
  --coords <file>                  Node coordinates file (x y z per line)
  --ordering <name>                Input ordering label: node-based or block-wise
  --dielectric-sign <mode>         Phi-Phi interpretation in full C: negated or as-is
  --scaling <mode>                 Preprocessing mode: none, field, or diag
  --eigensolve-path <mode>         auto, sa-amg, unpreconditioned, or explicit-schur-cusolver
  --grounded-dof <index>           Electrical DOF to ground (default: 0)
  --modes <count>                  Number of eigenpairs to compute (default: 3)
  --tol <value>                    Deflated PCG tolerance (default: 1e-3)
  --max-iters <count>              Deflated PCG iteration limit (default: 400)
  --sa-amg-regularization-epsilon <value>
  --sa-amg-pre-sweeps <count>
  --sa-amg-post-sweeps <count>
  --sa-amg-jacobi-damping <value>
  --sa-amg-prolongation-damping <value>
  --sa-amg-use-chebyshev <0|1>
  --verbose                        Print per-iteration eigensolver diagnostics
  --no-build                       Skip cmake configure/build step (use existing build)
  -h, --help                       Show this message
EOF
}

cmd_eigen() {
  local skip_build=0
  local args=()
  local full_mode=0
  local legacy_mode=0
  local stiffness_file=""
  local cu_file=""
  local cuphi_file=""
  local cphi_file=""
  local mass_file=""
  local coords_file=""

  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|--help)
        eigen_usage; exit 0 ;;
      --no-build)
        skip_build=1; shift ;;
      --synthetic)
        args+=("$1"); shift ;;
      --stiffness)
        stiffness_file="$2"; full_mode=1; args+=("$1" "$2"); shift 2 ;;
      --cu)
        cu_file="$2"; legacy_mode=1; args+=("$1" "$2"); shift 2 ;;
      --cuphi)
        cuphi_file="$2"; legacy_mode=1; args+=("$1" "$2"); shift 2 ;;
      --cphi)
        cphi_file="$2"; legacy_mode=1; args+=("$1" "$2"); shift 2 ;;
      --mass)
        mass_file="$2"; args+=("$1" "$2"); shift 2 ;;
      --coords)
        coords_file="$2"; args+=("$1" "$2"); shift 2 ;;
      --ordering|--dielectric-sign|--scaling|--eigensolve-path|--grounded-dof|--modes|--tol|--max-iters|--sa-amg-regularization-epsilon|--sa-amg-pre-sweeps|--sa-amg-post-sweeps|--sa-amg-jacobi-damping|--sa-amg-prolongation-damping|--sa-amg-use-chebyshev)
        args+=("$1" "$2"); shift 2 ;;
      --verbose)
        args+=("$1"); shift ;;
      --)
        shift; break ;;
      *)
        echo "Unknown option: $1" >&2
        echo "Try 'bin/r eigen -h' for more information." >&2
        exit 1 ;;
    esac
  done

  if [[ $full_mode -eq 1 && $legacy_mode -eq 1 ]]; then
    echo "Error: use either full-matrix mode (--stiffness --mass --coords) or legacy block mode (--cu --cuphi --cphi --mass --coords), not both" >&2
    exit 1
  fi

  if [[ $full_mode -eq 1 ]]; then
    if [[ -z "$stiffness_file" || -z "$mass_file" || -z "$coords_file" ]]; then
      echo "Error: full-matrix eigen mode requires --stiffness, --mass, and --coords" >&2
      exit 1
    fi
    for path in "$stiffness_file" "$mass_file" "$coords_file"; do
      if [[ ! -f "$path" ]]; then
        echo "Error: required file '$path' not found" >&2
        exit 1
      fi
    done
  elif [[ $legacy_mode -eq 1 ]]; then
    if [[ -z "$cu_file" || -z "$cuphi_file" || -z "$cphi_file" || -z "$mass_file" || -z "$coords_file" ]]; then
      echo "Error: legacy block eigen mode requires --cu, --cuphi, --cphi, --mass, and --coords" >&2
      exit 1
    fi
    for path in "$cu_file" "$cuphi_file" "$cphi_file" "$mass_file" "$coords_file"; do
      if [[ ! -f "$path" ]]; then
        echo "Error: required file '$path' not found" >&2
        exit 1
      fi
    done
  fi

  if [[ $skip_build -eq 0 ]]; then
    clean_build
  fi

  local exe="$BUILD_DIR/example_sa_amg_pcg_eigen"
  if [[ ! -x "$exe" ]]; then
    echo "Error: executable '$exe' not found" >&2
    exit 1
  fi

  "$exe" "${args[@]}"
}