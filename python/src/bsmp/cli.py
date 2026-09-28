#!/usr/bin/env python
import subprocess

import click

from bsmp.generators.eigen import generate_eigen_matrices
from bsmp.generators.matrix import generate_matrix
from bsmp.launchers.compare import (
    DEFAULT_PRECONDS,
    run_compare_all,
    run_compare_preconditioners,
    run_compare_solvers,
)
from bsmp.launchers.eigen import run_eigen
from bsmp.launchers.mult import run_mult
from bsmp.launchers.solve import run_solve


# This class is for command aliases
class AliasedGroup(click.Group):
    def __init__(self, *args, aliases=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._aliases = aliases or {}
    # That is how aliases work
    def get_command(self, ctx, cmd_name):
        cmd_name = self._aliases.get(cmd_name, cmd_name)
        return super().get_command(ctx, cmd_name)
    # Beautifying help text
    def format_commands(self, ctx, formatter):
        commands = []
        for subcommand in self.list_commands(ctx):
            cmd = self.get_command(ctx, subcommand)
            if cmd is None:
                continue
            help_text = cmd.get_short_help_str(limit=formatter.width)
            # Find all aliases that point to this subcommand
            aliases = [a for a, target in self._aliases.items() if target == subcommand]
            if aliases:
                subcommand = f"{subcommand} ({'/'.join(aliases)})"
            commands.append((subcommand, help_text))

        if commands:
            with formatter.section("Commands"):
                formatter.write_dl(commands)

@click.group(cls=AliasedGroup, aliases={
    "gen": "generate",
    "g": "generate",
    "run": "launch",
    "r": "launch"
})
@click.version_option()
def cli() -> None:
    """BSMP command-line tools"""


@cli.group()
def generate() -> None:
    """Generate matrices and eigenvalue sets"""


@click.command()
@click.argument("output_file")
@click.option("-n", "--n_blocks", default=1000, help="Number of blocks")
@click.option("-b", "--block_size", default=4, help="Block size")
@click.option("-r", "--random", default=False, help="")
def matrix_generator(output_file, n_blocks, block_size, random):
    """Generate matrix and rhs pack for testing SpMV"""
    try:
        generate_matrix(output_file, n_blocks, block_size, random)
    except ValueError as e:
        print(f"Invalid inputs error: {e}")
    except OSError as e:
        print(f"IO Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


@click.command()
@click.argument("output_file_a")
@click.argument("output_file_b")
@click.option("-n", "--size", default=100, help="Matrix size")
@click.option("--density-a", default=0.01, help="Density of matrix A")
@click.option("--density-b", default=0.005, help="Density of matrix B")
def eigenvalue_generator(output_file_a, output_file_b, size, density_a, density_b):
    """Generate matrix and rhs for eigenvalue search"""
    try:
        generate_eigen_matrices(
            output_file_a, output_file_b, size, density_a, density_b
        )
    except ValueError as e:
        print(f"Invalid inputs error: {e}")
    except OSError as e:
        print(f"IO Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


generate.add_command(matrix_generator, "matrix")
generate.add_command(eigenvalue_generator, "eigenvalue")


@cli.group()
def launch() -> None:
    """Launches some default usecases of BSMP"""


@click.command()
@click.argument("matrix_file")
@click.option("--rhs", help="RHS vector path", required=True)
@click.option(
    "--format", default="triplets", help="Input format: triplets|matrix-market"
)
@click.option("--mults", default=10, help="Number of chained multiplies")
@click.option("--output", help="Save resulting vector to path")
@click.option(
    "--verbose", is_flag=True, default=False, help="Verbose kernel statistics"
)
@click.option(
    "--no-build",
    is_flag=True,
    default=False,
    help="Use existing build if present",
)
def mult_launcher(matrix_file, rhs, format, mults, output, verbose, no_build):
    """Run the multiplication test on a matrix and RHS vector"""
    try:
        run_mult(matrix_file, rhs, format, mults, output, verbose, no_build)
    except ValueError as e:
        print(f"Invalid inputs error: {e}")
        raise SystemExit(1) from e
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}")
        raise SystemExit(1) from e
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise SystemExit(1) from e


@click.command()
@click.option("-m", "--method", help="Solver method (required)")
@click.option(
    "--precond",
    default="",
    help="Preconditioner: amg, scalar-jacobi, block-jacobi, none",
)
@click.option("--matrix", help="Input matrix in triplet format (required)")
@click.option("--rhs", help="RHS vector path (required)")
@click.option("--output", help="Save solution vector to path")
@click.option("--max-iters", help="Maximum number of iterations")
@click.option("--tol", help="Relative residual tolerance")
@click.option("--restart", help="Restart parameter for GMRES")
@click.option(
    "--no-build",
    is_flag=True,
    default=False,
    help="Use existing build if present",
)
def solve_launcher(
    method, precond, matrix, rhs, output, max_iters, tol, restart, no_build
):
    """Solve a system Ax=b using the selected method"""
    try:
        run_solve(
            method,
            matrix,
            rhs,
            precond,
            output,
            max_iters,
            tol,
            restart,
            no_build,
        )
    except ValueError as e:
        print(f"Invalid inputs error: {e}")
        raise SystemExit(1) from e
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}")
        raise SystemExit(1) from e
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise SystemExit(1) from e


@click.command()
@click.option(
    "--synthetic", is_flag=True, default=False, help="Force synthetic demo mode"
)
@click.option("--stiffness", help="Full mixed stiffness matrix C")
@click.option("--cu", help="Mechanical stiffness block C_u")
@click.option("--cuphi", help="Coupling block C_uphi")
@click.option("--cphi", help="Dielectric block C_phi")
@click.option(
    "--mass",
    help="Full mixed mass matrix M (or mechanical M in legacy block mode)",
)
@click.option("--coords", help="Node coordinates file (x y z per line)")
@click.option("--ordering", help="Input ordering label: node-based or block-wise")
@click.option(
    "--dielectric-sign",
    help="Phi-Phi interpretation in full C: negated or as-is",
)
@click.option("--scaling", help="Preprocessing mode: none, field, or diag")
@click.option(
    "--eigensolve-path",
    help="auto, sa-amg, unpreconditioned, or explicit-schur-cusolver",
)
@click.option("--grounded-dof", help="Electrical DOF to ground (default: 0)")
@click.option("--modes", help="Number of eigenpairs to compute (default: 3)")
@click.option("--tol", help="Deflated PCG tolerance (default: 1e-3)")
@click.option("--max-iters", help="Deflated PCG iteration limit (default: 400)")
@click.option("--sa-amg-regularization-epsilon", help="SA-AMG regularization epsilon")
@click.option("--sa-amg-pre-sweeps", help="SA-AMG pre-smoothing sweeps")
@click.option("--sa-amg-post-sweeps", help="SA-AMG post-smoothing sweeps")
@click.option("--sa-amg-jacobi-damping", help="SA-AMG Jacobi damping")
@click.option("--sa-amg-prolongation-damping", help="SA-AMG prolongation damping")
@click.option("--sa-amg-use-chebyshev", help="SA-AMG Chebyshev smoothing (0|1)")
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Print per-iteration eigensolver diagnostics",
)
@click.option(
    "--no-build",
    is_flag=True,
    default=False,
    help="Use existing build if present",
)
def eigen_launcher(
    synthetic,
    stiffness,
    cu,
    cuphi,
    cphi,
    mass,
    coords,
    ordering,
    dielectric_sign,
    scaling,
    eigensolve_path,
    grounded_dof,
    modes,
    tol,
    max_iters,
    sa_amg_regularization_epsilon,
    sa_amg_pre_sweeps,
    sa_amg_post_sweeps,
    sa_amg_jacobi_damping,
    sa_amg_prolongation_damping,
    sa_amg_use_chebyshev,
    verbose,
    no_build,
):
    """Run the SA-AMG + deflated PCG eigen example"""
    try:
        run_eigen(
            no_build=no_build,
            synthetic=synthetic,
            stiffness=stiffness,
            cu=cu,
            cuphi=cuphi,
            cphi=cphi,
            mass=mass,
            coords=coords,
            ordering=ordering,
            dielectric_sign=dielectric_sign,
            scaling=scaling,
            eigensolve_path=eigensolve_path,
            grounded_dof=grounded_dof,
            modes=modes,
            tol=tol,
            max_iters=max_iters,
            sa_amg_regularization_epsilon=sa_amg_regularization_epsilon,
            sa_amg_pre_sweeps=sa_amg_pre_sweeps,
            sa_amg_post_sweeps=sa_amg_post_sweeps,
            sa_amg_jacobi_damping=sa_amg_jacobi_damping,
            sa_amg_prolongation_damping=sa_amg_prolongation_damping,
            sa_amg_use_chebyshev=sa_amg_use_chebyshev,
            verbose=verbose,
        )
    except ValueError as e:
        print(f"Invalid inputs error: {e}")
        raise SystemExit(1) from e
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}")
        raise SystemExit(1) from e
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise SystemExit(1) from e


@click.command()
@click.option("--matrix", help="Input matrix file (required)", required=True)
@click.option("--rhs", help="RHS vector file (required)", required=True)
@click.option(
    "--preconds",
    default=DEFAULT_PRECONDS,
    help="Comma-separated list of preconditioners",
)
@click.option("--restart", default="30", help="GMRES restart size (default: 30)")
@click.option("--max-iters", default="1000", help="Maximum iterations (default: 1000)")
@click.option(
    "--tol", default="1e-6", help="Relative residual tolerance (default: 1e-6)"
)
@click.option(
    "--no-build",
    is_flag=True,
    default=False,
    help="Skip build step and use existing executables",
)
def compare_all_launcher(matrix, rhs, preconds, restart, max_iters, tol, no_build):
    """Run the full solver/preconditioner comparison matrix"""
    try:
        run_compare_all(matrix, rhs, preconds, restart, max_iters, tol, no_build)
    except ValueError as e:
        print(f"Invalid inputs error: {e}")
        raise SystemExit(1) from e
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}")
        raise SystemExit(1) from e
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise SystemExit(1) from e


@click.command()
@click.option(
    "--method",
    type=click.Choice(["bicgstab", "gmres"]),
    required=True,
    help="Solver: bicgstab or gmres (required)",
)
@click.option("--matrix", help="Input matrix file (required)", required=True)
@click.option("--rhs", help="RHS vector file (required)", required=True)
@click.option(
    "--preconds",
    default=DEFAULT_PRECONDS,
    help="Comma-separated list of preconditioners",
)
@click.option("--restart", default="30", help="GMRES restart size (default: 30)")
@click.option("--max-iters", default="1000", help="Maximum iterations (default: 1000)")
@click.option(
    "--tol", default="1e-6", help="Relative residual tolerance (default: 1e-6)"
)
@click.option(
    "--no-build",
    is_flag=True,
    default=False,
    help="Skip build step and use existing executables",
)
def compare_preconditioners_launcher(
    method, matrix, rhs, preconds, restart, max_iters, tol, no_build
):
    """Compare several preconditioners for one solver"""
    try:
        run_compare_preconditioners(
            method, matrix, rhs, preconds, restart, max_iters, tol, no_build
        )
    except ValueError as e:
        print(f"Invalid inputs error: {e}")
        raise SystemExit(1) from e
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}")
        raise SystemExit(1) from e
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise SystemExit(1) from e


@click.command()
@click.option("--matrix", help="Input matrix file (required)", required=True)
@click.option("--rhs", help="RHS vector file (required)", required=True)
@click.option(
    "--precond",
    default="amg",
    help="Preconditioner: amg, scalar-jacobi, block-jacobi, none (default: amg)",
)
@click.option("--restart", default="30", help="GMRES restart size (default: 30)")
@click.option("--max-iters", default="1000", help="Maximum iterations (default: 1000)")
@click.option(
    "--tol", default="1e-6", help="Relative residual tolerance (default: 1e-6)"
)
@click.option(
    "--no-build",
    is_flag=True,
    default=False,
    help="Skip build step and use existing executables",
)
def compare_solvers_launcher(matrix, rhs, precond, restart, max_iters, tol, no_build):
    """Compare BiCGStab and GMRES on the same linear system"""
    try:
        run_compare_solvers(matrix, rhs, precond, restart, max_iters, tol, no_build)
    except ValueError as e:
        print(f"Invalid inputs error: {e}")
        raise SystemExit(1) from e
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}")
        raise SystemExit(1) from e
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise SystemExit(1) from e


launch.add_command(mult_launcher, "mult")
launch.add_command(solve_launcher, "solve")
launch.add_command(eigen_launcher, "eigen")
launch.add_command(compare_all_launcher, "compare-all")
launch.add_command(compare_preconditioners_launcher, "compare-preconditioners")
launch.add_command(compare_solvers_launcher, "compare-solvers")


if __name__ == "__main__":
    cli()
