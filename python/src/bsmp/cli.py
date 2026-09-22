#!/usr/bin/env python
import click
import subprocess
from bsmp.generators.matrix import generate_matrix
from bsmp.generators.eigen import generate_eigen_matrices
from bsmp.launchers.mult import run_mult
from bsmp.launchers.solve import run_solve


@click.group()
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
    except IOError as e:
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
        generate_eigen_matrices(output_file_a, output_file_b, size, density_a, density_b)
    except ValueError as e:
        print(f"Invalid inputs error: {e}")
    except IOError as e:
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
@click.option("--format", default="triplets", help="Input format: triplets|matrix-market")
@click.option("--mults", default=10, help="Number of chained multiplies")
@click.option("--output", help="Save resulting vector to path")
@click.option("--verbose", is_flag=True, default=False, help="Verbose kernel statistics")
@click.option("--no-build", is_flag=True, default=False, help="Use existing build if present")
def mult_launcher(matrix_file, rhs, format, mults, output, verbose, no_build):
    """Run the multiplication test on a matrix and RHS vector"""
    try:
        run_mult(matrix_file, rhs, format, mults, output, verbose, no_build)
    except ValueError as e:
        print(f"Invalid inputs error: {e}")
        raise SystemExit(1)
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}")
        raise SystemExit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise SystemExit(1)


@click.command()
@click.option("-m", "--method", help="Solver method (required)")
@click.option("--precond", default="", help="Preconditioner: amg, scalar-jacobi, block-jacobi, none")
@click.option("--matrix", help="Input matrix in triplet format (required)")
@click.option("--rhs", help="RHS vector path (required)")
@click.option("--output", help="Save solution vector to path")
@click.option("--max-iters", help="Maximum number of iterations")
@click.option("--tol", help="Relative residual tolerance")
@click.option("--restart", help="Restart parameter for GMRES")
@click.option("--no-build", is_flag=True, default=False, help="Use existing build if present")
def solve_launcher(method, precond, matrix, rhs, output, max_iters, tol, restart, no_build):
    """Solve a system Ax=b using the selected method"""
    try:
        run_solve(method, matrix, rhs, precond, output, max_iters, tol, restart, no_build)
    except ValueError as e:
        print(f"Invalid inputs error: {e}")
        raise SystemExit(1)
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}")
        raise SystemExit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise SystemExit(1)

launch.add_command(mult_launcher, "mult")
launch.add_command(solve_launcher, "solve")



if __name__ == "__main__":
    cli()