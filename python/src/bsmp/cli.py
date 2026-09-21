#!/usr/bin/env python
import click
from bsmp.generators.matrix import generate_matrix
from bsmp.generators.eigen import generate_eigen_matrices


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


if __name__ == "__main__":
    cli()