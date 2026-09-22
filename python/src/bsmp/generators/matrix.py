#!/usr/bin/env python
from os.path import dirname, basename, exists, join
import random


# Generate a block-diagonal matrix and write it to file
def generate_matrix(
    output_filename: str, n_blocks: int, block_size: int, use_random: bool
) -> None:
    if n_blocks <= 0:
        raise ValueError(f"n_blocks must be positive, got {n_blocks}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if not output_filename:
        raise ValueError("output_filename is empty")
    output_dir = dirname(output_filename)
    if not exists(output_dir):
        raise ValueError(f"output directory does not exist: {output_dir}")

    with open(output_filename, "w") as f:
        for i in range(n_blocks):
            block_start = i * block_size
            value = random.random() if use_random else i + 1
            for j in range(block_size):
                for k in range(block_size):
                    row = block_start + j
                    col = block_start + k
                    f.write(f"{row} {col} {value}\n")

    n_rows = block_size * n_blocks
    rhs_output_filename = join(output_dir, f"rhs_{basename(output_filename)}")
    with open(rhs_output_filename, "w") as f:
        for i in range(n_rows):
            f.write(f"{1 + i / block_size}\n")
