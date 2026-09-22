#!/usr/bin/env python
import random
from os.path import dirname, exists


# Generate a symmetric sparse matrix as a list of [row, col, value] triplets
def generate_symmetric_sparse_matrix(
    n: int, density: float, scale_factor: float = 1.0
) -> list[list]:
    triplets = []

    for i in range(n):
        diag_value = (i + 1) * scale_factor + random.random() * 5.0
        triplets.append([i, i, diag_value])

    num_nonzeros = int(n * n * density / 2)

    for _ in range(num_nonzeros):
        i = random.randrange(n)
        j = random.randrange(n)
        if i == j:
            continue
        if i > j:
            continue

        value = (random.random() - 0.5) * 2.0 * scale_factor
        triplets.append([i, j, value])
        triplets.append([j, i, value])

    hash_map = {}
    for i, j, v in triplets:
        key = (i, j)
        hash_map[key] = hash_map.get(key, 0.0) + v

    result = []
    for (i, j), v in hash_map.items():
        if abs(v) > 1e-10:
            result.append([i, j, v])

    result.sort(key=lambda t: (t[0], t[1]))
    return result


# Write triplets to a file, one "row col value" per line
def save_triplets(filename: str, triplets: list[list]) -> None:
    with open(filename, "w") as f:
        for i, j, v in triplets:
            f.write(f"{i} {j} {v}\n")


# Generate the A and B matrices of a generalized eigenvalue problem
def generate_eigen_matrices(
    output_filename_A: str,
    output_filename_B: str,
    n: int,
    density_A: float,
    density_B: float,
) -> None:
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if density_A <= 0 or density_B <= 0:
        raise ValueError(
            f"densities must be positive, got {density_A}, {density_B}"
        )
    if not output_filename_A or not output_filename_B:
        raise ValueError("output filename is empty")
    if not exists(dirname(output_filename_A)):
        raise ValueError(
            f"output directory does not exist: {dirname(output_filename_A)}"
        )
    if not exists(dirname(output_filename_B)):
        raise ValueError(
            f"output directory does not exist: {dirname(output_filename_B)}"
        )

    triplets_A = generate_symmetric_sparse_matrix(n, density_A, 1.0)
    triplets_B = generate_symmetric_sparse_matrix(n, density_B, 0.5)

    # B is really good in this case, strongly diagonally dominant
    for triplet in triplets_B:
        if triplet[0] == triplet[1]:
            triplet[2] += 10.0

    print()
    print("Saving matrices...")
    save_triplets(output_filename_A, triplets_A)
    save_triplets(output_filename_B, triplets_B)

    print()
    print("=" * 60)
    print("Matrix statistics:")
    print("=" * 60)
    print("Matrix A:")
    print(f"  Size: {n} x {n}")
    print(f"  Nonzeros: {len(triplets_A)}")
    print(f"  Density: {round(len(triplets_A) / (n * n) * 100, 3)}%")
    print(f"  Estimated memory: {round(len(triplets_A) * 12 / 1024.0, 2)} KB")
    print()
    print("Matrix B:")
    print(f"  Size: {n} x {n}")
    print(f"  Nonzeros: {len(triplets_B)}")
    print(f"  Density: {round(len(triplets_B) / (n * n) * 100, 3)}%")
    print(f"  Estimated memory: {round(len(triplets_B) * 12 / 1024.0, 2)} KB")
    print()
