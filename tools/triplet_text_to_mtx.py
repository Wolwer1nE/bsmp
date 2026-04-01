#!/usr/bin/env python3
"""Convert plain text triplets into Matrix Market (.mtx).

Supported input format:
- zero-based or one-based row/column indices;
- whitespace or comma separators;
- comment lines starting with '#' or '%';
- duplicate entries are summed.

Example input:
    0, 0,  1.00000000000000E+00
    0, 3,  2.07271175287645E-01

Example usage:
    python3 tools/triplet_text_to_mtx.py in.txt out.mtx
    python3 tools/triplet_text_to_mtx.py in.txt out.mtx --one-based
    python3 tools/triplet_text_to_mtx.py in.txt out.mtx --symmetric
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert FE-style triplet text to Matrix Market")
    parser.add_argument("input", help="Input text file with triplets")
    parser.add_argument("output", help="Output Matrix Market .mtx file")
    parser.add_argument(
        "--one-based",
        action="store_true",
        help="Interpret input row/col indices as one-based instead of zero-based",
    )
    parser.add_argument(
        "--symmetric",
        action="store_true",
        help="Write Matrix Market header as symmetric and emit only upper triangle entries",
    )
    parser.add_argument(
        "--shape",
        nargs=2,
        type=int,
        metavar=("ROWS", "COLS"),
        help="Override inferred matrix shape",
    )
    return parser.parse_args()


def iter_triplets(path: Path, one_based: bool) -> Iterable[tuple[int, int, float]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("%"):
                continue

            normalized = stripped.replace(",", " ")
            parts = normalized.split()
            if len(parts) < 3:
                raise ValueError(f"Failed to parse line {line_number} in {path}: {raw_line.rstrip()}")

            row = int(parts[0])
            col = int(parts[1])
            value = float(parts[2])

            if one_based:
                row -= 1
                col -= 1

            if row < 0 or col < 0:
                raise ValueError(
                    f"Negative matrix index after normalization on line {line_number} in {path}: "
                    f"({row}, {col})"
                )

            yield row, col, value


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    values: dict[tuple[int, int], float] = defaultdict(float)
    max_row = -1
    max_col = -1

    for row, col, value in iter_triplets(input_path, one_based=args.one_based):
        if args.symmetric and row > col:
            row, col = col, row
        values[(row, col)] += value
        max_row = max(max_row, row)
        max_col = max(max_col, col)

    if args.shape is not None:
        num_rows, num_cols = args.shape
    else:
        num_rows = max_row + 1
        num_cols = max_col + 1

    if num_rows <= 0 or num_cols <= 0:
        raise ValueError("Input produced an empty matrix")

    entries = sorted(values.items())
    output_path.parent.mkdir(parents=True, exist_ok=True)

    symmetry = "symmetric" if args.symmetric else "general"
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(f"%%MatrixMarket matrix coordinate real {symmetry}\n")
        handle.write("% Converted from FE-style text triplets by triplet_text_to_mtx.py\n")
        handle.write(f"{num_rows} {num_cols} {len(entries)}\n")
        for (row, col), value in entries:
            handle.write(f"{row + 1} {col + 1} {value:.16e}\n")

    print(
        f"Wrote {output_path} with shape {num_rows}x{num_cols} and {len(entries)} stored entries"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
