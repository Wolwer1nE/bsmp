#!/usr/bin/env python3
"""Visualize the sparsity pattern of a triplet-format matrix, similar to MATLAB spy(m).

Supported input format:
- row, col, value triplets separated by commas and/or whitespace;
- zero-based indices by default, with optional one-based mode;
- comment lines starting with '#' or '%';
- files that mix omitted zeros with explicitly written zero entries.

Explicitly written zeros are always ignored when building the sparsity pattern.

Examples:
    python3 tools/sparse_spy.py data/bsmp1.txt
    python3 tools/sparse_spy.py data/martynova/big4/big4_O_phi_Ct.txt --output big4.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize sparse matrix structure from plain-text triplets"
    )
    parser.add_argument("input", help="Input text file with row, col, value triplets")
    parser.add_argument(
        "--output",
        help="Save the plot to an image file instead of opening an interactive window",
    )
    parser.add_argument(
        "--one-based",
        action="store_true",
        help="Interpret row/col indices in the file as one-based",
    )
    parser.add_argument(
        "--shape",
        nargs=2,
        type=int,
        metavar=("ROWS", "COLS"),
        help="Override inferred matrix shape",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=0.0,
        help="Treat |value| <= tol as zero (default: 0.0)",
    )
    parser.add_argument(
        "--nz-color",
        default="black",
        help="Color for structural nonzeros (default: black)",
    )
    parser.add_argument(
        "--marker",
        default=",",
        help="Matplotlib marker for entries (default: ',' pixel marker)",
    )
    parser.add_argument(
        "--markersize",
        type=float,
        default=1.0,
        help="Marker size passed to matplotlib (default: 1.0)",
    )
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=(8.0, 8.0),
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size in inches (default: 8 8)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Output DPI when saving the image (default: 200)",
    )
    parser.add_argument(
        "--title",
        help="Custom plot title (default: derived from file name)",
    )
    parser.add_argument(
        "--no-grid",
        action="store_true",
        help="Disable background grid",
    )
    return parser.parse_args()


def normalize_triplet_line(raw_line: str, *, line_number: int, path: Path) -> tuple[int, int, float] | None:
    stripped = raw_line.strip()
    if not stripped or stripped.startswith("#") or stripped.startswith("%"):
        return None

    parts = stripped.replace(",", " ").split()
    if len(parts) < 3:
        raise ValueError(f"Failed to parse line {line_number} in {path}: {raw_line.rstrip()}")

    row = int(parts[0])
    col = int(parts[1])
    value = float(parts[2])
    return row, col, value


def collect_pattern(
    path: Path,
    *,
    one_based: bool,
    tol: float,
) -> dict[str, object]:
    nz_rows: list[int] = []
    nz_cols: list[int] = []
    max_row = -1
    max_col = -1
    total_entries = 0
    skipped_zero_entries = 0
    
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            parsed = normalize_triplet_line(raw_line, line_number=line_number, path=path)
            if parsed is None:
                continue

            row, col, value = parsed
            if one_based:
                row -= 1
                col -= 1

            if row < 0 or col < 0:
                raise ValueError(
                    f"Negative matrix index after normalization on line {line_number} in {path}: ({row}, {col})"
                )

            total_entries += 1
            max_row = max(max_row, row)
            max_col = max(max_col, col)

            if abs(value) <= tol:
                skipped_zero_entries += 1
            else:
                nz_rows.append(row)
                nz_cols.append(col)

    return {
        "nz_rows": nz_rows,
        "nz_cols": nz_cols,
        "max_row": max_row,
        "max_col": max_col,
        "total_entries": total_entries,
        "skipped_zero_entries": skipped_zero_entries,
    }


def build_title(path: Path, rows: int, cols: int, nnz: int) -> str:
    return f"{path.name} — {rows}x{cols}, nnz={nnz}"


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else None

    if output_path is not None:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    pattern = collect_pattern(
        input_path,
        one_based=args.one_based,
        tol=args.tol,
    )

    max_row = int(pattern["max_row"])
    max_col = int(pattern["max_col"])
    if args.shape is not None:
        rows, cols = args.shape
    else:
        rows = max_row + 1
        cols = max_col + 1

    if rows <= 0 or cols <= 0:
        raise ValueError("Input produced an empty matrix")

    nz_rows = pattern["nz_rows"]
    nz_cols = pattern["nz_cols"]
    total_entries = int(pattern["total_entries"])
    skipped_zero_entries = int(pattern["skipped_zero_entries"])

    fig, ax = plt.subplots(figsize=tuple(args.figsize))

    if nz_rows:
        ax.plot(
            nz_cols,
            nz_rows,
            linestyle="None",
            marker=args.marker,
            color=args.nz_color,
            markersize=args.markersize,
            label="nonzeros",
        )

    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    ax.set_title(args.title or build_title(input_path, rows, cols, len(nz_rows)))

    if args.no_grid:
        ax.grid(False)
    else:
        ax.grid(True, linewidth=0.2, alpha=0.3)

    fig.tight_layout()

    print(f"Read {total_entries} triplets from {input_path}")
    print(f"Matrix shape: {rows} x {cols}")
    print(f"Structural nonzeros shown: {len(nz_rows)}")
    # print(f"Ignored explicit zero entries: {skipped_zero_entries}") это чисто для меня П.А.

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved sparsity plot to {output_path}")
    else:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())