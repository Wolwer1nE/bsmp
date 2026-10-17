#!/usr/bin/env python3
"""Plots over aggregated benchmark rows (requires matplotlib)."""

from collections import defaultdict
from pathlib import Path

_PLOT_IMPORT_ERROR = (
    "matplotlib is required for benchmark plots; install it with "
    "'pip install matplotlib' (or 'pip install -e .[plot]')"
)


def _series_key(row: dict) -> str:
    solver = row.get("solver") or row.get("type") or "?"
    precond = row.get("preconditioner")
    return f"{solver}({precond})" if precond else solver


def make_plots(rows: list[dict], output: str | Path) -> None:
    """Render bar charts and a scaling scatter plot to a PNG file."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(_PLOT_IMPORT_ERROR) from exc

    timed = [row for row in rows if row.get("time_ms") not in ("", None)]
    if not timed:
        raise ValueError("no rows with a time metric to plot")

    cases = sorted({row["case"] for row in timed})
    series = sorted({_series_key(row) for row in timed})

    total_time: dict[tuple[str, str], float] = defaultdict(float)
    count: dict[tuple[str, str], int] = defaultdict(int)
    worst_residual: dict[tuple[str, str], float] = {}
    for row in timed:
        key = (row["case"], _series_key(row))
        total_time[key] += float(row["time_ms"])
        count[key] += 1
        if row.get("residual") not in ("", None):
            worst_residual[key] = max(
                worst_residual.get(key, 0.0), float(row["residual"])
            )
    mean_time = {key: total_time[key] / count[key] for key in count}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x_positions = range(len(cases))
    width = 0.8 / max(1, len(series))

    for s, name in enumerate(series):
        time_values = [mean_time.get((case, name), 0.0) for case in cases]
        axes[0].bar(
            [x + s * width for x in x_positions],
            time_values,
            width=width,
            label=name,
        )
    axes[0].set_xticks(list(x_positions))
    axes[0].set_xticklabels(cases, rotation=30, ha="right")
    axes[0].set_ylabel("time, ms")
    axes[0].set_title("Mean time per case")
    axes[0].legend()

    if worst_residual:
        for s, name in enumerate(series):
            residual_values = [
                worst_residual.get((case, name), 0.0) for case in cases
            ]
            axes[1].bar(
                [x + s * width for x in x_positions],
                residual_values,
                width=width,
                label=name,
            )
        axes[1].set_xticks(list(x_positions))
        axes[1].set_xticklabels(cases, rotation=30, ha="right")
        axes[1].set_yscale("log")
        axes[1].set_ylabel("final residual")
        axes[1].set_title("Final residual per case")
        axes[1].legend()
    else:
        axes[1].set_visible(False)

    for name in series:
        xs = [
            float(row["rows"])
            for row in timed
            if _series_key(row) == name and row.get("rows") not in ("", None)
        ]
        ys = [
            float(row["time_ms"])
            for row in timed
            if _series_key(row) == name and row.get("rows") not in ("", None)
        ]
        if xs:
            axes[2].scatter(xs, ys, label=name)
    axes[2].set_xlabel("matrix rows")
    axes[2].set_ylabel("time, ms")
    axes[2].set_title("Time vs matrix size")
    if series:
        axes[2].legend()

    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
