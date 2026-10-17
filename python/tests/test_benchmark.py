#!/usr/bin/env python3
"""Tests for the bsmp benchmark module."""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from bsmp.benchmark.records import (
    COLUMNS,
    collect_records,
    print_table,
    record_to_rows,
    rows_from_records,
    write_csv,
)
from bsmp.benchmark.runner import run_benchmark
from bsmp.cli import cli

SAMPLE_RECORD = {
    "schema_version": "1.0.0",
    "hardware": {
        "gpu_model": "NVIDIA RTX 5060 Ti",
        "cuda_version": "13.4",
        "compute_capability": "120",
    },
    "problem": {
        "type": "solve",
        "matrix": {"name": "bfw782a", "rows": 782, "columns": 782, "nnz": 60},
        "rhs": {"size": 782, "from_task": False},
    },
    "runs": [
        {
            "parameters": {
                "solver": "bicgstab",
                "preconditioner": "amg",
                "max_iters": 1000,
                "tolerance": 1e-6,
            },
            "metrics": {"time": 503.7, "residual": 8.8e-07},
        }
    ],
}


@pytest.fixture
def runner():
    return CliRunner()


def write_sample(tmp_path, name="case1.json", record=None):
    path = tmp_path / name
    path.write_text(json.dumps(record or SAMPLE_RECORD), encoding="utf-8")
    return path


# --- records ---


def test_record_to_rows_flattens_runs():
    rows = record_to_rows("bfw782a", SAMPLE_RECORD)
    assert len(rows) == 1
    row = rows[0]
    assert row["case"] == "bfw782a"
    assert row["type"] == "solve"
    assert row["solver"] == "bicgstab"
    assert row["preconditioner"] == "amg"
    assert row["time_ms"] == 503.7
    assert row["residual"] == 8.8e-07
    assert row["matrix"] == "bfw782a"
    assert row["rows"] == 782


def test_record_to_rows_multiple_runs():
    record = dict(SAMPLE_RECORD)
    record["runs"] = [dict(run) for run in SAMPLE_RECORD["runs"]]
    record["runs"].append(
        {
            "parameters": {
                "solver": "gmres",
                "preconditioner": "none",
                "max_iters": 1000,
                "tolerance": 1e-6,
            },
            "metrics": {"time": 12.3, "residual": 1e-09},
        }
    )
    rows = record_to_rows("bfw782a", record)
    assert len(rows) == 2
    assert [row["solver"] for row in rows] == ["bicgstab", "gmres"]


def test_record_to_rows_empty_runs():
    record = dict(SAMPLE_RECORD, runs=[])
    assert record_to_rows("x", record) == []


def test_print_table_prints_header(capsys):
    rows = rows_from_records([("bfw782a", SAMPLE_RECORD)])
    print_table(rows)
    out = capsys.readouterr().out
    for expected in ("case", "solver", "bicgstab", "503.7"):
        assert expected in out


def test_write_csv(tmp_path):
    rows = rows_from_records([("bfw782a", SAMPLE_RECORD)])
    path = tmp_path / "table.csv"
    write_csv(rows, path)
    text = path.read_text(encoding="utf-8")
    assert text.splitlines()[0] == ",".join(COLUMNS)
    assert "bfw782a" in text
    assert "bicgstab" in text


def test_collect_records_from_directory_and_files(tmp_path):
    write_sample(tmp_path, "case1.json")
    write_sample(tmp_path, "other_dir.json", record=dict(SAMPLE_RECORD, runs=[]))
    collected = collect_records([tmp_path])
    assert {label for label, _ in collected} == {"case1", "other_dir"}
    assert len(collected) == 2


def test_collect_records_label_dedup(tmp_path):
    sub_a = tmp_path / "a"
    sub_b = tmp_path / "b"
    sub_a.mkdir()
    sub_b.mkdir()
    write_sample(sub_a, "same.json")
    write_sample(sub_b, "same.json")
    collected = collect_records([sub_a, sub_b])
    labels = [label for label, _ in collected]
    assert labels[0] != labels[1]
    assert "same" in labels[0]
    assert "same" in labels[1]


def test_collect_records_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        collect_records([tmp_path / "nope"])


# --- runner ---


def test_run_benchmark_collects_records(tmp_path, monkeypatch):
    matrix = tmp_path / "m.txt"
    matrix.write_text("0 0 1\n", encoding="utf-8")
    rhs = tmp_path / "r.txt"
    rhs.write_text("1\n", encoding="utf-8")

    def fake_run_solve(
        method,
        matrix_file,
        rhs_file,
        precond="",
        output_file="",
        max_iters="",
        tol="",
        restart="",
        no_build=False,
        cwd=None,
    ):
        log = Path(cwd) / "log_bicgstab.json"
        log.write_text(json.dumps(SAMPLE_RECORD), encoding="utf-8")

    monkeypatch.setattr("bsmp.benchmark.runner.run_solve", fake_run_solve)

    work_dir = tmp_path / "runs"
    records_dir = tmp_path / "records"
    collected = run_benchmark(
        [(str(matrix), str(rhs))],
        ("bicgstab",),
        work_dir=work_dir,
        records_dir=records_dir,
        no_build=True,
        table_path=tmp_path / "table.csv",
    )

    assert len(collected) == 1
    record = json.loads(Path(collected[0]).read_text(encoding="utf-8"))
    assert record["problem"]["matrix"]["name"] == "bfw782a"
    assert (tmp_path / "table.csv").is_file()


def test_run_benchmark_reports_failures_and_continues(tmp_path, monkeypatch, capsys):
    matrix = tmp_path / "m.txt"
    matrix.write_text("0 0 1\n", encoding="utf-8")
    rhs = tmp_path / "r.txt"
    rhs.write_text("1\n", encoding="utf-8")

    def fake_run_solve(
        method,
        matrix_file,
        rhs_file,
        precond="",
        output_file="",
        max_iters="",
        tol="",
        restart="",
        no_build=False,
        cwd=None,
    ):
        if method == "gmres":
            raise RuntimeError("exploded")
        log = Path(cwd) / "log_bicgstab.json"
        log.write_text(json.dumps(SAMPLE_RECORD), encoding="utf-8")

    monkeypatch.setattr("bsmp.benchmark.runner.run_solve", fake_run_solve)

    work_dir = tmp_path / "runs"
    records_dir = tmp_path / "records"
    collected = run_benchmark(
        [(str(matrix), str(rhs))],
        ("bicgstab", "gmres"),
        work_dir=work_dir,
        records_dir=records_dir,
        no_build=True,
    )

    assert len(collected) == 1  # gmres failed, bicgstab was collected
    assert "exploded" in capsys.readouterr().err


# --- CLI ---


def test_benchmark_help_lists_commands(runner):
    result = runner.invoke(cli, ["benchmark", "--help"])
    assert result.exit_code == 0
    for cmd in ("run", "table", "plot"):
        assert cmd in result.output


def test_benchmark_run_requires_matrix_and_rhs(runner, tmp_path):
    result = runner.invoke(cli, ["benchmark", "run", "--no-build"])
    assert result.exit_code == 1
    assert "--matrix and --rhs are required" in result.output


def test_benchmark_run_requires_paired_inputs(runner, tmp_path):
    matrix = tmp_path / "m.txt"
    matrix.write_text("0 0 1\n", encoding="utf-8")
    rhs = tmp_path / "r.txt"
    rhs.write_text("1\n", encoding="utf-8")
    result = runner.invoke(
        cli,
        [
            "benchmark",
            "run",
            "--matrix",
            str(matrix),
            "--rhs",
            str(rhs),
            "--rhs",
            str(rhs),
            "--no-build",
        ],
    )
    assert result.exit_code == 1
    assert "must match" in result.output


def test_benchmark_run_forwards_options(runner, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        "bsmp.cli.run_benchmark",
        lambda **kwargs: calls.append(kwargs) or [],
    )
    matrix_a = tmp_path / "a.txt"
    matrix_b = tmp_path / "b.txt"
    rhs_a = tmp_path / "ra.txt"
    rhs_b = tmp_path / "rb.txt"
    for path in (matrix_a, matrix_b, rhs_a, rhs_b):
        path.write_text("0 0 1\n", encoding="utf-8")

    result = runner.invoke(
        cli,
        [
            "benchmark",
            "run",
            "--matrix",
            str(matrix_a),
            "--matrix",
            str(matrix_b),
            "--rhs",
            str(rhs_a),
            "--rhs",
            str(rhs_b),
            "--method",
            "gmres",
            "--no-build",
            "--table",
            "out.csv",
            "--plot",
            "out.png",
        ],
    )

    assert result.exit_code == 0
    assert len(calls) == 1
    kwargs = calls[0]
    assert kwargs["cases"] == [(str(matrix_a), str(rhs_a)), (str(matrix_b), str(rhs_b))]
    assert kwargs["methods"] == ("gmres",)
    assert kwargs["no_build"] is True
    assert kwargs["table_path"] == "out.csv"
    assert kwargs["plot_path"] == "out.png"


def test_benchmark_table_prints_rows(runner, tmp_path):
    file = write_sample(tmp_path)
    result = runner.invoke(cli, ["benchmark", "table", str(file)])
    assert result.exit_code == 0
    for expected in ("case", "solver", "bicgstab", "503.7"):
        assert expected in result.output


def test_benchmark_plot_writes_png(runner, tmp_path):
    pytest.importorskip("matplotlib")
    file = write_sample(tmp_path)
    output = tmp_path / "plot.png"
    result = runner.invoke(cli, ["benchmark", "plot", str(file), "-o", str(output)])
    assert result.exit_code == 0
    assert output.is_file()
