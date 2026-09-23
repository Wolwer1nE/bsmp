from pathlib import Path

import pytest
from click.testing import CliRunner

from bsmp.cli import cli
from bsmp.launchers._build import is_windows

TRIPLET_MATRIX = "0 0 4.0\n1 1 2.0\n2 2 1.0\n"
RHS_VECTOR = "1.0\n2.0\n3.0\n"
MM_RHS_VECTOR = (
    "%%MatrixMarket matrix array real general\n"
    "% values used by the CLI tests\n"
    "3 1\n"
    f"{RHS_VECTOR}"
)


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def matrix_and_rhs(tmp_path):
    """A small triplet matrix and the matching Matrix Market RHS vector."""
    matrix = tmp_path / "matrix.txt"
    rhs = tmp_path / "rhs.mtx"
    matrix.write_text(TRIPLET_MATRIX, encoding="utf-8")
    rhs.write_text(MM_RHS_VECTOR, encoding="utf-8")
    return matrix, rhs


def spy(monkeypatch, target):
    """Replace an imported callable with a recorder, returning its call log."""
    calls = []
    monkeypatch.setattr(target, lambda *args, **kwargs: calls.append((args, kwargs)))
    return calls


# --- structure ---


def test_top_level_help(runner):
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "generate" in result.output
    assert "launch" in result.output


def test_generate_help(runner):
    result = runner.invoke(cli, ["generate", "--help"])
    assert result.exit_code == 0
    assert "matrix" in result.output
    assert "eigenvalue" in result.output


def test_launch_help(runner):
    result = runner.invoke(cli, ["launch", "--help"])
    assert result.exit_code == 0
    for cmd in ("mult", "solve", "eigen"):
        assert cmd in result.output


# --- generate ---


def test_matrix_generator_writes_matrix_and_rhs(runner, tmp_path):
    output = tmp_path / "matrix.txt"
    result = runner.invoke(
        cli, ["generate", "matrix", str(output), "-n", "2", "-b", "2"]
    )
    assert result.exit_code == 0
    assert output.read_text(encoding="utf-8").splitlines() == [
        "0 0 1",
        "0 1 1",
        "1 0 1",
        "1 1 1",
        "2 2 2",
        "2 3 2",
        "3 2 2",
        "3 3 2",
    ]
    rhs = tmp_path / "rhs_matrix.txt"
    assert rhs.read_text(encoding="utf-8").splitlines() == [
        "1.0",
        "1.5",
        "2.0",
        "2.5",
    ]


@pytest.mark.parametrize(
    ("command", "message"),
    [
        ("matrix", "n_blocks must be positive"),
        ("eigenvalue", "n must be positive"),
    ],
)
def test_generators_report_invalid_input(runner, tmp_path, command, message):
    args = [command, str(tmp_path / "out.txt")]
    if command == "eigenvalue":
        args.append(str(tmp_path / "other.txt"))
    args += ["-n", "0"]

    result = runner.invoke(cli, ["generate", *args])

    assert result.exit_code == 0  # generator commands report instead of failing
    assert f"Invalid inputs error: {message}" in result.output
    assert not list(tmp_path.iterdir())  # nothing is written for invalid inputs


def test_eigenvalue_generator_writes_both_matrices(runner, tmp_path):
    matrix_a = tmp_path / "matrix_a.txt"
    matrix_b = tmp_path / "matrix_b.txt"
    result = runner.invoke(
        cli,
        [
            "generate",
            "eigenvalue",
            str(matrix_a),
            str(matrix_b),
            "-n",
            "4",
            "--density-a",
            "0.2",
            "--density-b",
            "0.1",
        ],
    )
    assert result.exit_code == 0
    for path in (matrix_a, matrix_b):
        rows = [line.split() for line in path.read_text(encoding="utf-8").splitlines()]
        assert rows
        assert all(len(triplet) == 3 for triplet in rows)
        assert all(0 <= int(triplet[0]) < 4 for triplet in rows)


# --- launch mult ---


def test_mult_requires_rhs_option(runner, matrix_and_rhs):
    matrix, _ = matrix_and_rhs
    result = runner.invoke(cli, ["launch", "mult", str(matrix), "--no-build"])
    assert result.exit_code == 2
    assert "Missing option '--rhs'" in result.output


def test_mult_reports_missing_matrix_file(runner, matrix_and_rhs, tmp_path):
    _, rhs = matrix_and_rhs
    missing = tmp_path / "missing.txt"
    result = runner.invoke(
        cli, ["launch", "mult", str(missing), "--rhs", str(rhs), "--no-build"]
    )
    assert result.exit_code == 1
    assert f"matrix file '{missing}' not found" in result.output


def test_mult_forwards_options_to_launcher(
    runner, matrix_and_rhs, tmp_path, monkeypatch
):
    matrix, rhs = matrix_and_rhs
    output = tmp_path / "out.txt"
    calls = spy(monkeypatch, "bsmp.cli.run_mult")

    result = runner.invoke(
        cli,
        [
            "launch",
            "mult",
            str(matrix),
            "--rhs",
            str(rhs),
            "--mults",
            "3",
            "--output",
            str(output),
            "--verbose",
            "--no-build",
        ],
    )

    assert result.exit_code == 0
    expected = (str(matrix), str(rhs), "triplets", 3, str(output), True, True)
    assert calls == [(expected, {})]


# --- launch solve ---


def test_solve_rejects_unsupported_method(runner, matrix_and_rhs):
    matrix, rhs = matrix_and_rhs
    result = runner.invoke(
        cli,
        [
            "launch",
            "solve",
            "-m",
            "cg",
            "--matrix",
            str(matrix),
            "--rhs",
            str(rhs),
            "--no-build",
        ],
    )
    assert result.exit_code == 1
    assert "unsupported method 'cg'" in result.output


def test_solve_requires_matrix_and_rhs(runner):
    result = runner.invoke(cli, ["launch", "solve", "-m", "bicgstab", "--no-build"])
    assert result.exit_code == 1
    assert "--matrix and --rhs are required" in result.output


@pytest.mark.parametrize(
    ("method", "options", "expected_tail"),
    [
        (
            "bicgstab",
            ["--max-iters", "5", "--tol", "1e-8", "--precond", "amg"],
            ["", "5", "1e-8", "amg"],
        ),
        (
            "gmres",
            ["--restart", "20", "--max-iters", "5", "--tol", "1e-8"],
            ["", "20", "5", "1e-8"],
        ),
    ],
)
def test_solve_launches_expected_command(
    runner, matrix_and_rhs, monkeypatch, method, options, expected_tail
):
    matrix, rhs = matrix_and_rhs
    captured = {}

    def fake_run(command, **_kwargs):
        captured["command"] = command
        rhs_path = Path(command[2])
        captured["rhs_lines"] = rhs_path.read_text(encoding="utf-8").splitlines()

    # Only the executable lookup and the process launch are faked out.
    monkeypatch.setattr("bsmp.launchers.solve.isfile", lambda _path: True)
    monkeypatch.setattr("bsmp.launchers.solve.subprocess.run", fake_run)

    result = runner.invoke(
        cli,
        [
            "launch",
            "solve",
            "-m",
            method,
            "--matrix",
            str(matrix),
            "--rhs",
            str(rhs),
            *options,
            "--no-build",
        ],
    )

    assert result.exit_code == 0
    suffix = ".exe" if is_windows() else ""
    command = captured["command"]
    assert Path(command[0]).name == f"example_{method}{suffix}"
    assert command[1] == str(matrix)
    assert command[3:] == expected_tail
    # The Matrix Market header is stripped into a temporary RHS...
    assert captured["rhs_lines"] == RHS_VECTOR.splitlines()
    # ...which is removed once the solver returns.
    assert not Path(command[2]).exists()


# --- launch eigen ---


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (["--stiffness", "s.mtx", "--cu", "c.mtx"], "not both"),
        (["--stiffness", "s.mtx"], "requires --mass"),
    ],
)
def test_eigen_rejects_invalid_mode_combination(runner, options, message):
    result = runner.invoke(cli, ["launch", "eigen", *options, "--no-build"])
    assert result.exit_code == 1
    assert message in result.output


def test_eigen_forwards_options_to_launcher(runner, matrix_and_rhs, monkeypatch):
    matrix, _ = matrix_and_rhs
    calls = spy(monkeypatch, "bsmp.cli.run_eigen")

    result = runner.invoke(
        cli,
        [
            "launch",
            "eigen",
            "--synthetic",
            "--modes",
            "5",
            "--cu",
            str(matrix),
            "--no-build",
        ],
    )

    assert result.exit_code == 0
    args, kwargs = calls[0]
    assert args == ()
    assert kwargs["synthetic"] is True
    assert kwargs["modes"] == "5"
    assert kwargs["cu"] == str(matrix)
    assert kwargs["stiffness"] is None  # unused options stay unset
    assert kwargs["no_build"] is True
