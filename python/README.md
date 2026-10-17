# PYTHON for BSMP

A set of control scripts and utilities for BSMP. It consists of 4 subpackages:
- **Generators**: scripts for generating matrices, vectors and problems for test solutions
- **Launchers**: basic frontend for BSMP, potentially some edge cases launcher
- **Tracking**: performance data collecting and tracking utility. Includes figure builder
- **Benchmark**: run the examples over many (matrix, rhs) pairs, collect the produced JSON records into a single table, and plot graphs over them

## Installing

1. (Optional) Create a venv using:
```sh
python -m venv .venv 
```
2. Ensure you are in python folder. If not, `cd` to it;
3. Run:
```sh
pip install -e .
```
3. (Alternative) if you want to develop this package, install with dev dependencies:
```sh
pip install -e .[dev]
```

## Creating a wheel

Ensure you have installed .[dev] dependencies. Call this command:
```sh
python -m build --sdist
python -m build --wheel
```
Everything will be in dist/

## Benchmark

Run the examples over several matrices and turn the collected JSON records
into one table and a plot:

```sh
bsmp benchmark run --matrix F:\science\bsmp\temp\bfw782a.mtx --rhs F:\science\bsmp\temp\bfw782a.rhs --method mult --method bicgstab --no-build --work-dir F:\science\bsmp\benchmark_runs --table F:\science\bsmp\benchmark_runs\table.csv --plot F:\science\bsmp\benchmark_runs\plot.png
```

Reprocess records collected earlier:

```sh
bsmp benchmark table records_dir
bsmp benchmark plot records_dir -o results.png
```

Plots require `matplotlib` (`pip install -e .[plot]`).
