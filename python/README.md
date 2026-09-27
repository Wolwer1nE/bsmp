# PYTHON for BSMP

A set of control scripts and utilities for BSMP. It consists of 3 subpackages:
- **Generators**: scripts for generating matrices, vectors and problems for test solutions
- **Launchers**: basic frontend for BSMP, potentially some edge cases launcher
- **Tracking**: performance data collecting and tracking utility. Includes figure builder

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
4. To work, python binding for bsmp require having executables of bsmp examples next to it, inside `python/src/bsmp/bin`.
They can be put into a separate folder using cmake:
```sh
cmake --install build --prefix TEMP_FOLDER
```
Then `TEMP_FOLDER\bin` can be copied to  `python/src/bsmp/`.
5. There is an automated script to build python package wheel for current pc. The command:
```sh
python tools/pack.py
``
