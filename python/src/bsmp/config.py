#!/usr/bin/env python3
"""BSMP package configuration."""

import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    root_dir: Path = Path(__file__).resolve().parents[3]
    build_dir: Path = root_dir / "build"
    exe_dir: Path = Path(__file__).resolve().parent / "bin"
    is_source_tree: bool = (root_dir / "CMakeLists.txt").is_file()

    def find_exe(self, name: str) -> Path:
        exe_prefix = ".exe" if sys.platform.startswith("win") else ""
        exe_name = name + exe_prefix
        if (self.exe_dir / exe_name).is_file():
            return self.exe_dir / exe_name

        raise FileNotFoundError(f"Executable {name} not found")


CONFIG = Config()
