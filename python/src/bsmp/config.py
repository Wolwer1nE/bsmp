from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Config:
    root_dir: Path = Path(__file__).resolve().parents[3]
    build_dir: Path = root_dir / "build"

CONFIG = Config()