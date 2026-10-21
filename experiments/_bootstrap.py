"""Make ``import tinyengram`` work when running files from experiments/."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMPL = os.environ.get("TINYENGRAM_IMPL", "solutions")
sys.path.insert(0, str(ROOT / IMPL))

RUNS = ROOT / ".runs"
RUNS.mkdir(exist_ok=True)


def write_markdown(name: str, text: str) -> Path:
    path = RUNS / name
    path.write_text(text)
    print(f"\n写入 {path}")
    return path
