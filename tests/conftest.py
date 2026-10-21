"""Select the implementation under test.

    pytest                          -> src/tinyengram      (your TODO skeleton)
    TINYENGRAM_IMPL=solutions pytest -> solutions/tinyengram (reference)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMPL = os.environ.get("TINYENGRAM_IMPL", "src")

impl_path = ROOT / IMPL
if not impl_path.exists():
    raise RuntimeError(f"找不到实现目录: {impl_path}")

# Make `_utils` importable from every test module.
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(impl_path))
