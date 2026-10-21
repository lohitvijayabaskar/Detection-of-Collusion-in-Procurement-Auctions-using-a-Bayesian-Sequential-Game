"""Shared helpers for scripts: repo-root import path and seed parsing."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"


def parse_seeds(text: str) -> list[int]:
    """'0-9' -> [0..9]; '0,3,5' -> [0, 3, 5]."""
    if "-" in text:
        a, b = text.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(s) for s in text.split(",")]
