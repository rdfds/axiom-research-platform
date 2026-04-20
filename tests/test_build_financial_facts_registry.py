from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _write_financial_rows(root: Path, year: int, rows: list[dict]) -> Path:
    year_dir = root / f"year={year}"
    year_dir.mkdir(parents=True, exist_ok=True)
    out = year_dir / "part_test.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    return out


