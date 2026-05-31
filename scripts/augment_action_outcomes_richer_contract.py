#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.config import load_config


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    raw = df.get(column)
    if isinstance(raw, pd.Series):
        return pd.to_numeric(raw, errors="coerce")
    return pd.Series([pd.NA] * len(df), index=df.index, dtype="Float64")


