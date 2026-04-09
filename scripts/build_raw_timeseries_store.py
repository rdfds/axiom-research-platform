#!/usr/bin/env python
"""
Build RawTimeSeriesStore by combining prices, macro, and estimates into a
single point-in-time table with provenance.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True)


def load_parquet_cols(path: Path, cols: List[str]) -> pd.DataFrame:
    pf = pq.ParquetFile(path)
    available = [c for c in cols if c in pf.schema.names]
    if not available:
        return pd.DataFrame()
    return pd.read_parquet(path, columns=available)


