#!/usr/bin/env python
"""
Build EventRegistry from the unified corporate actions master dataset.

This is a baseline generator: it normalizes core fields and preserves provenance.
Parameters/evidence_links are left null for now (can be enriched later).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True)


def coalesce(series_list: Iterable[pd.Series]) -> pd.Series:
    out = None
    for s in series_list:
        if s is None:
            continue
        out = s if out is None else out.combine_first(s)
    return out


def prefixed_str(s: pd.Series, prefix: str) -> pd.Series:
    s = s.astype("string")
    s = s.where(s.notna(), pd.NA)
    return prefix + s


