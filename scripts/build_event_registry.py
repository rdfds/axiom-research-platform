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


