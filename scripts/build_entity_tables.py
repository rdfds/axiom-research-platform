#!/usr/bin/env python
"""
Build canonical Entity + EntityIdentifier tables from the ID mapping file.
Also emits stub EntityRelationship + EntityCorrectionLog tables.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def utc_now() :
    return datetime.now(timezone.utc).isoformat()


def parse_dt(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


