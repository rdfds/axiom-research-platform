#!/usr/bin/env python3
"""Build a lightweight quantitative peer-screen view from the core market-grade export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flat-path", required=True, help="Input core-market-grade parquet path")
    parser.add_argument("--out-parquet", required=True, help="Output parquet path")
    parser.add_argument("--out-csv", help="Optional output CSV path")
    parser.add_argument("--summary-out", help="Optional output summary JSON path")
    return parser.parse_args()


def _support_counts(series: pd.Series) :
    support = series.fillna("unsupported").astype(str)
    return {str(k): int(v) for k, v in support.value_counts(dropna=False).to_dict().items()}


def _convert_percent(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce") * 100.0


def _convert_bps(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce") * 10000.0


