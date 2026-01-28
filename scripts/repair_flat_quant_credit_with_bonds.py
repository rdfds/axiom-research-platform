#!/usr/bin/env python3
"""Overlay issuer bond-spread truth onto a flat quantitative export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import duckdb
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flat-path", required=True, help="Input flat parquet export")
    parser.add_argument("--entity-identifier-path", required=True, help="Entity identifier parquet")
    parser.add_argument("--trace-daily-path", required=True, help="TRACE daily parquet")
    parser.add_argument("--bond-issuances-path", required=True, help="FISD bond issuances parquet")
    parser.add_argument("--raw-timeseries-path", required=True, help="Raw timeseries parquet with Treasury yields")
    parser.add_argument("--as-of-date", default="2024-12-31", help="As-of date in YYYY-MM-DD")
    parser.add_argument(
        "--current-lookback-days",
        type=int,
        default=90,
        help="Recency window for the current spread level",
    )
    parser.add_argument(
        "--history-lookback-days",
        type=int,
        default=730,
        help="History window for issuer spread percentile",
    )
    parser.add_argument("--out-parquet", required=True, help="Output parquet path")
    parser.add_argument("--out-csv", help="Optional output CSV path")
    parser.add_argument("--summary-out", help="Optional summary JSON path")
    return parser.parse_args()


def _sql_path(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def _support_counts(series: pd.Series) -> Dict[str, int]:
    support = series.fillna("unsupported").astype(str)
    return {
        "exact": int((support == "exact").sum()),
        "proxy_missing_component": int((support == "proxy_missing_component").sum()),
        "unsupported": int((support == "unsupported").sum()),
    }


