#!/usr/bin/env python3
"""Overlay USD public-bond maturity schedule lower bounds onto a flat quantitative export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def parse_args() :
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flat-path", required=True, help="Input flat parquet export")
    parser.add_argument("--entity-identifier-path", required=True, help="Entity identifier parquet")
    parser.add_argument("--bond-issuances-path", required=True, help="FISD bond issuances parquet")
    parser.add_argument("--bond-redemptions-path", help="Optional FISD bond redemptions parquet")
    parser.add_argument("--as-of-date", default="2024-12-31", help="As-of date in YYYY-MM-DD")
    parser.add_argument("--out-parquet", required=True, help="Output parquet path")
    parser.add_argument("--out-csv", help="Optional output CSV path")
    parser.add_argument("--summary-out", help="Optional summary JSON path")
    return parser.parse_args()


def _support_counts(series: pd.Series) -> Dict[str, int]:
    support = series.fillna("unsupported").astype(str)
    return {
        "exact": int((support == "exact").sum()),
        "proxy_missing_component": int((support == "proxy_missing_component").sum()),
        "unsupported": int((support == "unsupported").sum()),
    }


def _json_scalar(value):
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            item = value.item()
            if isinstance(item, pd.Timestamp):
                return item.isoformat()
            return item
        except Exception:
            pass
    return value


