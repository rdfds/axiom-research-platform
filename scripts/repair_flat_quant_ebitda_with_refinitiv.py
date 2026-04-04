#!/usr/bin/env python3
"""Overlay market-grade Refinitiv market/pricing metrics onto a flat quantitative export."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict

import pandas as pd

from scripts.repair_rating_state_artifact import (
    _resolve_ratings_path,
    build_rating_index,
    load_issuer_ratings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flat-path", required=True, help="Input flat parquet export")
    parser.add_argument("--provider-reference-path", required=True, help="Refinitiv fundamentals parquet")
    parser.add_argument("--entity-identifier-path", required=True, help="Entity identifier parquet")
    parser.add_argument("--ratings-path", help="Optional issuer ratings parquet/csv.gz path")
    parser.add_argument("--out-parquet", required=True, help="Output parquet path")
    parser.add_argument("--out-csv", help="Optional output CSV path")
    parser.add_argument("--summary-out", help="Optional summary JSON path")
    return parser.parse_args()


