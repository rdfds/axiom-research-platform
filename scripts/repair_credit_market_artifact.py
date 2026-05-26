#!/usr/bin/env python3
"""Repair credit-market metrics in a materialized company-state artifact.

This pass fills the still-empty credit regime / credit window layer using:
- exact macro IG/HY OAS anchors already present in the artifact
- macro OAS history from the raw timeseries parquet for percentiles
- company risk signals already materialized in the artifact

The repaired company-level spread metrics are intentionally tagged as heuristic
proxy values; they are not direct traded bond/CDS spreads.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd
import pyarrow.parquet as pq


REPAIR_METRICS = [
    "macro.us_ig_oas",
    "macro.us_ig_oas_percentile_history",
    "market.credit_spread_level",
    "market.credit_spread_percentile_2y",
    "market.credit_window_proxy",
]

IG_OAS_INSTRUMENT = "BAMLC0A0CM"
HY_OAS_INSTRUMENT = "BAMLH0A0HYM2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-path", required=True, help="Input company-state JSONL artifact")
    parser.add_argument("--macro-timeseries-path", help="Optional raw timeseries parquet path")
    parser.add_argument("--out", required=True, help="Output repaired JSONL artifact")
    parser.add_argument("--summary-out", help="Optional summary JSON")
    return parser.parse_args()


