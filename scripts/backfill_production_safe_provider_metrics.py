#!/usr/bin/env python3
"""Backfill production-safe direct provider metrics into snapshot JSONL rows.

This keeps the contract narrow on purpose: only direct provider fields with a
single stable meaning are emitted. No in-house reconstructed adjusted metrics
are added here.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd


METRIC_SPECS = {
    "operating.ebitda_ltm_provider_direct": {
        "source_column": "EBITDA",
        "unit": "usd",
        "quality_flags": None,
    },
    "liquidity.cash_and_short_term_investments_provider_direct": {
        "source_column": "Cash and Short Term Investments",
        "unit": "usd",
        "quality_flags": None,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-path", required=True, help="Input snapshot JSONL path")
    parser.add_argument("--taxonomy-reference-path", required=True, help="Provider reference parquet")
    parser.add_argument("--entity-identifier-path", required=True, help="Entity identifier parquet with ticker rows")
    parser.add_argument("--out", required=True, help="Output JSONL path")
    return parser.parse_args()


