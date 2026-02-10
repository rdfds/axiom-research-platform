#!/usr/bin/env python3
"""Patch known as-of-safe artifact issues without re-running the full pipeline.

This repair pass is intentionally narrow:

1. Recompute direct market price / return metrics from exact CRSP daily data
   when available, and fail honestly when they are not.
2. Reject negative TTM revenue outputs and demote dependent margin metrics.

The underlying builders have also been updated, but this script lets us repair
already-materialized artifacts quickly and deterministically.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

try:
    from backfill_market_macro_input_layer_v1 import (
        DEFAULT_LOCAL_CRSP_DAILY_ROOT,
        _build_price_metrics,
        _build_price_metrics_from_crsp,
        _load_crsp_daily_from_repo,
        _load_crsp_market_cache,
    )
except Exception:  # noqa: BLE001
    try:
        from scripts.backfill_market_macro_input_layer_v1 import (
            DEFAULT_LOCAL_CRSP_DAILY_ROOT,
            _build_price_metrics,
            _build_price_metrics_from_crsp,
            _load_crsp_daily_from_repo,
            _load_crsp_market_cache,
        )
    except Exception:  # noqa: BLE001
        DEFAULT_LOCAL_CRSP_DAILY_ROOT = None
        _build_price_metrics = None
        _build_price_metrics_from_crsp = None
        _load_crsp_daily_from_repo = None
        _load_crsp_market_cache = None


TARGET_ARTIFACTS = [
    "company_state_snapshots_asof=2024-12-31.input_layer_v1.asofsafe.jsonl",
    "company_state_snapshots_asof=2024-12-31.input_layer_v1_market_macro.asofsafe.jsonl",
    "company_state_snapshots_asof=2024-12-31.input_layer_v1_market_macro_statement_optional.asofsafe.jsonl",
    "company_state_snapshots_asof=2024-12-31.input_layer_v1_with_sec_components.asofsafe.jsonl",
    "company_state_snapshots_asof=2024-12-31.input_layer_v1_smart_normalized_with_sec.asofsafe.jsonl",
]

TARGET_MARKET_METRICS = [
    "market.price_spot",
    "market.total_return_1m_standardized",
    "market.total_return_3m_standardized",
    "market.total_return_6m_standardized",
    "market.total_return_12m_standardized",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--entity-identifier-path", required=True)
    parser.add_argument("--raw-timeseries-path", required=True)
    parser.add_argument("--crsp-market-cache-path")
    parser.add_argument(
        "--crsp-daily-root",
        help="Optional CRSP daily parquet folder. Defaults to the local canonical WRDS CRSP folder when present.",
    )
    parser.add_argument(
        "--allow-monthly-market-proxy",
        action="store_true",
        help="Allow the older monthly raw-timeseries proxy path when exact CRSP daily data is unavailable.",
    )
    return parser.parse_args()


