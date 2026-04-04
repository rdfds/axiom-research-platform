#!/usr/bin/env python3
"""Repair PIT-unsafe flat quant profitability/cash-flow metrics.

This pass replaces the date-less Refinitiv overlay on top of a flat export with
point-in-time-safe metrics rebuilt from SEC companyfacts:

1. `operating__ebitda_margin_ttm__*`
2. `market__ev_ebitda__*`
3. `market__fcf_yield__*`
4. `operating__fcf_conversion__*`

It also writes transparent raw SEC-backed columns for TTM revenue, EBITDA, free
cash flow, and cash/short-term-investments so the repaired metrics are easy to
audit.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict

import duckdb
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import backfill_input_layer_v1_metrics as core
import backfill_market_macro_input_layer_v1 as market_macro
import backfill_sec_companyfacts_components as seccomp
import repair_cash_flow_artifact as cashflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flat-path", required=True, help="Input flat parquet export")
    parser.add_argument("--companyfacts-root", required=True, help="SEC companyfacts folder")
    parser.add_argument("--entity-identifier-path", help="Entity identifier parquet for permno mapping")
    parser.add_argument("--raw-timeseries-path", help="Raw timeseries parquet for PIT prices")
    parser.add_argument("--out-parquet", required=True, help="Output parquet path")
    parser.add_argument("--out-csv", help="Optional output CSV path")
    parser.add_argument("--summary-out", help="Optional summary JSON path")
    return parser.parse_args()


def _support_counts(series: pd.Series) -> Dict[str, int]:
    values = series.fillna("unsupported").astype(str)
    return {
        "exact": int((values == "exact").sum()),
        "proxy_missing_component": int((values == "proxy_missing_component").sum()),
        "unsupported": int((values == "unsupported").sum()),
    }


def _json_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _raw_support(value: float | None, support_mode: str | None) -> str:
    if value is None:
        return "unsupported"
    return support_mode or "exact"


def _derived_support(*support_modes: str) -> str:
    if not support_modes or any(mode == "unsupported" for mode in support_modes):
        return "unsupported"
    return "exact" if all(mode == "exact" for mode in support_modes) else "proxy_missing_component"


def _exact_or_proxy_support(*support_modes: str) -> str:
    if not support_modes or any(mode == "unsupported" for mode in support_modes):
        return "unsupported"
    return "exact" if all(mode == "exact" for mode in support_modes) else "proxy_missing_component"


