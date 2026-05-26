#!/usr/bin/env python3
"""Backfill market and macro inputs into the v1 company input-layer artifact.

Preferred path:
- CRSP daily market cache for point-in-time-safe prices, returns, and shares

Fallback path:
- local month-end parquet, which remains a proxy-only source
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
import signal
from typing import Any, Dict, Iterable

import duckdb
import pandas as pd
from bs4 import BeautifulSoup

try:
    from repair_statement_debt_override_artifact import _fetch_sec_primary_document, _latest_sec_filing, _sec_session
except Exception:  # noqa: BLE001
    try:
        from scripts.repair_statement_debt_override_artifact import _fetch_sec_primary_document, _latest_sec_filing, _sec_session
    except Exception:  # noqa: BLE001
        _fetch_sec_primary_document = None
        _latest_sec_filing = None
        _sec_session = None


MAX_SEC_FACT_AGE_DAYS = 550
MAX_MONTHLY_GAP_DAYS = 45
MAX_DAILY_ANCHOR_GAP_DAYS = 7
MAX_ISSUER_SHARES_AGE_DAYS = 130
ISSUER_SHARES_OVERRIDE_MIN_RATIO = 1.05
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_COMPANYFACTS_ROOT = REPO_ROOT / "data" / "sec" / "companyfacts"
DEFAULT_LOCAL_CRSP_DAILY_ROOT = REPO_ROOT / "data" / "wrds" / "crsp"
SHARES_OUT_CONCEPTS = [
    ("dei", "EntityCommonStockSharesOutstanding"),
    ("us-gaap", "CommonStockSharesOutstanding"),
]
SHARE_CLASS_SEGMENT_RE = re.compile(
    r"(statementclassofstockaxis|classesofsharecapitalaxis|commonclass[a-z0-9]*member|class[a-z0-9]*member|ordinaryshareclass)",
    re.IGNORECASE,
)

MARKET_METRICS = {
    "market.price_spot": {"unit": "usd_per_share"},
    "market.total_return_1m_standardized": {"unit": "ratio", "months": 1},
    "market.total_return_3m_standardized": {"unit": "ratio", "months": 3},
    "market.total_return_6m_standardized": {"unit": "ratio", "months": 6},
    "market.total_return_12m_standardized": {"unit": "ratio", "months": 12},
}

MACRO_SERIES_SPECS = {
    "macro.fed_funds_effective": {"instrument_id": "DFF", "unit": "pct"},
    "macro.sofr": {"instrument_id": "SOFR", "unit": "pct"},
    "macro.ust_2y_yield": {"instrument_id": "DGS2", "unit": "pct"},
    "macro.ust_10y_yield": {"instrument_id": "DGS10", "unit": "pct"},
    "macro.ig_oas": {"instrument_id": "BAMLC0A0CM", "unit": "pct"},
    "macro.hy_oas": {"instrument_id": "BAMLH0A0HYM2", "unit": "pct"},
    "macro.unemployment_rate": {"instrument_id": "UNRATE", "unit": "pct"},
    "macro.wti_crude": {"instrument_id": "DCOILWTICO", "unit": "usd_bbl"},
}

MACRO_YOY_SPECS = {
    "macro.cpi_yoy": {"instrument_id": "CPIAUCSL", "unit": "ratio"},
    "macro.retail_sales_yoy": {"instrument_id": "RSAFS", "unit": "ratio"},
    "macro.real_gdp_growth_yoy": {"instrument_id": "GDPC1", "unit": "ratio", "lag_observations": 4},
}

MACRO_BASE_LOOKBACK_DAYS = 400
MACRO_LAGGED_SERIES_LOOKBACK_DAYS = max(
    MACRO_BASE_LOOKBACK_DAYS,
    max(int(spec.get("lag_observations") or 0) for spec in MACRO_YOY_SPECS.values()) * 120 + 180,
)

MACRO_METRIC_UNITS = {
    "macro.sofr_or_fed_funds": "pct",
    **{metric_name: spec["unit"] for metric_name, spec in MACRO_SERIES_SPECS.items()},
    "macro.curve_2s10s": "pct",
    **{metric_name: spec["unit"] for metric_name, spec in MACRO_YOY_SPECS.items()},
}


class _CompanyProcessingTimeout(RuntimeError):
    """Raised when one company exceeds the allowed processing timeout."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-path", required=True, help="Input company snapshot JSONL")
    parser.add_argument("--entity-identifier-path", required=True, help="Entity identifier parquet")
    parser.add_argument("--raw-timeseries-path", required=True, help="Local raw_timeseries parquet")
    parser.add_argument("--crsp-market-cache-path", help="Optional filtered CRSP daily market parquet cache")
    parser.add_argument(
        "--crsp-daily-root",
        help="Optional CRSP daily parquet folder. Defaults to the local canonical WRDS CRSP folder when present.",
    )
    parser.add_argument(
        "--allow-monthly-market-proxy",
        action="store_true",
        help="Allow the older monthly raw-timeseries proxy path when exact CRSP daily data is unavailable.",
    )
    parser.add_argument(
        "--companyfacts-root",
        help="Optional SEC companyfacts folder. Defaults to the local canonical companyfacts root when present.",
    )
    parser.add_argument(
        "--sec-filing-cache-root",
        default="/tmp/sec_filing_debt_cache",
        help="Optional SEC filing HTML cache for issuer-level shares fallback",
    )
    parser.add_argument(
        "--company-processing-timeout-seconds",
        type=float,
        default=30.0,
        help="Fail open on a single company if market-cap construction exceeds this timeout. Use 0 to disable.",
    )
    parser.add_argument(
        "--price-load-batch-size",
        type=int,
        default=32,
        help="Number of rows to batch together when loading CRSP/monthly price history. Smaller batches write earlier; larger batches reduce repeated parquet scans.",
    )
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument("--summary-out", help="Optional summary JSON path")
    return parser.parse_args()


