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


def _support_rank(support_mode: str | None) -> int:
    if support_mode == "exact":
        return 2
    if support_mode == "proxy_missing_component":
        return 1
    return 0


def _permno_map(entity_identifier_path: Path) -> dict[str, str]:
    ids = pd.read_parquet(entity_identifier_path)
    ids = ids[ids["identifier_type"].astype(str).str.lower() == "permno"].copy()
    ids["permno"] = ids["identifier_value"].astype(str).str.strip()
    return {
        str(entity_id): permno
        for entity_id, permno in ids[["entity_id", "permno"]].drop_duplicates().itertuples(index=False)
    }


def _load_price_history(raw_timeseries_path: Path, permnos: list[str]) -> dict[str, pd.DataFrame]:
    if not permnos:
        return {}
    permno_sql = ",".join(f"'{permno}'" for permno in sorted(set(permnos)))
    query = f"""
        SELECT
            CAST(entity_id AS VARCHAR) AS permno,
            CAST(trade_date AS DATE) AS trade_date,
            close
        FROM read_parquet('{raw_timeseries_path}')
        WHERE series_type = 'price'
          AND CAST(entity_id AS VARCHAR) IN ({permno_sql})
    """
    prices = duckdb.sql(query).fetchdf()
    if prices.empty:
        return {}
    prices["trade_date"] = pd.to_datetime(prices["trade_date"], utc=True).dt.normalize()
    prices["date_key"] = prices["trade_date"]
    prices = prices.sort_values(["permno", "trade_date"]).drop_duplicates(["permno", "trade_date"], keep="last")
    return {
        permno: frame.reset_index(drop=True)
        for permno, frame in prices.groupby("permno")
    }


def _latest_row_on_or_before(df: pd.DataFrame, date_key: pd.Timestamp) -> pd.Series | None:
    eligible = df[df["date_key"] <= date_key]
    if eligible.empty:
        return None
    return eligible.iloc[-1]


