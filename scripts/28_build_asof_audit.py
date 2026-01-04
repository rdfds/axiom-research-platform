#!/usr/bin/env python
"""
As-Of Coverage Audit
====================
Generates coverage reports for warehouse tables:
- Missing required fields
- Missing months (prices)
- Missing sizes by action type (corporate actions)
"""

from __future__ import annotations

import argparse
import os
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import sys

try:
    import duckdb  # type: ignore
except Exception:  # pragma: no cover
    duckdb = None

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.asof_store import AsOfWarehouse


DATA_DIR = Path(__file__).parent.parent / "data"
WAREHOUSE_DIR = DATA_DIR / "warehouse"


REQUIRED_FIELDS = {
    "warehouse_financials": [
        "company_id",
        "event_time",
        "available_time",
        "statement_type",
        "line_item",
        "value",
        "fiscal_period_end",
        "fiscal_year",
        "fiscal_quarter",
    ],
    "warehouse_prices": ["entity_id", "event_time", "available_time", "close", "adjusted_close", "volume"],
    "warehouse_prices_daily": [
        "entity_id",
        "event_time",
        "available_time",
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
        "total_return_index",
    ],
    "warehouse_prices_daily_rdp": [
        "entity_id",
        "event_time",
        "available_time",
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
        "total_return_index",
    ],
    "warehouse_macro": [
        "entity_id",
        "event_time",
        "available_time",
        "instrument_id",
        "instrument_type",
        "tenor",
        "value",
        "units",
    ],
    "warehouse_estimates": [
        "entity_id",
        "event_time",
        "available_time",
        "metric",
        "period",
        "consensus_value",
        # "num_estimates" is optional for some sources (e.g., FMP).
        # Set AUDIT_ESTIMATES_REQUIRE_NUM=1 to enforce it.
    ],
    "warehouse_press_releases": [
        "entity_id",
        "event_time",
        "available_time",
        "document_id",
        "release_date",
        "headline",
        "text",
    ],
    "warehouse_corp_actions": ["entity_id", "event_time", "available_time", "action_type", "announcement_date", "effective_date", "size"],
    "warehouse_mna_deals": ["deal_id", "announcement_date", "deal_value", "status"],
    "warehouse_13f_holdings": [
        "company_id",
        "event_time",
        "available_time",
        "holding_cusip",
        "value_k",
        "shares",
    ],
    "warehouse_13f_filings": [
        "company_id",
        "event_time",
        "available_time",
    ],
}

if os.getenv("AUDIT_ESTIMATES_REQUIRE_NUM", "0") in ("1", "true", "True"):
    REQUIRED_FIELDS["warehouse_estimates"].append("num_estimates")


def log(msg: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}")


def missingness(df: pd.DataFrame, fields: List[str]) -> Dict[str, float]:
    stats = {}
    total = len(df)
    if total == 0:
        return {f"missing_{f}_pct": 1.0 for f in fields}
    for field in fields:
        if field not in df.columns:
            stats[f"missing_{field}_pct"] = 1.0
        else:
            stats[f"missing_{field}_pct"] = float(df[field].isna().mean())
    return stats


def months_between(start: pd.Timestamp, end: pd.Timestamp) -> int:
    if pd.isna(start) or pd.isna(end):
        return 0
    return (end.year - start.year) * 12 + (end.month - start.month) + 1


def audit_prices(df: pd.DataFrame) -> Dict[str, float]:
    df = df.copy()
    df["event_time"] = pd.to_datetime(df["event_time"])
    coverage = df.groupby("entity_id")["event_time"].agg(["min", "max", "nunique"]).reset_index()
    coverage["expected_months"] = coverage.apply(lambda r: months_between(r["min"], r["max"]), axis=1)
    coverage["missing_months"] = coverage["expected_months"] - coverage["nunique"]
    coverage["missing_months"] = coverage["missing_months"].clip(lower=0)

    total_missing = coverage["missing_months"].sum()
    avg_missing = coverage["missing_months"].mean()
    coverage_pct = (coverage["nunique"].sum() / coverage["expected_months"].sum()) if coverage["expected_months"].sum() else 0
    pct_with_gaps = (coverage["missing_months"] > 0).mean()

    coverage_path = WAREHOUSE_DIR / "coverage_prices_missing_months.csv"
    coverage.to_csv(coverage_path, index=False)

    return {
        "price_missing_months_total": float(total_missing),
        "price_missing_months_avg": float(avg_missing),
        "price_coverage_pct": float(coverage_pct),
        "price_entities_with_gaps_pct": float(pct_with_gaps),
    }


def audit_corp_actions(df: pd.DataFrame) -> None:
    if "action_type" not in df.columns or "size" not in df.columns:
        return
    summary = (
        df.groupby("action_type")["size"]
        .apply(lambda s: float(s.isna().mean()))
        .reset_index()
        .rename(columns={"size": "missing_size_pct"})
    )
    summary.to_csv(WAREHOUSE_DIR / "coverage_corp_actions_missing_size_by_type.csv", index=False)


