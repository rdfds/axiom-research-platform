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


def _permno_map(entity_identifier_path: Path) -> dict[str, str]:
    ids = pd.read_parquet(entity_identifier_path)
    ids = ids[ids["identifier_type"].astype(str).str.lower() == "permno"].copy()
    ids["permno"] = ids["identifier_value"].astype(str).str.strip()
    return {
        str(entity_id): permno
        for entity_id, permno in ids[["entity_id", "permno"]].drop_duplicates().itertuples(index=False)
    }


def _load_monthly_price_history(raw_timeseries_path: Path, permnos: list[str]) -> dict[str, pd.DataFrame]:
    if not permnos:
        return {}
    permno_sql = ",".join(f"'{permno}'" for permno in sorted(set(permnos)))
    query = f"""
        SELECT
            CAST(entity_id AS VARCHAR) AS permno,
            CAST(trade_date AS DATE) AS trade_date,
            close,
            ret,
            retx
        FROM read_parquet('{raw_timeseries_path}')
        WHERE series_type = 'price'
          AND CAST(entity_id AS VARCHAR) IN ({permno_sql})
    """
    prices = duckdb.sql(query).fetchdf()
    prices["trade_date"] = pd.to_datetime(prices["trade_date"], utc=True).dt.normalize()
    prices = prices.sort_values(["permno", "trade_date"]).drop_duplicates(["permno", "trade_date"], keep="last")
    return {
        permno: frame.reset_index(drop=True)
        for permno, frame in prices.groupby("permno")
    }


def _latest_row_on_or_before(df: pd.DataFrame, date_key: pd.Timestamp) -> pd.Series | None:
    eligible = df[df["trade_date"] <= date_key]
    if eligible.empty:
        return None
    return eligible.iloc[-1]


def _compound_trailing_return(price_history: pd.DataFrame, as_of_ts: pd.Timestamp, months: int) -> tuple[float | None, dict[str, Any], str | None]:
    current_row = _latest_row_on_or_before(price_history, as_of_ts)
    if current_row is None:
        return None, {}, "market_timeseries_unavailable"
    current_trade_date = current_row["trade_date"]
    target_date = current_trade_date - pd.DateOffset(months=months)
    window = price_history[(price_history["trade_date"] > target_date) & (price_history["trade_date"] <= current_trade_date)].copy()
    if window.empty:
        return None, {
            "current_trade_date": str(current_trade_date.date()),
            "target_trade_date": str(target_date.date()),
        }, "market_timeseries_unavailable"
    if window["ret"].notna().all():
        compounded = float((1.0 + window["ret"].astype(float)).prod() - 1.0)
        return compounded, {
            "current_close": None if pd.isna(current_row["close"]) else float(current_row["close"]),
            "lookback_months": months,
            "current_trade_date": str(current_trade_date.date()),
            "target_trade_date": str(target_date.date()),
            "periods_used": int(len(window)),
            "return_column": "ret",
            "formula": "compound_ret_over_monthly_window",
        }, None
    if window["retx"].notna().all():
        compounded = float((1.0 + window["retx"].astype(float)).prod() - 1.0)
        return compounded, {
            "current_close": None if pd.isna(current_row["close"]) else float(current_row["close"]),
            "lookback_months": months,
            "current_trade_date": str(current_trade_date.date()),
            "target_trade_date": str(target_date.date()),
            "periods_used": int(len(window)),
            "return_column": "retx",
            "formula": "compound_retx_over_monthly_window",
        }, None
    return None, {
        "current_trade_date": str(current_trade_date.date()),
        "target_trade_date": str(target_date.date()),
        "periods_used": int(len(window)),
    }, "market_return_series_unavailable"


def _normalize_quality_flags(flags: Any, new_flag: str | None = None) -> list[str] | None:
    values = [str(flag) for flag in (flags or []) if str(flag)]
    if new_flag and new_flag not in values:
        values.append(new_flag)
    return values or None


def _demote_metric(feature: dict[str, Any], *, missing_reason: str) -> None:
    feature["value"] = None
    feature["confidence"] = None
    feature["support_mode"] = "unsupported"
    feature["missing_reason"] = missing_reason
    feature["quality_flags"] = _normalize_quality_flags(feature.get("quality_flags"), missing_reason)


def _repair_negative_revenue(features: dict[str, Any]) -> None:
    revenue = features.get("operating.revenue_ttm_provider_direct")
    if not revenue:
        return
    value = revenue.get("value")
    if value is None or float(value) >= 0:
        return
    _demote_metric(revenue, missing_reason="negative_ttm_revenue_rejected")
    breakdown = revenue.get("component_breakdown") or {}
    breakdown["guard_reason"] = "negative_ttm_revenue_rejected"
    revenue["component_breakdown"] = breakdown
    for metric_name in [
        "operating.ebitda_margin_standardized",
        "earnings.net_margin_standardized",
    ]:
        feature = features.get(metric_name)
        if feature:
            _demote_metric(feature, missing_reason="dependent_on_negative_revenue")


def _exact_market_repairs(
    *,
    permno: str | None,
    price_history: pd.DataFrame | None,
    as_of_time: str,
    computed_at: str,
    provenance_source: str,
) -> dict[str, dict[str, Any]]:
    if _build_price_metrics_from_crsp is None:
        return {}
    return _build_price_metrics_from_crsp(
        permno=permno,
        price_history=price_history,
        as_of_time=as_of_time,
        computed_at=computed_at,
        provenance_source=provenance_source,
    )


