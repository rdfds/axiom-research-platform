#!/usr/bin/env python3
"""Build a lightweight quantitative peer-screen view from the core market-grade export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flat-path", required=True, help="Input core-market-grade parquet path")
    parser.add_argument("--out-parquet", required=True, help="Output parquet path")
    parser.add_argument("--out-csv", help="Optional output CSV path")
    parser.add_argument("--summary-out", help="Optional output summary JSON path")
    return parser.parse_args()


def _support_counts(series: pd.Series) -> Dict[str, int]:
    support = series.fillna("unsupported").astype(str)
    return {str(k): int(v) for k, v in support.value_counts(dropna=False).to_dict().items()}


def _convert_percent(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce") * 100.0


def _convert_bps(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce") * 10000.0


def build_peer_screen_view(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "company_id": df["company_id"].astype(str),
            "company_name": df["company_name"],
            "available_liquidity_usd": pd.to_numeric(
                df["liquidity__available_liquidity_normalized__value"], errors="coerce"
            ),
            "available_liquidity_support_mode": df["liquidity__available_liquidity_normalized__support_mode"],
            "debt_like_usd": pd.to_numeric(
                df["capital_structure__debt_like_obligations_normalized__value"], errors="coerce"
            ),
            "debt_like_support_mode": df["capital_structure__debt_like_obligations_normalized__support_mode"],
            "net_debt_usd": pd.to_numeric(df["capital_structure__net_debt_normalized__value"], errors="coerce"),
            "net_debt_support_mode": df["capital_structure__net_debt_normalized__support_mode"],
            "gross_leverage_x": pd.to_numeric(
                df["capital_structure__gross_leverage_normalized__value"], errors="coerce"
            ),
            "gross_leverage_support_mode": df["capital_structure__gross_leverage_normalized__support_mode"],
            "net_leverage_x": pd.to_numeric(
                df["capital_structure__net_leverage_normalized__value"], errors="coerce"
            ),
            "net_leverage_support_mode": df["capital_structure__net_leverage_normalized__support_mode"],
            "revenue_yoy_last_q_pct": _convert_percent(df["operating__revenue_yoy_last_q__value"]),
            "revenue_yoy_last_q_support_mode": df["operating__revenue_yoy_last_q__support_mode"],
            "revenue_cagr_3y_pct": _convert_percent(df["operating__revenue_cagr_3y__value"]),
            "revenue_cagr_3y_support_mode": df["operating__revenue_cagr_3y__support_mode"],
            "ebitda_margin_ttm_pct": _convert_percent(df["operating__ebitda_margin_ttm__value"]),
            "ebitda_margin_ttm_support_mode": df["operating__ebitda_margin_ttm__support_mode"],
            "ev_ebitda_x": pd.to_numeric(df["market__ev_ebitda__value"], errors="coerce"),
            "ev_ebitda_support_mode": df["market__ev_ebitda__support_mode"],
            "fcf_yield_pct": _convert_percent(df["market__fcf_yield__value"]),
            "fcf_yield_support_mode": df["market__fcf_yield__support_mode"],
            "fcf_conversion_pct": _convert_percent(df["operating__fcf_conversion__value"]),
            "fcf_conversion_support_mode": df["operating__fcf_conversion__support_mode"],
            "rating": df["capital_structure__rating_state__rating"],
            "rating_support_mode": df["capital_structure__rating_state__rating_support_mode"],
            "rating_score": pd.to_numeric(df["capital_structure__rating_state__score__value"], errors="coerce"),
            "rating_score_support_mode": df["capital_structure__rating_state__score__support_mode"],
            "credit_spread_bps": _convert_bps(df["market__credit_spread_level__value"]),
            "credit_spread_support_mode": df["market__credit_spread_level__support_mode"],
            "credit_truth_tier": df["market__credit_truth_tier"],
            "credit_truth_tier_rank": pd.to_numeric(df["market__credit_truth_tier_rank__value"], errors="coerce"),
            "credit_spread_percentile_2y_pct": _convert_percent(
                df["market__credit_spread_percentile_2y__value"]
            ),
            "credit_spread_percentile_2y_support_mode": df[
                "market__credit_spread_percentile_2y__support_mode"
            ],
        }
    )
    return out


