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


def _maybe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _pit_market_cap_metrics(
    *,
    company_id: str,
    as_of_time: str,
    companyfacts: dict | None,
    permno_by_company: dict[str, str],
    price_history_by_permno: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    unsupported = {
        "market_cap": None,
        "market_cap_support": "unsupported",
        "market_cap_fallback": "pit_market_cap_unavailable",
        "market_cap_components": None,
    }
    if companyfacts is None:
        return unsupported

    permno = permno_by_company.get(company_id)
    if not permno:
        return unsupported
    price_history = price_history_by_permno.get(permno)
    if price_history is None or price_history.empty:
        return unsupported

    as_of_ts = pd.Timestamp(as_of_time)
    if as_of_ts.tzinfo is None:
        as_of_ts = as_of_ts.tz_localize("UTC")
    else:
        as_of_ts = as_of_ts.tz_convert("UTC")
    as_of_date = as_of_ts.normalize()
    current_row = _latest_row_on_or_before(price_history, as_of_date)
    if current_row is None:
        return unsupported

    current_trade_date = current_row["trade_date"]
    close_price = _maybe_float(current_row.get("close"))
    if close_price is None:
        return unsupported
    if not market_macro._recent_enough_trade_date(current_trade_date, as_of_date):
        return {
            "market_cap": None,
            "market_cap_support": "unsupported",
            "market_cap_fallback": "market_timeseries_stale",
            "market_cap_components": {
                "permno": permno,
                "trade_date": str(current_trade_date.date()),
                "close_price": close_price,
            },
        }

    shares_out, shares_meta = market_macro._latest_shares_outstanding(companyfacts, as_of_time[:10])
    if shares_out is None:
        return unsupported

    reference_date = None
    if shares_meta and shares_meta.get("end"):
        try:
            reference_date = date.fromisoformat(str(shares_meta["end"]))
        except ValueError:
            reference_date = None
    shares_support, shares_missing_reason = market_macro._shares_support_mode(
        reference_date=reference_date,
        as_of_date=as_of_date.date(),
    )
    support_mode = _exact_or_proxy_support("exact", shares_support)
    return {
        "market_cap": close_price * float(shares_out),
        "market_cap_support": support_mode,
        "market_cap_fallback": "raw_timeseries_close_x_companyfacts_shares",
        "market_cap_components": {
            "permno": permno,
            "trade_date": str(current_trade_date.date()),
            "close_price": close_price,
            "shares_outstanding": float(shares_out),
            "shares_support_mode": shares_support,
            "shares_missing_reason": shares_missing_reason,
            "shares_meta": shares_meta,
            "formula": "close_price * latest_shares_outstanding_on_or_before_asof",
        },
    }


def _sec_cash_direct_metric(companyfacts: dict | None, as_of_date: str) -> tuple[float | None, str]:
    if companyfacts is None:
        return None, "unsupported"
    candidates = core._instant_candidates(
        companyfacts,
        core.CASH_CONCEPTS,
        as_of_date=as_of_date,
        unit_filter="USD",
    )
    if not candidates:
        return None, "unsupported"
    candidate = candidates[0]
    age_days = (date.fromisoformat(as_of_date) - candidate["end_dt"]).days
    support_mode = "exact" if age_days <= core.EXACT_BALANCE_SHEET_MAX_AGE_DAYS else "proxy_missing_component"
    return float(candidate["value"]), support_mode


def _can_promote_cash_only_exact(companyfacts: dict | None, as_of_date: str) -> bool:
    if companyfacts is None:
        return False
    combined_candidates = core._instant_candidates(
        companyfacts,
        core.COMBINED_CASH_STI_CONCEPTS,
        as_of_date=as_of_date,
        unit_filter="USD",
    )
    sti_candidates = core._instant_candidates(
        companyfacts,
        core.STI_CONCEPTS,
        as_of_date=as_of_date,
        unit_filter="USD",
    )
    return not combined_candidates and not sti_candidates


def _companyfacts_metrics(
    *,
    company_id: str,
    as_of_date: str,
    companyfacts_root: Path,
    cache: dict[str, dict | None],
) -> dict[str, Any]:
    if company_id not in cache:
        cache[company_id] = core._load_companyfacts(companyfacts_root / f"CIK{company_id}.json")
    companyfacts = cache[company_id]
    if companyfacts is None:
        return {
            "sec_revenue_ttm": None,
            "sec_revenue_support": "unsupported",
            "sec_ebitda_ttm": None,
            "sec_ebitda_support": "unsupported",
            "sec_fcf_ttm": None,
            "sec_fcf_support": "unsupported",
            "sec_cash_sti": None,
            "sec_cash_sti_support": "unsupported",
        }

    revenue, revenue_support, _missing, _breakdown, _quality = core._build_sec_core_metric(
        "operating.revenue_ttm_provider_direct",
        companyfacts,
        as_of_date,
    )
    ebitda, ebitda_support, _missing, _breakdown, _quality = core._build_sec_core_metric(
        "operating.ebitda_ltm_provider_direct",
        companyfacts,
        as_of_date,
    )
    fcf, _ocf, _capex, _fcf_breakdown = cashflow._repairable_fcf_inputs(
        companyfacts=companyfacts,
        as_of_date=as_of_date,
    )
    cash_direct, cash_direct_support = _sec_cash_direct_metric(companyfacts, as_of_date)
    cash_sti, cash_sti_support, _missing, _breakdown, _quality = core._build_sec_core_metric(
        "liquidity.cash_and_short_term_investments_provider_direct",
        companyfacts,
        as_of_date,
    )

    return {
        "sec_revenue_ttm": revenue,
        "sec_revenue_support": _raw_support(revenue, revenue_support),
        "sec_ebitda_ttm": ebitda,
        "sec_ebitda_support": _raw_support(ebitda, ebitda_support),
        "sec_fcf_ttm": fcf,
        "sec_fcf_support": "exact" if fcf is not None else "unsupported",
        "sec_cash_direct": cash_direct,
        "sec_cash_direct_support": cash_direct_support,
        "sec_cash_sti": cash_sti,
        "sec_cash_sti_support": _raw_support(cash_sti, cash_sti_support),
    }


