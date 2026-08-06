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


def _sec_debt_like_metrics(companyfacts: dict | None, as_of_date: str) -> dict[str, Any]:
    unsupported = {
        "debt_like": None,
        "debt_like_support": "unsupported",
        "debt_like_fallback": "sec_debt_like_unavailable",
    }
    if companyfacts is None:
        return unsupported

    total_debt, total_debt_support, _missing, _meta, _quality = core._build_sec_core_metric(
        "capital_structure.total_debt_provider_direct",
        companyfacts,
        as_of_date,
    )
    total_debt_support = _raw_support(total_debt, total_debt_support)

    lease_value, _lease_meta = seccomp._extract_lease_liabilities(companyfacts, as_of_date)
    lease_missing_reason = (
        "sec_concept_unavailable"
        if seccomp._has_any_us_gaap_concepts(
            companyfacts,
            seccomp.OPERATING_LEASE_ANY_CONCEPTS
            | seccomp.FINANCE_LEASE_ANY_CONCEPTS
            | seccomp.LEASE_AGGREGATE_TOTAL_EXACT_CONCEPTS,
        )
        else "sec_concept_absent"
    )

    lease_support = "unsupported"
    fallback = "sec_total_debt_plus_sec_lease_liabilities"
    if lease_value is not None:
        lease_support = "exact"
    elif lease_missing_reason == "sec_concept_absent":
        lease_value = 0.0
        lease_support = "exact"
        fallback = "sec_total_debt_plus_0_absent_lease"

    if total_debt is None or lease_value is None:
        return unsupported

    return {
        "debt_like": float(total_debt) + float(lease_value),
        "debt_like_support": _exact_or_proxy_support(total_debt_support, lease_support),
        "debt_like_fallback": fallback,
    }


def overlay_pit_safe_fundamentals(
    df: pd.DataFrame,
    companyfacts_root: Path,
    *,
    entity_identifier_path: Path | None = None,
    raw_timeseries_path: Path | None = None,
) -> pd.DataFrame:
    out = df.copy()
    out["company_id"] = out["company_id"].astype(str)

    cache: dict[str, dict | None] = {}
    permno_by_company: dict[str, str] = {}
    price_history_by_permno: dict[str, pd.DataFrame] = {}
    if entity_identifier_path and raw_timeseries_path:
        permno_by_company = _permno_map(entity_identifier_path)
        needed_permnos = [permno_by_company[cid] for cid in out["company_id"] if cid in permno_by_company]
        price_history_by_permno = _load_price_history(raw_timeseries_path, needed_permnos)

    raw_metric_columns = {
        "operating__revenue_ttm_sec__value": [],
        "operating__revenue_ttm_sec__unit": [],
        "operating__revenue_ttm_sec__support_mode": [],
        "operating__ebitda_ttm_sec__value": [],
        "operating__ebitda_ttm_sec__unit": [],
        "operating__ebitda_ttm_sec__support_mode": [],
        "operating__free_cash_flow_ttm_sec__value": [],
        "operating__free_cash_flow_ttm_sec__unit": [],
        "operating__free_cash_flow_ttm_sec__support_mode": [],
        "liquidity__cash_and_short_term_investments_sec__value": [],
        "liquidity__cash_and_short_term_investments_sec__unit": [],
        "liquidity__cash_and_short_term_investments_sec__support_mode": [],
        "market__market_cap_pit_safe__value": [],
        "market__market_cap_pit_safe__unit": [],
        "market__market_cap_pit_safe__support_mode": [],
    }

    new_metric_columns = {
        "capital_structure__debt_like_obligations_normalized__value": [],
        "capital_structure__debt_like_obligations_normalized__unit": [],
        "capital_structure__debt_like_obligations_normalized__support_mode": [],
        "capital_structure__debt_like_obligations_normalized__fallback_used": [],
        "liquidity__cash_and_short_term_investments_provider_direct__value": [],
        "liquidity__cash_and_short_term_investments_provider_direct__unit": [],
        "liquidity__cash_and_short_term_investments_provider_direct__support_mode": [],
        "market__market_cap_provider_direct__value": [],
        "market__market_cap_provider_direct__unit": [],
        "market__market_cap_provider_direct__support_mode": [],
        "market__enterprise_value__value": [],
        "market__enterprise_value__unit": [],
        "market__enterprise_value__support_mode": [],
        "market__enterprise_value__fallback_used": [],
        "operating__ebitda_margin_ttm__value": [],
        "operating__ebitda_margin_ttm__support_mode": [],
        "operating__ebitda_margin_ttm__fallback_used": [],
        "market__ev_ebitda__value": [],
        "market__ev_ebitda__support_mode": [],
        "market__ev_ebitda__fallback_used": [],
        "market__fcf_yield__value": [],
        "market__fcf_yield__support_mode": [],
        "market__fcf_yield__fallback_used": [],
        "operating__fcf_conversion__value": [],
        "operating__fcf_conversion__support_mode": [],
        "operating__fcf_conversion__fallback_used": [],
    }

    for row in out.itertuples(index=False):
        company_id = str(row.company_id)
        as_of_time = str(row.as_of_time)
        as_of_date = as_of_time[:10]
        companyfacts_metrics = _companyfacts_metrics(
            company_id=company_id,
            as_of_date=as_of_date,
            companyfacts_root=companyfacts_root,
            cache=cache,
        )
        companyfacts = cache.get(company_id)
        pit_market_cap_metrics = _pit_market_cap_metrics(
            company_id=company_id,
            as_of_time=as_of_time,
            companyfacts=companyfacts,
            permno_by_company=permno_by_company,
            price_history_by_permno=price_history_by_permno,
        )

        revenue = companyfacts_metrics["sec_revenue_ttm"]
        revenue_support = companyfacts_metrics["sec_revenue_support"]
        ebitda = companyfacts_metrics["sec_ebitda_ttm"]
        ebitda_support = companyfacts_metrics["sec_ebitda_support"]
        fcf = companyfacts_metrics["sec_fcf_ttm"]
        fcf_support = companyfacts_metrics["sec_fcf_support"]
        sec_cash_direct = companyfacts_metrics["sec_cash_direct"]
        sec_cash_direct_support = companyfacts_metrics["sec_cash_direct_support"]
        sec_cash_sti = companyfacts_metrics["sec_cash_sti"]
        sec_cash_sti_support = companyfacts_metrics["sec_cash_sti_support"]
        pit_market_cap = pit_market_cap_metrics["market_cap"]
        pit_market_cap_support = pit_market_cap_metrics["market_cap_support"]
        marketable_securities = _maybe_float(
            getattr(row, "liquidity__marketable_securities_sec_exact__value", None)
        )
        marketable_securities_support = str(
            getattr(row, "liquidity__marketable_securities_sec_exact__support_mode", None) or "unsupported"
        )
        rebuilt_debt_like = _sec_debt_like_metrics(companyfacts, as_of_date)
        existing_debt_like = _maybe_float(
            getattr(row, "capital_structure__debt_like_obligations_normalized__value", None)
        )
        existing_debt_like_support = str(
            getattr(row, "capital_structure__debt_like_obligations_normalized__support_mode", None) or "unsupported"
        )
        existing_debt_like_fallback = str(
            getattr(row, "capital_structure__debt_like_obligations_normalized__fallback_used", None)
            or "existing_flat_value"
        )
        debt_like_value = existing_debt_like
        debt_like_support = existing_debt_like_support
        debt_like_fallback = existing_debt_like_fallback
        if (
            rebuilt_debt_like["debt_like"] is not None
            and _support_rank(rebuilt_debt_like["debt_like_support"]) >= _support_rank(existing_debt_like_support)
        ):
            debt_like_value = rebuilt_debt_like["debt_like"]
            debt_like_support = rebuilt_debt_like["debt_like_support"]
            debt_like_fallback = rebuilt_debt_like["debt_like_fallback"]

        repaired_cash_sti = sec_cash_sti
        repaired_cash_sti_support = sec_cash_sti_support
        if sec_cash_direct is not None and marketable_securities is not None:
            repaired_cash_sti = sec_cash_direct + marketable_securities
            repaired_cash_sti_support = _exact_or_proxy_support(
                sec_cash_direct_support,
                marketable_securities_support,
            )
        elif (
            sec_cash_direct is not None
            and sec_cash_direct_support == "exact"
            and _can_promote_cash_only_exact(companyfacts, as_of_date)
        ):
            repaired_cash_sti = sec_cash_direct
            repaired_cash_sti_support = "exact"

        raw_metric_columns["operating__revenue_ttm_sec__value"].append(revenue)
        raw_metric_columns["operating__revenue_ttm_sec__unit"].append("usd")
        raw_metric_columns["operating__revenue_ttm_sec__support_mode"].append(revenue_support)
        raw_metric_columns["operating__ebitda_ttm_sec__value"].append(ebitda)
        raw_metric_columns["operating__ebitda_ttm_sec__unit"].append("usd")
        raw_metric_columns["operating__ebitda_ttm_sec__support_mode"].append(ebitda_support)
        raw_metric_columns["operating__free_cash_flow_ttm_sec__value"].append(fcf)
        raw_metric_columns["operating__free_cash_flow_ttm_sec__unit"].append("usd")
        raw_metric_columns["operating__free_cash_flow_ttm_sec__support_mode"].append(fcf_support)
        raw_metric_columns["liquidity__cash_and_short_term_investments_sec__value"].append(repaired_cash_sti)
        raw_metric_columns["liquidity__cash_and_short_term_investments_sec__unit"].append("usd")
        raw_metric_columns["liquidity__cash_and_short_term_investments_sec__support_mode"].append(
            repaired_cash_sti_support
        )
        raw_metric_columns["market__market_cap_pit_safe__value"].append(pit_market_cap)
        raw_metric_columns["market__market_cap_pit_safe__unit"].append("usd")
        raw_metric_columns["market__market_cap_pit_safe__support_mode"].append(pit_market_cap_support)

        new_metric_columns["capital_structure__debt_like_obligations_normalized__value"].append(debt_like_value)
        new_metric_columns["capital_structure__debt_like_obligations_normalized__unit"].append("usd")
        new_metric_columns["capital_structure__debt_like_obligations_normalized__support_mode"].append(
            debt_like_support
        )
        new_metric_columns["capital_structure__debt_like_obligations_normalized__fallback_used"].append(
            debt_like_fallback
        )

        if revenue not in (None, 0) and ebitda is not None:
            new_metric_columns["operating__ebitda_margin_ttm__value"].append(float(ebitda) / float(revenue))
            new_metric_columns["operating__ebitda_margin_ttm__support_mode"].append(
                _derived_support(revenue_support, ebitda_support)
            )
            new_metric_columns["operating__ebitda_margin_ttm__fallback_used"].append(
                "sec_companyfacts_ttm_ebitda_over_revenue"
            )
        else:
            new_metric_columns["operating__ebitda_margin_ttm__value"].append(None)
            new_metric_columns["operating__ebitda_margin_ttm__support_mode"].append("unsupported")
            new_metric_columns["operating__ebitda_margin_ttm__fallback_used"].append("sec_companyfacts_ttm_unavailable")

        total_debt = _maybe_float(getattr(row, "capital_structure__total_debt_provider_direct__value", None))
        total_debt_support = str(
            getattr(row, "capital_structure__total_debt_provider_direct__support_mode", None) or "unsupported"
        )
        debt_for_ev = debt_like_value if debt_like_value is not None else total_debt
        debt_for_ev_support = debt_like_support if debt_like_value is not None else total_debt_support
        repaired_enterprise_value = None
        repaired_enterprise_value_support = "unsupported"
        repaired_enterprise_value_fallback = "pit_components_unavailable"
        if pit_market_cap is not None and debt_for_ev is not None and repaired_cash_sti is not None:
            repaired_enterprise_value = pit_market_cap + debt_for_ev - repaired_cash_sti
            repaired_enterprise_value_support = _exact_or_proxy_support(
                pit_market_cap_support,
                debt_for_ev_support,
                repaired_cash_sti_support,
            )
            repaired_enterprise_value_fallback = (
                "pit_market_cap_plus_debt_like_minus_sec_cash_sti"
                if debt_like_value is not None
                else "pit_market_cap_plus_total_debt_minus_sec_cash_sti"
            )

        new_metric_columns["liquidity__cash_and_short_term_investments_provider_direct__value"].append(
            repaired_cash_sti
        )
        new_metric_columns["liquidity__cash_and_short_term_investments_provider_direct__unit"].append("usd")
        new_metric_columns["liquidity__cash_and_short_term_investments_provider_direct__support_mode"].append(
            repaired_cash_sti_support
        )
        new_metric_columns["market__market_cap_provider_direct__value"].append(pit_market_cap)
        new_metric_columns["market__market_cap_provider_direct__unit"].append("usd")
        new_metric_columns["market__market_cap_provider_direct__support_mode"].append(pit_market_cap_support)
        new_metric_columns["market__enterprise_value__value"].append(repaired_enterprise_value)
        new_metric_columns["market__enterprise_value__unit"].append("usd")
        new_metric_columns["market__enterprise_value__support_mode"].append(repaired_enterprise_value_support)
        new_metric_columns["market__enterprise_value__fallback_used"].append(repaired_enterprise_value_fallback)

        if repaired_enterprise_value is not None and ebitda not in (None, 0) and float(ebitda) > 0:
            new_metric_columns["market__ev_ebitda__value"].append(float(repaired_enterprise_value) / float(ebitda))
            new_metric_columns["market__ev_ebitda__support_mode"].append(
                _exact_or_proxy_support(repaired_enterprise_value_support, ebitda_support)
            )
            new_metric_columns["market__ev_ebitda__fallback_used"].append(
                "sec_companyfacts_ttm_ebitda_over_pit_safe_enterprise_value"
            )
        else:
            new_metric_columns["market__ev_ebitda__value"].append(None)
            new_metric_columns["market__ev_ebitda__support_mode"].append("unsupported")
            new_metric_columns["market__ev_ebitda__fallback_used"].append("sec_companyfacts_ttm_unavailable")

        if pit_market_cap not in (None, 0) and fcf is not None:
            new_metric_columns["market__fcf_yield__value"].append(float(fcf) / float(pit_market_cap))
            new_metric_columns["market__fcf_yield__support_mode"].append(
                _exact_or_proxy_support(pit_market_cap_support, fcf_support)
            )
            new_metric_columns["market__fcf_yield__fallback_used"].append(
                "sec_companyfacts_ttm_fcf_over_pit_safe_market_cap"
            )
        else:
            new_metric_columns["market__fcf_yield__value"].append(None)
            new_metric_columns["market__fcf_yield__support_mode"].append("unsupported")
            new_metric_columns["market__fcf_yield__fallback_used"].append("sec_companyfacts_ttm_unavailable")

        if fcf is not None and ebitda not in (None, 0):
            new_metric_columns["operating__fcf_conversion__value"].append(float(fcf) / float(ebitda))
            new_metric_columns["operating__fcf_conversion__support_mode"].append(
                _derived_support(fcf_support, ebitda_support)
            )
            new_metric_columns["operating__fcf_conversion__fallback_used"].append(
                "sec_companyfacts_ttm_fcf_over_ebitda"
            )
        else:
            new_metric_columns["operating__fcf_conversion__value"].append(None)
            new_metric_columns["operating__fcf_conversion__support_mode"].append("unsupported")
            new_metric_columns["operating__fcf_conversion__fallback_used"].append("sec_companyfacts_ttm_unavailable")

    for column, values in raw_metric_columns.items():
        out[column] = values
    for column, values in new_metric_columns.items():
        out[column] = values

    return out


def main() -> None:
    args = parse_args()
    flat_path = Path(args.flat_path)
    companyfacts_root = Path(args.companyfacts_root)
    entity_identifier_path = Path(args.entity_identifier_path) if args.entity_identifier_path else None
    raw_timeseries_path = Path(args.raw_timeseries_path) if args.raw_timeseries_path else None
    out_parquet = Path(args.out_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(flat_path)
    repaired = overlay_pit_safe_fundamentals(
        df,
        companyfacts_root,
        entity_identifier_path=entity_identifier_path,
        raw_timeseries_path=raw_timeseries_path,
    )
    repaired.to_parquet(out_parquet, index=False)

    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        repaired.to_csv(out_csv, index=False)

    if args.summary_out:
        summary = {
            "rows": int(len(repaired)),
            "capital_structure.debt_like_obligations_normalized": _support_counts(
                repaired["capital_structure__debt_like_obligations_normalized__support_mode"]
            ),
            "operating.ebitda_margin_ttm": _support_counts(repaired["operating__ebitda_margin_ttm__support_mode"]),
            "liquidity.cash_and_short_term_investments_provider_direct": _support_counts(
                repaired["liquidity__cash_and_short_term_investments_provider_direct__support_mode"]
            ),
            "market.market_cap_provider_direct": _support_counts(
                repaired["market__market_cap_provider_direct__support_mode"]
            ),
            "market.ev_ebitda": _support_counts(repaired["market__ev_ebitda__support_mode"]),
            "market.fcf_yield": _support_counts(repaired["market__fcf_yield__support_mode"]),
            "operating.fcf_conversion": _support_counts(repaired["operating__fcf_conversion__support_mode"]),
            "operating.ebitda_ttm_sec": _support_counts(repaired["operating__ebitda_ttm_sec__support_mode"]),
            "operating.free_cash_flow_ttm_sec": _support_counts(
                repaired["operating__free_cash_flow_ttm_sec__support_mode"]
            ),
        }
        sample_ids = [
            "0000006201",
            "0000909832",
            "0001637459",
            "0000003453",
            "0000003197",
            "0001639438",
            "0000001750",
            "0000004281",
        ]
        sample = {}
        for cid in sample_ids:
            sub = repaired[repaired["company_id"] == cid]
            if sub.empty:
                continue
            row = sub.iloc[0]
            sample[cid] = {
                "company_name": row["company_name"],
                "cash_and_short_term_investments_sec": _json_scalar(
                    row.get("liquidity__cash_and_short_term_investments_sec__value")
                ),
                "market_cap_pit_safe": _json_scalar(row.get("market__market_cap_pit_safe__value")),
                "enterprise_value": _json_scalar(row.get("market__enterprise_value__value")),
                "ebitda_ttm_sec": _json_scalar(row.get("operating__ebitda_ttm_sec__value")),
                "ebitda_ttm_sec_support_mode": _json_scalar(row.get("operating__ebitda_ttm_sec__support_mode")),
                "free_cash_flow_ttm_sec": _json_scalar(row.get("operating__free_cash_flow_ttm_sec__value")),
                "free_cash_flow_ttm_sec_support_mode": _json_scalar(
                    row.get("operating__free_cash_flow_ttm_sec__support_mode")
                ),
                "ebitda_margin_ttm": _json_scalar(row.get("operating__ebitda_margin_ttm__value")),
                "ev_ebitda": _json_scalar(row.get("market__ev_ebitda__value")),
                "fcf_yield": _json_scalar(row.get("market__fcf_yield__value")),
                "fcf_conversion": _json_scalar(row.get("operating__fcf_conversion__value")),
            }
        summary["sample"] = sample
        Path(args.summary_out).write_text(json.dumps(summary, indent=2))

    print(f"Overlayed PIT-safe SEC fundamentals -> {out_parquet}")


if __name__ == "__main__":
    main()
