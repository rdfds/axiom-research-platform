#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import date
import duckdb
from functools import lru_cache
import gzip
import json
import math
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("RECO_DISABLE_PRECEDENT_NARRATIVE", "1")
sys.path.insert(0, str(REPO_ROOT))

from src.model_feature_bundle import _STATE_VECTOR_V1_FEATURES, build_model_feature_bundle
from src.pipeline.historical_price_metric_backfill import backfill_historical_price_window_metrics
from src.pipeline.precedent_brain import (
    _PRECEDENT_DISTANCE_WEIGHTS_CACHE,
    _PRECEDENT_DISTANCE_V2_WEIGHTS_CACHE,
    _enrich_missing_historical_taxonomy,
    _effective_action_subtype,
    _historical_taxonomy_for_ticker,
    augment_precedent_state_vector_columns,
    build_precedent_pack_v2,
    build_precedent_retrieval_index,
)
from src.pipeline.run import (
    _default_precedent_outcomes_path,
    adapt_snapshot,
    attach_model_feature_bundle,
    feature_view_from_snapshot,
)

SNAPSHOT_PATH = (
    REPO_ROOT
    / "out/materialized_feedback_20260405/company_state_snapshots_input_complete_catalog.asof_safe_v1.jsonl.gz"
)
TARGET_RAW_METRIC_SPECS: list[tuple[str, tuple[str, ...]]] = [
    ("operating.revenue_ttm_provider_direct", ("operating.revenue_ttm_provider_direct", "operating.revenue_ttm")),
    ("operating.revenue_ttm_lag_1y", ("operating.revenue_ttm_lag_1y", "operating.revenue_ttm_prior_year", "operating.revenue_ttm_prev_year")),
    ("operating.ebitda_ltm_provider_direct", ("operating.ebitda_ltm_provider_direct", "operating.ebitda_ttm", "operating.operating_earnings_normalized")),
    ("operating.ebitda_margin_ttm", ("operating.ebitda_margin_ttm",)),
    ("cash_flow.free_cash_flow_ttm", ("cash_flow.free_cash_flow_ttm", "operating.free_cash_flow_ttm", "cash_flow.free_cash_flow", "free_cash_flow_ttm")),
    ("capital_structure.total_debt_provider_direct", ("capital_structure.total_debt_provider_direct", "capital_structure.total_debt_reported", "capital_structure.total_debt")),
    ("capital_structure.net_debt_normalized", ("capital_structure.net_debt_normalized", "capital_structure.net_debt_standardized", "capital_structure.net_debt")),
    ("capital_structure.lease_liabilities_sec_exact", ("capital_structure.lease_liabilities_sec_exact",)),
    ("capital_structure.combined_retirement_liability", ("capital_structure.combined_retirement_liability",)),
    ("liquidity.cash_and_short_term_investments_provider_direct", ("liquidity.cash_and_short_term_investments_provider_direct", "liquidity.cash")),
    ("liquidity.marketable_securities_sec_exact", ("liquidity.marketable_securities_sec_exact", "liquidity.marketable_securities")),
    ("liquidity.revolver_undrawn", ("liquidity.revolver_undrawn",)),
    ("liquidity.available_liquidity_normalized", ("liquidity.available_liquidity_normalized",)),
    ("capital_structure.debt_due_next_24m", ("capital_structure.debt_due_next_24m", "capital_structure.debt_due_0_12m")),
    ("capital_structure.current_debt_statement_direct", ("capital_structure.current_debt_statement_direct", "capital_structure.current_debt")),
    ("capital_structure.current_debt_provider_direct", ("capital_structure.current_debt_provider_direct", "capital_structure.current_debt")),
    ("capital_structure.interest_expense_statement_direct", ("capital_structure.interest_expense_statement_direct", "capital_structure.interest_expense")),
    ("capital_structure.interest_coverage", ("capital_structure.interest_coverage",)),
    ("market.market_cap_provider_direct", ("market.market_cap_provider_direct", "market.market_cap")),
    ("market.enterprise_value", ("market.enterprise_value", "market.enterprise_value_provider_direct")),
    ("market.ev_ebitda", ("market.ev_ebitda",)),
    ("market.fcf_yield", ("market.fcf_yield",)),
    ("market.volatility_90d", ("market.volatility_90d",)),
    ("market.drawdown_90d", ("market.drawdown_90d",)),
    ("market.vix", ("market.vix",)),
    ("market.credit_window_proxy", ("market.credit_window_proxy",)),
    ("market.equity_window_proxy", ("market.equity_window_proxy",)),
    ("market.credit_spread_level", ("market.credit_spread_level",)),
    ("macro.fed_funds_effective", ("macro.fed_funds_effective",)),
    ("macro.hy_oas", ("macro.hy_oas",)),
]
HISTORICAL_RAW_FIELD_ORDER = [
    "ticker",
    "base_revenue_ttm",
    "base_revenue_ttm_lag_1y",
    "base_revenue_growth_yoy",
    "base_ebitda_ttm",
    "base_total_debt",
    "base_current_debt",
    "base_cash",
    "base_available_liquidity",
    "base_interest_expense",
    "base_market_cap",
    "base_ev_ebitda",
    "base_fcf_yield",
    "base_volatility_30d",
    "base_volatility_90d",
    "base_drawdown_90d",
    "base_credit_spread_level",
    "base_equity_window_proxy",
    "base_credit_window_proxy",
    "base_net_debt",
    "base_leverage",
    "base_margin",
    "macro_fed_funds_effective",
    "macro_hy_oas",
    "macro_real_gdp_growth_yoy",
]
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a one-company precedent audit packet.")
    parser.add_argument("--company-id", required=True)
    parser.add_argument("--company-name", required=True)
    parser.add_argument("--action-id", required=True)
    parser.add_argument("--source-company-id")
    parser.add_argument("--target-ticker")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--snapshot-path", default=str(SNAPSHOT_PATH))
    parser.add_argument("--snapshot-as-of-time")
    parser.add_argument("--snapshot-row-path")
    parser.add_argument("--snapshot-source-note")
    parser.add_argument("--outcomes-path", default=_default_precedent_outcomes_path())
    parser.add_argument("--out-path", required=True)
    return parser.parse_args()


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and not math.isfinite(value))


def _fmt_value(value: Any) -> str:
    if _is_missing(value):
        return "`null`"
    if isinstance(value, bool):
        return f"`{str(value).lower()}`"
    if isinstance(value, int):
        return f"`{value}`"
    if isinstance(value, float):
        return f"`{value:.4f}`"
    return f"`{value}`"


def _row_as_of_sort_key(row: Dict[str, Any]) -> str:
    return str(row.get("as_of_time") or "")


def _normalize_as_of_time(value: str | None) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    stamp = pd.to_datetime(text, utc=True, errors="coerce")
    if pd.isna(stamp):
        return text
    return stamp.isoformat()


def _calendar_year_end_timestamp(year: int) -> str:
    return pd.Timestamp(year=year, month=12, day=31, tz="UTC").isoformat()


def _calendar_year_exclusive_upper_bound(year: int) -> str:
    return pd.Timestamp(year=year + 1, month=1, day=1, tz="UTC").isoformat()


def _resolve_snapshot_policy(action_id: str, snapshot_as_of_time: str | None) -> Dict[str, str]:
    normalized_requested = _normalize_as_of_time(snapshot_as_of_time)
    target_snapshot_as_of_time = normalized_requested
    historical_precedent_cutoff_time = normalized_requested
    cutoff_policy = "requested_snapshot_as_of_time"
    target_snapshot_cutoff_date = ""
    if normalized_requested:
        stamp = pd.to_datetime(normalized_requested, utc=True, errors="coerce")
        if pd.notna(stamp):
            target_snapshot_cutoff_date = str(stamp.date())
    return {
        "requested_snapshot_as_of_time": normalized_requested,
        "target_snapshot_as_of_time": target_snapshot_as_of_time,
        "historical_precedent_cutoff_time": historical_precedent_cutoff_time,
        "target_snapshot_cutoff_date": target_snapshot_cutoff_date,
        "cutoff_policy": cutoff_policy,
    }


def _load_snapshot_row(snapshot_path: Path, company_id: str, snapshot_as_of_time: str | None = None) -> Dict[str, Any]:
    matches: list[Dict[str, Any]] = []
    normalized_target_time = _normalize_as_of_time(snapshot_as_of_time)
    with gzip.open(snapshot_path, "rt") as handle:
        for line in handle:
            row = json.loads(line)
            if str(row.get("company_id") or "") != company_id:
                continue
            if normalized_target_time and _normalize_as_of_time(str(row.get("as_of_time") or "")) == normalized_target_time:
                return row
            matches.append(row)
    if not matches:
        raise ValueError(f"company_id={company_id} not found in snapshot file {snapshot_path}")
    if normalized_target_time:
        raise ValueError(
            f"company_id={company_id} snapshot_as_of_time={snapshot_as_of_time} not found in snapshot file {snapshot_path}"
        )
    return max(matches, key=_row_as_of_sort_key)


def _load_snapshot_row_from_json(snapshot_row_path: Path, company_id: str) -> Dict[str, Any]:
    row = json.loads(snapshot_row_path.read_text())
    if str(row.get("company_id") or "") != company_id:
        raise ValueError(
            f"company_id mismatch for snapshot row {snapshot_row_path}: "
            f"expected {company_id}, found {row.get('company_id')}"
        )
    return row


def _is_flat_outcome_row(row: Dict[str, Any]) -> bool:
    if not isinstance(row, dict) or not row:
        return False
    features = row.get("features")
    if isinstance(features, dict) and features:
        return False
    outcome_markers = (
        "normalized_action_id",
        "normalized_action_family",
        "action_date",
        "base_revenue_ttm",
        "base_total_debt",
        "base_market_cap",
        "base_ebitda_ttm",
    )
    return any(marker in row for marker in outcome_markers)


def _coerce_snapshot_row_for_audit(
    row: Dict[str, Any],
    *,
    company_id: str,
    snapshot_as_of_time: str | None,
    outcomes_path: Path,
) -> Dict[str, Any]:
    if not _is_flat_outcome_row(row):
        return row
    as_of_time = str(
        snapshot_as_of_time
        or _normalize_as_of_time(str(row.get("as_of_time") or row.get("action_date") or ""))
    )
    return _synthesized_snapshot_row_from_outcome_row(
        row,
        company_id=str(company_id or ""),
        as_of_time=as_of_time,
        outcomes_path=outcomes_path,
    )


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _feature_record(
    value: Any,
    *,
    support_mode: str = "historical_outcome_fallback",
    quality_flags: list[str] | tuple[str, ...] = (),
) -> Dict[str, Any]:
    record = {
        "value": value,
        "support_mode": support_mode,
    }
    flags = [str(flag) for flag in quality_flags if flag]
    if flags:
        record["quality_flags"] = flags
    return record


def _action_params_from_outcome_row(outcome_row: Dict[str, Any]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for key in (
        "amount_usd",
        "amount",
        "transaction_value_usd",
        "deal_value_usd",
        "estimated_proceeds_usd",
        "draw_amount_usd",
        "resize_amount_usd",
        "amount_refinanced_usd",
        "action_size",
    ):
        value = _safe_float(outcome_row.get(key))
        if value is None:
            continue
        params["amount_usd"] = value
        params["action_size"] = value
        break
    raw_subtype = str(outcome_row.get("raw_action_subtype") or outcome_row.get("action_subtype") or "").strip()
    if raw_subtype:
        params["source_action_subtype"] = raw_subtype
        effective_subtype = _effective_action_subtype(
            outcome_row.get("normalized_action_id") or outcome_row.get("action_id"),
            raw_subtype,
            {"source_action_subtype": raw_subtype},
        )
        if effective_subtype == "refinancing_term_loan_family":
            params["instrument_type"] = "term_loan"
        elif effective_subtype == "refinancing_revolver_family":
            params["instrument_type"] = "revolver"
        elif effective_subtype == "refinancing_bond_family":
            params["instrument_type"] = "bond"
    return params


@lru_cache(maxsize=2048)
def _historical_company_taxonomy_from_outcomes(
    outcomes_path_text: str,
    company_id: str,
    action_id: str,
    ticker: str,
) -> Dict[str, str]:
    outcomes_path = Path(str(outcomes_path_text or ""))
    company_id_text = str(company_id or "").strip()
    action_id_text = str(action_id or "").strip()
    ticker_text = str(ticker or "").strip().upper()
    if not outcomes_path.exists() or (not company_id_text and not ticker_text):
        return {}

    clauses = []
    params: List[str] = [str(outcomes_path)]
    if company_id_text:
        clauses.append("CAST(company_id AS VARCHAR) = ?")
        params.append(company_id_text)
    elif ticker_text:
        clauses.append("UPPER(CAST(ticker AS VARCHAR)) = ?")
        params.append(ticker_text)
    if action_id_text:
        clauses.append("CAST(normalized_action_id AS VARCHAR) = ?")
        params.append(action_id_text)
    query = "SELECT * FROM read_parquet(?)"
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    try:
        frame = duckdb.execute(query, params).df()
    except Exception:
        return {}
    if frame.empty:
        return {}
    frame = _enrich_missing_historical_taxonomy(frame)
    frame = augment_precedent_state_vector_columns(frame)

    sector_votes: Dict[str, int] = {}
    subsector_votes: Dict[str, int] = {}
    for row in frame.to_dict(orient="records"):
        sector = str(
            row.get("taxonomy.sector")
            or row.get("sector")
            or row.get("base_sector")
            or ""
        ).strip()
        subsector = str(
            row.get("taxonomy.subsector")
            or row.get("subsector")
            or row.get("industry")
            or row.get("base_industry")
            or ""
        ).strip()
        if sector:
            sector_votes[sector] = sector_votes.get(sector, 0) + 1
        if subsector:
            subsector_votes[subsector] = subsector_votes.get(subsector, 0) + 1

    best_sector = max(sector_votes.items(), key=lambda item: item[1])[0] if sector_votes else ""
    best_subsector = max(subsector_votes.items(), key=lambda item: item[1])[0] if subsector_votes else ""
    if not best_sector and not best_subsector:
        return {}
    return {
        "taxonomy.sector": best_sector,
        "taxonomy.subsector": best_subsector,
    }


def _synthesized_snapshot_row_from_outcome_row(
    outcome_row: Dict[str, Any],
    *,
    company_id: str,
    as_of_time: str,
    outcomes_path: Path,
) -> Dict[str, Any]:
    frame = _enrich_missing_historical_taxonomy(pd.DataFrame([dict(outcome_row)]))
    frame = backfill_historical_price_window_metrics(frame)
    frame = augment_precedent_state_vector_columns(frame)
    row = frame.iloc[0].to_dict()

    def _scaled_monetary(field_name: str) -> float | None:
        value = _safe_float(row.get(field_name))
        if value is None:
            return None
        return value * 1_000_000.0

    def _derived_feature(value: Any) -> Dict[str, Any]:
        return _feature_record(value, support_mode="historical_outcome_fallback_derived")

    revenue = _scaled_monetary("base_revenue_ttm")
    revenue_lag = _scaled_monetary("base_revenue_ttm_lag_1y")
    ebitda = _scaled_monetary("base_ebitda_ttm")
    total_debt = _scaled_monetary("base_total_debt")
    net_debt = _scaled_monetary("base_net_debt")
    cash = _scaled_monetary("base_cash")
    available_liquidity = _scaled_monetary("base_available_liquidity")
    current_debt = _scaled_monetary("base_current_debt")
    interest_expense = _scaled_monetary("base_interest_expense")
    market_cap = _scaled_monetary("base_market_cap")
    ev_ebitda = _safe_float(row.get("base_ev_ebitda"))
    fcf_margin = _safe_float(row.get("base_fcf_margin"))
    free_cash_flow = (
        revenue * fcf_margin
        if revenue is not None and fcf_margin is not None
        else None
    )
    enterprise_value_from_components = (
        market_cap + total_debt - cash
        if market_cap is not None and total_debt is not None and cash is not None
        else None
    )
    enterprise_value_from_multiple = (
        ebitda * ev_ebitda
        if ebitda is not None and ev_ebitda is not None
        else None
    )
    enterprise_value = enterprise_value_from_components
    if enterprise_value is None:
        enterprise_value = enterprise_value_from_multiple
    interest_coverage = (
        ebitda / interest_expense
        if ebitda is not None and interest_expense not in (None, 0.0) and interest_expense > 0.0
        else None
    )
    sector = str(
        row.get("taxonomy.sector")
        or row.get("sector")
        or row.get("base_sector")
        or ""
    ).strip()
    subsector = str(
        row.get("taxonomy.subsector")
        or row.get("subsector")
        or row.get("industry")
        or row.get("base_industry")
        or ""
    ).strip()
    if not sector or not subsector:
        allow_sec_identity_heuristics = (
            str(row.get("normalized_action_id") or outcome_row.get("normalized_action_id") or "").strip().lower()
            == "capital_structure.equity_issuance"
        )
        ticker_taxonomy = _historical_taxonomy_for_ticker(
            str(row.get("ticker") or ""),
            allow_sec_identity_heuristics=allow_sec_identity_heuristics,
        )
        if not sector:
            sector = str(ticker_taxonomy.get("taxonomy.sector") or "").strip()
        if not subsector:
            subsector = str(ticker_taxonomy.get("taxonomy.subsector") or "").strip()
    if not sector or not subsector:
        company_history_taxonomy = _historical_company_taxonomy_from_outcomes(
            str(outcomes_path),
            str(company_id or ""),
            str(row.get("normalized_action_id") or outcome_row.get("normalized_action_id") or ""),
            str(row.get("ticker") or ""),
        )
        if not sector:
            sector = str(company_history_taxonomy.get("taxonomy.sector") or "").strip()
        if not subsector:
            subsector = str(company_history_taxonomy.get("taxonomy.subsector") or "").strip()

    features: Dict[str, Dict[str, Any]] = {
        "operating.revenue_ttm_provider_direct": _feature_record(revenue),
        "operating.revenue_ttm_lag_1y": _feature_record(revenue_lag),
        "operating.ebitda_ltm_provider_direct": _feature_record(ebitda),
        "operating.ebitda_margin_ttm": _feature_record(row.get("base_margin")),
        "cash_flow.free_cash_flow_ttm": _feature_record(free_cash_flow),
        "capital_structure.total_debt_provider_direct": _feature_record(total_debt),
        "capital_structure.net_debt_normalized": _feature_record(net_debt),
        "liquidity.cash_and_short_term_investments_provider_direct": _feature_record(cash),
        "liquidity.available_liquidity_normalized": _feature_record(available_liquidity),
        "capital_structure.current_debt_statement_direct": _feature_record(current_debt),
        "capital_structure.current_debt_provider_direct": _derived_feature(current_debt),
        "capital_structure.debt_due_next_24m": _feature_record(
            current_debt,
            support_mode="proxy_missing_component",
            quality_flags=["current_debt_fallback"],
        ),
        "capital_structure.interest_expense_statement_direct": _feature_record(interest_expense),
        "capital_structure.interest_coverage": _derived_feature(interest_coverage),
        "market.market_cap_provider_direct": _feature_record(market_cap),
        "market.enterprise_value": _derived_feature(enterprise_value),
        "market.enterprise_value_provider_direct": _derived_feature(enterprise_value),
        "market.ev_ebitda": _feature_record(ev_ebitda),
        "market.fcf_yield": _feature_record(row.get("base_fcf_yield")),
        "market.volatility_30d": _feature_record(row.get("base_volatility_30d")),
        "market.volatility_90d": _feature_record(row.get("base_volatility_90d")),
        "market.drawdown_90d": _feature_record(row.get("base_drawdown_90d")),
        "market.momentum_60d": _feature_record(row.get("base_momentum_60d")),
        "market.vix": _feature_record(row.get("macro_vix")),
        "market.credit_spread_level": _feature_record(row.get("base_credit_spread_level")),
        "market.credit_window_proxy": _feature_record(row.get("base_credit_window_proxy")),
        "market.equity_window_proxy": _feature_record(row.get("base_equity_window_proxy")),
        "macro.fed_funds_effective": _feature_record(row.get("macro_fed_funds_effective")),
        "macro.hy_oas": _feature_record(row.get("macro_hy_oas")),
        "macro.ig_oas": _feature_record(row.get("macro_ig_oas")),
        "macro.real_gdp_growth_yoy": _feature_record(row.get("macro_real_gdp_growth_yoy")),
        "macro.sofr": _feature_record(row.get("macro_sofr")),
        "macro.ust_10y_yield": _feature_record(row.get("macro_rate_10y")),
        "macro.ust_2y_yield": _feature_record(row.get("macro_rate_2y")),
        "operating.revenue_yoy_last_q": _feature_record(row.get("base_revenue_growth_yoy")),
        "taxonomy.sector": _feature_record(sector),
        "taxonomy.subsector": _feature_record(subsector),
    }
    for feature in _STATE_VECTOR_V1_FEATURES:
        features[feature] = _feature_record(row.get(feature))

    return {
        "company_id": str(company_id or ""),
        "as_of_time": str(as_of_time or ""),
        "snapshot_id": f"historical_outcome_fallback:{company_id}:{as_of_time}",
        "action_params": _action_params_from_outcome_row(outcome_row),
        "features": features,
        "snapshot_catalog_source": "historical_outcome_fallback",
        "snapshot_catalog_path": str(outcomes_path),
    }


