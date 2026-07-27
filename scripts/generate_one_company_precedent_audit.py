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


def _load_historical_outcome_target_row(
    outcomes_path: Path,
    *,
    action_id: str,
    company_id: str,
    snapshot_as_of_time: str | None,
    source_company_id: str = "",
    target_ticker: str = "",
) -> Dict[str, Any] | None:
    if not outcomes_path.exists():
        return None
    frame = pd.read_parquet(outcomes_path, filters=[[("normalized_action_id", "==", str(action_id))]])
    if frame.empty:
        return None
    if source_company_id:
        frame = frame[frame.get("company_id", pd.Series("", index=frame.index)).astype(str) == str(source_company_id)]
    elif target_ticker and "ticker" in frame.columns:
        frame = frame[frame["ticker"].astype(str) == str(target_ticker)]
    if frame.empty:
        return None
    frame = frame.copy()
    frame["action_date"] = pd.to_datetime(frame["action_date"], utc=True, errors="coerce")
    if snapshot_as_of_time:
        target_ts = pd.to_datetime(snapshot_as_of_time, utc=True, errors="coerce")
        if pd.notna(target_ts):
            delta = (frame["action_date"] - target_ts).dt.total_seconds()
            forward = delta.where(delta >= 0.0)
            if forward.notna().any():
                frame = frame.assign(_priority=forward)
            else:
                frame = frame.assign(_priority=delta.abs())
            frame = frame.sort_values(["_priority", "action_date"], ascending=[True, True])
    else:
        frame = frame.sort_values("action_date", ascending=False)
    if frame.empty:
        return None
    return _synthesized_snapshot_row_from_outcome_row(
        frame.iloc[0].to_dict(),
        company_id=company_id,
        as_of_time=str(snapshot_as_of_time or ""),
        outcomes_path=outcomes_path,
    )


def _target_context_lines(row: Dict[str, Any], bundle: Dict[str, Any]) -> List[str]:
    features = row.get("features") or {}
    support = bundle["state_vector_v1"]["support"]
    proxy = [key for key in _STATE_VECTOR_V1_FEATURES if (support.get(key) or {}).get("support_mode") == "proxy_missing_component"]
    missing = [
        key
        for key in _STATE_VECTOR_V1_FEATURES
        if (support.get(key) or {}).get("support_mode") in {None, "unsupported"}
        and _is_missing(bundle["state_vector_v1"]["values"].get(key))
    ]
    sector = (features.get("taxonomy.sector") or {}).get("value")
    subsector = (features.get("taxonomy.subsector") or {}).get("value")
    regime = (features.get("capital_structure.retirement_obligation_regime") or {}).get("value")
    return [
        f"- Sector: `{sector}`",
        f"- Subsector: `{subsector}`",
        f"- Retirement regime: `{regime}`",
        f"- Proxy compact features: {', '.join(f'`{key}`' for key in proxy) if proxy else '`None`'}",
        f"- Missing compact features: {', '.join(f'`{key}`' for key in missing) if missing else '`None`'}",
    ]


def _resolve_metric_record(features: Dict[str, Any], aliases: tuple[str, ...]) -> tuple[str, Dict[str, Any]]:
    for alias in aliases:
        record = features.get(alias)
        if isinstance(record, dict) and record.get("value") is not None:
            return alias, record
    for alias in aliases:
        record = features.get(alias)
        if isinstance(record, dict):
            return alias, record
    return aliases[0], {}


def _target_raw_table(row: Dict[str, Any]) -> str:
    features = row.get("features") or {}
    lines = ["| Raw metric | Value | Support | Resolved from |", "|---|---:|---|---|"]
    for metric, aliases in TARGET_RAW_METRIC_SPECS:
        resolved_metric, record = _resolve_metric_record(features, aliases)
        lines.append(
            f"| `{metric}` | {_fmt_value(record.get('value'))} | `{record.get('support_mode') or 'unsupported'}` | `{resolved_metric}` |"
        )
    return "\n".join(lines)


def _compact_table(values: Dict[str, Any]) -> str:
    lines = ["| Compact feature | Value |", "|---|---:|"]
    for key in _STATE_VECTOR_V1_FEATURES:
        lines.append(f"| `{key}` | {_fmt_value(values.get(key))} |")
    return "\n".join(lines)


def _compact_comparison_table(target_values: Dict[str, Any], match_row: pd.Series) -> str:
    lines = ["| Compact feature | Target | Match |", "|---|---:|---:|"]
    for key in _STATE_VECTOR_V1_FEATURES:
        lines.append(f"| `{key}` | {_fmt_value(target_values.get(key))} | {_fmt_value(match_row.get(key))} |")
    return "\n".join(lines)


def _historical_raw_table(match_row: pd.Series) -> str:
    lines = ["| Historical raw field | Match value |", "|---|---:|"]
    for field in HISTORICAL_RAW_FIELD_ORDER:
        lines.append(f"| `{field}` | {_fmt_value(match_row.get(field))} |")
    return "\n".join(lines)


def _nonnull_compact_count(match_row: pd.Series) -> int:
    return sum(0 if _is_missing(match_row.get(key)) else 1 for key in _STATE_VECTOR_V1_FEATURES)


def _compact_feature_scale_map(historical_df: pd.DataFrame) -> Dict[str, float]:
    scales: Dict[str, float] = {}
    for key in _STATE_VECTOR_V1_FEATURES:
        if key not in historical_df.columns:
            scales[key] = 1.0
            continue
        series = pd.to_numeric(historical_df.get(key), errors="coerce").dropna()
        if series.empty:
            scales[key] = 1.0
            continue
        q25 = float(series.quantile(0.25))
        q75 = float(series.quantile(0.75))
        iqr = float(q75 - q25)
        std = float(series.std()) if len(series) > 1 else 0.0
        scale = iqr if math.isfinite(iqr) and iqr > 1e-8 else std
        scales[key] = scale if math.isfinite(scale) and scale > 1e-8 else 1.0
    return scales


def _compact_feature_label(feature_name: str) -> str:
    suffix = str(feature_name).split(".")[-1]
    return suffix.replace("_", " ")


def _match_explanation_lines(
    *,
    action_id: str,
    target_values: Dict[str, Any],
    match_row: pd.Series,
    feature_scales: Dict[str, float],
) -> List[str]:
    scored: List[Tuple[str, float, float]] = []
    for key in _STATE_VECTOR_V1_FEATURES:
        target_value = target_values.get(key)
        match_value = match_row.get(key)
        if _is_missing(target_value) or _is_missing(match_value):
            continue
        scale = float(feature_scales.get(key, 1.0) or 1.0)
        gap = abs(float(target_value) - float(match_value)) / max(scale, 1e-8)
        scored.append((key, gap, 1.0))
    if not scored:
        return ["- Why it matched: `insufficient comparable compact features`"]

    closest = sorted(
        scored,
        key=lambda item: (-(item[2] / (1.0 + item[1])), item[1], item[0]),
    )[:3]
    farthest = sorted(
        scored,
        key=lambda item: (-(item[2] * item[1]), item[0]),
    )[:3]

    def _fmt_feature_triplet(feature_name: str) -> str:
        return (
            f"`{_compact_feature_label(feature_name)}` "
            f"({_fmt_value(target_values.get(feature_name))} vs {_fmt_value(match_row.get(feature_name))})"
        )

    closest_text = ", ".join(_fmt_feature_triplet(feature_name) for feature_name, _, _ in closest)
    farthest_text = ", ".join(_fmt_feature_triplet(feature_name) for feature_name, _, _ in farthest)
    return [
        f"- Why it matched: {closest_text}",
        f"- Main gaps: {farthest_text}",
    ]


def _confidence_label(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.55:
        return "medium"
    return "low"


def _locate_match_row(historical_df: pd.DataFrame, case: Any) -> pd.Series:
    action_date = pd.to_datetime(case.decision_time, utc=True, errors="coerce")
    if pd.isna(action_date):
        raise RuntimeError(
            f"Could not normalize decision_time for company_id={case.company_id} decision_time={case.decision_time}"
        )
    action_date = action_date.normalize()
    mask = historical_df["company_id"].astype(str).eq(str(case.company_id))
    mask &= pd.to_datetime(historical_df["action_date"], utc=True, errors="coerce").dt.normalize().eq(action_date)
    normalized_action_id = historical_df.get("normalized_action_id")
    if normalized_action_id is not None and str(case.action_id or "").strip():
        id_mask = normalized_action_id.fillna("").astype(str).eq(str(case.action_id))
        if bool((mask & id_mask).any()):
            mask &= id_mask
    matches = historical_df.loc[mask]
    if matches.empty:
        raise RuntimeError(
            f"Could not locate historical row for company_id={case.company_id} action_date={action_date} action_id={case.action_id}"
        )
    return matches.iloc[0]


def _build_target_payload(row: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    adapted_row, _ = adapt_snapshot(row)
    adapted_row = attach_model_feature_bundle(adapted_row)
    bundle = build_model_feature_bundle(adapted_row)
    precedent_features = dict(feature_view_from_snapshot(adapted_row, view_name="precedent") or {})
    canonical = dict(bundle.get("canonical", {}) or {})
    state_meta = dict((bundle.get("state_vector_v1", {}) or {}).get("meta", {}) or {})
    if precedent_features.get("market_cap") is None:
        market_cap_value = canonical.get("scale.market_cap")
        if market_cap_value is not None:
            precedent_features["market_cap"] = {"value": market_cap_value, "support_mode": "canonical_alias"}
    if precedent_features.get("sector") is None:
        sector_value = state_meta.get("sector")
        if sector_value:
            precedent_features["sector"] = {"value": sector_value, "support_mode": "canonical_alias"}
    if precedent_features.get("subsector") is None:
        subsector_value = state_meta.get("subsector")
        if subsector_value:
            precedent_features["subsector"] = {"value": subsector_value, "support_mode": "canonical_alias"}
    regime = adapted_row.get("regime", {}) if isinstance(adapted_row.get("regime"), dict) else {}
    return adapted_row, bundle, {
        "precedent_features": precedent_features,
        "regime": regime,
        "target_values": bundle["state_vector_v1"]["values"],
    }


def _target_payload_nonnull_count(payload: Dict[str, Any]) -> int:
    values = payload.get("target_values") if isinstance(payload, dict) else None
    if not isinstance(values, dict):
        return 0
    count = 0
    for value in values.values():
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        count += 1
    return count


def _filter_historical_precedents_as_of(
    historical_df: pd.DataFrame,
    *,
    snapshot_as_of_time: str | None,
) -> pd.DataFrame:
    normalized_target_time = _normalize_as_of_time(snapshot_as_of_time)
    if not normalized_target_time or historical_df.empty or "action_date" not in historical_df.columns:
        return historical_df
    target_ts = pd.to_datetime(normalized_target_time, utc=True, errors="coerce")
    if pd.isna(target_ts):
        return historical_df
    action_dates = pd.to_datetime(historical_df["action_date"], utc=True, errors="coerce")
    filtered = historical_df.loc[action_dates.lt(target_ts)].copy()
    if filtered.empty:
        return historical_df
    return filtered


def _retrieve_variant(
    company_id: str,
    action_id: str,
    action_params: Dict[str, Any],
    candidate_features: Dict[str, Any],
    candidate_regime: Dict[str, Any],
    target_values: Dict[str, Any],
    retrieval_index: Any,
    historical_df: pd.DataFrame,
    disable_learned: bool,
    top_k: int,
) -> Dict[str, Any]:
    if disable_learned:
        os.environ["PRECEDENT_DISABLE_LEARNED_DISTANCE_WEIGHTS"] = "1"
        os.environ["PRECEDENT_DISABLE_DISTANCE_V2"] = "1"
    else:
        os.environ.pop("PRECEDENT_DISABLE_LEARNED_DISTANCE_WEIGHTS", None)
        os.environ.pop("PRECEDENT_DISABLE_DISTANCE_V2", None)
    _PRECEDENT_DISTANCE_WEIGHTS_CACHE.clear()
    _PRECEDENT_DISTANCE_V2_WEIGHTS_CACHE.clear()
    pack = build_precedent_pack_v2(
        candidate_id=f"audit:{company_id}",
        run_id=f"audit:{uuid.uuid4()}",
        company_id=company_id,
        action_id=action_id,
        action_subtype=None,
        action_params=action_params,
        candidate_features=candidate_features,
        candidate_regime=candidate_regime,
        retrieval_index=retrieval_index,
        top_k=top_k,
        min_k=top_k,
    )
    feature_scales = _compact_feature_scale_map(historical_df)
    matches = []
    for case in pack.retrieved_cohorts[:top_k]:
        hist_row = _locate_match_row(historical_df, case)
        matches.append(
            {
                "precedent_id": case.precedent_id,
                "company_id": case.company_id,
                "action_id": case.action_id,
                "decision_time": case.decision_time,
                "similarity_score": float(case.similarity_score),
                "nonnull_compact_features": _nonnull_compact_count(hist_row),
                "explanation_lines": _match_explanation_lines(
                    action_id=action_id,
                    target_values=target_values,
                    match_row=hist_row,
                    feature_scales=feature_scales,
                ),
                "historical_row": hist_row,
            }
        )
    return {
        "state_weight_scope": pack.mismatch_diagnostics.get("state_weight_scope"),
        "calibration_confidence": float(pack.calibration_confidence),
        "confidence_label": _confidence_label(float(pack.calibration_confidence)),
        "out_of_sample_flag": bool(pack.mismatch_diagnostics.get("out_of_sample_flag")),
        "retrieval_tier": pack.mismatch_diagnostics.get("retrieval_tier"),
        "exact_match_count": int(pack.mismatch_diagnostics.get("exact_match_count") or 0),
        "minimum_exact_support": int(pack.mismatch_diagnostics.get("minimum_exact_support") or 0),
        "top_similarity_mean": pack.mismatch_diagnostics.get("top_similarity_mean"),
        "top_weighted_feature_coverage": pack.mismatch_diagnostics.get("top_weighted_feature_coverage"),
        "top_critical_feature_coverage": pack.mismatch_diagnostics.get("top_critical_feature_coverage"),
        "top_action_match_score": pack.mismatch_diagnostics.get("top_action_match_score"),
        "matches": matches,
    }


def _confidence_lines(label: str, payload: Dict[str, Any]) -> List[str]:
    confidence = float(payload.get("calibration_confidence") or 0.0)
    return [
        f"- {label} confidence: `{confidence:.4f}` (`{payload.get('confidence_label')}`)",
        f"- {label} retrieval tier: `{payload.get('retrieval_tier')}`",
        f"- {label} out-of-sample flag: `{str(bool(payload.get('out_of_sample_flag'))).lower()}`",
        f"- {label} exact-support depth: `{int(payload.get('exact_match_count') or 0)}` / `{int(payload.get('minimum_exact_support') or 0)}`",
        f"- {label} top-support similarity mean: `{float(payload.get('top_similarity_mean') or 0.0):.4f}`",
        f"- {label} top-support weighted coverage: `{float(payload.get('top_weighted_feature_coverage') or 0.0):.4f}`",
        f"- {label} top-support critical coverage: `{float(payload.get('top_critical_feature_coverage') or 0.0):.4f}`",
        f"- {label} top-support action match score: `{float(payload.get('top_action_match_score') or 0.0):.4f}`",
    ]


def _support_tier_label(
    *,
    target_company_id: str,
    match_company_id: str,
    similarity_score: float,
    top_similarity_score: float,
) -> str:
    same_company = str(match_company_id or "") == str(target_company_id or "")
    primary_floor = max(0.40, top_similarity_score - 0.03)
    secondary_floor = max(0.30, top_similarity_score - 0.08)
    if same_company:
        if similarity_score >= primary_floor:
            return "same_company_history_primary"
        if similarity_score >= secondary_floor:
            return "same_company_history_secondary"
        return "same_company_history_context"
    if similarity_score >= primary_floor:
        return "peer_primary"
    if similarity_score >= secondary_floor:
        return "peer_secondary"
    return "context"


def _support_tier_summary_lines(
    *,
    company_id: str,
    payload: Dict[str, Any],
) -> List[str]:
    matches = list(payload.get("matches") or [])
    if not matches:
        return [
            "- Support summary: `no matches retrieved`",
        ]
    top_similarity_score = max(float(match.get("similarity_score") or 0.0) for match in matches)
    tier_buckets: Dict[str, List[str]] = {}
    for match in matches:
        tier_label = _support_tier_label(
            target_company_id=company_id,
            match_company_id=str(match.get("company_id") or ""),
            similarity_score=float(match.get("similarity_score") or 0.0),
            top_similarity_score=top_similarity_score,
        )
        hist_row = match.get("historical_row")
        ticker = ""
        if hasattr(hist_row, "get"):
            ticker = str(hist_row.get("ticker") or "").strip()
        if not ticker:
            ticker = str(match.get("company_id") or "")
        decision_date = str(match.get("decision_time") or "")[:10]
        label = f"{ticker} ({decision_date})" if decision_date else ticker
        tier_buckets.setdefault(tier_label, []).append(label)
    ordered_tiers = (
        "peer_primary",
        "same_company_history_primary",
        "peer_secondary",
        "same_company_history_secondary",
        "context",
        "same_company_history_context",
    )
    lines = []
    for tier in ordered_tiers:
        values = tier_buckets.get(tier) or []
        if not values:
            continue
        lines.append(f"- {tier.replace('_', ' ').title()}: `{'; '.join(values)}`")
    if not lines:
        lines.append("- Support summary: `no tiered support available`")
    return lines


def _ambiguity_read_lines(
    *,
    company_id: str,
    payload: Dict[str, Any],
) -> List[str]:
    matches = list(payload.get("matches") or [])
    confidence = float(payload.get("calibration_confidence") or 0.0)
    if not matches:
        return ["- No precedent support was retrieved for this target."]
    scores = [float(match.get("similarity_score") or 0.0) for match in matches]
    top_score = max(scores)
    second_score = scores[1] if len(scores) > 1 else None
    same_company_count = sum(1 for match in matches if str(match.get("company_id") or "") == str(company_id or ""))
    lines: List[str] = []
    if confidence < 0.20:
        lines.append(
            "- No high-confidence precedent set was found; this read relies on partial analogs rather than a tight peer neighborhood."
        )
    elif confidence < 0.35:
        lines.append(
            "- Precedent support is usable but still mixed; the top names are better treated as directional analogs than exact comps."
        )
    else:
        lines.append("- Precedent support is reasonably concentrated around the top neighborhood.")
    if second_score is not None and (top_score - second_score) <= 0.03:
        lines.append(
            "- The top scores cluster tightly together, which usually means the model sees several partial analogs instead of one clearly dominant match."
        )
    if same_company_count > 0:
        lines.append(
            f"- Same-company history contributes `{same_company_count}` of the top `{len(matches)}` learned matches, which is useful context but not a full substitute for external peers."
        )
    if float(payload.get("top_critical_feature_coverage") or 0.0) < 0.70:
        lines.append(
            "- Critical feature coverage is thin here, so the read should lean more on broad borrower pattern alignment than on any single exact ratio match."
        )
    return lines


def _render_doc(
    company_name: str,
    company_id: str,
    action_id: str,
    row: Dict[str, Any],
    bundle: Dict[str, Any],
    learned: Dict[str, Any],
    prior_only: Dict[str, Any],
    outcomes_path: Path,
    snapshot_path: Path,
    snapshot_source_note: str | None,
    target_snapshot_cutoff_date: str,
    historical_precedent_cutoff_time: str,
    cutoff_policy: str,
) -> str:
    target_values = bundle["state_vector_v1"]["values"]
    learned_ids = [match["precedent_id"] for match in learned["matches"]]
    prior_ids = [match["precedent_id"] for match in prior_only["matches"]]
    top1_same = learned_ids[:1] == prior_ids[:1]
    topk_same = learned_ids == prior_ids
    audit_date = date.today().isoformat()
    historical_cutoff_label = historical_precedent_cutoff_time or "latest available"
    lines = [
        f"# {company_name} Precedent Audit",
        "",
        f"This packet shows the target company state and the retrieved precedent matches for `{action_id}`.",
        "",
        f"- Audit generated on: `{audit_date}`",
        f"- Target snapshot cutoff date: `{target_snapshot_cutoff_date or 'latest available'}`",
        f"- Historical precedent cutoff: `< {historical_cutoff_label}`",
        f"- Cutoff policy: `{cutoff_policy}`",
        f"- Company id: `{company_id}`",
        f"- Snapshot artifact: `{snapshot_path}`",
        f"- Historical precedent artifact: `{outcomes_path}`",
        (
            f"- Snapshot source note: {snapshot_source_note}"
            if snapshot_source_note
            else None
        ),
        f"- Learned-vs-prior top-1 same: `{str(top1_same).lower()}`",
        f"- Learned-vs-prior full top-{len(learned['matches'])} same: `{str(topk_same).lower()}`",
        "",
        "## Target Compact State Vector",
        "",
        _compact_table(target_values),
        "",
        "## Target Raw Inputs Behind The Compact Features",
        "",
        _target_raw_table(row),
        "",
        "## Support Context",
        "",
        *_target_context_lines(row, bundle),
        "",
        "## Match Confidence",
        "",
        *_confidence_lines("Learned", learned),
        *_confidence_lines("Prior", prior_only),
        "",
        "## Support Tiers",
        "",
        *_support_tier_summary_lines(company_id=company_id, payload=learned),
        "",
        "## Ambiguity Read",
        "",
        *_ambiguity_read_lines(company_id=company_id, payload=learned),
        "",
        f"## Learned Weights Enabled (`{learned['state_weight_scope']}`)",
        "",
    ]
    learned_top_score = max((float(match.get("similarity_score") or 0.0) for match in learned["matches"]), default=0.0)
    for idx, payload in enumerate(learned["matches"], start=1):
        support_tier = _support_tier_label(
            target_company_id=company_id,
            match_company_id=str(payload.get("company_id") or ""),
            similarity_score=float(payload.get("similarity_score") or 0.0),
            top_similarity_score=learned_top_score,
        )
        lines.extend(
            [
                f"### Learned Match {idx}",
                "",
                f"- Precedent id: `{payload['precedent_id']}`",
                f"- Historical company id: `{payload['company_id']}`",
                f"- Action: `{payload['action_id']}`",
                f"- Decision time: `{payload['decision_time']}`",
                f"- Similarity score: `{payload['similarity_score']:.6f}`",
                f"- Support tier: `{support_tier}`",
                f"- Non-null compact features on match row: `{payload['nonnull_compact_features']}/{len(_STATE_VECTOR_V1_FEATURES)}`",
                *payload.get("explanation_lines", []),
                "",
                _compact_comparison_table(target_values, payload["historical_row"]),
                "",
                _historical_raw_table(payload["historical_row"]),
                "",
            ]
        )
    lines.extend(
        [
            "## Prior Only",
            "",
        ]
    )
    prior_top_score = max((float(match.get("similarity_score") or 0.0) for match in prior_only["matches"]), default=0.0)
    for idx, payload in enumerate(prior_only["matches"], start=1):
        support_tier = _support_tier_label(
            target_company_id=company_id,
            match_company_id=str(payload.get("company_id") or ""),
            similarity_score=float(payload.get("similarity_score") or 0.0),
            top_similarity_score=prior_top_score,
        )
        lines.extend(
            [
                f"### Prior Match {idx}",
                "",
                f"- Precedent id: `{payload['precedent_id']}`",
                f"- Historical company id: `{payload['company_id']}`",
                f"- Action: `{payload['action_id']}`",
                f"- Decision time: `{payload['decision_time']}`",
                f"- Similarity score: `{payload['similarity_score']:.6f}`",
                f"- Support tier: `{support_tier}`",
                f"- Non-null compact features on match row: `{payload['nonnull_compact_features']}/{len(_STATE_VECTOR_V1_FEATURES)}`",
                *payload.get("explanation_lines", []),
                "",
                _compact_comparison_table(target_values, payload["historical_row"]),
                "",
                _historical_raw_table(payload["historical_row"]),
                "",
            ]
        )
    if topk_same:
        lines.extend(
            [
                "## Read",
                "",
                "The learned weights did not change the retrieved top match set here; they only nudged similarity scores.",
                "",
            ]
        )
    else:
        learned_only = [precedent_id for precedent_id in learned_ids if precedent_id not in prior_ids]
        prior_only_ids = [precedent_id for precedent_id in prior_ids if precedent_id not in learned_ids]
        lines.extend(
            [
                "## Read",
                "",
                f"- Learned-only precedents: `{', '.join(learned_only) if learned_only else 'None'}`",
                f"- Prior-only precedents: `{', '.join(prior_only_ids) if prior_only_ids else 'None'}`",
                "",
            ]
        )
    return "\n".join(line for line in lines if line is not None).rstrip() + "\n"


def main() -> None:
    args = _parse_args()
    snapshot_path = Path(args.snapshot_row_path or args.snapshot_path)
    outcomes_path = Path(args.outcomes_path)
    out_path = Path(args.out_path)
    snapshot_policy = _resolve_snapshot_policy(str(args.action_id), args.snapshot_as_of_time)
    target_snapshot_as_of_time = snapshot_policy.get("target_snapshot_as_of_time") or args.snapshot_as_of_time
    historical_precedent_cutoff_time = snapshot_policy.get("historical_precedent_cutoff_time") or args.snapshot_as_of_time
    target_snapshot_cutoff_date = str(snapshot_policy.get("target_snapshot_cutoff_date") or "")
    cutoff_policy = str(snapshot_policy.get("cutoff_policy") or "requested_snapshot_as_of_time")
    fallback_row = None

    if args.snapshot_row_path:
        row = _load_snapshot_row_from_json(snapshot_path, company_id=str(args.company_id))
        snapshot_source_note = args.snapshot_source_note
        if not target_snapshot_as_of_time and _is_flat_outcome_row(row):
            inferred_snapshot_time = _normalize_as_of_time(str(row.get("as_of_time") or row.get("action_date") or ""))
            if inferred_snapshot_time:
                target_snapshot_as_of_time = inferred_snapshot_time
                historical_precedent_cutoff_time = inferred_snapshot_time
                stamp = pd.to_datetime(inferred_snapshot_time, utc=True, errors="coerce")
                if pd.notna(stamp):
                    target_snapshot_cutoff_date = str(stamp.date())
                cutoff_policy = "snapshot_row_action_date"
        row = _coerce_snapshot_row_for_audit(
            row,
            company_id=str(args.company_id),
            snapshot_as_of_time=target_snapshot_as_of_time,
            outcomes_path=outcomes_path,
        )
        _, bundle, target_payload = _build_target_payload(row)
    else:
        snapshot_source_note = args.snapshot_source_note
        fallback_row = None
        try:
            row = _load_snapshot_row(
                snapshot_path,
                company_id=str(args.company_id),
                snapshot_as_of_time=target_snapshot_as_of_time,
            )
        except ValueError:
            fallback_row = _load_historical_outcome_target_row(
                outcomes_path,
                action_id=str(args.action_id),
                company_id=str(args.company_id),
                snapshot_as_of_time=target_snapshot_as_of_time,
                source_company_id=str(args.source_company_id or ""),
                target_ticker=str(args.target_ticker or ""),
            )
            if fallback_row is None:
                raise
            row = fallback_row
            _, bundle, target_payload = _build_target_payload(row)
        else:
            _, bundle, target_payload = _build_target_payload(row)
            fallback_row = _load_historical_outcome_target_row(
                outcomes_path,
                action_id=str(args.action_id),
                company_id=str(args.company_id),
                snapshot_as_of_time=target_snapshot_as_of_time,
                source_company_id=str(args.source_company_id or ""),
                target_ticker=str(args.target_ticker or ""),
            )
            if fallback_row is not None:
                _, fallback_bundle, fallback_target_payload = _build_target_payload(fallback_row)
                if _target_payload_nonnull_count(fallback_target_payload) > _target_payload_nonnull_count(target_payload):
                    row = fallback_row
                    bundle = fallback_bundle
                    target_payload = fallback_target_payload
                    snapshot_source_note = "historical_outcome_fallback_preferred_for_completeness"
    if not snapshot_source_note and row.get("snapshot_catalog_source"):
        source_name = str(row.get("snapshot_catalog_source"))
        catalog_path = row.get("snapshot_catalog_path")
        snapshot_source_note = f"{source_name}: {catalog_path}" if catalog_path else source_name
    try:
        raw_historical_df = pd.read_parquet(
            outcomes_path,
            filters=[[("normalized_action_id", "==", str(args.action_id))]],
        )
        if raw_historical_df.empty:
            raw_historical_df = pd.read_parquet(outcomes_path)
    except Exception:
        raw_historical_df = pd.read_parquet(outcomes_path)
    raw_historical_df = _filter_historical_precedents_as_of(
        raw_historical_df,
        snapshot_as_of_time=historical_precedent_cutoff_time,
    )
    retrieval_index = build_precedent_retrieval_index(raw_historical_df)
    historical_df = retrieval_index.df
    target_action_params = dict(row.get("action_params") or {})
    if not target_action_params and fallback_row is not None:
        target_action_params = dict(fallback_row.get("action_params") or {})

    learned = _retrieve_variant(
        company_id=str(args.company_id),
        action_id=str(args.action_id),
        action_params=target_action_params,
        candidate_features=target_payload["precedent_features"],
        candidate_regime=target_payload["regime"],
        target_values=target_payload["target_values"],
        retrieval_index=retrieval_index,
        historical_df=historical_df,
        disable_learned=False,
        top_k=int(args.top_k),
    )
    prior_only = _retrieve_variant(
        company_id=str(args.company_id),
        action_id=str(args.action_id),
        action_params=target_action_params,
        candidate_features=target_payload["precedent_features"],
        candidate_regime=target_payload["regime"],
        target_values=target_payload["target_values"],
        retrieval_index=retrieval_index,
        historical_df=historical_df,
        disable_learned=True,
        top_k=int(args.top_k),
    )

    doc = _render_doc(
        company_name=str(args.company_name),
        company_id=str(args.company_id),
        action_id=str(args.action_id),
        row=row,
        bundle=bundle,
        learned=learned,
        prior_only=prior_only,
        outcomes_path=outcomes_path,
        snapshot_path=snapshot_path,
        snapshot_source_note=snapshot_source_note,
        target_snapshot_cutoff_date=target_snapshot_cutoff_date,
        historical_precedent_cutoff_time=str(historical_precedent_cutoff_time or ""),
        cutoff_policy=cutoff_policy,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc)
    print(json.dumps({"out_path": str(out_path), "top_k": int(args.top_k)}))


if __name__ == "__main__":
    main()
