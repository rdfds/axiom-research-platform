"""
CompanyStateSnapshot builder (World Model spec).

This module assembles a point-in-time company snapshot using only
information available as-of a given timestamp. It is intentionally
conservative and auditable: every feature carries provenance and
confidence metadata. Missing data is explicit.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import re
import subprocess
import time
import uuid

import duckdb
import numpy as np
import pandas as pd

from scripts.backfill_sec_companyfacts_components import (
    _extract_lease_liabilities as _extract_sec_companyfacts_lease_liabilities,
    _extract_marketable_securities as _extract_sec_companyfacts_marketable_securities,
    _extract_restricted_cash as _extract_sec_companyfacts_restricted_cash,
    _extract_revolver_undrawn as _extract_sec_companyfacts_revolver_undrawn,
)
from scripts.backfill_input_layer_v1_metrics import (
    DEBT_CURRENT_CONCEPTS,
    DEBT_NONCURRENT_CONCEPTS,
    SHORT_TERM_BORROWINGS_CONCEPTS,
    _build_sec_core_metric,
    _instant_candidates,
)
from scripts.backfill_smart_normalized_metrics_v1 import materialize_smart_metrics_for_row

from .company_state_input_source_registry import CompanyStateInputSourceRegistry
from .company_state_components import RegimeClassifier, PeerSetResolver
from .data_paths import resolve_companyfacts_root, resolve_data_path
from .metric_policy import MetricPolicyEngine, TaxonomyContext


ROOT = Path(__file__).resolve().parents[1]
MAX_SEC_FACT_AGE_DAYS = 550
COMPANYFACTS_FRESHER_OVERRIDE_MIN_DAYS = 7
COMPANYFACTS_LOAD_TIMEOUT_SECONDS = 1.5
COMPANYFACTS_CASH_EQ_CONCEPTS = ["CashAndCashEquivalentsAtCarryingValue", "Cash"]
INTEREST_EXPENSE_TTM_EXACT_CONCEPTS = ["InterestExpense"]


def _smart_metric_registry_path() -> Path:
    env = str(os.environ.get("AXIOM_SMART_METRIC_REGISTRY_PATH", "") or "").strip()
    if env:
        return Path(env)
    return ROOT / "out" / "smart_metric_registry_v1.json"


def _market_availability_overrides_path() -> Path:
    env = str(os.environ.get("AXIOM_MARKET_AVAILABILITY_OVERRIDES_PATH", "") or "").strip()
    if env:
        return Path(env)
    return ROOT / "configs" / "liquidity_market_availability_overrides.json"

EXACT_SUPPORT_MODES = {"exact", "exact_not_applicable", "exact_structural_zero"}
PROXY_COMPONENT_SUPPORT_FLAGS = {
    "decision_uses_reported_view",
    "lease_adjustment_missing_assumed_zero",
    "lease_adjusted_denominator_fallback_to_ebitda",
    "lease_adjusted_denominator_missing_lease_expense",
    "lease_expense_estimated_from_liabilities",
    "lease_fixed_charge_proxy_from_liability",
    "marketable_securities_included_at_par_proxy",
    "minimum_cash_policy_proxy_not_applied_to_market_view",
    "mixed_quarter_value_basis",
    "pension_excluded_from_debt",
    "preferred_equity_excluded_pending_hybrid_review",
    "provider_fcf_fallback",
    "quarter_value_derived_from_ytd_delta",
    "reference_ebitda_fallback",
    "reference_total_debt_fallback",
    "reference_total_debt_used_for_completeness",
    "recent_total_debt_peak_used_for_completeness",
    "restricted_cash_missing_assumed_zero",
    "supplier_finance_included_without_payables_extension_test",
    "convertibles_excluded_pending_hybrid_review",
}
NON_PROXY_DIAGNOSTIC_FLAGS = {
    "companyfacts_cash_fresher",
    "companyfacts_total_debt_fresher",
    "fixed_charge_coverage_preferred",
    "latest_recurring_dividend_outside_active_window",
    "multiple_price_series_candidates",
    "no_recurring_dividend_events_in_history",
    "no_strategic_actions_in_window",
    "non_price_series_filtered",
    "pe_history_unavailable",
    "price_shares_fallback",
    "provider_market_cap_missing",
    "reference_market_cap_fallback",
    "reference_market_cap_preferred_over_stale_price_shares",
    "shares_basic_fallback",
}


# -------------------------------
# Data classes / schemas
# -------------------------------


@dataclass
class InputReference:
    artifact_type: str
    artifact_id: str
    source: Optional[str]
    published_at: Optional[str]
    ingested_at: Optional[str]
    hash: Optional[str]


@dataclass
class FeatureRecord:
    name: str
    value: Any
    unit: Optional[str]
    computed_at: str
    as_of_time: str
    window: Optional[Dict[str, Any]]
    confidence: Optional[float]
    provenance: List[InputReference]
    missing_reason: Optional[str]
    fallback_used: Optional[str]
    metric_policy_id: Optional[str] = None
    market_owner: Optional[str] = None
    primary_source_basis: Optional[str] = None
    methodology_registry_id: Optional[str] = None
    methodology_metric_id: Optional[str] = None
    canonical_owner_id: Optional[str] = None
    canonical_owner_name: Optional[str] = None
    canonical_classification: Optional[str] = None
    market_layer_status: Optional[str] = None
    current_alignment_status: Optional[str] = None
    primary_source_document_id: Optional[str] = None
    recommended_metric_name: Optional[str] = None
    input_source_registry_id: Optional[str] = None
    input_source_owner_id: Optional[str] = None
    input_source_owner_name: Optional[str] = None
    input_source_classification: Optional[str] = None
    input_source_formula_basis: Optional[str] = None
    input_source_alignment_status: Optional[str] = None
    input_source_document_ids: Optional[List[str]] = None
    definition_requirement: Optional[str] = None
    definition_requirement_reason: Optional[str] = None
    methodology_execution_decision: Optional[str] = None
    methodology_execution_reason: Optional[str] = None
    input_layer_bucket: Optional[str] = None
    input_layer_bucket_reason: Optional[str] = None
    strict_market_defined: Optional[bool] = None
    archetype: Optional[str] = None
    sector: Optional[str] = None
    subsector: Optional[str] = None
    override_level_applied: Optional[str] = None
    support_mode: Optional[str] = None
    applicability_status: Optional[str] = None
    component_breakdown: Optional[Dict[str, Any]] = None
    quality_flags: Optional[List[str]] = None
    view_type: Optional[str] = None


def _alias_feature_record(
    source: FeatureRecord,
    *,
    name: str,
    value: Any,
    unit: Optional[str] = None,
    primary_source_basis: Optional[str] = None,
    component_breakdown: Optional[Dict[str, Any]] = None,
    extra_quality_flags: Optional[List[str]] = None,
    missing_reason: Optional[str] = None,
) -> FeatureRecord:
    quality_flags = list(source.quality_flags or [])
    for flag in extra_quality_flags or []:
        if flag not in quality_flags:
            quality_flags.append(flag)
    return replace(
        source,
        name=name,
        value=value,
        unit=unit if unit is not None else source.unit,
        primary_source_basis=primary_source_basis if primary_source_basis is not None else source.primary_source_basis,
        component_breakdown=component_breakdown if component_breakdown is not None else source.component_breakdown,
        quality_flags=quality_flags or None,
        missing_reason=missing_reason,
    )


@dataclass
class ConstraintObject:
    name: str
    value: Any
    hardness: str  # "hard" or "soft"
    confidence: Optional[float]
    valid_from: Optional[str]
    valid_to: Optional[str]
    evidence: List[InputReference]


@dataclass
class PeerSet:
    peer_set_id: str
    members: List[str]
    method: str
    version: int


@dataclass
class CompanyStateSnapshot:
    snapshot_id: str
    company_id: str
    as_of_time: str
    features: Dict[str, Dict[str, Any]]
    regime: Dict[str, Any]
    constraint_set: Dict[str, List[Dict[str, Any]]]
    peer_set: Dict[str, Any]
    provenance: Dict[str, Any]


# -------------------------------
# Utilities
# -------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(val: Any) -> Optional[float]:
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        return float(val)
    except Exception:
        return None


def _to_float(val: Any, default: Optional[float] = None) -> Optional[float]:
    parsed = _safe_float(val)
    return default if parsed is None else parsed


def _null_if_na(val: Any) -> Any:
    try:
        if val is None or pd.isna(val):
            return None
    except Exception:
        pass
    return val


def _pick_first_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _pick_first_populated_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c not in df.columns:
            continue
        series = df[c]
        if not series.isna().all():
            return c
    return _pick_first_col(df, candidates)


def _pick_time_col(df: pd.DataFrame) -> Optional[str]:
    return _pick_first_col(
        df,
        [
            "observation_time",
            "event_time",
            "trade_date",
            "effective_at",
            "published_at",
            "available_time",
            "ingestion_time",
            "date",
            "as_of_date",
            "timestamp",
        ],
    )


def _parse_companyfacts_date(value: Any) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    try:
        parsed = pd.to_datetime(value, utc=True, errors="coerce")
    except Exception:
        return None
    if parsed is None or pd.isna(parsed):
        return None
    return pd.Timestamp(parsed)


def _companyfacts_units_map(companyfacts: Dict[str, Any], concept_name: str) -> Optional[Dict[str, Any]]:
    for taxonomy in ("us-gaap", "dei", "ifrs-full"):
        facts = (companyfacts.get("facts") or {}).get(taxonomy) or {}
        if concept_name in facts:
            return facts[concept_name].get("units") or {}
    return None


def _collect_companyfacts_duration_entries(
    companyfacts: Dict[str, Any],
    concept_name: str,
    as_of: pd.Timestamp,
) -> List[Dict[str, Any]]:
    units_map = _companyfacts_units_map(companyfacts, concept_name)
    if not units_map:
        return []
    as_of_date = as_of.date()
    rows: List[Dict[str, Any]] = []
    for unit, entries in units_map.items():
        if str(unit or "").upper() != "USD":
            continue
        for entry in entries:
            start_ts = _parse_companyfacts_date(entry.get("start"))
            end_ts = _parse_companyfacts_date(entry.get("end"))
            filed_ts = _parse_companyfacts_date(entry.get("filed"))
            value = entry.get("val")
            if start_ts is None or end_ts is None or value is None:
                continue
            if end_ts.date() > as_of_date:
                continue
            if filed_ts is not None and filed_ts.date() > as_of_date:
                continue
            if (as_of_date - end_ts.date()).days > MAX_SEC_FACT_AGE_DAYS:
                continue
            duration_days = max(1, (end_ts.date() - start_ts.date()).days + 1)
            rows.append(
                {
                    "concept": concept_name,
                    "start": start_ts,
                    "end": end_ts,
                    "filed": filed_ts or end_ts,
                    "value": float(value),
                    "fy": entry.get("fy"),
                    "fp": entry.get("fp"),
                    "frame": entry.get("frame"),
                    "form": entry.get("form"),
                    "duration_days": duration_days,
                }
            )
    rows.sort(key=lambda item: (item["end"], item["filed"], item["duration_days"]))
    return rows


def _collect_companyfacts_instant_entries(
    companyfacts: Dict[str, Any],
    concept_names: List[str],
    as_of: pd.Timestamp,
) -> List[Dict[str, Any]]:
    as_of_date = as_of.date()
    rows: List[Dict[str, Any]] = []
    for concept_name in concept_names:
        units_map = _companyfacts_units_map(companyfacts, concept_name)
        if not units_map:
            continue
        for unit, entries in units_map.items():
            if str(unit or "").upper() != "USD":
                continue
            for entry in entries:
                end_ts = _parse_companyfacts_date(entry.get("end"))
                filed_ts = _parse_companyfacts_date(entry.get("filed"))
                value = entry.get("val")
                if end_ts is None or value is None:
                    continue
                if end_ts.date() > as_of_date:
                    continue
                if filed_ts is not None and filed_ts.date() > as_of_date:
                    continue
                if (as_of_date - end_ts.date()).days > MAX_SEC_FACT_AGE_DAYS:
                    continue
                rows.append(
                    {
                        "concept": concept_name,
                        "end": end_ts,
                        "filed": filed_ts or end_ts,
                        "value": float(value),
                        "fy": entry.get("fy"),
                        "fp": entry.get("fp"),
                        "frame": entry.get("frame"),
                        "form": entry.get("form"),
                    }
                )
    rows.sort(key=lambda item: (item["end"], item["filed"]))
    return rows


def _latest_companyfacts_point_value(
    companyfacts: Dict[str, Any],
    concept_names: List[str],
    as_of: pd.Timestamp,
) -> tuple[Optional[float], Optional[Dict[str, Any]]]:
    entries = _collect_companyfacts_instant_entries(companyfacts, concept_names, as_of)
    if not entries:
        return None, None
    latest = max(entries, key=lambda item: (item["end"], item["filed"]))
    return float(latest["value"]), {
        "concept": latest.get("concept"),
        "mode": "latest_balance_sheet_point",
        "end": latest["end"].date().isoformat(),
        "filed": latest["filed"].date().isoformat(),
        "fy": latest.get("fy"),
        "fp": latest.get("fp"),
        "frame": latest.get("frame"),
        "form": latest.get("form"),
        "formula": "latest_companyfacts_point_value",
    }


def _latest_companyfacts_meta_timestamp(meta: Any) -> Optional[pd.Timestamp]:
    latest: Optional[pd.Timestamp] = None

    def _visit(node: Any) -> None:
        nonlocal latest
        if isinstance(node, dict):
            for key in ("filed", "published_at", "end"):
                ts = _parse_companyfacts_date(node.get(key))
                if ts is not None and (latest is None or ts > latest):
                    latest = ts
            for value in node.values():
                _visit(value)
        elif isinstance(node, list):
            for item in node:
                _visit(item)

    _visit(meta)
    return latest


def _should_use_fresher_companyfacts_value(
    current_value: Optional[float],
    current_published_at: Optional[str],
    companyfacts_value: Optional[float],
    companyfacts_meta: Any,
) -> bool:
    if companyfacts_value is None:
        return False
    if current_value is None:
        return True
    companyfacts_ts = _latest_companyfacts_meta_timestamp(companyfacts_meta)
    if companyfacts_ts is None:
        return False
    current_ts = _parse_companyfacts_date(current_published_at)
    if current_ts is None:
        return True
    return companyfacts_ts >= current_ts + pd.Timedelta(days=COMPANYFACTS_FRESHER_OVERRIDE_MIN_DAYS)


def _companyfacts_input_reference(
    companyfacts_path: Optional[Path],
    meta: Any,
) -> Optional[Dict[str, Any]]:
    if companyfacts_path is None:
        return None
    published_ts = _latest_companyfacts_meta_timestamp(meta)
    published_at = published_ts.isoformat() if published_ts is not None else None
    return {
        "artifact_type": "SecCompanyFacts",
        "artifact_id": f"sec_companyfacts:{companyfacts_path.name}",
        "source": str(companyfacts_path),
        "published_at": published_at,
        "ingested_at": published_at,
        "hash": None,
    }


def _compute_companyfacts_ttm_from_concept(
    companyfacts: Dict[str, Any],
    concept_name: str,
    as_of: pd.Timestamp,
) -> tuple[Optional[float], Optional[Dict[str, Any]]]:
    entries = _collect_companyfacts_duration_entries(companyfacts, concept_name, as_of)
    if not entries:
        return None, None

    latest = max(entries, key=lambda item: (item["end"], item["filed"], item["duration_days"]))
    latest_fp = str(latest.get("fp") or "").upper()
    if latest_fp == "FY" or latest["duration_days"] >= 300:
        return float(latest["value"]), {
            "concept": concept_name,
            "mode": "latest_fy",
            "end": latest["end"].date().isoformat(),
            "filed": latest["filed"].date().isoformat(),
            "fy": latest.get("fy"),
            "fp": latest.get("fp"),
            "frame": latest.get("frame"),
            "form": latest.get("form"),
            "formula": "latest_fiscal_year_value",
        }

    if latest_fp not in {"Q1", "Q2", "Q3"}:
        return None, None

    current_fy = latest.get("fy")
    if current_fy is None:
        return None, None
    try:
        prior_fy = int(current_fy) - 1
    except Exception:
        return None, None

    annual = None
    prior_same = None
    for entry in entries:
        entry_fp = str(entry.get("fp") or "").upper()
        if entry.get("fy") == prior_fy and entry_fp == "FY":
            if annual is None or (entry["end"], entry["filed"], entry["duration_days"]) > (
                annual["end"],
                annual["filed"],
                annual["duration_days"],
            ):
                annual = entry
        if entry.get("fy") == prior_fy and entry_fp == latest_fp:
            if prior_same is None or (entry["end"], entry["filed"], entry["duration_days"]) > (
                prior_same["end"],
                prior_same["filed"],
                prior_same["duration_days"],
            ):
                prior_same = entry

    if annual is None or prior_same is None:
        return None, None

    return float(latest["value"] + annual["value"] - prior_same["value"]), {
        "concept": concept_name,
        "mode": "ytd_plus_prior_fy_minus_prior_ytd",
        "latest": {
            "end": latest["end"].date().isoformat(),
            "filed": latest["filed"].date().isoformat(),
            "fy": latest.get("fy"),
            "fp": latest.get("fp"),
            "frame": latest.get("frame"),
            "form": latest.get("form"),
            "value": latest["value"],
        },
        "prior_fy": {
            "end": annual["end"].date().isoformat(),
            "filed": annual["filed"].date().isoformat(),
            "fy": annual.get("fy"),
            "fp": annual.get("fp"),
            "frame": annual.get("frame"),
            "form": annual.get("form"),
            "value": annual["value"],
        },
        "prior_same_period": {
            "end": prior_same["end"].date().isoformat(),
            "filed": prior_same["filed"].date().isoformat(),
            "fy": prior_same.get("fy"),
            "fp": prior_same.get("fp"),
            "frame": prior_same.get("frame"),
            "form": prior_same.get("form"),
            "value": prior_same["value"],
        },
        "formula": "latest_ytd + prior_fy - prior_same_period_ytd",
    }


def _support_mode_is_exact_like(mode: Optional[str]) -> bool:
    return str(mode or "").strip().lower() in EXACT_SUPPORT_MODES


def _is_exact_structural_zero_metric(
    metric_id: Optional[str],
    value: Any,
    component_breakdown: Optional[Dict[str, Any]],
) -> bool:
    if metric_id != "capital_structure.total_debt":
        return False
    value_f = _safe_float(value)
    if value_f is None or abs(value_f) > 1e-9:
        return False
    breakdown = component_breakdown or {}
    for key in (
        "local_reported_debt",
        "lease_liabilities",
        "included_lease_liabilities",
        "supplier_finance",
        "included_supplier_finance",
        "preferred_equity",
        "convertibles",
        "unfunded_pension",
    ):
        component_val = _safe_float(breakdown.get(key))
        if component_val not in (None, 0.0):
            return False
    return True


def _classify_metric_support_mode(
    *,
    base_mode: Optional[str],
    metric_id: Optional[str],
    value: Any,
    quality_flags: Optional[List[str]],
    component_breakdown: Optional[Dict[str, Any]],
) -> str:
    mode = str(base_mode or "exact").strip().lower() or "exact"
    if mode in {"unsupported", "inferred", "proxy", "proxy_missing_component"}:
        return mode
    if not _support_mode_is_exact_like(mode):
        return mode

    flags = [str(flag).strip().lower() for flag in (quality_flags or []) if flag is not None]
    effective_flags = [flag for flag in flags if flag not in NON_PROXY_DIAGNOSTIC_FLAGS]
    if effective_flags:
        if any(
            flag in PROXY_COMPONENT_SUPPORT_FLAGS
            or any(token in flag for token in ("missing", "fallback", "proxy", "estimated", "assumed_zero"))
            for flag in effective_flags
        ):
            return "proxy_missing_component"
        return "proxy"

    if _is_exact_structural_zero_metric(metric_id, value, component_breakdown):
        return "exact_structural_zero"
    return mode


def _pick_value_col(df: pd.DataFrame) -> Optional[str]:
    return _pick_first_col(
        df,
        [
            "value",
            "close",
            "adjusted_close",
            "consensus_value",
            "fact_value",
            "numeric_value",
            "amount",
        ],
    )


def _pick_price_col(df: pd.DataFrame) -> Optional[str]:
    return _pick_first_col(df, ["adjusted_close", "close", "value"])


def _pick_price_time_col(df: pd.DataFrame) -> Optional[str]:
    # Price history should prefer the actual trading date over generic event /
    # availability timestamps so rolling windows are aligned to market sessions.
    return _pick_first_col(
        df,
        [
            "trade_date",
            "observation_time",
            "event_time",
            "effective_at",
            "published_at",
            "available_time",
            "ingestion_time",
            "date",
            "as_of_date",
            "timestamp",
        ],
    )


def _parse_dealscan_ratio_value(val: Any) -> Optional[float]:
    raw = _null_if_na(val)
    if raw in (None, ""):
        return None
    parsed = _safe_float(raw)
    if parsed is not None:
        return float(parsed)
    text = str(raw).strip().replace(",", "")
    if not text:
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", text, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(0))
    except Exception:
        return None


def _looks_like_equity_instrument(text: Any) -> bool:
    lowered = str(text or "").strip().lower()
    if not lowered:
        return False
    return any(token in lowered for token in ["equity", "stock", "common", "ordinary", "share"])


def _looks_like_debt_instrument(text: Any) -> bool:
    lowered = str(text or "").strip().lower()
    if not lowered:
        return False
    return any(token in lowered for token in ["bond", "note", "debt", "loan", "credit"])


def _price_group_col(df: pd.DataFrame) -> Optional[str]:
    for candidate in ["security_id", "instrument_id", "series_id", "field_name", "metric"]:
        if candidate in df.columns and df[candidate].notna().any():
            return candidate
    return None


def _prepare_price_series(
    ts: pd.DataFrame,
) -> Tuple[Optional[pd.DataFrame], Optional[float], Optional[str], Optional[str], Dict[str, Any], List[str]]:
    if ts is None or ts.empty:
        return None, None, None, None, {}, ["price_history_unavailable"]

    time_col = _pick_price_time_col(ts)
    if time_col is None:
        return None, None, None, None, {}, ["price_history_unavailable"]

    candidate_specs: List[Tuple[str, str, pd.DataFrame]] = []
    for price_col in ("adjusted_close", "close"):
        if price_col in ts.columns:
            candidate_specs.append(("wide", price_col, ts))

    if "value" in ts.columns:
        long_df = ts.copy()
        selector = pd.Series(False, index=long_df.index)
        if "series_type" in long_df.columns:
            selector |= long_df["series_type"].astype(str).str.lower().eq("price")
        for candidate_col in ("series_id", "field_name", "metric"):
            if candidate_col in long_df.columns:
                selector |= long_df[candidate_col].astype(str).str.contains(
                    "adjusted[_ ]?close|close|price",
                    case=False,
                    na=False,
                )
        if selector.any():
            candidate_specs.append(("long", "value", long_df[selector].copy()))

    candidates: List[Dict[str, Any]] = []
    series_type_filtered = False
    for source_kind, price_col, raw_df in candidate_specs:
        if raw_df.empty:
            continue
        df = raw_df.copy()
        if "series_type" in df.columns:
            price_mask = df["series_type"].astype(str).str.lower().eq("price")
            if price_mask.any():
                if (~price_mask).any():
                    series_type_filtered = True
                df = df[price_mask].copy()
        df["obs_time"] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
        df["price"] = pd.to_numeric(df[price_col], errors="coerce")
        df = df.dropna(subset=["obs_time", "price"])
        df = df[df["price"] > 0].copy()
        if df.empty:
            continue

        group_col = _price_group_col(df)
        grouped = [("__all__", df)] if group_col is None else list(df.groupby(group_col, dropna=False, sort=False))
        for group_value, group_df in grouped:
            group_df = group_df.sort_values("obs_time").drop_duplicates(subset=["obs_time"], keep="last")
            if group_df.empty:
                continue
            instrument_type = None
            if "instrument_type" in group_df.columns:
                instrument_type = _null_if_na(group_df["instrument_type"].dropna().astype(str).iloc[0]) if not group_df["instrument_type"].dropna().empty else None
            series_type = None
            if "series_type" in group_df.columns:
                series_type = _null_if_na(group_df["series_type"].dropna().astype(str).iloc[0]) if not group_df["series_type"].dropna().empty else None
            latest_obs = group_df["obs_time"].max()
            score = (
                1 if price_col == "adjusted_close" else 0,
                1 if str(series_type or "").lower() == "price" else 0,
                1 if _looks_like_equity_instrument(instrument_type) else 0,
                1 if not _looks_like_debt_instrument(instrument_type) else 0,
                int(len(group_df)),
                int(latest_obs.value) if pd.notna(latest_obs) else -1,
            )
            candidates.append(
                {
                    "df": group_df[["obs_time", "price"]].copy(),
                    "price_col": price_col,
                    "source_kind": source_kind,
                    "time_col": time_col,
                    "group_col": group_col,
                    "group_value": _null_if_na(group_value),
                    "instrument_type": instrument_type,
                    "series_type": series_type,
                    "score": score,
                }
            )

    if not candidates:
        return None, None, None, None, {}, ["price_history_unavailable"]

    best = max(candidates, key=lambda item: item["score"])
    price_df = best["df"].sort_values("obs_time").reset_index(drop=True)
    latest_row = price_df.iloc[-1]
    breakdown: Dict[str, Any] = {
        "source_kind": best["source_kind"],
        "price_field": best["price_col"],
        "time_field": best["time_col"],
        "group_field": best["group_col"],
        "group_value": best["group_value"],
        "series_type": best["series_type"],
        "instrument_type": best["instrument_type"],
        "candidate_series_evaluated": int(len(candidates)),
        "price_observations": int(len(price_df)),
        "latest_price": _safe_float(latest_row.get("price")),
        "latest_observation_time": str(latest_row.get("obs_time")) if latest_row.get("obs_time") is not None else None,
    }
    flags: List[str] = []
    if series_type_filtered:
        flags.append("non_price_series_filtered")
    if len(candidates) > 1:
        flags.append("multiple_price_series_candidates")
    return price_df, _safe_float(latest_row.get("price")), best["price_col"], "obs_time", breakdown, flags


def _as_of_ts_literal(as_of_dt: pd.Timestamp) -> str:
    # DuckDB TIMESTAMP literal (naive). Avoid TIMESTAMPTZ comparisons against timestamp_ns.
    try:
        naive = as_of_dt.tz_convert(None)
    except Exception:
        naive = as_of_dt
    return naive.isoformat()


def _sql_quote(s: str) -> str:
    return "'" + str(s).replace("'", "''") + "'"


def _is_readable_file(path: Path) -> bool:
    try:
        st = path.stat()
        if st.st_size <= 0:
            return False
        return True
    except Exception:
        return False


def _zscore(series: pd.Series) -> Optional[float]:
    if series is None or series.empty:
        return None
    s = series.dropna().astype(float)
    if len(s) < 10:
        return None
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0:
        return None
    return float((s.iloc[-1] - mu) / sd)


def _percentile(series: pd.Series) -> Optional[float]:
    if series is None or series.empty:
        return None
    s = series.dropna().astype(float)
    if len(s) < 10:
        return None
    return float((s.rank(pct=True).iloc[-1]) * 100.0)


# -------------------------------
# Builder
# -------------------------------


class CompanyStateBuilder:
    def __init__(
        self,
        raw_timeseries_path: Path | str = "data/inputs_layer/raw_timeseries.parquet",
        macro_timeseries_path: Path | str | None = None,
        event_store_path: Path | str = "data/inputs_layer/event_store.parquet",
        corporate_actions_master_path: Path | str = "data/curated/corporate_actions_master.parquet",
        facts_path: Path | str = "data/inputs_layer/extracted_fact_registry_validity",
        dealscan_revolver_path: Path | str = "data/wrds/dealscan/loanconnector_revolver_facilities.parquet",
        ownership_summary_path: Path | str = "data/inputs_layer/ownership_13f_summary.parquet",
        issuer_ratings_path: Path | str = "data/inputs_layer/issuer_rating_history.parquet",
        estimates_path: Path | str = "data/warehouse/warehouse_estimates.parquet",
        entity_graph_path: Path | str = "data/inputs_layer/entity_graph.parquet",
        entity_identifier_path: Path | str = "data/inputs_layer/entity_identifier.parquet",
        entity_table_path: Path | str = "data/inputs_layer/entity.parquet",
        taxonomy_reference_path: Path | str = "data/refinitiv/fundamentals_all.parquet",
        metric_policy_path: Path | str | None = None,
        methodology_registry_path: Path | str | None = None,
        input_source_registry_path: Path | str | None = None,
        use_duckdb: bool = True,
        skip_timeseries: bool = False,
        skip_macro: bool = False,
        facts_years: Optional[List[int]] = None,
        skip_events: bool = False,
        skip_peer_context: bool = False,
        debug: bool = False,
        cache_facts: bool = False,
        cache_events: bool = False,
        cache_timeseries: bool = False,
        cache_ownership: bool = False,
        cache_ratings: bool = False,
        historical_backfill_mode: bool = False,
        companyfacts_root: Path | str | None = None,
        enable_market_relevant_smart_normalized_inputs: bool = False,
    ) -> None:
        self.raw_timeseries_path = resolve_data_path(raw_timeseries_path)
        self.macro_timeseries_path = resolve_data_path(macro_timeseries_path) if macro_timeseries_path else None
        self.event_store_path = resolve_data_path(event_store_path)
        self.corporate_actions_master_path = resolve_data_path(corporate_actions_master_path)
        self.facts_path = resolve_data_path(facts_path)
        self.dealscan_revolver_path = resolve_data_path(dealscan_revolver_path)
        self.ownership_summary_path = resolve_data_path(ownership_summary_path)
        self.issuer_ratings_path = resolve_data_path(issuer_ratings_path)
        self.estimates_path = resolve_data_path(estimates_path)
        self.entity_graph_path = resolve_data_path(entity_graph_path)
        self.entity_identifier_path = resolve_data_path(entity_identifier_path)
        self.entity_table_path = resolve_data_path(entity_table_path)
        self.taxonomy_reference_path = resolve_data_path(taxonomy_reference_path)
        self.metric_policy = MetricPolicyEngine(
            metric_policy_path,
            methodology_registry_path=methodology_registry_path,
        )
        self.input_source_registry = CompanyStateInputSourceRegistry(input_source_registry_path)
        self.use_duckdb = use_duckdb
        self._entity_table_cache: Optional[pd.DataFrame] = None
        self._taxonomy_reference_cache: Optional[pd.DataFrame] = None
        self._parquet_columns_cache: Dict[Path, set[str]] = {}
        self._macro_cache: Dict[str, pd.DataFrame] = {}
        self._identifier_to_entity: Optional[Dict[str, str]] = None
        self._entity_to_identifiers: Optional[Dict[str, List[str]]] = None
        self.skip_timeseries = skip_timeseries
        self.skip_macro = skip_macro
        self.facts_years = facts_years
        self.skip_events = skip_events
        self.skip_peer_context = skip_peer_context
        self.debug = debug
        self.cache_facts = cache_facts
        self.cache_events = cache_events
        self.cache_timeseries = cache_timeseries
        self.cache_ownership = cache_ownership
        self.cache_ratings = cache_ratings
        self.historical_backfill_mode = historical_backfill_mode
        self.companyfacts_root = resolve_companyfacts_root(companyfacts_root)
        self.enable_market_relevant_smart_normalized_inputs = enable_market_relevant_smart_normalized_inputs
        self._facts_cache: Optional[pd.DataFrame] = None
        self._events_cache: Optional[pd.DataFrame] = None
        self._corporate_actions_cache: Optional[pd.DataFrame] = None
        self._timeseries_cache: Optional[pd.DataFrame] = None
        self._ownership_cache: Optional[pd.DataFrame] = None
        self._issuer_ratings_cache: Optional[pd.DataFrame] = None
        self._estimates_cache: Optional[pd.DataFrame] = None
        self._dealscan_revolver_cache: Optional[pd.DataFrame] = None
        self._facts_files_cache: Dict[str, List[str]] = {}
        self._companyfacts_cache: Dict[Path, Optional[Dict[str, Any]]] = {}
        self._smart_metric_registry_cache: Optional[Dict[str, Any]] = None
        self._market_availability_overrides_cache: Optional[Dict[str, Any]] = None

        self.fact_map = {
            "cash": ["financial.cash", "financial.cash_and_equivalents", "cash_and_equivalents"],
            "restricted_cash": [
                "financial.restricted_cash",
                "financial.cash_restricted",
                "financial.restricted_cash_and_equivalents",
                "financial.cash_and_cash_equivalents_restricted",
                "financial.restricted_and_escrowed_cash",
                "financial.escrow_cash",
                "restricted_cash",
            ],
            "restricted_cash_current": [
                "financial.restricted_cash_current",
                "financial.cash_restricted_current",
                "financial.escrow_cash_current",
            ],
            "restricted_cash_noncurrent": [
                "financial.restricted_cash_noncurrent",
                "financial.cash_restricted_noncurrent",
                "financial.escrow_cash_noncurrent",
            ],
            "marketable_securities": [
                "financial.marketable_securities",
                "financial.current_marketable_securities",
                "financial.marketable_securities_current",
                "financial.short_term_investments",
                "financial.short_term_investment",
                "financial.current_investments",
                "financial.investments_current",
                "financial.available_for_sale_securities_current",
                "marketable_securities",
            ],
            "cash_and_short_term_investments": [
                "financial.cash_and_short_term_investments",
                "financial.cash_short_term_investments",
                "financial.cash_and_investments_current",
            ],
            "unavailable_cash": [
                "financial.trapped_cash",
                "financial.unavailable_cash",
                "financial.cash_not_freely_available",
            ],
            "revolver_undrawn": [
                "liquidity.revolver_undrawn",
                "financial.revolver_undrawn",
                "financial.revolving_credit_facility_undrawn",
                "financial.revolving_credit_facility_available",
                "financial.credit_facility_undrawn",
                "financial.credit_facility_available",
                "financial.line_of_credit_available",
                "financial.unused_committed_credit_lines",
                "financial.undrawn_revolver_capacity",
                "revolver_undrawn",
            ],
            "debt_current": ["financial.debt_current", "financial.short_term_debt"],
            "debt_long": ["financial.debt_long_term", "financial.long_term_debt"],
            "total_debt": ["financial.total_debt", "financial.debt_total"],
            "lease_current": [
                "financial.lease_liability_current",
                "financial.operating_lease_liability_current",
                "financial.finance_lease_liability_current",
                "financial.lease_debt_current",
            ],
            "lease_long": [
                "financial.lease_liability_noncurrent",
                "financial.operating_lease_liability_noncurrent",
                "financial.finance_lease_liability_noncurrent",
                "financial.lease_debt_noncurrent",
            ],
            "supplier_finance": [
                "financial.supplier_finance_obligation",
                "financial.supply_chain_finance",
                "financial.vendor_financing",
            ],
            "preferred_equity": [
                "financial.preferred_equity",
                "financial.preferred_stock",
                "financial.redeemable_preferred",
            ],
            "convertibles": [
                "financial.convertible_debt",
                "financial.convertible_notes",
            ],
            "unfunded_pension": [
                "financial.unfunded_pension",
                "financial.pension_deficit",
                "financial.net_pension_liability",
            ],
            "revenue": ["financial.revenue", "financial.total_revenue"],
            "ebitda": ["financial.ebitda", "financial.ebitda_ttm"],
            "ebit": ["financial.ebit", "financial.operating_income"],
            "net_income": ["financial.net_income"],
            "diluted_eps": [
                "financial.diluted_eps",
                "financial.eps_diluted",
                "financial.diluted_eps_ttm",
                "financial.eps_diluted_ttm",
            ],
            "basic_eps": [
                "financial.basic_eps",
                "financial.eps_basic",
                "financial.basic_eps_ttm",
                "financial.eps_basic_ttm",
            ],
            "fcf": ["financial.free_cash_flow", "financial.fcf"],
            "operating_cash_flow": [
                "financial.operating_cash_flow",
                "financial.cash_flow_from_operations",
                "financial.net_cash_from_operations",
            ],
            "interest_expense": ["financial.interest_expense"],
            "interest_income": ["financial.interest_income", "financial.interest_received"],
            "associate_dividends": [
                "financial.associate_dividends",
                "financial.equity_method_dividends",
                "financial.recurring_dividends_from_associates",
            ],
            "minority_dividends": [
                "financial.dividends_paid_to_minorities",
                "financial.net_income_attributable_to_non_controlling_interests",
            ],
            "preferred_dividends": [
                "financial.preferred_dividends_paid",
                "financial.preferred_dividend",
            ],
            "common_dividends_cash": [
                "financial.dividends_cash",
                "financial.common_dividends_cash",
                "financial.cash_dividends",
                "financial.dividends_paid_common",
            ],
            "dividends_per_share_cash": [
                "financial.dividends_per_share_cash",
                "financial.common_dividends_per_share_cash",
                "financial.dividend_per_share",
            ],
            "lease_expense": [
                "financial.lease_expense",
                "financial.operating_lease_expense",
                "financial.rent_expense",
            ],
            "capex": ["financial.capex"],
            "working_capital": ["financial.working_capital"],
            # Many upstream runs populate shares as financial.shares_out.
            "shares_basic": [
                "financial.shares_basic",
                "financial.shares_outstanding",
                "financial.shares_out",
                "shares_outstanding",
            ],
            "shares_diluted": [
                "financial.shares_diluted",
                "financial.diluted_shares_outstanding",
                "diluted_shares_outstanding",
            ],
        }
        self.fact_pattern_map = {
            "restricted_cash": {
                "contains_any": [
                    "restricted_cash",
                    "cash_restricted",
                    "restricted_and_escrow",
                    "escrow_cash",
                    "cash_equivalents_restricted",
                ],
                "exclude_any": [
                    "unrestricted",
                    "restricted_cash_current",
                    "cash_restricted_current",
                    "escrow_cash_current",
                    "restricted_cash_noncurrent",
                    "cash_restricted_noncurrent",
                    "escrow_cash_noncurrent",
                ],
            },
            "restricted_cash_current": {
                "contains_any": [
                    "restricted_cash_current",
                    "cash_restricted_current",
                    "escrow_cash_current",
                ],
            },
            "restricted_cash_noncurrent": {
                "contains_any": [
                    "restricted_cash_noncurrent",
                    "cash_restricted_noncurrent",
                    "escrow_cash_noncurrent",
                ],
            },
            "marketable_securities": {
                "contains_any": [
                    "marketable_securit",
                    "short_term_invest",
                    "current_invest",
                    "available_for_sale_securit",
                ],
                "exclude_any": [
                    "cash_and_short_term_invest",
                    "cash_short_term_invest",
                    "long_term",
                    "noncurrent",
                ],
            },
            "cash_and_short_term_investments": {
                "contains_any": [
                    "cash_and_short_term_invest",
                    "cash_short_term_invest",
                    "cash_and_investments_current",
                ],
            },
            "revolver_undrawn": {
                "contains_any": [
                    "revolver",
                    "revolving_credit",
                    "credit_facilit",
                    "line_of_credit",
                ],
                "require_any": ["undrawn", "available", "unused", "unutilized", "remaining"],
                "exclude_any": ["drawn", "outstanding", "borrowed", "debt"],
            },
        }

        self.macro_series = {
            "fed_funds_effective": "DFF",
            "hy_oas": "BAMLH0A0HYM2",
            "hy_all_in_yield": "BAMLH0A0HYM2EY",
            "ig_oas": "BAMLC0A0CM",
            "sofr": "SOFR",
            "vix": "VIXCLS",
            "sp500": "SP500",
            "sp500_pe_ttm": "SP500_PE_RATIO",
            "rate_2y": "DGS2",
            "rate_10y": "DGS10",
            "real_gdp": "GDPC1",
        }
        self.regime_classifier = RegimeClassifier(self.macro_series)
        self.peer_resolver = PeerSetResolver()

    def _artifact_ref(
        self,
        artifact_type: str,
        artifact_id: str,
        source: Optional[str],
        as_of_dt: Optional[pd.Timestamp] = None,
    ) -> InputReference:
        ts = as_of_dt.isoformat() if as_of_dt is not None else None
        return InputReference(
            artifact_type=artifact_type,
            artifact_id=artifact_id,
            source=source,
            published_at=ts,
            ingested_at=ts,
            hash=None,
        )

    def _default_input_refs(self, as_of_dt: pd.Timestamp) -> Dict[str, InputReference]:
        date_key = as_of_dt.strftime("%Y-%m-%d")
        return {
            "facts": self._artifact_ref(
                "ExtractedFact",
                f"facts:{date_key}",
                str(self.facts_path),
                as_of_dt,
            ),
            "timeseries": self._artifact_ref(
                "RawTimeseries",
                f"timeseries:{date_key}",
                str(self.raw_timeseries_path),
                as_of_dt,
            ),
            "macro": self._artifact_ref(
                "RawTimeseries",
                f"macro:{date_key}",
                str(self.macro_timeseries_path or self.raw_timeseries_path),
                as_of_dt,
            ),
            "events": self._artifact_ref(
                "Event",
                f"events:{date_key}",
                str(self.event_store_path),
                as_of_dt,
            ),
            "ownership": self._artifact_ref(
                "RawDocument",
                f"ownership:{date_key}",
                str(self.ownership_summary_path),
                as_of_dt,
            ),
            "issuer_ratings": self._artifact_ref(
                "ExtractedFact",
                f"issuer_ratings:{date_key}",
                str(self.issuer_ratings_path),
                as_of_dt,
            ),
            "entity": self._artifact_ref(
                "RawDocument",
                f"entity:{date_key}",
                str(self.entity_table_path),
                as_of_dt,
            ),
        }

    def _fundamentals_reference_input_ref(
        self,
        reference_row: Optional[Dict[str, Any]],
        field_name: str,
        as_of_dt: Optional[pd.Timestamp],
    ) -> Optional[InputReference]:
        if not reference_row:
            return None
        instrument = _null_if_na(reference_row.get("Instrument"))
        if instrument is None:
            return None
        field_value = _null_if_na(reference_row.get(field_name))
        if field_value is None:
            return None
        return self._artifact_ref(
            "FundamentalsReference",
            f"{instrument}:{field_name}",
            str(self.taxonomy_reference_path),
            as_of_dt=as_of_dt,
        )

    def _fallback_provenance_for_feature(
        self,
        feature_name: str,
        as_of_dt: pd.Timestamp,
    ) -> List[InputReference]:
        refs = self._default_input_refs(as_of_dt)
        if feature_name.startswith("liquidity."):
            return [refs["facts"]]
        if feature_name.startswith("capital_structure."):
            out = [refs["facts"]]
            if "rating" in feature_name:
                out.append(refs["issuer_ratings"])
                out.append(refs["events"])
            if "debt_due_" in feature_name or "maturity" in feature_name or "refi" in feature_name:
                out.append(refs["events"])
            return out
        if feature_name.startswith("market."):
            out = [refs["timeseries"], refs["facts"]]
            if "window_proxy" in feature_name:
                out.append(refs["macro"])
            return out
        if feature_name.startswith("macro."):
            return [refs["macro"]]
        if feature_name.startswith("operating."):
            out = [refs["facts"]]
            if "cyclicality" in feature_name:
                out.append(refs["macro"])
            return out
        if feature_name.startswith("ownership_governance."):
            return [refs["ownership"], refs["events"], refs["facts"]]
        if feature_name.startswith("strategic."):
            return [refs["facts"], refs["events"]]
        if feature_name.startswith("peer_context."):
            return [refs["entity"], refs["events"], refs["facts"]]
        return [refs["facts"]]

    def _feature_transform_steps(self, feature_name: str, feat: FeatureRecord) -> List[str]:
        steps: List[str] = []
        if feature_name.startswith("liquidity."):
            steps.extend(["extract_latest_facts", "compute_liquidity_metrics"])
        elif feature_name.startswith("capital_structure."):
            steps.extend(["extract_latest_facts", "compute_capital_structure_metrics"])
        elif feature_name.startswith("market."):
            steps.extend(["extract_market_timeseries", "compute_market_metrics"])
        elif feature_name.startswith("macro."):
            steps.extend(["extract_macro_timeseries", "compute_macro_metrics"])
        elif feature_name.startswith("operating."):
            steps.extend(["extract_operating_facts", "compute_operating_metrics"])
        elif feature_name.startswith("ownership_governance."):
            steps.extend(["extract_ownership_signals", "compute_ownership_governance_metrics"])
        elif feature_name.startswith("strategic."):
            steps.extend(["extract_strategic_signals", "compute_strategic_metrics"])
        elif feature_name.startswith("peer_context."):
            steps.extend(["resolve_peer_set", "compute_peer_relative_metrics"])
        else:
            steps.append("compute_feature")
        if feat.fallback_used not in (None, "", "none"):
            steps.append(f"fallback:{feat.fallback_used}")
        if feat.missing_reason:
            steps.append(f"missing:{feat.missing_reason}")
        return steps

    def _finalize_feature_provenance(
        self,
        features: Dict[str, FeatureRecord],
        as_of_dt: pd.Timestamp,
    ) -> Dict[str, Dict[str, Any]]:
        lineage: Dict[str, Dict[str, Any]] = {}
        for name, feat in features.items():
            if not isinstance(feat, FeatureRecord):
                continue
            if not feat.provenance:
                feat.provenance = self._fallback_provenance_for_feature(name, as_of_dt)
            lineage[name] = {
                "inputs": [asdict(r) for r in feat.provenance],
                "transforms": self._feature_transform_steps(name, feat),
                "computation_version": "state_builder_v5",
                "metric_context": {
                    "metric_policy_id": feat.metric_policy_id,
                    "market_owner": feat.market_owner,
                    "primary_source_basis": feat.primary_source_basis,
                    "methodology_registry_id": feat.methodology_registry_id,
                    "methodology_metric_id": feat.methodology_metric_id,
                    "canonical_owner_id": feat.canonical_owner_id,
                    "canonical_owner_name": feat.canonical_owner_name,
                    "canonical_classification": feat.canonical_classification,
                    "market_layer_status": feat.market_layer_status,
                    "current_alignment_status": feat.current_alignment_status,
                    "primary_source_document_id": feat.primary_source_document_id,
                    "recommended_metric_name": feat.recommended_metric_name,
                    "input_source_registry_id": feat.input_source_registry_id,
                    "input_source_owner_id": feat.input_source_owner_id,
                    "input_source_owner_name": feat.input_source_owner_name,
                    "input_source_classification": feat.input_source_classification,
                    "input_source_formula_basis": feat.input_source_formula_basis,
                    "input_source_alignment_status": feat.input_source_alignment_status,
                    "input_source_document_ids": feat.input_source_document_ids or [],
                    "definition_requirement": feat.definition_requirement,
                    "definition_requirement_reason": feat.definition_requirement_reason,
                    "methodology_execution_decision": feat.methodology_execution_decision,
                    "methodology_execution_reason": feat.methodology_execution_reason,
                    "input_layer_bucket": feat.input_layer_bucket,
                    "input_layer_bucket_reason": feat.input_layer_bucket_reason,
                    "strict_market_defined": feat.strict_market_defined,
                    "archetype": feat.archetype,
                    "sector": feat.sector,
                    "subsector": feat.subsector,
                    "override_level_applied": feat.override_level_applied,
                    "support_mode": feat.support_mode,
                    "applicability_status": feat.applicability_status,
                    "view_type": feat.view_type,
                    "component_breakdown": feat.component_breakdown or {},
                    "quality_flags": feat.quality_flags or [],
                },
            }
        return lineage

    def _load_taxonomy_reference(self) -> pd.DataFrame:
        if self._taxonomy_reference_cache is not None:
            return self._taxonomy_reference_cache
        if not self.taxonomy_reference_path.exists():
            self._taxonomy_reference_cache = pd.DataFrame()
            return self._taxonomy_reference_cache
        try:
            con = duckdb.connect()
            self._taxonomy_reference_cache = con.execute(
                f"SELECT * FROM read_parquet('{self.taxonomy_reference_path.as_posix()}', union_by_name=True)"
            ).df()
        except Exception:
            self._taxonomy_reference_cache = pd.DataFrame()
        return self._taxonomy_reference_cache

    def _taxonomy_reference_row(self, aliases: List[str]) -> Dict[str, Any]:
        df = self._load_taxonomy_reference()
        if df is None or df.empty or "Instrument" not in df.columns:
            return {}
        instrument_series = df["Instrument"].astype(str)
        instrument_root = instrument_series.str.split(".").str[0].str.upper()
        alias_set = {str(alias).upper() for alias in aliases if alias is not None}
        if not alias_set:
            return {}
        matched = df[instrument_root.isin(alias_set)].copy()
        if matched.empty:
            return {}
        row = matched.iloc[0]
        return {
            "sector": _null_if_na(row.get("GICS Sector Name")),
            "gics_sector": _null_if_na(row.get("GICS Sector Name")),
            "industry": _null_if_na(row.get("GICS Industry Name")),
            "subsector": _null_if_na(row.get("GICS Industry Name")),
            "gics_industry": _null_if_na(row.get("GICS Industry Name")),
            "gics_sub_industry": _null_if_na(row.get("GICS Industry Name")),
            "company_common_name": _null_if_na(row.get("Company Common Name")),
            "instrument": _null_if_na(row.get("Instrument")),
        }

    def _fundamentals_reference_row(self, aliases: List[str]) -> Dict[str, Any]:
        df = self._load_taxonomy_reference()
        if df is None or df.empty or "Instrument" not in df.columns:
            return {}
        instrument_series = df["Instrument"].astype(str)
        instrument_root = instrument_series.str.split(".").str[0].str.upper()
        alias_set = {str(alias).upper() for alias in aliases if alias is not None}
        if not alias_set:
            return {}
        matched = df[instrument_root.isin(alias_set)].copy()
        if matched.empty:
            return {}
        row = matched.iloc[0]
        return row.to_dict()

    def _reference_aliases(self, reference_row: Optional[Dict[str, Any]]) -> List[str]:
        row = reference_row or {}
        aliases: List[str] = []
        for key in (
            "Instrument",
            "Company Common Name",
            "Issuer Name",
            "Company Name",
            "company_common_name",
            "instrument",
            "issuer_name",
            "company_name",
        ):
            value = _null_if_na(row.get(key))
            if value in (None, ""):
                continue
            text = str(value).strip()
            if not text:
                continue
            aliases.append(text)
            if "." in text:
                aliases.append(text.split(".")[0])
        return aliases

    def _entity_context_row(self, company_id: str, aliases: Optional[List[str]] = None) -> Dict[str, Any]:
        entity_table = self._load_entity_table()
        base_row: Dict[str, Any] = {}
        if entity_table is not None and not entity_table.empty and "entity_id" in entity_table.columns:
            df = entity_table.copy()
            df["entity_id"] = df["entity_id"].astype(str)
            row = df[df["entity_id"] == str(company_id)]
            if not row.empty:
                base_row = row.iloc[0].to_dict()
        taxonomy_row = self._taxonomy_reference_row(aliases or [])
        if not taxonomy_row:
            return base_row
        merged = dict(base_row)
        for key, value in taxonomy_row.items():
            if _null_if_na(merged.get(key)) in (None, ""):
                merged[key] = value
        return merged

    def _metric_feature(
        self,
        *,
        name: str,
        value: Any,
        unit: Optional[str],
        as_of: pd.Timestamp,
        window: Optional[Dict[str, Any]],
        confidence: Optional[float],
        provenance: List[InputReference],
        missing_reason: Optional[str],
        fallback_used: Optional[str],
        metric_id: Optional[str] = None,
        taxonomy: Optional[TaxonomyContext] = None,
        component_breakdown: Optional[Dict[str, Any]] = None,
        quality_flags: Optional[List[str]] = None,
        support_mode: Optional[str] = None,
        view_type: Optional[str] = None,
    ) -> FeatureRecord:
        meta: Dict[str, Any] = {}
        if metric_id and taxonomy is not None:
            meta = self.metric_policy.metric_metadata(
                metric_id,
                taxonomy,
                view_type=view_type or "decision",
                support_mode=support_mode,
                component_breakdown=component_breakdown,
                quality_flags=quality_flags,
            )
        return FeatureRecord(
            name=name,
            value=value,
            unit=unit,
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window=window,
            confidence=confidence,
            provenance=provenance,
            missing_reason=missing_reason,
            fallback_used=fallback_used,
            metric_policy_id=meta.get("metric_policy_id"),
            market_owner=meta.get("market_owner"),
            primary_source_basis=meta.get("primary_source_basis"),
            methodology_registry_id=meta.get("methodology_registry_id"),
            methodology_metric_id=meta.get("methodology_metric_id"),
            canonical_owner_id=meta.get("canonical_owner_id"),
            canonical_owner_name=meta.get("canonical_owner_name"),
            canonical_classification=meta.get("canonical_classification"),
            market_layer_status=meta.get("market_layer_status"),
            current_alignment_status=meta.get("current_alignment_status"),
            primary_source_document_id=meta.get("primary_source_document_id"),
            recommended_metric_name=meta.get("recommended_metric_name"),
            archetype=meta.get("archetype"),
            sector=meta.get("sector"),
            subsector=meta.get("subsector"),
            override_level_applied=meta.get("override_level_applied"),
            support_mode=meta.get("support_mode", support_mode),
            applicability_status=meta.get("applicability_status"),
            component_breakdown=meta.get("component_breakdown", component_breakdown),
            quality_flags=meta.get("quality_flags", quality_flags),
            view_type=meta.get("view_type", view_type),
        )

    def _apply_input_source_registry(
        self,
        features: Dict[str, FeatureRecord],
        taxonomy: TaxonomyContext,
    ) -> None:
        registry_id = self.input_source_registry.registry_id
        for name, feat in features.items():
            if not isinstance(feat, FeatureRecord):
                continue
            registry_metric_id = self._registry_metric_id_for_feature(name)
            record = self.input_source_registry.metric(registry_metric_id)
            if not record:
                continue
            owner_id = record.get("canonical_owner_id")
            owner = self.input_source_registry.owner(owner_id)
            feat.input_source_registry_id = registry_id
            feat.input_source_owner_id = owner_id
            feat.input_source_owner_name = owner.get("name")
            feat.input_source_classification = record.get("classification")
            feat.input_source_formula_basis = record.get("formula_basis")
            feat.input_source_alignment_status = record.get("current_alignment_status")
            feat.input_source_document_ids = list(record.get("raw_source_document_ids") or [])
            feat.definition_requirement = record.get("definition_requirement")
            feat.definition_requirement_reason = record.get("definition_requirement_reason")
            feat.methodology_execution_decision = record.get("methodology_execution_decision")
            feat.methodology_execution_reason = record.get("methodology_execution_reason")
            feat.input_layer_bucket = self.input_source_registry.input_layer_bucket(registry_metric_id)
            feat.input_layer_bucket_reason = self.input_source_registry.input_layer_bucket_reason(registry_metric_id)
            feat.strict_market_defined = feat.input_layer_bucket == "strict_market_defined"
            if feat.archetype is None:
                feat.archetype = taxonomy.archetype
            if feat.sector is None:
                feat.sector = taxonomy.sector
            if feat.subsector is None:
                feat.subsector = taxonomy.subsector
            if feat.override_level_applied is None:
                feat.override_level_applied = taxonomy.override_level_applied

    def _apply_support_mode_semantics(self, features: Dict[str, FeatureRecord]) -> None:
        for name, feat in features.items():
            if not isinstance(feat, FeatureRecord):
                continue
            if feat.view_type == "reported":
                feat.support_mode = "unsupported" if feat.value is None else "exact"
                continue
            if feat.value is None:
                feat.support_mode = "unsupported"
                continue
            if feat.input_layer_bucket == "internal_inference":
                feat.support_mode = "inferred"
                continue

            base_mode = feat.support_mode or "exact"
            feat.support_mode = _classify_metric_support_mode(
                base_mode=base_mode,
                metric_id=self._registry_metric_id_for_feature(name),
                value=feat.value,
                quality_flags=feat.quality_flags,
                component_breakdown=feat.component_breakdown,
            )

    def _registry_metric_id_for_feature(self, feature_name: str) -> str:
        registry_metric_id = feature_name
        for suffix in ("_reported", "_market"):
            if registry_metric_id.endswith(suffix):
                registry_metric_id = registry_metric_id[: -len(suffix)]
                break
        return registry_metric_id

    def _input_layer_views(self, features: Dict[str, FeatureRecord]) -> Dict[str, Dict[str, Any]]:
        summary = self.input_source_registry.input_layer_summary()
        feature_names_by_metric: Dict[str, List[str]] = {}
        for name, feat in features.items():
            if not isinstance(feat, FeatureRecord):
                continue
            metric_id = self._registry_metric_id_for_feature(name)
            feature_names_by_metric.setdefault(metric_id, []).append(name)

        views: Dict[str, Dict[str, Any]] = {}
        for bucket, bucket_summary in summary.items():
            metric_ids = list(bucket_summary.get("registry_metric_ids") or [])
            snapshot_metric_ids_present = [metric_id for metric_id in metric_ids if metric_id in feature_names_by_metric]
            snapshot_feature_names_present = sorted(
                {
                    feature_name
                    for metric_id in snapshot_metric_ids_present
                    for feature_name in feature_names_by_metric.get(metric_id, [])
                }
            )
            views[bucket] = {
                **bucket_summary,
                "snapshot_input_metric_ids_present": snapshot_metric_ids_present,
                "snapshot_input_metric_count_present": len(snapshot_metric_ids_present),
                "snapshot_feature_names_present": snapshot_feature_names_present,
                "snapshot_feature_count_present": len(snapshot_feature_names_present),
            }
        return views

    def _market_taxonomy_context(self, company_id: str, facts: pd.DataFrame, aliases: Optional[List[str]] = None) -> TaxonomyContext:
        entity_row = self._entity_context_row(company_id, aliases=aliases)
        lease_current, _ = self._latest_fact(facts, self.fact_map.get("lease_current", []))
        lease_long, _ = self._latest_fact(facts, self.fact_map.get("lease_long", []))
        total_debt, _ = self._latest_fact(facts, self.fact_map.get("total_debt", []))
        if total_debt is None:
            debt_current, _ = self._latest_fact(facts, self.fact_map.get("debt_current", []))
            debt_long, _ = self._latest_fact(facts, self.fact_map.get("debt_long", []))
            if debt_current is not None or debt_long is not None:
                total_debt = (debt_current or 0.0) + (debt_long or 0.0)
        lease_liabilities = (lease_current or 0.0) + (lease_long or 0.0)
        lease_ratio = None
        if total_debt not in (None, 0):
            lease_ratio = float(lease_liabilities) / float(total_debt)
        return self.metric_policy.resolve_taxonomy(
            company_id,
            entity_row=entity_row,
            fingerprints={
                "lease_liability_to_reported_debt": lease_ratio,
            },
        )

    def _taxonomy_features(self, taxonomy: TaxonomyContext, as_of: pd.Timestamp) -> Dict[str, FeatureRecord]:
        quality_flags = list(taxonomy.quality_flags or [])
        common = dict(
            as_of=as_of,
            window={"type": "asof", "length_days": 0},
            confidence=taxonomy.confidence,
            provenance=[],
            missing_reason=None,
            fallback_used="heuristic" if taxonomy.override_level_applied in {"fingerprint", "base"} else None,
        )
        return {
            "taxonomy.archetype": self._metric_feature(
                name="taxonomy.archetype",
                value=taxonomy.archetype,
                unit="label",
                quality_flags=quality_flags,
                support_mode=taxonomy.support_mode,
                view_type="decision",
                **common,
            ),
            "taxonomy.sector": self._metric_feature(
                name="taxonomy.sector",
                value=taxonomy.sector,
                unit="label",
                quality_flags=quality_flags,
                support_mode=taxonomy.support_mode,
                view_type="decision",
                **common,
            ),
            "taxonomy.subsector": self._metric_feature(
                name="taxonomy.subsector",
                value=taxonomy.subsector,
                unit="label",
                quality_flags=quality_flags,
                support_mode=taxonomy.support_mode,
                view_type="decision",
                **common,
            ),
            "taxonomy.override_level_applied": self._metric_feature(
                name="taxonomy.override_level_applied",
                value=taxonomy.override_level_applied,
                unit="label",
                quality_flags=quality_flags,
                support_mode=taxonomy.support_mode,
                view_type="decision",
                **common,
            ),
        }

    def _emit_metric_views(
        self,
        features: Dict[str, FeatureRecord],
        *,
        metric_id: str,
        base_name: str,
        reported_value: Any,
        market_value: Any,
        unit: Optional[str],
        as_of: pd.Timestamp,
        window: Optional[Dict[str, Any]],
        taxonomy: TaxonomyContext,
        reported_provenance: List[InputReference],
        market_provenance: List[InputReference],
        reported_missing_reason: Optional[str],
        market_missing_reason: Optional[str],
        fallback_used: Optional[str],
        component_breakdown: Optional[Dict[str, Any]] = None,
        quality_flags: Optional[List[str]] = None,
        decision_fallback_to_reported: bool = True,
    ) -> None:
        applicability = self.metric_policy.resolve_applicability(metric_id, taxonomy)
        market_quality_flags = list(quality_flags or [])

        reported_feature = self._metric_feature(
            name=f"{base_name}_reported",
            value=reported_value,
            unit=unit,
            as_of=as_of,
            window=window,
            confidence=None,
            provenance=reported_provenance,
            missing_reason=reported_missing_reason,
            fallback_used=fallback_used,
            metric_id=metric_id,
            taxonomy=taxonomy,
            component_breakdown=component_breakdown,
            quality_flags=market_quality_flags,
            support_mode="exact",
            view_type="reported",
        )
        features[f"{base_name}_reported"] = reported_feature

        market_support_mode = taxonomy.support_mode if market_value is not None else "unsupported"
        if market_value is not None and market_missing_reason is None:
            market_support_mode = _classify_metric_support_mode(
                base_mode=market_support_mode,
                metric_id=metric_id,
                value=market_value,
                quality_flags=market_quality_flags,
                component_breakdown=component_breakdown,
            )
        market_feature = self._metric_feature(
            name=f"{base_name}_market",
            value=market_value,
            unit=unit,
            as_of=as_of,
            window=window,
            confidence=None,
            provenance=market_provenance,
            missing_reason=market_missing_reason,
            fallback_used=fallback_used,
            metric_id=metric_id,
            taxonomy=taxonomy,
            component_breakdown=component_breakdown,
            quality_flags=market_quality_flags,
            support_mode=market_support_mode,
            view_type="market",
        )
        features[f"{base_name}_market"] = market_feature

        decision_value = market_value
        decision_missing_reason = market_missing_reason
        decision_fallback = fallback_used
        decision_quality_flags = list(market_quality_flags)
        decision_support_mode = market_support_mode
        if applicability == "unsupported":
            decision_value = None
            decision_missing_reason = "unsupported_for_archetype"
            decision_quality_flags.append("unsupported_metric")
            decision_support_mode = "unsupported"
        elif decision_value is None and decision_fallback_to_reported and reported_value is not None:
            decision_value = reported_value
            decision_missing_reason = None
            decision_fallback = "reported_view_fallback"
            decision_quality_flags.append("decision_uses_reported_view")
            if _support_mode_is_exact_like(decision_support_mode):
                decision_support_mode = "proxy_missing_component"

        features[base_name] = self._metric_feature(
            name=base_name,
            value=decision_value,
            unit=unit,
            as_of=as_of,
            window=window,
            confidence=None,
            provenance=market_provenance if decision_value == market_value else reported_provenance,
            missing_reason=decision_missing_reason,
            fallback_used=decision_fallback,
            metric_id=metric_id,
            taxonomy=taxonomy,
            component_breakdown=component_breakdown,
            quality_flags=decision_quality_flags,
            support_mode=decision_support_mode,
            view_type="decision",
        )

    # ---------------------------
    # Public API
    # ---------------------------

    def build(
        self,
        company_id: str,
        as_of_time: str | datetime,
        extra_aliases: Optional[List[str]] = None,
    ) -> CompanyStateSnapshot:
        if isinstance(as_of_time, str):
            as_of_dt = pd.to_datetime(as_of_time, utc=True)
        else:
            as_of_dt = pd.to_datetime(as_of_time, utc=True)

        resolved_id, alias_ids = self._resolve_entity_aliases(company_id, extra_aliases=extra_aliases)
        if self.debug:
            print(f"[build] resolved_id={resolved_id} alias_ids={len(alias_ids)}")

        t0 = pd.Timestamp.utcnow()
        # Pull fact rows across the resolved entity id plus any point-in-time aliases
        # in one pass. Historical replay cases often carry financial facts under the
        # ticker/source alias while the normalized entity id still has other metadata
        # rows; only falling back to aliases on an empty primary hit drops those
        # financial inputs entirely.
        fact_ids = alias_ids or [resolved_id]
        facts = self._load_facts(fact_ids, as_of_dt)
        if self.debug:
            print(f"[build] facts rows={len(facts)} dt={(pd.Timestamp.utcnow()-t0).total_seconds():.2f}s")
        t0 = pd.Timestamp.utcnow()
        ts = self._load_timeseries(alias_ids, as_of_dt)
        if self.debug:
            print(f"[build] timeseries rows={len(ts)} dt={(pd.Timestamp.utcnow()-t0).total_seconds():.2f}s")
        t0 = pd.Timestamp.utcnow()
        macro = self._load_macro(as_of_dt)
        if self.debug:
            print(f"[build] macro rows={len(macro)} dt={(pd.Timestamp.utcnow()-t0).total_seconds():.2f}s")
        t0 = pd.Timestamp.utcnow()
        events = self._load_events(alias_ids, as_of_dt)
        if self.debug:
            print(f"[build] events rows={len(events)} dt={(pd.Timestamp.utcnow()-t0).total_seconds():.2f}s")
        t0 = pd.Timestamp.utcnow()
        ownership = self._load_ownership_summary(alias_ids, as_of_dt)
        if self.debug:
            print(f"[build] ownership rows={len(ownership)} dt={(pd.Timestamp.utcnow()-t0).total_seconds():.2f}s")
        t0 = pd.Timestamp.utcnow()
        issuer_ratings = self._load_issuer_ratings(alias_ids, as_of_dt)
        if self.debug:
            print(f"[build] issuer_ratings rows={len(issuer_ratings)} dt={(pd.Timestamp.utcnow()-t0).total_seconds():.2f}s")
        t0 = pd.Timestamp.utcnow()
        estimates = self._load_estimates(alias_ids, as_of_dt)
        if self.debug:
            print(f"[build] estimates rows={len(estimates)} dt={(pd.Timestamp.utcnow()-t0).total_seconds():.2f}s")

        taxonomy = self._market_taxonomy_context(resolved_id, facts, aliases=alias_ids)
        features: Dict[str, FeatureRecord] = {}
        features.update(self._taxonomy_features(taxonomy, as_of_dt))
        t0 = pd.Timestamp.utcnow()
        features.update(self._compute_liquidity(resolved_id, facts, as_of_dt, taxonomy))
        if self.debug:
            print(f"[build] liquidity dt={(pd.Timestamp.utcnow()-t0).total_seconds():.2f}s")
        t0 = pd.Timestamp.utcnow()
        features.update(self._compute_capital_structure(resolved_id, facts, events, issuer_ratings, as_of_dt, taxonomy))
        if self.debug:
            print(f"[build] capital_structure dt={(pd.Timestamp.utcnow()-t0).total_seconds():.2f}s")
        t0 = pd.Timestamp.utcnow()
        features.update(self._compute_market(resolved_id, ts, facts, as_of_dt, features, taxonomy))
        if self.debug:
            print(f"[build] market dt={(pd.Timestamp.utcnow()-t0).total_seconds():.2f}s")
        t0 = pd.Timestamp.utcnow()
        features.update(self._compute_macro(macro, as_of_dt))
        if self.debug:
            print(f"[build] macro_features dt={(pd.Timestamp.utcnow()-t0).total_seconds():.2f}s")
        t0 = pd.Timestamp.utcnow()
        features.update(self._compute_operating(resolved_id, facts, as_of_dt))
        if self.debug:
            print(f"[build] operating dt={(pd.Timestamp.utcnow()-t0).total_seconds():.2f}s")
        t0 = pd.Timestamp.utcnow()
        features.update(self._compute_capital_return_context(facts, as_of_dt, features))
        if self.debug:
            print(f"[build] capital_return_context dt={(pd.Timestamp.utcnow()-t0).total_seconds():.2f}s")
        t0 = pd.Timestamp.utcnow()
        if self.enable_market_relevant_smart_normalized_inputs:
            self._apply_market_relevant_smart_normalized_inputs(
                resolved_id,
                alias_ids,
                facts,
                as_of_dt,
                features,
            )
            if self.debug:
                print(f"[build] smart_normalized dt={(pd.Timestamp.utcnow()-t0).total_seconds():.2f}s")
        t0 = pd.Timestamp.utcnow()
        features.update(self._compute_ownership_governance(facts, events, ownership, as_of_dt))
        if self.debug:
            print(f"[build] ownership_governance dt={(pd.Timestamp.utcnow()-t0).total_seconds():.2f}s")
        t0 = pd.Timestamp.utcnow()
        features.update(self._compute_strategic(facts, events, as_of_dt))
        if self.debug:
            print(f"[build] strategic dt={(pd.Timestamp.utcnow()-t0).total_seconds():.2f}s")

        t0 = pd.Timestamp.utcnow()
        regime = self._compute_regime(macro, as_of_dt)
        if self.debug:
            print(f"[build] regime dt={(pd.Timestamp.utcnow()-t0).total_seconds():.2f}s")
        t0 = pd.Timestamp.utcnow()
        constraints = self._build_constraint_set(facts, as_of_dt)
        if self.debug:
            print(f"[build] constraints dt={(pd.Timestamp.utcnow()-t0).total_seconds():.2f}s")
        if self.skip_peer_context:
            peer_set = PeerSet(
                peer_set_id=str(resolved_id),
                members=[],
                method="skipped",
                version=1,
            )
        else:
            t0 = pd.Timestamp.utcnow()
            peer_set = self._resolve_peers(resolved_id, facts, ts, as_of_dt)
            if self.debug:
                print(f"[build] peers resolve dt={(pd.Timestamp.utcnow()-t0).total_seconds():.2f}s")
            t0 = pd.Timestamp.utcnow()
            features.update(self._compute_peer_context(resolved_id, peer_set, as_of_dt))
            if self.debug:
                print(f"[build] peer_context dt={(pd.Timestamp.utcnow()-t0).total_seconds():.2f}s")

        t0 = pd.Timestamp.utcnow()
        features.update(self._compute_segment_portfolio_context(facts, events, as_of_dt, features, taxonomy))
        if self.debug:
            print(f"[build] segment_portfolio dt={(pd.Timestamp.utcnow()-t0).total_seconds():.2f}s")
        t0 = pd.Timestamp.utcnow()
        features.update(self._compute_expectations_revisions(estimates, as_of_dt))
        if self.debug:
            print(f"[build] expectations dt={(pd.Timestamp.utcnow()-t0).total_seconds():.2f}s")

        # Standardized confidence framework:
        # confidence = recency * source_reliability * proxy_penalty * parse_certainty
        self._apply_input_source_registry(features, taxonomy)
        self._apply_support_mode_semantics(features)
        self._apply_confidence_framework(features, as_of_dt)
        constraints = self._augment_constraint_set_with_feature_proxies(constraints, features, as_of_dt)
        feature_lineage = self._finalize_feature_provenance(features, as_of_dt)
        input_layer_views = self._input_layer_views(features)

        snapshot = CompanyStateSnapshot(
            snapshot_id=str(uuid.uuid4()),
            company_id=str(resolved_id),
            as_of_time=as_of_dt.isoformat(),
            features={k: asdict(v) for k, v in features.items()},
            regime=regime,
            constraint_set={
                "hard": [asdict(c) for c in constraints.get("hard", [])],
                "soft": [asdict(c) for c in constraints.get("soft", [])],
            },
            peer_set=asdict(peer_set),
            provenance={
                "inputs_used": {
                    "facts": str(self.facts_path),
                    "timeseries": str(self.raw_timeseries_path),
                    "macro": str(self.macro_timeseries_path or self.raw_timeseries_path),
                    "events": str(self.event_store_path),
                    "ownership": str(self.ownership_summary_path),
                    "issuer_ratings": str(self.issuer_ratings_path),
                    "estimates": str(self.estimates_path),
                },
                "data_cutoff": {"as_of_time": as_of_dt.isoformat()},
                "computation_version": "state_builder_v5",
                "input_company_id": str(company_id),
                "market_metric_context": {
                    "policy_id": self.metric_policy.policy_id,
                    "methodology_registry_id": self.metric_policy.methodology_registry.registry_id,
                    "input_source_registry_id": self.input_source_registry.registry_id,
                    "archetype": taxonomy.archetype,
                    "sector": taxonomy.sector,
                    "subsector": taxonomy.subsector,
                    "override_level_applied": taxonomy.override_level_applied,
                    "support_mode": taxonomy.support_mode,
                    "quality_flags": taxonomy.quality_flags,
                    "confidence": taxonomy.confidence,
                    "strict_market_defined_metric_count": input_layer_views["strict_market_defined"]["registry_metric_count"],
                    "secondary_externally_anchored_metric_count": input_layer_views["secondary_externally_anchored"]["registry_metric_count"],
                    "internal_inference_metric_count": input_layer_views["internal_inference"]["registry_metric_count"],
                },
                "input_layer_views": input_layer_views,
                "feature_lineage": feature_lineage,
                "missing_data_flags": {
                    k: v.missing_reason
                    for k, v in (features.items() if isinstance(features, dict) else {})
                    if isinstance(v, FeatureRecord) and v.missing_reason is not None
                },
            },
        )
        return snapshot

    # ---------------------------
    # Loaders
    # ---------------------------

    def _get_parquet_columns(self, path: Path) -> set[str]:
        if path in self._parquet_columns_cache:
            return self._parquet_columns_cache[path]
        if not path.exists():
            self._parquet_columns_cache[path] = set()
            return self._parquet_columns_cache[path]
        con = duckdb.connect()
        try:
            df = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{path.as_posix()}')").df()
            cols = set(df["column_name"].astype(str))
        except Exception:
            cols = set()
        self._parquet_columns_cache[path] = cols
        return cols

    def _facts_columns(self) -> List[str]:
        # Keep facts projection tight to reduce per-company parquet scan cost.
        # Include note-extract aliases so we can union SEC note-pattern rows with
        # canonical fact-registry rows without a separate normalization pass.
        return [
            "fact_id",
            "document_id",
            "entity_id",
            "fact_type",
            "metric_key",
            "fact_value",
            "value",
            "fact_time",
            "context_norm",
            "bucket_label",
            "contradiction_group_id",
            "confidence_score",
            "extraction_confidence",
            "source_type",
            "published_at",
            "ingested_at",
            "valid_from",
            "valid_to",
            "effective_at",
        ]

    def _fact_files(self) -> List[str]:
        if self.facts_path.is_file():
            return [self.facts_path.as_posix()] if _is_readable_file(self.facts_path) else []
        if not self.facts_path.exists():
            return []
        key = ",".join(str(y) for y in sorted(self.facts_years)) if self.facts_years else "ALL"
        if key in self._facts_files_cache:
            return self._facts_files_cache[key]

        files: List[str] = []
        if self.facts_years:
            candidates = [self.facts_path / f"year={int(y)}" / "part.parquet" for y in self.facts_years]
        else:
            candidates = list(self.facts_path.glob("year=*/part.parquet"))
            if not candidates:
                candidates = list(self.facts_path.rglob("*.parquet"))

        for p in candidates:
            name = p.name.lower()
            if "tmp_part" in name or name.startswith("tmp_"):
                continue
            try:
                if p.stat().st_size < 256:
                    continue
            except OSError:
                continue
            if not _is_readable_file(p):
                if self.debug:
                    print(f"[facts] skip unreadable {p}")
                continue
            files.append(p.as_posix())

        self._facts_files_cache[key] = files
        return files

    def _available_columns_for_files(self, files: List[str], desired: List[str]) -> List[str]:
        if not files:
            return desired
        try:
            cols = self._get_parquet_columns(Path(files[0]))
        except Exception:
            cols = set()
        if not cols:
            return desired
        selected = [c for c in desired if c in cols]
        return selected if selected else desired

    def _to_datetimes(self, df: pd.DataFrame, cols: List[str], utc: bool = True) -> pd.DataFrame:
        for col in cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=utc, errors="coerce")
        return df

    def _normalize_fact_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        out = df.copy()
        if "fact_type" not in out.columns and "metric_key" in out.columns:
            out["fact_type"] = out["metric_key"]
        elif "metric_key" in out.columns:
            missing = out["fact_type"].isna() if "fact_type" in out.columns else pd.Series(True, index=out.index)
            if missing.any():
                out.loc[missing, "fact_type"] = out.loc[missing, "metric_key"]
        if "fact_value" not in out.columns and "value" in out.columns:
            out["fact_value"] = pd.to_numeric(out["value"], errors="coerce")
        elif "value" in out.columns:
            fact_numeric = pd.to_numeric(out.get("fact_value"), errors="coerce")
            value_numeric = pd.to_numeric(out["value"], errors="coerce")
            out["fact_value"] = fact_numeric.fillna(value_numeric)
        if "confidence_score" not in out.columns and "extraction_confidence" in out.columns:
            out["confidence_score"] = pd.to_numeric(out["extraction_confidence"], errors="coerce")
        elif "extraction_confidence" in out.columns:
            conf_numeric = pd.to_numeric(out.get("confidence_score"), errors="coerce")
            extract_numeric = pd.to_numeric(out["extraction_confidence"], errors="coerce")
            out["confidence_score"] = conf_numeric.fillna(extract_numeric)
        if "fact_id" not in out.columns:
            out["fact_id"] = None
        if "document_id" in out.columns:
            doc_ids = out["document_id"].astype(str)
            metric_ids = out.get("metric_key", out.get("fact_type", pd.Series("", index=out.index))).astype(str)
            bucket_ids = out.get("bucket_label", pd.Series("", index=out.index)).fillna("").astype(str)
            missing_fact_id = out["fact_id"].isna() | (out["fact_id"].astype(str).str.strip() == "")
            if missing_fact_id.any():
                synthesized = doc_ids + "::" + metric_ids
                non_empty_bucket = bucket_ids.str.strip() != ""
                synthesized = synthesized.where(~non_empty_bucket, synthesized + "::" + bucket_ids)
                out.loc[missing_fact_id, "fact_id"] = synthesized.loc[missing_fact_id]
        if "bucket_label" in out.columns:
            bucket_series = out["bucket_label"].fillna("").astype(str).str.strip()
            if "context_norm" not in out.columns:
                out["context_norm"] = None
            context_series = out["context_norm"].fillna("").astype(str)
            has_bucket = bucket_series != ""
            needs_bucket = ~context_series.str.contains("bucket_label=", regex=False)
            fill_mask = has_bucket & needs_bucket
            if fill_mask.any():
                idx = fill_mask[fill_mask].index
                appended = context_series.loc[idx].str.rstrip(";")
                empty_mask = appended.str.strip().eq("")
                if empty_mask.any():
                    empty_idx = empty_mask[empty_mask].index
                    out.loc[empty_idx, "context_norm"] = "bucket_label=" + bucket_series.loc[empty_idx]
                non_empty_mask = ~empty_mask
                if non_empty_mask.any():
                    non_empty_idx = non_empty_mask[non_empty_mask].index
                    out.loc[non_empty_idx, "context_norm"] = (
                        appended.loc[non_empty_idx] + "; bucket_label=" + bucket_series.loc[non_empty_idx]
                    )
        return out

    def _augment_fact_context_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        ctx_col = _pick_first_col(df, ["context_norm", "context"])
        if ctx_col is None:
            if "period_end" not in df.columns and "effective_at" in df.columns:
                out = df.copy()
                out["period_end"] = pd.to_datetime(out["effective_at"], utc=True, errors="coerce")
                return out
            return df

        out = df.copy()
        ctx = out[ctx_col].fillna("").astype(str)

        parsed_period_end = pd.to_datetime(
            ctx.str.extract(r"(?:^|;\s*)fiscal_period_end=([^;]+)")[0],
            utc=True,
            errors="coerce",
        )
        if "period_end" in out.columns:
            out["period_end"] = pd.to_datetime(out["period_end"], utc=True, errors="coerce")
            out["period_end"] = out["period_end"].fillna(parsed_period_end)
        else:
            out["period_end"] = parsed_period_end

        parsed_fiscal_year = pd.to_numeric(
            ctx.str.extract(r"(?:^|;\s*)fiscal_year=([0-9]{4})")[0],
            errors="coerce",
        )
        if "fiscal_year" in out.columns:
            out["fiscal_year"] = pd.to_numeric(out["fiscal_year"], errors="coerce").fillna(parsed_fiscal_year)
        else:
            out["fiscal_year"] = parsed_fiscal_year

        parsed_fiscal_quarter = pd.to_numeric(
            ctx.str.extract(r"(?:^|;\s*)fiscal_quarter=([0-9]+)")[0],
            errors="coerce",
        )
        if "fiscal_quarter" in out.columns:
            out["fiscal_quarter"] = pd.to_numeric(out["fiscal_quarter"], errors="coerce").fillna(parsed_fiscal_quarter)
        else:
            out["fiscal_quarter"] = parsed_fiscal_quarter

        return out

    def _first_present(self, row: pd.Series, keys: List[str]) -> Any:
        for key in keys:
            if key in row and row.get(key) is not None and not pd.isna(row.get(key)):
                return row.get(key)
        return None

    def _revision_dedupe(
        self,
        df: pd.DataFrame,
        key_candidates: List[str],
        order_preference: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        if df.empty:
            return df
        keys = [k for k in key_candidates if k in df.columns]
        if not keys:
            return df
        if order_preference is None:
            # Bitemporal-friendly ordering: "latest effective value available as-of",
            # then latest ingested/publication revision.
            order_preference = [
                "effective_at",
                "valid_from",
                "fact_time",
                "rating_date",
                "report_date",
                "filing_date",
                "announced_at",
                "event_time",
                "trade_date",
                "observation_time",
                "date",
                "period_end",
                "ingested_at",
                "ingestion_time",
                "created_at",
                "published_at",
                "available_time",
            ]
        order_cols = [c for c in order_preference if c in df.columns]
        if order_cols:
            df = self._to_datetimes(df.copy(), order_cols)
            df = df.sort_values(order_cols, ascending=[False] * len(order_cols), na_position="last")
        return df.drop_duplicates(keys, keep="first")

    def _select_latest_fact_revision(self, df: pd.DataFrame) -> pd.DataFrame:
        # Keep historical rows, but collapse duplicate revisions of the same fact observation.
        if df.empty:
            return df
        if "contradiction_group_id" in df.columns and df["contradiction_group_id"].notna().all():
            keys = ["entity_id", "fact_type", "contradiction_group_id"]
        elif "contradiction_group_id" in df.columns and df["contradiction_group_id"].notna().any():
            with_group = df[df["contradiction_group_id"].notna()].copy()
            without_group = df[df["contradiction_group_id"].isna()].copy()
            deduped_parts: List[pd.DataFrame] = []
            if not with_group.empty:
                deduped_parts.append(
                    self._revision_dedupe(
                        with_group,
                        ["entity_id", "fact_type", "contradiction_group_id"],
                    )
                )
            if not without_group.empty:
                if "fact_time" in without_group.columns:
                    keys = ["entity_id", "fact_type", "fact_time"]
                    if "context_norm" in without_group.columns:
                        keys.append("context_norm")
                elif "effective_at" in without_group.columns:
                    keys = ["entity_id", "fact_type", "effective_at"]
                    if "context_norm" in without_group.columns:
                        keys.append("context_norm")
                elif "valid_from" in without_group.columns:
                    keys = ["entity_id", "fact_type", "valid_from"]
                    if "context_norm" in without_group.columns:
                        keys.append("context_norm")
                else:
                    keys = (
                        ["entity_id", "fact_type", "fact_value"]
                        if "fact_value" in without_group.columns
                        else ["entity_id", "fact_type"]
                    )
                deduped_parts.append(self._revision_dedupe(without_group, keys))
            if not deduped_parts:
                return df
            return pd.concat(deduped_parts, ignore_index=True)
        elif "fact_time" in df.columns:
            keys = ["entity_id", "fact_type", "fact_time"]
            if "context_norm" in df.columns:
                keys.append("context_norm")
        elif "effective_at" in df.columns:
            keys = ["entity_id", "fact_type", "effective_at"]
            if "context_norm" in df.columns:
                keys.append("context_norm")
        elif "valid_from" in df.columns:
            keys = ["entity_id", "fact_type", "valid_from"]
            if "context_norm" in df.columns:
                keys.append("context_norm")
        else:
            # Last-resort fallback: preserve one row per (entity, fact_type, value).
            keys = ["entity_id", "fact_type", "fact_value"] if "fact_value" in df.columns else ["entity_id", "fact_type"]
        return self._revision_dedupe(df, keys)

    def _select_latest_timeseries_revision(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._revision_dedupe(
            df,
            [
                "entity_id",
                "company_id",
                "security_id",
                "series_id",
                "instrument_id",
                "metric",
                "field_name",
                "series_type",
                "trade_date",
                "observation_time",
                "event_time",
                "effective_at",
                "date",
                "period_end",
            ],
            order_preference=[
                "effective_at",
                "trade_date",
                "observation_time",
                "event_time",
                "date",
                "period_end",
                "ingested_at",
                "ingestion_time",
                "published_at",
                "available_time",
            ],
        )

    def _select_latest_event_revision(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        keys: List[str]
        if "event_id" in df.columns:
            keys = ["event_id"]
        else:
            keys = ["company_id", "event_type", "event_subtype", "announced_at", "effective_at"]
        return self._revision_dedupe(
            df,
            keys,
            order_preference=[
                "effective_at",
                "announced_at",
                "event_time",
                "ingested_at",
                "created_at",
                "published_at",
            ],
        )

    def _select_latest_ownership_snapshot(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = self._revision_dedupe(
            df,
            ["company_id", "report_date", "filing_date", "effective_at"],
            order_preference=[
                "report_date",
                "effective_at",
                "filing_date",
                "ingested_at",
                "published_at",
            ],
        )
        if "company_id" not in df.columns:
            return df

        # Ownership summary is quarter-level and can legitimately contain many
        # rows for the same issuer within the latest reporting period. We want
        # to preserve that latest-quarter slice so downstream feature assembly
        # can choose the richest aggregate row instead of blindly collapsing to
        # the last filing, which is often a thin single-holder update.
        if "report_date" in df.columns and df["report_date"].notna().any():
            latest_report = df.groupby("company_id")["report_date"].transform("max")
            df = df[df["report_date"].eq(latest_report)].copy()
            return df

        if "effective_at" in df.columns and df["effective_at"].notna().any():
            latest_effective = df.groupby("company_id")["effective_at"].transform("max")
            df = df[df["effective_at"].eq(latest_effective)].copy()
            return df

        df = self._revision_dedupe(
            df,
            ["company_id"],
            order_preference=[
                "filing_date",
                "ingested_at",
                "published_at",
            ],
        )
        return df

    def _select_latest_rating_snapshot(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = self._revision_dedupe(
            df,
            ["company_id", "rating_type_code", "rating_date", "effective_at"],
            order_preference=[
                "rating_date",
                "effective_at",
                "ingested_at",
                "published_at",
            ],
        )
        # Preserve the latest row per company/agency so downstream selection can
        # still apply the canonical-owner preference (for consumer/industrials,
        # Fitch) instead of collapsing everything to the single most recent
        # agency update.
        if "company_id" in df.columns:
            agency_keys = ["company_id"]
            for col in ("agency", "rating_agency", "source_type", "provider"):
                if col in df.columns:
                    agency_keys.append(col)
            if len(agency_keys) > 1:
                df = self._revision_dedupe(
                    df,
                    agency_keys,
                    order_preference=[
                        "rating_date",
                        "effective_at",
                        "ingested_at",
                        "published_at",
                    ],
                )
        return df

    def _apply_bitemporal_filters(
        self, df: pd.DataFrame, ids: List[str] | None, as_of_dt: pd.Timestamp
    ) -> pd.DataFrame:
        if df.empty:
            return df
        cutoff = pd.to_datetime(as_of_dt, utc=True)
        df = self._normalize_fact_columns(df)
        df = self._to_datetimes(df, ["published_at", "ingested_at", "valid_from", "valid_to", "effective_at"])
        if "entity_id" in df.columns and ids:
            df = df[df["entity_id"].astype(str).isin(ids)]
        cond = pd.Series(True, index=df.index)
        if "published_at" in df.columns:
            cond &= df["published_at"].isna() | (df["published_at"] <= cutoff)
        if "ingested_at" in df.columns and not self.historical_backfill_mode:
            cond &= df["ingested_at"].isna() | (df["ingested_at"] <= cutoff)
        if "valid_from" in df.columns:
            cond &= df["valid_from"].isna() | (df["valid_from"] <= cutoff)
        if "valid_to" in df.columns:
            cond &= df["valid_to"].isna() | (df["valid_to"] > cutoff)
        df = df[cond]
        return self._select_latest_fact_revision(df)

    def _load_facts(self, company_id: str | List[str], as_of_dt: pd.Timestamp) -> pd.DataFrame:
        if not self.facts_path.exists():
            return pd.DataFrame()

        ids = [str(company_id)] if not isinstance(company_id, list) else [str(i) for i in company_id if i is not None]
        ids = list(dict.fromkeys(ids))

        if self.cache_facts and self._facts_cache is not None:
            df = self._apply_bitemporal_filters(self._facts_cache, ids, as_of_dt)
            return self._augment_fact_context_fields(df)

        # Fast path for single small parquet file (avoid DuckDB overhead)
        if self.facts_path.is_file():
            try:
                if self.facts_path.stat().st_size < 5_000_000:
                    df = pd.read_parquet(self.facts_path)
                    if self.debug:
                        print(f"[facts] pandas read {self.facts_path} rows={len(df)}")
                    return self._augment_fact_context_fields(self._apply_bitemporal_filters(df, ids, as_of_dt))
            except Exception:
                pass

        if self.cache_facts and self._facts_cache is None:
            # Load once using DuckDB to keep column set reasonable
            source = self.facts_path.as_posix()
            files = self._fact_files() if self.facts_path.is_dir() else [source]
            if not files:
                return pd.DataFrame()
            cols = self._available_columns_for_files(files, self._facts_columns())
            file_list = ", ".join([_sql_quote(p) for p in files])
            source = f"[{file_list}]"
            if self.debug:
                start = time.time()
                print(f"[facts] caching from {len(files)} file(s)...", flush=True)
            con = duckdb.connect()
            query = f"SELECT {', '.join(cols)} FROM read_parquet({source}, union_by_name=True)"
            self._facts_cache = con.execute(query).df()
            self._facts_cache = self._to_datetimes(
                self._facts_cache,
                ["published_at", "ingested_at", "valid_from", "valid_to", "effective_at"],
            )
            if "entity_id" in self._facts_cache.columns:
                self._facts_cache["entity_id"] = self._facts_cache["entity_id"].astype(str)
            if self.debug:
                print(f"[facts] cache rows={len(self._facts_cache)} dt={time.time()-start:.2f}s", flush=True)
            return self._load_facts(company_id, as_of_dt)

        con = duckdb.connect()
        cutoff = _as_of_ts_literal(as_of_dt)
        files = self._fact_files() if self.facts_path.is_dir() else [self.facts_path.as_posix()]
        if not files:
            return pd.DataFrame()
        file_list = ", ".join([_sql_quote(p) for p in files])
        source = f"[{file_list}]"
        if self.debug:
            print(f"[facts] duckdb read files={len(files)}")
        cols = self._available_columns_for_files(files, self._facts_columns())
        id_list = ", ".join([_sql_quote(i) for i in ids])
        ingested_clause = (
            f"AND (ingested_at IS NULL OR CAST(ingested_at AS TIMESTAMP) <= TIMESTAMP '{cutoff}')"
            if not self.historical_backfill_mode
            else ""
        )
        query = f"""
        SELECT {", ".join(cols)}
        FROM read_parquet({source}, union_by_name=True)
        WHERE entity_id IN ({id_list})
          AND (published_at IS NULL OR CAST(published_at AS TIMESTAMP) <= TIMESTAMP '{cutoff}')
          {ingested_clause}
          AND (
                (valid_from IS NULL OR CAST(valid_from AS TIMESTAMP) <= TIMESTAMP '{cutoff}')
                AND (valid_to IS NULL OR CAST(valid_to AS TIMESTAMP) > TIMESTAMP '{cutoff}')
              OR (valid_from IS NULL AND valid_to IS NULL)
          )
        """
        try:
            df = con.execute(query).df()
        except Exception as exc:
            if self.debug:
                print(f"[facts] duckdb read failed; falling back to pandas file scans: {exc}", flush=True)
            frames: List[pd.DataFrame] = []
            parquet_module = None
            try:
                import pyarrow.parquet as parquet_module  # type: ignore
            except Exception:
                parquet_module = None

            for file_path in files:
                try:
                    file_cols = cols
                    if parquet_module is not None:
                        available = set(parquet_module.ParquetFile(file_path).schema.names)
                        file_cols = [col for col in cols if col in available]
                    if not file_cols:
                        continue
                    read_kwargs: Dict[str, Any] = {"columns": file_cols}
                    if "entity_id" in file_cols:
                        read_kwargs["filters"] = [("entity_id", "in", ids)]
                    part = pd.read_parquet(file_path, **read_kwargs)
                    if not part.empty:
                        frames.append(part)
                except Exception as file_exc:
                    if self.debug:
                        print(f"[facts] pandas fallback skip {file_path}: {file_exc}", flush=True)
                    continue

            if not frames:
                raise exc
            df = pd.concat(frames, ignore_index=True, sort=False)
        return self._augment_fact_context_fields(self._apply_bitemporal_filters(df, ids, as_of_dt))

    def _load_timeseries(self, company_id: str | List[str], as_of_dt: pd.Timestamp) -> pd.DataFrame:
        if self.skip_timeseries:
            return pd.DataFrame()
        if not self.raw_timeseries_path.exists():
            return pd.DataFrame()
        if not _is_readable_file(self.raw_timeseries_path):
            if self.debug:
                print(f"[timeseries] unreadable file {self.raw_timeseries_path}")
            return pd.DataFrame()
        ids = [str(company_id)] if not isinstance(company_id, list) else [str(i) for i in company_id if i is not None]
        ids = list(dict.fromkeys(ids))
        if self.cache_timeseries and self._timeseries_cache is None:
            if self.debug:
                start = time.time()
                print(f"[timeseries] caching from {self.raw_timeseries_path}", flush=True)
            con = duckdb.connect()
            self._timeseries_cache = con.execute(
                f"SELECT * FROM read_parquet('{self.raw_timeseries_path.as_posix()}', union_by_name=True)"
            ).df()
            if self.debug:
                print(f"[timeseries] cache rows={len(self._timeseries_cache)} dt={time.time()-start:.2f}s", flush=True)
        if self.cache_timeseries and self._timeseries_cache is not None:
            df = self._timeseries_cache
            cutoff = pd.to_datetime(as_of_dt, utc=True)
            if "entity_id" in df.columns:
                df = df[df["entity_id"].astype(str).isin(ids)]
            elif "company_id" in df.columns:
                df = df[df["company_id"].astype(str).isin(ids)]
            pub_col = "published_at" if "published_at" in df.columns else ("available_time" if "available_time" in df.columns else None)
            ing_col = None if self.historical_backfill_mode else ("ingested_at" if "ingested_at" in df.columns else ("ingestion_time" if "ingestion_time" in df.columns else None))
            if pub_col:
                df = df[(df[pub_col].isna()) | (pd.to_datetime(df[pub_col], utc=True, errors="coerce") <= cutoff)]
            if ing_col:
                df = df[(df[ing_col].isna()) | (pd.to_datetime(df[ing_col], utc=True, errors="coerce") <= cutoff)]
            if "series_type" in df.columns:
                df = df[df["series_type"].astype(str) != "macro"]
            eff_col = None
            for c in ["effective_at", "event_time", "trade_date"]:
                if c in df.columns:
                    eff_col = c
                    break
            if eff_col:
                df = df[(df[eff_col].isna()) | (pd.to_datetime(df[eff_col], utc=True, errors="coerce") <= cutoff)]
            return self._select_latest_timeseries_revision(df)
        con = duckdb.connect()
        cutoff = _as_of_ts_literal(as_of_dt)
        cols = self._get_parquet_columns(self.raw_timeseries_path)
        filters = []
        pub_col = "published_at" if "published_at" in cols else ("available_time" if "available_time" in cols else None)
        ing_col = None if self.historical_backfill_mode else ("ingested_at" if "ingested_at" in cols else ("ingestion_time" if "ingestion_time" in cols else None))
        eff_col = "effective_at" if "effective_at" in cols else ("event_time" if "event_time" in cols else ("trade_date" if "trade_date" in cols else None))
        if pub_col:
            filters.append(f"({pub_col} IS NULL OR CAST({pub_col} AS TIMESTAMP) <= TIMESTAMP '{cutoff}')")
        if ing_col:
            filters.append(f"({ing_col} IS NULL OR CAST({ing_col} AS TIMESTAMP) <= TIMESTAMP '{cutoff}')")
        if eff_col:
            filters.append(f"({eff_col} IS NULL OR CAST({eff_col} AS TIMESTAMP) <= TIMESTAMP '{cutoff}')")
        if "entity_id" in cols:
            id_clause = " OR ".join([f"entity_id = '{i}'" for i in ids])
            filters.append(f"({id_clause})")
        elif "company_id" in cols:
            id_clause = " OR ".join([f"company_id = '{i}'" for i in ids])
            filters.append(f"({id_clause})")
        if "series_type" in cols:
            filters.append("series_type <> 'macro'")
        where_clause = " AND ".join(filters)
        query = f"""
        SELECT *
        FROM read_parquet('{self.raw_timeseries_path.as_posix()}')
        WHERE {where_clause}
        """
        df = con.execute(query).df()
        if not cols:
            if "entity_id" in df.columns:
                df = df[df["entity_id"].astype(str).isin(ids)]
            elif "company_id" in df.columns:
                df = df[df["company_id"].astype(str).isin(ids)]
            if "series_type" in df.columns:
                df = df[df["series_type"].astype(str) != "macro"]
        return self._select_latest_timeseries_revision(df)

    def _load_macro(self, as_of_dt: pd.Timestamp) -> pd.DataFrame:
        if self.skip_macro:
            return pd.DataFrame()
        macro_path = self.macro_timeseries_path or self.raw_timeseries_path
        if not macro_path.exists():
            return pd.DataFrame()
        if not _is_readable_file(macro_path):
            if self.debug:
                print(f"[macro] unreadable file {macro_path}")
            return pd.DataFrame()
        if self.cache_timeseries and self._timeseries_cache is not None:
            df = self._timeseries_cache
            if "series_type" in df.columns:
                df = df[df["series_type"].astype(str) == "macro"]
            cutoff = pd.to_datetime(as_of_dt, utc=True)
            pub_col = "published_at" if "published_at" in df.columns else ("available_time" if "available_time" in df.columns else None)
            ing_col = None if self.historical_backfill_mode else ("ingested_at" if "ingested_at" in df.columns else ("ingestion_time" if "ingestion_time" in df.columns else None))
            eff_col = None
            for c in ["effective_at", "event_time", "trade_date", "date"]:
                if c in df.columns:
                    eff_col = c
                    break
            if pub_col:
                df = df[(df[pub_col].isna()) | (pd.to_datetime(df[pub_col], utc=True, errors="coerce") <= cutoff)]
            if ing_col:
                df = df[(df[ing_col].isna()) | (pd.to_datetime(df[ing_col], utc=True, errors="coerce") <= cutoff)]
            if eff_col:
                df = df[(df[eff_col].isna()) | (pd.to_datetime(df[eff_col], utc=True, errors="coerce") <= cutoff)]
            return self._select_latest_timeseries_revision(df)
        cache_key = as_of_dt.normalize().isoformat()
        if cache_key in self._macro_cache:
            return self._macro_cache[cache_key].copy()
        con = duckdb.connect()
        cutoff = _as_of_ts_literal(as_of_dt)
        cols = self._get_parquet_columns(macro_path)
        filters = []
        pub_col = "published_at" if "published_at" in cols else ("available_time" if "available_time" in cols else None)
        ing_col = None if self.historical_backfill_mode else ("ingested_at" if "ingested_at" in cols else ("ingestion_time" if "ingestion_time" in cols else None))
        eff_col = "effective_at" if "effective_at" in cols else ("event_time" if "event_time" in cols else ("trade_date" if "trade_date" in cols else None))
        if pub_col:
            filters.append(f"({pub_col} IS NULL OR CAST({pub_col} AS TIMESTAMP) <= TIMESTAMP '{cutoff}')")
        if ing_col:
            filters.append(f"({ing_col} IS NULL OR CAST({ing_col} AS TIMESTAMP) <= TIMESTAMP '{cutoff}')")
        if eff_col:
            filters.append(f"({eff_col} IS NULL OR CAST({eff_col} AS TIMESTAMP) <= TIMESTAMP '{cutoff}')")
        if "series_type" in cols:
            filters.append("series_type = 'macro'")
        where_clause = " AND ".join(filters)
        query = f"""
        SELECT *
        FROM read_parquet('{macro_path.as_posix()}')
        WHERE {where_clause}
        """
        df = con.execute(query).df()
        if "series_type" in df.columns:
            df = df[df["series_type"].astype(str) == "macro"]
        df = self._select_latest_timeseries_revision(df)
        self._macro_cache[cache_key] = df
        return df

    def _load_events(self, company_id: str | List[str], as_of_dt: pd.Timestamp) -> pd.DataFrame:
        if self.skip_events:
            return pd.DataFrame()
        ids = [str(company_id)] if not isinstance(company_id, list) else [str(i) for i in company_id if i is not None]
        ids = list(dict.fromkeys(ids))
        if not self.event_store_path.exists():
            return self._load_corporate_actions_event_fallback(ids, as_of_dt)
        if not _is_readable_file(self.event_store_path):
            if self.debug:
                print(f"[events] unreadable file {self.event_store_path}")
            return self._load_corporate_actions_event_fallback(ids, as_of_dt)
        if self.cache_events and self._events_cache is None:
            if self.debug:
                start = time.time()
                print(f"[events] caching from {self.event_store_path}", flush=True)
            con = duckdb.connect()
            self._events_cache = con.execute(
                f"SELECT * FROM read_parquet('{self.event_store_path.as_posix()}', union_by_name=True)"
            ).df()
            if self.debug:
                print(f"[events] cache rows={len(self._events_cache)} dt={time.time()-start:.2f}s", flush=True)
        if self.cache_events and self._events_cache is not None:
            df = self._events_cache
            if "company_id" in df.columns:
                df = df[df["company_id"].astype(str).isin(ids)]
            cutoff = pd.to_datetime(as_of_dt, utc=True)
            pub_col = "published_at" if "published_at" in df.columns else ("announced_at" if "announced_at" in df.columns else None)
            ing_col = None if self.historical_backfill_mode else ("ingested_at" if "ingested_at" in df.columns else ("created_at" if "created_at" in df.columns else None))
            eff_col = "effective_at" if "effective_at" in df.columns else self._event_time_col(df)
            if pub_col:
                pub = pd.to_datetime(df[pub_col], utc=True, errors="coerce")
                df = df[pub.isna() | (pub <= cutoff)]
            if ing_col:
                ing = pd.to_datetime(df[ing_col], utc=True, errors="coerce")
                df = df[ing.isna() | (ing <= cutoff)]
            if eff_col:
                eff = pd.to_datetime(df[eff_col], utc=True, errors="coerce")
                df = df[eff.isna() | (eff <= cutoff)]
            df = self._select_latest_event_revision(df)
            if df.empty:
                return self._load_corporate_actions_event_fallback(ids, as_of_dt)
            return df
        con = duckdb.connect()
        cutoff = _as_of_ts_literal(as_of_dt)
        cols = self._get_parquet_columns(self.event_store_path)
        pub_col = "published_at" if "published_at" in cols else ("announced_at" if "announced_at" in cols else None)
        ing_col = None if self.historical_backfill_mode else ("ingested_at" if "ingested_at" in cols else ("created_at" if "created_at" in cols else None))
        eff_col = "effective_at" if "effective_at" in cols else ("announced_at" if "announced_at" in cols else None)
        filters = []
        if "company_id" in cols:
            id_clause = " OR ".join([f"company_id = '{i}'" for i in ids])
            filters.append(f"({id_clause})")
        if pub_col:
            filters.append(f"({pub_col} IS NULL OR try_cast({pub_col} AS TIMESTAMP) <= TIMESTAMP '{cutoff}')")
        if ing_col:
            filters.append(f"({ing_col} IS NULL OR try_cast({ing_col} AS TIMESTAMP) <= TIMESTAMP '{cutoff}')")
        if eff_col:
            filters.append(f"({eff_col} IS NULL OR try_cast({eff_col} AS TIMESTAMP) <= TIMESTAMP '{cutoff}')")
        where_clause = " AND ".join(filters) if filters else "TRUE"
        query = f"""
        SELECT *
        FROM read_parquet('{self.event_store_path.as_posix()}', union_by_name=True)
        WHERE {where_clause}
        """
        df = con.execute(query).df()
        if not cols:
            if "company_id" in df.columns:
                df = df[df["company_id"].astype(str).isin(ids)]
            cutoff_pd = pd.to_datetime(as_of_dt, utc=True)
            pub_col_fb = "published_at" if "published_at" in df.columns else ("announced_at" if "announced_at" in df.columns else None)
            ing_col_fb = None if self.historical_backfill_mode else ("ingested_at" if "ingested_at" in df.columns else ("created_at" if "created_at" in df.columns else None))
            eff_col_fb = "effective_at" if "effective_at" in df.columns else self._event_time_col(df)
            if pub_col_fb:
                pub = pd.to_datetime(df[pub_col_fb], utc=True, errors="coerce")
                df = df[pub.isna() | (pub <= cutoff_pd)]
            if ing_col_fb:
                ing = pd.to_datetime(df[ing_col_fb], utc=True, errors="coerce")
                df = df[ing.isna() | (ing <= cutoff_pd)]
            if eff_col_fb:
                eff = pd.to_datetime(df[eff_col_fb], utc=True, errors="coerce")
                df = df[eff.isna() | (eff <= cutoff_pd)]
        df = self._select_latest_event_revision(df)
        if df.empty:
            return self._load_corporate_actions_event_fallback(ids, as_of_dt)
        return df

    def _load_corporate_actions_event_fallback(self, ids: List[str], as_of_dt: pd.Timestamp) -> pd.DataFrame:
        if not ids:
            return pd.DataFrame()
        if not self.corporate_actions_master_path.exists():
            return pd.DataFrame()
        if not _is_readable_file(self.corporate_actions_master_path):
            if self.debug:
                print(f"[events-fallback] unreadable file {self.corporate_actions_master_path}")
            return pd.DataFrame()

        ticker_aliases = sorted(
            {
                alias.upper()
                for alias in ids
                if alias
                and not alias.lower().startswith(("permno:", "permco:"))
                and not alias.isdigit()
                and 1 <= len(alias) <= 10
            }
        )
        cik_aliases = sorted(
            {
                alias.zfill(10)
                for alias in ids
                if alias and alias.isdigit() and 6 <= len(alias.zfill(10)) <= 10
            }
        )
        permno_aliases = sorted(
            {
                alias.split(":", 1)[1]
                for alias in ids
                if alias and alias.lower().startswith("permno:") and alias.split(":", 1)[1]
            }
        )

        def _normalize_corp_actions(df: pd.DataFrame) -> pd.DataFrame:
            if df.empty:
                return df
            out = df.copy()
            for col in ("action_date", "dclrdt", "exdt", "paydt", "filing_date", "accepted_date"):
                if col in out.columns:
                    out[col] = pd.to_datetime(out[col], utc=True, errors="coerce")
            out = out.dropna(subset=["action_type", "action_date"])
            if out.empty:
                return pd.DataFrame()
            out["event_type"] = out["action_type"].astype(str)
            if "action_subtype" in out.columns:
                out["event_subtype"] = out["action_subtype"]
            else:
                out["event_subtype"] = None
            out["announced_at"] = out["dclrdt"] if "dclrdt" in out.columns else pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
            if "filing_date" in out.columns:
                out["announced_at"] = out["announced_at"].combine_first(out["filing_date"])
            if "accepted_date" in out.columns:
                out["announced_at"] = out["announced_at"].combine_first(out["accepted_date"])
            out["announced_at"] = out["announced_at"].combine_first(out["action_date"])
            out["effective_at"] = out["exdt"] if "exdt" in out.columns else pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
            if "paydt" in out.columns:
                out["effective_at"] = out["effective_at"].combine_first(out["paydt"])
            out["effective_at"] = out["effective_at"].combine_first(out["action_date"])
            out["created_at"] = out["announced_at"].combine_first(out["effective_at"]).combine_first(out["action_date"])
            company_key = str(ids[0])
            out["company_id"] = company_key
            out["source_type"] = "corporate_actions_master_fallback"

            def _row_params(row: pd.Series) -> Dict[str, Any]:
                payload: Dict[str, Any] = {}
                for key in (
                    "amount",
                    "ratio",
                    "buyback_amount_qtr",
                    "deal_value",
                    "company_name",
                    "ticker",
                    "tic",
                    "cik",
                    "source",
                    "source_action_type",
                    "source_action_subtype",
                ):
                    value = _null_if_na(row.get(key))
                    if value is not None:
                        payload[key] = value
                return payload

            out["params"] = out.apply(_row_params, axis=1)

            def _event_id(row: pd.Series) -> str:
                key = "|".join(
                    [
                        str(company_key),
                        str(_null_if_na(row.get("ticker")) or _null_if_na(row.get("tic")) or ""),
                        str(_null_if_na(row.get("cik")) or ""),
                        str(_null_if_na(row.get("action_type")) or ""),
                        str(_null_if_na(row.get("action_subtype")) or ""),
                        str(_null_if_na(row.get("action_date")) or ""),
                        str(_null_if_na(row.get("amount")) or _null_if_na(row.get("deal_value")) or _null_if_na(row.get("buyback_amount_qtr")) or ""),
                    ]
                )
                return f"evt_corp_action:{uuid.uuid5(uuid.NAMESPACE_URL, key).hex}"

            out["event_id"] = out.apply(_event_id, axis=1)
            keep_cols = [
                "event_id",
                "company_id",
                "event_type",
                "event_subtype",
                "params",
                "announced_at",
                "effective_at",
                "created_at",
                "source_type",
            ]
            return out[keep_cols]

        if self.cache_events and self._corporate_actions_cache is None:
            if self.debug:
                start = time.time()
                print(f"[events-fallback] caching from {self.corporate_actions_master_path}", flush=True)
            con = duckdb.connect()
            self._corporate_actions_cache = con.execute(
                f"SELECT * FROM read_parquet('{self.corporate_actions_master_path.as_posix()}', union_by_name=True)"
            ).df()
            if self.debug:
                print(f"[events-fallback] cache rows={len(self._corporate_actions_cache)} dt={time.time()-start:.2f}s", flush=True)

        if self.cache_events and self._corporate_actions_cache is not None:
            df = self._corporate_actions_cache.copy()
            mask = pd.Series(False, index=df.index)
            if "ticker" in df.columns and ticker_aliases:
                mask = mask | df["ticker"].astype(str).str.upper().isin(ticker_aliases)
            if "tic" in df.columns and ticker_aliases:
                mask = mask | df["tic"].astype(str).str.upper().isin(ticker_aliases)
            if "cik" in df.columns and cik_aliases:
                cik_norm = df["cik"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)
                mask = mask | cik_norm.isin(cik_aliases)
            if "permno" in df.columns and permno_aliases:
                permno_norm = df["permno"].astype(str).str.replace(r"\.0$", "", regex=True)
                mask = mask | permno_norm.isin(permno_aliases)
            df = df[mask]
            if "action_date" in df.columns:
                action_time = pd.to_datetime(df["action_date"], utc=True, errors="coerce")
                df = df[action_time.isna() | (action_time <= pd.to_datetime(as_of_dt, utc=True))]
            return self._select_latest_event_revision(_normalize_corp_actions(df))

        con = duckdb.connect()
        cols = self._get_parquet_columns(self.corporate_actions_master_path)
        filters: List[str] = []
        alias_filters: List[str] = []
        if "ticker" in cols and ticker_aliases:
            alias_filters.append(f"upper(ticker) IN ({', '.join(_sql_quote(v) for v in ticker_aliases)})")
        if "tic" in cols and ticker_aliases:
            alias_filters.append(f"upper(tic) IN ({', '.join(_sql_quote(v) for v in ticker_aliases)})")
        if "cik" in cols and cik_aliases:
            alias_filters.append(f"lpad(replace(cast(cik as varchar), '.0', ''), 10, '0') IN ({', '.join(_sql_quote(v) for v in cik_aliases)})")
        if "permno" in cols and permno_aliases:
            alias_filters.append(f"replace(cast(permno as varchar), '.0', '') IN ({', '.join(_sql_quote(v) for v in permno_aliases)})")
        if not alias_filters:
            return pd.DataFrame()
        filters.append("(" + " OR ".join(alias_filters) + ")")
        cutoff = _as_of_ts_literal(as_of_dt)
        if "action_date" in cols:
            filters.append(f"(action_date IS NULL OR try_cast(action_date AS TIMESTAMP) <= TIMESTAMP '{cutoff}')")
        if "filing_date" in cols:
            filters.append(f"(filing_date IS NULL OR try_cast(filing_date AS TIMESTAMP) <= TIMESTAMP '{cutoff}')")
        if "accepted_date" in cols:
            filters.append(f"(accepted_date IS NULL OR try_cast(accepted_date AS TIMESTAMP) <= TIMESTAMP '{cutoff}')")
        where_clause = " AND ".join(filters)
        query = f"""
        SELECT *
        FROM read_parquet('{self.corporate_actions_master_path.as_posix()}', union_by_name=True)
        WHERE {where_clause}
        """
        df = con.execute(query).df()
        return self._select_latest_event_revision(_normalize_corp_actions(df))

    def _load_ownership_summary(self, company_id: str | List[str], as_of_dt: pd.Timestamp) -> pd.DataFrame:
        if not self.ownership_summary_path.exists():
            return pd.DataFrame()
        if not _is_readable_file(self.ownership_summary_path):
            if self.debug:
                print(f"[ownership] unreadable file {self.ownership_summary_path}")
            return pd.DataFrame()

        ids = [str(company_id)] if not isinstance(company_id, list) else [str(i) for i in company_id if i is not None]
        ids = list(dict.fromkeys(ids))

        if self.cache_ownership and self._ownership_cache is None:
            if self.debug:
                start = time.time()
                print(f"[ownership] caching from {self.ownership_summary_path}", flush=True)
            con = duckdb.connect()
            self._ownership_cache = con.execute(
                f"SELECT * FROM read_parquet('{self.ownership_summary_path.as_posix()}', union_by_name=True)"
            ).df()
            if self.debug:
                print(f"[ownership] cache rows={len(self._ownership_cache)} dt={time.time()-start:.2f}s", flush=True)

        df = self._ownership_cache.copy() if (self.cache_ownership and self._ownership_cache is not None) else None
        if df is None:
            con = duckdb.connect()
            id_list = ", ".join([_sql_quote(i) for i in ids])
            cutoff = _as_of_ts_literal(as_of_dt)
            query = f"""
            SELECT *
            FROM read_parquet('{self.ownership_summary_path.as_posix()}', union_by_name=True)
            WHERE company_id IN ({id_list})
              AND (published_at IS NULL OR try_cast(published_at AS TIMESTAMP) <= TIMESTAMP '{cutoff}')
              {"AND (ingested_at IS NULL OR try_cast(ingested_at AS TIMESTAMP) <= TIMESTAMP '" + cutoff + "')" if not self.historical_backfill_mode else ""}
              AND (effective_at IS NULL OR try_cast(effective_at AS TIMESTAMP) <= TIMESTAMP '{cutoff}')
            """
            return self._select_latest_ownership_snapshot(con.execute(query).df())

        if "company_id" in df.columns:
            df = df[df["company_id"].astype(str).isin(ids)]
        cutoff = pd.to_datetime(as_of_dt, utc=True)
        for col in ("published_at", "ingested_at", "effective_at", "filing_date", "report_date"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
        if "published_at" in df.columns:
            df = df[df["published_at"].isna() | (df["published_at"] <= cutoff)]
        if "ingested_at" in df.columns and not self.historical_backfill_mode:
            df = df[df["ingested_at"].isna() | (df["ingested_at"] <= cutoff)]
        if "effective_at" in df.columns:
            df = df[df["effective_at"].isna() | (df["effective_at"] <= cutoff)]
        return self._select_latest_ownership_snapshot(df)

    def _load_issuer_ratings(self, company_id: str | List[str], as_of_dt: pd.Timestamp) -> pd.DataFrame:
        if not self.issuer_ratings_path.exists():
            return pd.DataFrame()
        if not _is_readable_file(self.issuer_ratings_path):
            if self.debug:
                print(f"[issuer_ratings] unreadable file {self.issuer_ratings_path}")
            return pd.DataFrame()

        ids = [str(company_id)] if not isinstance(company_id, list) else [str(i) for i in company_id if i is not None]
        ids = list(dict.fromkeys(ids))

        if self.cache_ratings and self._issuer_ratings_cache is None:
            if self.debug:
                start = time.time()
                print(f"[issuer_ratings] caching from {self.issuer_ratings_path}", flush=True)
            con = duckdb.connect()
            self._issuer_ratings_cache = con.execute(
                f"SELECT * FROM read_parquet('{self.issuer_ratings_path.as_posix()}', union_by_name=True)"
            ).df()
            if self.debug:
                print(
                    f"[issuer_ratings] cache rows={len(self._issuer_ratings_cache)} dt={time.time()-start:.2f}s",
                    flush=True,
                )

        df = self._issuer_ratings_cache.copy() if (self.cache_ratings and self._issuer_ratings_cache is not None) else None
        if df is None:
            con = duckdb.connect()
            id_list = ", ".join([_sql_quote(i) for i in ids])
            cutoff = _as_of_ts_literal(as_of_dt)
            query = f"""
            SELECT *
            FROM read_parquet('{self.issuer_ratings_path.as_posix()}', union_by_name=True)
            WHERE company_id IN ({id_list})
              AND (published_at IS NULL OR try_cast(published_at AS TIMESTAMP) <= TIMESTAMP '{cutoff}')
              {"AND (ingested_at IS NULL OR try_cast(ingested_at AS TIMESTAMP) <= TIMESTAMP '" + cutoff + "')" if not self.historical_backfill_mode else ""}
              AND (effective_at IS NULL OR try_cast(effective_at AS TIMESTAMP) <= TIMESTAMP '{cutoff}')
            """
            return self._select_latest_rating_snapshot(con.execute(query).df())

        if "company_id" in df.columns:
            df = df[df["company_id"].astype(str).isin(ids)]
        cutoff = pd.to_datetime(as_of_dt, utc=True)
        for col in ("rating_date", "published_at", "ingested_at", "effective_at"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
        if "published_at" in df.columns:
            df = df[df["published_at"].isna() | (df["published_at"] <= cutoff)]
        if "ingested_at" in df.columns and not self.historical_backfill_mode:
            df = df[df["ingested_at"].isna() | (df["ingested_at"] <= cutoff)]
        if "effective_at" in df.columns:
            df = df[df["effective_at"].isna() | (df["effective_at"] <= cutoff)]
        if "rating_date" in df.columns:
            df = df[df["rating_date"].isna() | (df["rating_date"] <= cutoff)]
        return self._select_latest_rating_snapshot(df)

    def _load_estimates(self, company_id: str | List[str], as_of_dt: pd.Timestamp) -> pd.DataFrame:
        if str(os.environ.get("AXIOM_SKIP_ESTIMATES", "")).strip().lower() in {"1", "true", "yes", "on"}:
            if self.debug:
                print("[estimates] skipped via AXIOM_SKIP_ESTIMATES")
            return pd.DataFrame()
        if not self.estimates_path.exists():
            return pd.DataFrame()
        if not _is_readable_file(self.estimates_path):
            if self.debug:
                print(f"[estimates] unreadable file {self.estimates_path}")
            return pd.DataFrame()

        ids = [str(company_id)] if not isinstance(company_id, list) else [str(i) for i in company_id if i is not None]
        ids = list(dict.fromkeys(ids))
        if not ids:
            return pd.DataFrame()

        cols = self._get_parquet_columns(self.estimates_path)
        filters: List[str] = []
        id_list = ", ".join([_sql_quote(i) for i in ids])
        for col in ("company_id", "entity_id", "security_id"):
            if col in cols:
                filters.append(f"cast({col} as varchar) IN ({id_list})")
        if not filters:
            return pd.DataFrame()

        cutoff = _as_of_ts_literal(as_of_dt)
        query = f"""
        SELECT *
        FROM read_parquet('{self.estimates_path.as_posix()}', union_by_name=True)
        WHERE ({' OR '.join(filters)})
          AND (available_time IS NULL OR try_cast(available_time AS TIMESTAMP) <= TIMESTAMP '{cutoff}')
          AND (event_time IS NULL OR try_cast(event_time AS TIMESTAMP) <= TIMESTAMP '{cutoff}')
        """
        try:
            con = duckdb.connect()
            df = con.execute(query).df()
        except Exception:
            return pd.DataFrame()
        if "available_time" in df.columns:
            df["available_time"] = pd.to_datetime(df["available_time"], utc=True, errors="coerce")
        if "event_time" in df.columns:
            df["event_time"] = pd.to_datetime(df["event_time"], utc=True, errors="coerce")
        if "period_end" in df.columns:
            df["period_end"] = pd.to_datetime(df["period_end"], utc=True, errors="coerce")
        return df

    def _load_entity_table(self) -> pd.DataFrame:
        if self._entity_table_cache is not None:
            return self._entity_table_cache
        if not self.entity_table_path.exists():
            self._entity_table_cache = pd.DataFrame()
            return self._entity_table_cache
        try:
            con = duckdb.connect()
            self._entity_table_cache = con.execute(
                f"SELECT * FROM read_parquet('{self.entity_table_path.as_posix()}', union_by_name=True)"
            ).df()
        except Exception:
            self._entity_table_cache = pd.DataFrame()
        return self._entity_table_cache

    def _load_identifier_maps(self) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
        if self._identifier_to_entity is not None and self._entity_to_identifiers is not None:
            return self._identifier_to_entity, self._entity_to_identifiers
        identifier_to_entity: Dict[str, str] = {}
        entity_to_identifiers: Dict[str, List[str]] = {}
        if self.entity_identifier_path.exists():
            try:
                con = duckdb.connect()
                df = con.execute(
                    "SELECT entity_id, identifier_value, identifier_type "
                    f"FROM read_parquet('{self.entity_identifier_path.as_posix()}', union_by_name=True)"
                ).df()
                df = df.dropna(subset=["entity_id", "identifier_value"])
                df["entity_id"] = df["entity_id"].astype(str)
                df["identifier_value"] = df["identifier_value"].astype(str)
                for _, row in df.iterrows():
                    ent = row["entity_id"]
                    ident = row["identifier_value"]
                    ident_type = str(row.get("identifier_type", "")).lower() if row.get("identifier_type") is not None else ""
                    aliases = {ident}
                    if ident_type == "ticker":
                        aliases.add(ident.upper())
                    if ident_type in ("cusip", "isin", "sedol"):
                        aliases.add(ident.upper())
                    if ident.isdigit():
                        stripped = ident.lstrip("0")
                        if stripped:
                            aliases.add(stripped)
                            for w in (6, 8, 10):
                                aliases.add(stripped.zfill(w))
                        # common widths from original
                        for w in (6, 8, 10):
                            aliases.add(ident.zfill(w))
                    if ident_type == "permno":
                        aliases.add(f"permno:{ident}")
                        if ident.isdigit():
                            stripped = ident.lstrip("0")
                            if stripped:
                                aliases.add(f"permno:{stripped}")
                    if ident_type == "permco":
                        aliases.add(f"permco:{ident}")
                        if ident.isdigit():
                            stripped = ident.lstrip("0")
                            if stripped:
                                aliases.add(f"permco:{stripped}")
                    for alias in aliases:
                        identifier_to_entity[alias] = ent
                        if ent not in entity_to_identifiers:
                            entity_to_identifiers[ent] = []
                        entity_to_identifiers[ent].append(alias)
                # ensure identity mapping
                for ent in list(entity_to_identifiers.keys()):
                    identifier_to_entity[ent] = ent
                    if ent not in entity_to_identifiers[ent]:
                        entity_to_identifiers[ent].append(ent)
            except Exception:
                identifier_to_entity = {}
                entity_to_identifiers = {}
        self._identifier_to_entity = identifier_to_entity
        self._entity_to_identifiers = entity_to_identifiers
        return self._identifier_to_entity, self._entity_to_identifiers

    def _resolve_entity_aliases(
        self,
        company_id: str,
        extra_aliases: Optional[List[str]] = None,
    ) -> Tuple[str, List[str]]:
        if company_id is None:
            return "", []
        cid = str(company_id)
        identifier_to_entity, entity_to_identifiers = self._load_identifier_maps()

        canonical = identifier_to_entity.get(cid)
        if canonical is None:
            for alias in list(extra_aliases or []):
                if alias is None:
                    continue
                mapped = identifier_to_entity.get(str(alias))
                if mapped is not None:
                    canonical = mapped
                    break
        if canonical is None:
            canonical = cid

        aliases = list(entity_to_identifiers.get(canonical, []))
        # always include input, canonical, and any explicit aliases provided by the caller
        for alias in [cid, canonical, *(list(extra_aliases or []))]:
            if alias is None:
                continue
            text = str(alias)
            if text not in aliases:
                aliases = aliases + [text]
        # dedupe preserve order
        seen = set()
        uniq = []
        for a in aliases:
            if a is None:
                continue
            s = str(a)
            if s in seen:
                continue
            seen.add(s)
            uniq.append(s)
        return canonical, uniq

    def _companyfacts_candidate_paths(self, identifiers: List[str]) -> List[Path]:
        if self.companyfacts_root is None:
            return []
        candidates: List[Path] = []
        seen: set[Path] = set()
        for identifier in identifiers:
            text = str(identifier or "").strip()
            if not text:
                continue
            digits = re.sub(r"\D", "", text)
            if not digits:
                continue
            padded = (digits.lstrip("0") or "0").zfill(10)
            candidate = self.companyfacts_root / f"CIK{padded}.json"
            if candidate.exists() and candidate not in seen:
                seen.add(candidate)
                candidates.append(candidate)
        return candidates

    def _load_companyfacts(self, path: Path) -> Optional[Dict[str, Any]]:
        if path in self._companyfacts_cache:
            return self._companyfacts_cache[path]
        if not path.exists():
            self._companyfacts_cache[path] = None
            return None
        try:
            completed = subprocess.run(
                ["/bin/cat", str(path)],
                capture_output=True,
                check=True,
                timeout=COMPANYFACTS_LOAD_TIMEOUT_SECONDS,
            )
            companyfacts = json.loads(completed.stdout)
        except Exception:
            try:
                companyfacts = json.loads(path.read_text())
            except Exception:
                companyfacts = None
        self._companyfacts_cache[path] = companyfacts
        return companyfacts

    def _load_first_companyfacts_bundle(
        self,
        company_id: str,
        aliases: List[str],
    ) -> tuple[Optional[Dict[str, Any]], Optional[Path]]:
        for companyfacts_path in self._companyfacts_candidate_paths([company_id, *aliases]):
            companyfacts = self._load_companyfacts(companyfacts_path)
            if companyfacts:
                return companyfacts, companyfacts_path
        return None, None

    def _load_smart_metric_registry(self) -> Dict[str, Any]:
        if self._smart_metric_registry_cache is not None:
            return self._smart_metric_registry_cache
        registry_path = _smart_metric_registry_path()
        fallback = {
            "metrics": {
                "debt_like_obligations_normalized": {"status": "enabled", "promotion_rule": "builder_inline"},
                "available_liquidity_normalized": {"status": "enabled", "promotion_rule": "builder_inline"},
                "operating_earnings_normalized": {"status": "enabled", "promotion_rule": "builder_inline"},
                "net_debt_normalized": {"status": "enabled", "promotion_rule": "builder_inline"},
                "gross_leverage_normalized": {"status": "enabled", "promotion_rule": "builder_inline"},
                "net_leverage_normalized": {"status": "enabled", "promotion_rule": "builder_inline"},
            }
        }
        if not registry_path.exists():
            self._smart_metric_registry_cache = fallback
            return self._smart_metric_registry_cache
        try:
            self._smart_metric_registry_cache = json.loads(registry_path.read_text())
        except Exception:
            self._smart_metric_registry_cache = fallback
        return self._smart_metric_registry_cache

    def _load_market_availability_overrides(self) -> Dict[str, Any]:
        if self._market_availability_overrides_cache is not None:
            return self._market_availability_overrides_cache
        overrides_path = _market_availability_overrides_path()
        if not overrides_path.exists():
            self._market_availability_overrides_cache = {}
            return self._market_availability_overrides_cache
        try:
            payload = json.loads(overrides_path.read_text())
        except Exception:
            payload = {}
        self._market_availability_overrides_cache = payload if isinstance(payload, dict) else {}
        return self._market_availability_overrides_cache

    def _simple_smart_node(
        self,
        *,
        value: Optional[float],
        support_mode: Optional[str] = None,
        missing_reason: Optional[str] = None,
        component_breakdown: Optional[Dict[str, Any]] = None,
        quality_flags: Optional[List[str]] = None,
        provenance: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        resolved_support_mode = support_mode
        if resolved_support_mode is None:
            resolved_support_mode = "exact" if value is not None else "unsupported"
        return {
            "support_mode": resolved_support_mode,
            "value": None if value is None else float(value),
            "missing_reason": missing_reason,
            "component_breakdown": component_breakdown or {},
            "quality_flags": quality_flags,
            "provenance": provenance or [],
        }

    def _smart_node_from_feature(self, feat: Optional[FeatureRecord]) -> Dict[str, Any]:
        if feat is None:
            return self._simple_smart_node(value=None, missing_reason="component_unavailable")
        value = _safe_float(feat.value)
        if value is None:
            return self._simple_smart_node(
                value=None,
                support_mode="unsupported",
                missing_reason=feat.missing_reason or "component_unavailable",
                component_breakdown=feat.component_breakdown,
                quality_flags=feat.quality_flags,
                provenance=[asdict(ref) for ref in (feat.provenance or [])],
            )
        support_mode = _classify_metric_support_mode(
            base_mode=feat.support_mode or "exact",
            metric_id=self._registry_metric_id_for_feature(feat.name),
            value=value,
            quality_flags=feat.quality_flags,
            component_breakdown=feat.component_breakdown,
        )
        return self._simple_smart_node(
            value=value,
            support_mode=support_mode,
            component_breakdown=feat.component_breakdown,
            quality_flags=feat.quality_flags,
            provenance=[asdict(ref) for ref in (feat.provenance or [])],
        )

    def _smart_feature_record_from_node(self, metric_name: str, node: Dict[str, Any]) -> FeatureRecord:
        provenance = [
            InputReference(
                artifact_type=str(ref.get("artifact_type") or "DerivedMetric"),
                artifact_id=str(ref.get("artifact_id") or metric_name),
                source=ref.get("source"),
                published_at=ref.get("published_at"),
                ingested_at=ref.get("ingested_at"),
                hash=ref.get("hash"),
            )
            for ref in (node.get("provenance") or [])
        ]
        return FeatureRecord(
            name=metric_name,
            value=_safe_float(node.get("value")),
            unit=node.get("unit"),
            computed_at=str(node.get("computed_at") or _now_iso()),
            as_of_time=str(node.get("as_of_time")),
            window=node.get("window"),
            confidence=_safe_float(node.get("confidence")),
            provenance=provenance,
            missing_reason=node.get("missing_reason"),
            fallback_used=node.get("fallback_used"),
            primary_source_basis=node.get("primary_source_basis"),
            input_source_classification=node.get("input_source_classification"),
            input_source_alignment_status=node.get("input_source_alignment_status"),
            input_layer_bucket=node.get("input_layer_bucket"),
            input_layer_bucket_reason=node.get("input_layer_bucket_reason"),
            support_mode=node.get("support_mode"),
            applicability_status=node.get("applicability_status"),
            component_breakdown=node.get("component_breakdown"),
            quality_flags=node.get("quality_flags"),
            view_type=node.get("view_type"),
        )

    def _merge_input_references(
        self,
        *reference_groups: Optional[List[InputReference]],
    ) -> List[InputReference]:
        merged: List[InputReference] = []
        for refs in reference_groups:
            for ref in refs or []:
                if ref not in merged:
                    merged.append(ref)
        return merged

    def _overlay_feature_with_smart_node(
        self,
        features: Dict[str, FeatureRecord],
        target_name: str,
        smart_node: Optional[Dict[str, Any]],
    ) -> None:
        if smart_node is None or smart_node.get("value") is None:
            return
        existing = features.get(target_name)
        if existing is None:
            return
        normalized = self._smart_feature_record_from_node(target_name, smart_node)
        existing.value = normalized.value
        existing.unit = normalized.unit or existing.unit
        existing.confidence = None
        existing.provenance = self._merge_input_references(existing.provenance, normalized.provenance)
        existing.missing_reason = normalized.missing_reason
        existing.fallback_used = (
            None if _support_mode_is_exact_like(normalized.support_mode) else "smart_normalized_market_inputs"
        )
        existing.primary_source_basis = normalized.primary_source_basis or "smart_normalized_policy"
        existing.component_breakdown = normalized.component_breakdown
        existing.quality_flags = normalized.quality_flags
        existing.support_mode = normalized.support_mode

    def _apply_market_relevant_smart_normalized_inputs(
        self,
        company_id: str,
        aliases: List[str],
        facts: pd.DataFrame,
        as_of: pd.Timestamp,
        features: Dict[str, FeatureRecord],
    ) -> None:
        if not features:
            return

        registry = self._load_smart_metric_registry()
        market_availability_overrides = self._load_market_availability_overrides()
        companyfacts, companyfacts_path = self._load_first_companyfacts_bundle(company_id, aliases)

        as_of_time = as_of.isoformat()
        computed_at = _now_iso()
        override_company_id = str(company_id)
        for identifier in [company_id, *aliases]:
            digits = re.sub(r"\D", "", str(identifier or ""))
            if digits:
                override_company_id = digits.lstrip("0").zfill(10)
                break

        combined_cash_value, _ = self._latest_fact_with_patterns(
            facts,
            self.fact_map.get("cash_and_short_term_investments", []),
            self.fact_pattern_map.get("cash_and_short_term_investments"),
        )
        current_debt_value, _ = self._latest_fact(facts, self.fact_map.get("debt_current", []))
        long_term_debt_value, _ = self._latest_fact(facts, self.fact_map.get("debt_long", []))
        current_debt_companyfacts_meta = None
        long_term_debt_companyfacts_meta = None
        ebitda_ttm, _, ebitda_ttm_breakdown, ebitda_ttm_flags = self._latest_ttm_statement_value(
            facts,
            self.fact_map.get("ebitda", []),
        )
        net_income_ttm, _, net_income_ttm_breakdown, net_income_ttm_flags = self._latest_ttm_statement_value(
            facts,
            self.fact_map.get("net_income", []),
        )
        interest_expense_ttm, _, interest_expense_ttm_breakdown, interest_expense_ttm_flags = self._latest_ttm_statement_value(
            facts,
            self.fact_map.get("interest_expense", []),
        )
        if interest_expense_ttm is None:
            repaired_interest_expense, _, repaired_interest_context = self._repaired_statement_direct_interest_expense(
                company_id=str(company_id),
                aliases=aliases,
                as_of=as_of,
            )
            if repaired_interest_expense is not None:
                interest_expense_ttm = repaired_interest_expense
                interest_expense_ttm_breakdown = repaired_interest_context or {
                    "formula": "companyfacts_interest_expense_ttm",
                }
                interest_expense_ttm_flags = []

        restricted_cash_sec_value = None
        restricted_cash_sec_meta = None
        marketable_sec_value = None
        marketable_sec_meta = None
        revolver_sec_value = None
        revolver_sec_meta = None
        lease_sec_value = None
        lease_sec_meta = None
        companyfacts_provenance = []
        if companyfacts is not None and companyfacts_path is not None:
            restricted_cash_sec_value, restricted_cash_sec_meta = _extract_sec_companyfacts_restricted_cash(
                companyfacts,
                as_of_time[:10],
            )
            marketable_sec_value, marketable_sec_meta = _extract_sec_companyfacts_marketable_securities(
                companyfacts,
                as_of_time[:10],
            )
            revolver_sec_value, revolver_sec_meta = _extract_sec_companyfacts_revolver_undrawn(
                companyfacts,
                as_of_time[:10],
            )
            lease_sec_value, lease_sec_meta = _extract_sec_companyfacts_lease_liabilities(
                companyfacts,
                as_of_time[:10],
            )
            companyfacts_provenance = [
                {
                    "artifact_type": "SecCompanyFacts",
                    "artifact_id": f"sec_companyfacts:{companyfacts_path.name}",
                    "source": str(companyfacts_path),
                    "published_at": as_of_time,
                    "ingested_at": computed_at,
                    "hash": None,
                }
            ]
            if current_debt_value is None:
                current_debt_candidates = _instant_candidates(
                    companyfacts,
                    DEBT_CURRENT_CONCEPTS + SHORT_TERM_BORROWINGS_CONCEPTS,
                    as_of_date=as_of_time[:10],
                    unit_filter="USD",
                )
                if current_debt_candidates:
                    current_debt_value = float(current_debt_candidates[0]["value"])
                    current_debt_companyfacts_meta = current_debt_candidates[0]["meta"]
            if long_term_debt_value is None:
                long_term_debt_candidates = _instant_candidates(
                    companyfacts,
                    DEBT_NONCURRENT_CONCEPTS,
                    as_of_date=as_of_time[:10],
                    unit_filter="USD",
                )
                if long_term_debt_candidates:
                    long_term_debt_value = float(long_term_debt_candidates[0]["value"])
                    long_term_debt_companyfacts_meta = long_term_debt_candidates[0]["meta"]

        row = {
            "company_id": override_company_id,
            "as_of_time": as_of_time,
            "features": {
                "capital_structure.total_debt_provider_direct": self._smart_node_from_feature(
                    features.get("capital_structure.total_debt_reported")
                ),
                "capital_structure.net_pension_liability": self._smart_node_from_feature(
                    features.get("capital_structure.net_pension_liability")
                ),
                "liquidity.cash_and_short_term_investments_provider_direct": self._simple_smart_node(
                    value=combined_cash_value,
                    support_mode="exact" if combined_cash_value is not None else "unsupported",
                    missing_reason="component_unavailable" if combined_cash_value is None else None,
                    component_breakdown={"formula": "financial.cash_and_short_term_investments"},
                ),
                "liquidity.cash_and_equivalents_statement_direct": self._smart_node_from_feature(
                    features.get("liquidity.cash")
                ),
                "liquidity.restricted_cash_sec_exact": self._simple_smart_node(
                    value=restricted_cash_sec_value,
                    support_mode="exact" if restricted_cash_sec_value is not None else "unsupported",
                    missing_reason="sec_concept_unavailable" if restricted_cash_sec_value is None else None,
                    component_breakdown=(
                        dict(restricted_cash_sec_meta or {}, companyfacts_path=str(companyfacts_path))
                        if companyfacts_path is not None
                        else restricted_cash_sec_meta
                    ),
                    provenance=companyfacts_provenance,
                ),
                "liquidity.marketable_securities_sec_exact": self._simple_smart_node(
                    value=marketable_sec_value,
                    support_mode="exact" if marketable_sec_value is not None else "unsupported",
                    missing_reason="sec_concept_unavailable" if marketable_sec_value is None else None,
                    component_breakdown=(
                        dict(marketable_sec_meta or {}, companyfacts_path=str(companyfacts_path))
                        if companyfacts_path is not None
                        else marketable_sec_meta
                    ),
                    provenance=companyfacts_provenance,
                ),
                "liquidity.revolver_undrawn_sec_exact": self._simple_smart_node(
                    value=revolver_sec_value,
                    support_mode="exact" if revolver_sec_value is not None else "unsupported",
                    missing_reason="sec_concept_unavailable" if revolver_sec_value is None else None,
                    component_breakdown=(
                        dict(revolver_sec_meta or {}, companyfacts_path=str(companyfacts_path))
                        if companyfacts_path is not None
                        else revolver_sec_meta
                    ),
                    provenance=companyfacts_provenance,
                ),
                "capital_structure.lease_liabilities_sec_exact": self._simple_smart_node(
                    value=lease_sec_value,
                    support_mode="exact" if lease_sec_value is not None else "unsupported",
                    missing_reason="sec_concept_unavailable" if lease_sec_value is None else None,
                    component_breakdown=(
                        dict(lease_sec_meta or {}, companyfacts_path=str(companyfacts_path))
                        if companyfacts_path is not None
                        else lease_sec_meta
                    ),
                    provenance=companyfacts_provenance,
                ),
                "liquidity.restricted_cash": self._smart_node_from_feature(features.get("liquidity.restricted_cash")),
                "liquidity.marketable_securities": self._smart_node_from_feature(
                    features.get("liquidity.marketable_securities")
                ),
                "liquidity.revolver_undrawn": self._smart_node_from_feature(features.get("liquidity.revolver_undrawn")),
                "operating.ebitda_ltm_provider_direct": self._simple_smart_node(
                    value=ebitda_ttm,
                    support_mode="exact" if ebitda_ttm is not None and not ebitda_ttm_flags else (
                        "proxy_missing_component" if ebitda_ttm is not None else "unsupported"
                    ),
                    missing_reason="component_unavailable" if ebitda_ttm is None else None,
                    component_breakdown=ebitda_ttm_breakdown,
                    quality_flags=ebitda_ttm_flags or None,
                ),
                "earnings.net_income_ttm_provider_direct": self._simple_smart_node(
                    value=net_income_ttm,
                    support_mode="exact" if net_income_ttm is not None and not net_income_ttm_flags else (
                        "proxy_missing_component" if net_income_ttm is not None else "unsupported"
                    ),
                    missing_reason="component_unavailable" if net_income_ttm is None else None,
                    component_breakdown=net_income_ttm_breakdown,
                    quality_flags=net_income_ttm_flags or None,
                ),
                "capital_structure.interest_expense_statement_direct": self._simple_smart_node(
                    value=interest_expense_ttm,
                    support_mode=(
                        "exact"
                        if interest_expense_ttm is not None and not interest_expense_ttm_flags
                        else ("proxy_missing_component" if interest_expense_ttm is not None else "unsupported")
                    ),
                    missing_reason="component_unavailable" if interest_expense_ttm is None else None,
                    component_breakdown=interest_expense_ttm_breakdown,
                    quality_flags=interest_expense_ttm_flags or None,
                ),
                "capital_structure.current_debt_statement_direct": self._simple_smart_node(
                    value=current_debt_value,
                    support_mode="exact" if current_debt_value is not None else "unsupported",
                    missing_reason="component_unavailable" if current_debt_value is None else None,
                    component_breakdown=(
                        dict(current_debt_companyfacts_meta or {}, companyfacts_path=str(companyfacts_path))
                        if current_debt_companyfacts_meta is not None and companyfacts_path is not None
                        else current_debt_companyfacts_meta
                    ),
                    provenance=companyfacts_provenance if current_debt_companyfacts_meta is not None else None,
                ),
                "capital_structure.long_term_debt_statement_direct": self._simple_smart_node(
                    value=long_term_debt_value,
                    support_mode="exact" if long_term_debt_value is not None else "unsupported",
                    missing_reason="component_unavailable" if long_term_debt_value is None else None,
                    component_breakdown=(
                        dict(long_term_debt_companyfacts_meta or {}, companyfacts_path=str(companyfacts_path))
                        if long_term_debt_companyfacts_meta is not None and companyfacts_path is not None
                        else long_term_debt_companyfacts_meta
                    ),
                    provenance=companyfacts_provenance if long_term_debt_companyfacts_meta is not None else None,
                ),
            },
        }

        repaired = materialize_smart_metrics_for_row(
            row=row,
            registry=registry,
            computed_at=computed_at,
            provenance_sources=[str(_smart_metric_registry_path())],
            companyfacts=companyfacts,
            market_availability_overrides=market_availability_overrides,
        )
        repaired_features = repaired.get("features") or {}
        for metric_name in (
            "capital_structure.debt_like_obligations_normalized",
            "liquidity.available_liquidity_normalized",
            "operating.operating_earnings_normalized",
            "capital_structure.net_debt_normalized",
            "capital_structure.gross_leverage_normalized",
            "capital_structure.net_leverage_normalized",
            "capital_structure.net_pension_liability",
            "capital_structure.debt_like_obligations_including_pension",
            "capital_structure.net_debt_including_pension",
            "capital_structure.gross_leverage_including_pension",
            "capital_structure.net_leverage_including_pension",
        ):
            node = repaired_features.get(metric_name)
            if node is not None:
                if (
                    metric_name == "capital_structure.net_pension_liability"
                    and _safe_float(node.get("value")) is None
                    and _safe_float(getattr(features.get(metric_name), "value", None)) is not None
                ):
                    continue
                features[metric_name] = self._smart_feature_record_from_node(metric_name, node)

    def _repaired_statement_direct_interest_expense(
        self,
        *,
        company_id: str,
        aliases: List[str],
        as_of: pd.Timestamp,
    ) -> tuple[Optional[float], List[InputReference], Optional[Dict[str, Any]]]:
        for companyfacts_path in self._companyfacts_candidate_paths([company_id, *aliases]):
            companyfacts = self._load_companyfacts(companyfacts_path)
            if not companyfacts:
                continue
            for concept_name in INTEREST_EXPENSE_TTM_EXACT_CONCEPTS:
                repaired_value, repaired_meta = _compute_companyfacts_ttm_from_concept(
                    companyfacts,
                    concept_name,
                    as_of,
                )
                if repaired_value is None or repaired_meta is None:
                    continue
                return (
                    float(repaired_value),
                    [
                        self._artifact_ref(
                            "SecCompanyFacts",
                            f"sec_companyfacts:{companyfacts_path.name}:{concept_name}",
                            str(companyfacts_path),
                            as_of,
                        )
                    ],
                    {
                        "mode": "companyfacts_ttm_interest_expense",
                        "concept": concept_name,
                        "ttm_context": repaired_meta,
                        "formula": "companyfacts_interest_expense_ttm",
                    },
                )
        return None, [], None

    # ---------------------------
    # Feature computation helpers
    # ---------------------------

    def _event_time_col(self, events: pd.DataFrame) -> Optional[str]:
        for c in ["announced_at", "effective_at", "created_at", "event_time"]:
            if c in events.columns:
                return c
        return None

    def _event_param_value(self, row: pd.Series, key: str) -> Any:
        params = row.get("params")
        if isinstance(params, dict):
            return params.get(key)
        return None

    def _event_param_float(self, row: pd.Series, keys: List[str]) -> Optional[float]:
        for key in keys:
            if key in row and row.get(key) is not None and not pd.isna(row.get(key)):
                val = _safe_float(row.get(key))
                if val is not None:
                    return val
            val = _safe_float(self._event_param_value(row, key))
            if val is not None:
                return val
        return None

    def _event_param_datetime(self, row: pd.Series, keys: List[str]) -> Optional[pd.Timestamp]:
        for key in keys:
            val = None
            if key in row and row.get(key) is not None and not pd.isna(row.get(key)):
                val = row.get(key)
            if val is None:
                val = self._event_param_value(row, key)
            if val is None:
                continue
            ts = pd.to_datetime(val, utc=True, errors="coerce")
            if pd.notna(ts):
                return ts
        return None

    def _event_amount(self, row: pd.Series) -> Optional[float]:
        # Prefer explicit amount columns first.
        amt = self._event_param_float(
            row,
            [
                "amount",
                "deal_value",
                "dealamount",
                "offering_amount",
                "amount_sold",
                "call_amount",
            ],
        )
        if amt is not None:
            return amt
        # FISD offering_amt_k is in thousands.
        offering_k = self._event_param_float(row, ["offering_amt_k"])
        if offering_k is not None:
            return offering_k * 1000.0
        return self._event_param_float(row, ["principal_amt"])

    def _event_refs(self, events: pd.DataFrame, limit: int = 5) -> List[InputReference]:
        refs: List[InputReference] = []
        if events is None or events.empty:
            return refs
        rows = events.head(limit)
        for _, row in rows.iterrows():
            refs.append(
                InputReference(
                    artifact_type="Event",
                    artifact_id=str(_null_if_na(row.get("event_id"))) if _null_if_na(row.get("event_id")) is not None else "",
                    source=str(_null_if_na(row.get("source_type"))) if _null_if_na(row.get("source_type")) is not None else None,
                    published_at=str(_null_if_na(row.get("announced_at"))) if _null_if_na(row.get("announced_at")) is not None else None,
                    ingested_at=str(_null_if_na(row.get("created_at"))) if _null_if_na(row.get("created_at")) is not None else None,
                    hash=None,
                )
            )
        return refs

    def _fact_refs(self, facts: pd.DataFrame, limit: int = 5) -> List[InputReference]:
        refs: List[InputReference] = []
        if facts is None or facts.empty:
            return refs
        rows = facts.head(limit)
        for _, row in rows.iterrows():
            refs.append(
                InputReference(
                    artifact_type="ExtractedFact",
                    artifact_id=str(_null_if_na(row.get("fact_id"))) if _null_if_na(row.get("fact_id")) is not None else "",
                    source=str(_null_if_na(row.get("source_type"))) if _null_if_na(row.get("source_type")) is not None else None,
                    published_at=str(_null_if_na(row.get("published_at"))) if _null_if_na(row.get("published_at")) is not None else None,
                    ingested_at=str(_null_if_na(row.get("ingested_at"))) if _null_if_na(row.get("ingested_at")) is not None else None,
                    hash=None,
                )
            )
        return refs

    def _rating_score(self, rating: Optional[str]) -> Optional[float]:
        if rating is None:
            return None
        r = str(rating).upper().strip()
        if not r:
            return None
        mapping = {
            "AAA": 1,
            "AA+": 2,
            "AA": 3,
            "AA-": 4,
            "A+": 5,
            "A": 6,
            "A-": 7,
            "BBB+": 8,
            "BBB": 9,
            "BBB-": 10,
            "BB+": 11,
            "BB": 12,
            "BB-": 13,
            "B+": 14,
            "B": 15,
            "B-": 16,
            "CCC+": 17,
            "CCC": 18,
            "CCC-": 19,
            "CC": 20,
            "C": 21,
            "D": 22,
        }
        for key, score in mapping.items():
            if key in r:
                return float(score)
        return None

    def _latest_fact(self, facts: pd.DataFrame, fact_types: List[str]) -> Tuple[Optional[float], Optional[Dict[str, Any]]]:
        if facts is None or facts.empty:
            return None, None
        df = facts[facts["fact_type"].isin(fact_types)].copy() if "fact_type" in facts.columns else pd.DataFrame()
        if df.empty:
            return None, None
        order_cols = [
            c
            for c in ["effective_at", "valid_from", "fact_time", "ingested_at", "published_at"]
            if c in df.columns
        ]
        if order_cols:
            df = self._to_datetimes(df, order_cols)
            df = df.sort_values(order_cols, ascending=[False] * len(order_cols), na_position="last")
        row = df.iloc[0]
        val = _safe_float(row.get("fact_value", None))
        if val is None:
            val = _safe_float(row.get("value", None))
        if val is None:
            val = _safe_float(row.get("numeric_value", None))
        prov = {
            "artifact_type": "ExtractedFact",
            "artifact_id": str(row.get("fact_id")) if row.get("fact_id") is not None else "",
            "source": str(row.get("source_type")) if row.get("source_type") is not None else None,
            "published_at": str(row.get("published_at")) if row.get("published_at") is not None else None,
            "ingested_at": str(row.get("ingested_at")) if row.get("ingested_at") is not None else None,
            "hash": None,
        }
        return val, prov

    def _fact_row_provenance(self, row: Optional[pd.Series]) -> Optional[Dict[str, Any]]:
        if row is None:
            return None
        artifact_id = _null_if_na(row.get("fact_id"))
        return {
            "artifact_type": "ExtractedFact",
            "artifact_id": str(artifact_id) if artifact_id is not None else "",
            "source": str(_null_if_na(row.get("source_type"))) if _null_if_na(row.get("source_type")) is not None else None,
            "published_at": str(_null_if_na(row.get("published_at"))) if _null_if_na(row.get("published_at")) is not None else None,
            "ingested_at": str(_null_if_na(row.get("ingested_at"))) if _null_if_na(row.get("ingested_at")) is not None else None,
            "hash": None,
        }

    def _dated_fact_series(
        self,
        facts: pd.DataFrame,
        fact_types: List[str],
        date_candidates: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, Optional[str]]:
        series = self._fact_series(facts, fact_types)
        if series.empty:
            return pd.DataFrame(), None
        series = self._select_latest_fact_revision(series)
        series = self._augment_fact_context_fields(series)
        candidates = date_candidates or ["period_end", "effective_at", "fact_time", "valid_from", "published_at"]
        date_col = _pick_first_col(series, candidates)
        if date_col is None:
            return series, None
        order_cols: List[str] = []
        for col in [date_col, "published_at", "ingested_at", "effective_at", "valid_from", "fact_time"]:
            if col in series.columns and col not in order_cols:
                order_cols.append(col)
        series = self._to_datetimes(series.copy(), order_cols)
        series = series.dropna(subset=[date_col])
        if not series.empty:
            series = series.sort_values(order_cols, ascending=[True] * len(order_cols), na_position="last")
        return series, date_col

    def _statement_metric_frame(
        self,
        facts: pd.DataFrame,
        fact_types: List[str],
    ) -> Tuple[pd.DataFrame, Optional[str]]:
        series, date_col = self._dated_fact_series(
            facts,
            fact_types,
            date_candidates=["period_end", "effective_at", "fact_time", "valid_from", "published_at"],
        )
        if date_col is None or series.empty:
            return pd.DataFrame(), None
        frame = series.copy()
        frame["period_date"] = pd.to_datetime(frame[date_col], utc=True, errors="coerce").dt.normalize()
        frame["fact_value"] = pd.to_numeric(frame.get("fact_value"), errors="coerce")
        if "fiscal_quarter" in frame.columns:
            frame["fiscal_quarter"] = pd.to_numeric(frame["fiscal_quarter"], errors="coerce")
        if "fiscal_year" in frame.columns:
            frame["fiscal_year"] = pd.to_numeric(frame["fiscal_year"], errors="coerce")
        frame = frame.dropna(subset=["period_date", "fact_value"]).sort_values("period_date")
        return frame, date_col

    def _latest_statement_interim_pair(
        self,
        frame: pd.DataFrame,
    ) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
        if frame.empty or "fiscal_quarter" not in frame.columns:
            return None, None
        interim = frame[frame["fiscal_quarter"].isin([1, 2, 3])].copy()
        if interim.empty:
            return None, None
        current_row = interim.sort_values("period_date").iloc[-1]
        quarter = _safe_float(current_row.get("fiscal_quarter"))
        if quarter is None:
            return current_row, None
        target_date = current_row["period_date"] - pd.Timedelta(days=365)
        candidates = interim[
            (interim["period_date"] <= current_row["period_date"] - pd.Timedelta(days=270))
            & (interim["period_date"] >= current_row["period_date"] - pd.Timedelta(days=460))
            & (interim["fiscal_quarter"] == current_row["fiscal_quarter"])
        ].copy()
        if candidates.empty:
            return current_row, None
        candidates["date_distance"] = (candidates["period_date"] - target_date).abs()
        prior_row = candidates.sort_values(["date_distance", "period_date"]).iloc[0]
        return current_row, prior_row

    def _normalize_interim_single_quarter_values(
        self,
        frame: pd.DataFrame,
    ) -> pd.DataFrame:
        if frame.empty or "fiscal_quarter" not in frame.columns:
            return frame
        normalized = frame.copy()
        normalized["single_quarter_value"] = pd.to_numeric(normalized.get("fact_value"), errors="coerce")
        normalized["single_quarter_basis"] = "as_reported"
        normalized["single_quarter_quality_flag"] = pd.Series([None] * len(normalized), dtype=object)
        interim = normalized[normalized["fiscal_quarter"].isin([1, 2, 3])].copy()
        if interim.empty:
            return normalized
        for idx, row in interim.sort_values(["fiscal_year", "fiscal_quarter", "period_date"]).iterrows():
            quarter = _safe_float(row.get("fiscal_quarter"))
            current_value = _safe_float(row.get("fact_value"))
            fiscal_year = _safe_float(row.get("fiscal_year"))
            if quarter in (None, 1) or current_value is None or fiscal_year is None:
                continue
            prior_candidates = interim[
                (interim["fiscal_year"] == fiscal_year)
                & (interim["fiscal_quarter"] == quarter - 1)
                & (interim["period_date"] < row.get("period_date"))
            ].copy()
            if prior_candidates.empty:
                normalized.at[idx, "single_quarter_basis"] = "as_reported_without_prior_quarter"
                normalized.at[idx, "single_quarter_quality_flag"] = "prior_quarter_missing_for_normalization"
                continue
            prior_row = prior_candidates.sort_values("period_date").iloc[-1]
            prior_value = _safe_float(prior_row.get("fact_value"))
            if prior_value is None:
                normalized.at[idx, "single_quarter_basis"] = "as_reported_without_prior_quarter"
                normalized.at[idx, "single_quarter_quality_flag"] = "prior_quarter_value_missing_for_normalization"
                continue
            # When the reported value jumps materially above the prior quarter,
            # treat it as a YTD cumulative amount and back into the standalone quarter.
            if current_value > prior_value * 1.15:
                normalized.at[idx, "single_quarter_value"] = current_value - prior_value
                normalized.at[idx, "single_quarter_basis"] = "derived_from_ytd_delta"
                normalized.at[idx, "single_quarter_quality_flag"] = "quarter_value_derived_from_ytd_delta"
            else:
                normalized.at[idx, "single_quarter_basis"] = "as_reported_quarter"
        return normalized

    def _latest_annual_statement_row(self, frame: pd.DataFrame) -> Optional[pd.Series]:
        if frame.empty:
            return None
        if "fiscal_quarter" not in frame.columns:
            return frame.sort_values("period_date").iloc[-1]
        annual = frame[~frame["fiscal_quarter"].isin([1, 2, 3])].copy()
        if annual.empty:
            return None
        return annual.sort_values("period_date").iloc[-1]

    def _latest_ttm_statement_value(
        self,
        facts: pd.DataFrame,
        fact_types: List[str],
    ) -> Tuple[Optional[float], List[Dict[str, Any]], Dict[str, Any], List[str]]:
        frame, _ = self._statement_metric_frame(facts, fact_types)
        if frame.empty:
            return None, [], {"formula": None}, ["statement_metric_unavailable"]

        direct_ttm = frame[frame["fact_type"].astype(str).str.endswith("_ttm")].copy()
        if not direct_ttm.empty:
            row = direct_ttm.sort_values("period_date").iloc[-1]
            return (
                _safe_float(row.get("fact_value")),
                [self._fact_row_provenance(row)] if self._fact_row_provenance(row) else [],
                {
                    "formula": "direct_ttm_fact",
                    "period_end": str(row.get("period_date")),
                    "fact_type": row.get("fact_type"),
                },
                [],
            )

        annual_row = self._latest_annual_statement_row(frame)
        current_row, prior_row = self._latest_statement_interim_pair(frame)

        if annual_row is not None and current_row is not None and prior_row is not None:
            current_quarter = _safe_float(current_row.get("fiscal_quarter"))
            annual_value = _safe_float(annual_row.get("fact_value"))
            current_value = _safe_float(current_row.get("fact_value"))
            prior_value = _safe_float(prior_row.get("fact_value"))
            if (
                current_quarter == 1
                and annual_value is not None
                and current_value is not None
                and prior_value is not None
                and annual_row.get("period_date") is not None
                and annual_row["period_date"] < current_row["period_date"]
            ):
                provenance = [
                    self._fact_row_provenance(row)
                    for row in [annual_row, current_row, prior_row]
                    if self._fact_row_provenance(row)
                ]
                return (
                    annual_value + current_value - prior_value,
                    provenance,
                    {
                        "formula": "latest_annual + current_q1 - prior_year_q1",
                        "annual_value": annual_value,
                        "annual_period": str(annual_row.get("period_date")),
                        "current_interim_value": current_value,
                        "current_interim_period": str(current_row.get("period_date")),
                        "prior_interim_value": prior_value,
                        "prior_interim_period": str(prior_row.get("period_date")),
                    },
                    [],
                )

        if annual_row is not None:
            annual_value = _safe_float(annual_row.get("fact_value"))
            return (
                annual_value,
                [self._fact_row_provenance(annual_row)] if self._fact_row_provenance(annual_row) else [],
                {
                    "formula": "latest_annual_statement_value",
                    "annual_value": annual_value,
                    "annual_period": str(annual_row.get("period_date")),
                },
                [],
            )

        return None, [], {"formula": None}, ["ttm_unavailable_without_annual_bridge"]

    def _extract_series_history(
        self,
        df: pd.DataFrame,
        *,
        exact_ids: Optional[List[str]] = None,
        contains_any: Optional[List[str]] = None,
        series_candidates: Optional[List[str]] = None,
        time_candidates: Optional[List[str]] = None,
        value_candidates: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=["obs_time", "value", "series_id"])
        series_col = _pick_first_populated_col(
            df,
            series_candidates or ["series_id", "field_name", "metric", "instrument_id"],
        )
        time_col = _pick_first_populated_col(
            df,
            time_candidates or ["observation_time", "event_time", "trade_date", "effective_at", "published_at", "available_time", "date"],
        )
        value_col = _pick_first_populated_col(
            df,
            value_candidates or ["value", "close", "adjusted_close", "fact_value", "numeric_value"],
        )
        if series_col is None or time_col is None or value_col is None:
            return pd.DataFrame(columns=["obs_time", "value", "series_id"])
        out = df[[series_col, time_col, value_col]].copy()
        series_values = out[series_col].astype(str)
        mask = pd.Series(True, index=out.index)
        if exact_ids:
            exact = {str(v).upper() for v in exact_ids if v}
            mask &= series_values.str.upper().isin(exact)
        if contains_any:
            contains_mask = pd.Series(False, index=out.index)
            lower = series_values.str.lower()
            for token in contains_any:
                contains_mask |= lower.str.contains(str(token).lower(), regex=False, na=False)
            mask &= contains_mask
        out = out[mask].copy()
        if out.empty:
            return pd.DataFrame(columns=["obs_time", "value", "series_id"])
        out["obs_time"] = pd.to_datetime(out[time_col], utc=True, errors="coerce")
        out["value"] = pd.to_numeric(out[value_col], errors="coerce")
        out["series_id"] = series_values.loc[out.index].astype(str)
        out = out.dropna(subset=["obs_time", "value"]).sort_values("obs_time")
        return out[["obs_time", "value", "series_id"]]

    def _periodic_series(self, history: pd.DataFrame, freq: str, periods: Optional[int] = None) -> pd.Series:
        if history is None or history.empty:
            return pd.Series(dtype=float)
        series = history.dropna(subset=["obs_time", "value"]).copy()
        if series.empty:
            return pd.Series(dtype=float)
        s = series.sort_values("obs_time").set_index("obs_time")["value"].astype(float)
        s = s[~s.index.duplicated(keep="last")]
        pandas_freq = {"M": "ME", "Q": "QE"}.get(freq, freq)
        s = s.resample(pandas_freq).last().dropna()
        if periods is not None:
            s = s.tail(periods)
        return s

    def _diluted_eps_history(self, facts: pd.DataFrame) -> pd.DataFrame:
        eps_df, _ = self._statement_metric_frame(facts, self.fact_map.get("diluted_eps", []))
        if eps_df.empty:
            return pd.DataFrame(columns=["obs_time", "value"])

        direct_ttm = eps_df[eps_df["fact_type"].astype(str).str.endswith("_ttm")].copy()
        annual = pd.DataFrame()
        if "fiscal_quarter" in eps_df.columns:
            annual = eps_df[~eps_df["fiscal_quarter"].isin([1, 2, 3])].copy()

        histories: List[pd.DataFrame] = []
        for candidate in [direct_ttm, annual]:
            if candidate is None or candidate.empty:
                continue
            hist = candidate.copy()
            hist["obs_time"] = pd.to_datetime(hist["period_date"], utc=True, errors="coerce")
            hist["value"] = pd.to_numeric(hist.get("fact_value"), errors="coerce")
            hist = hist.dropna(subset=["obs_time", "value"]).sort_values("obs_time")
            if not hist.empty:
                histories.append(hist[["obs_time", "value"]])

        if not histories:
            return pd.DataFrame(columns=["obs_time", "value"])

        merged = pd.concat(histories, ignore_index=True)
        merged = merged.sort_values("obs_time").drop_duplicates(subset=["obs_time"], keep="last")
        return merged[["obs_time", "value"]]

    def _peer_percentile_from_entity_table(
        self,
        company_id: str,
        metric_value: Optional[float],
        taxonomy: TaxonomyContext,
        candidate_columns: List[str],
        minimum_peers: int = 8,
    ) -> Tuple[Optional[float], Dict[str, Any], List[str]]:
        breakdown: Dict[str, Any] = {
            "minimum_peer_count": minimum_peers,
            "candidate_columns": candidate_columns,
        }
        flags: List[str] = []
        if metric_value in (None, 0):
            flags.append("company_metric_unavailable")
            return None, breakdown, flags
        entity_table = self._load_entity_table()
        if entity_table is None or entity_table.empty or "entity_id" not in entity_table.columns:
            flags.append("entity_table_unavailable")
            return None, breakdown, flags
        df = entity_table.copy()
        df["entity_id"] = df["entity_id"].astype(str)
        company_row = df[df["entity_id"] == str(company_id)]
        group_fields = [
            ("gics_sub_industry", taxonomy.subsector),
            ("subsector", taxonomy.subsector),
            ("gics_industry", taxonomy.subsector),
            ("industry", taxonomy.subsector),
            ("gics_sector", taxonomy.sector),
            ("sector", taxonomy.sector),
        ]
        chosen_group_col = None
        chosen_group_value = None
        peer_df = pd.DataFrame()
        for group_col, taxonomy_value in group_fields:
            if group_col not in df.columns:
                continue
            company_group_value = None
            if not company_row.empty:
                company_group_value = _null_if_na(company_row.iloc[0].get(group_col))
            group_value = company_group_value or taxonomy_value
            if group_value is None:
                continue
            candidate_df = df[df[group_col].astype(str) == str(group_value)].copy()
            if len(candidate_df) >= minimum_peers or chosen_group_col is None:
                chosen_group_col = group_col
                chosen_group_value = group_value
                peer_df = candidate_df
            if len(candidate_df) >= minimum_peers:
                break

        if chosen_group_col is None or peer_df.empty:
            flags.append("peer_group_unavailable")
            return None, breakdown, flags

        chosen_value_col = None
        for col in candidate_columns:
            if col in peer_df.columns:
                chosen_value_col = col
                break
        if chosen_value_col is None:
            flags.append("peer_metric_column_unavailable")
            breakdown.update({
                "peer_group_col": chosen_group_col,
                "peer_group_value": chosen_group_value,
            })
            return None, breakdown, flags

        values = pd.to_numeric(peer_df[chosen_value_col], errors="coerce")
        peer_df = peer_df.assign(_metric_value=values)
        peer_df = peer_df[peer_df["_metric_value"].notna()].copy()
        peer_df = peer_df[peer_df["_metric_value"] > 0]
        breakdown.update({
            "peer_group_col": chosen_group_col,
            "peer_group_value": chosen_group_value,
            "peer_metric_column": chosen_value_col,
            "peer_row_count": int(len(peer_df)),
        })
        if peer_df.empty:
            flags.append("peer_metric_values_unavailable")
            return None, breakdown, flags

        if str(company_id) not in peer_df["entity_id"].values:
            peer_df = pd.concat(
                [
                    peer_df[["entity_id", "_metric_value"]],
                    pd.DataFrame([{"entity_id": str(company_id), "_metric_value": float(metric_value)}]),
                ],
                ignore_index=True,
            )
        rank = peer_df["_metric_value"].rank(pct=True)
        company_match = peer_df.index[peer_df["entity_id"] == str(company_id)]
        if len(company_match) == 0:
            flags.append("company_not_in_peer_rank")
            return None, breakdown, flags
        percentile = float(rank.loc[company_match[-1]]) * 100.0
        breakdown["peer_percentile"] = percentile
        return percentile, breakdown, flags

    def _source_reliability_score(self, source: Optional[str]) -> float:
        if source is None:
            return 0.70
        s = str(source).strip().lower()
        if not s:
            return 0.70
        mapping = {
            "sec": 0.95,
            "sec_10q": 0.95,
            "sec_10k": 0.95,
            "wrds_13f": 0.90,
            "wrds": 0.90,
            "fisd": 0.88,
            "fisd_ratings": 0.88,
            "ciq": 0.86,
            "ciq_ratings": 0.86,
            "lseg": 0.90,
            "refinitiv": 0.90,
            "worldscope": 0.90,
            "event_store": 0.80,
            "market_data": 0.82,
            "fred": 0.90,
            "fmp": 0.75,
            "heuristic": 0.60,
            "entity_table": 0.80,
        }
        if s in mapping:
            return mapping[s]
        for key, val in mapping.items():
            if key in s:
                return val
        return 0.70

    def _recency_score(self, ts: Optional[pd.Timestamp], as_of: pd.Timestamp) -> float:
        if ts is None or pd.isna(ts):
            return 0.65
        age_days = max(0.0, float((as_of - pd.to_datetime(ts, utc=True)).days))
        if age_days <= 30:
            return 1.00
        if age_days <= 90:
            return 0.95
        if age_days <= 180:
            return 0.85
        if age_days <= 365:
            return 0.72
        if age_days <= 365 * 2:
            return 0.58
        return 0.45

    def _parse_certainty_score(self, raw: Optional[float]) -> float:
        if raw is None:
            return 0.82
        return float(np.clip(float(raw), 0.35, 1.0))

    def _combined_confidence(
        self,
        recency_score: float,
        source_reliability_score: float,
        proxy_penalty: float,
        parse_certainty: float,
    ) -> float:
        return float(
            np.clip(recency_score * source_reliability_score * proxy_penalty * parse_certainty, 0.0, 1.0)
        )

    def _confidence_from_row(self, row: Optional[pd.Series], as_of: pd.Timestamp, proxy_used: bool = False) -> Optional[float]:
        if row is None:
            return None
        dt_val = None
        for c in ["effective_at", "published_at", "ingested_at", "fact_time", "report_date", "rating_date"]:
            if c in row and row.get(c) is not None and not pd.isna(row.get(c)):
                dt_val = pd.to_datetime(row.get(c), utc=True, errors="coerce")
                if pd.notna(dt_val):
                    break
        recency = self._recency_score(dt_val, as_of)
        source_rel = self._source_reliability_score(_null_if_na(row.get("source_type")))
        parse = self._parse_certainty_score(_safe_float(row.get("confidence_score")))
        proxy_penalty = 0.65 if proxy_used else 1.0
        return round(self._combined_confidence(recency, source_rel, proxy_penalty, parse), 4)

    def _apply_confidence_framework(self, features: Dict[str, FeatureRecord], as_of: pd.Timestamp) -> None:
        for feat in features.values():
            if feat is None:
                continue
            if feat.missing_reason is not None:
                feat.confidence = None
                continue

            # recency from newest provenance timestamp available
            newest_ts = None
            for ref in feat.provenance or []:
                for c in ("published_at", "ingested_at"):
                    v = getattr(ref, c, None)
                    if v is None:
                        continue
                    ts = pd.to_datetime(v, utc=True, errors="coerce")
                    if pd.isna(ts):
                        continue
                    if newest_ts is None or ts > newest_ts:
                        newest_ts = ts
            recency = self._recency_score(newest_ts, as_of)

            # average source reliability for multi-input features
            src_scores: List[float] = []
            for ref in feat.provenance or []:
                src_scores.append(self._source_reliability_score(getattr(ref, "source", None)))
            source_rel = float(np.mean(src_scores)) if src_scores else 0.70

            proxy_penalty = 0.65 if feat.fallback_used not in (None, "", "none") else 1.0

            if feat.provenance:
                parse_certainty = 0.85
            else:
                parse_certainty = 0.75 if feat.fallback_used not in (None, "", "none") else 0.70
            feat.confidence = round(
                self._combined_confidence(recency, source_rel, proxy_penalty, parse_certainty),
                4,
            )

    def _fact_series(self, facts: pd.DataFrame, fact_types: List[str]) -> pd.DataFrame:
        if facts is None or facts.empty or "fact_type" not in facts.columns:
            return pd.DataFrame()
        return facts[facts["fact_type"].isin(fact_types)].copy()

    def _fact_pattern_series(
        self,
        facts: pd.DataFrame,
        pattern_spec: Optional[Dict[str, List[str]]],
    ) -> pd.DataFrame:
        if facts is None or facts.empty or "fact_type" not in facts.columns or not pattern_spec:
            return pd.DataFrame()
        df = facts.copy()
        fact_type_text = df["fact_type"].astype(str).str.lower()
        fact_type_norm = fact_type_text.str.replace(r"[^a-z0-9]+", "", regex=True)
        fact_id_text = (
            df["fact_id"].astype(str).str.lower()
            if "fact_id" in df.columns
            else pd.Series("", index=df.index, dtype="object")
        )
        fact_id_norm = fact_id_text.str.replace(r"[^a-z0-9]+", "", regex=True)

        def _normalized_token(token: str) -> str:
            return re.sub(r"[^a-z0-9]+", "", str(token).lower())

        contains_any = [str(token).lower() for token in pattern_spec.get("contains_any", []) if token]
        require_any = [str(token).lower() for token in pattern_spec.get("require_any", []) if token]
        exclude_any = [str(token).lower() for token in pattern_spec.get("exclude_any", []) if token]

        mask = pd.Series(True, index=df.index)
        if contains_any:
            contains_mask = pd.Series(False, index=df.index)
            for token in contains_any:
                norm_token = _normalized_token(token)
                contains_mask |= fact_type_text.str.contains(token, regex=False, na=False)
                if norm_token:
                    contains_mask |= fact_type_norm.str.contains(norm_token, regex=False, na=False)
                if "fact_id" in df.columns:
                    contains_mask |= fact_id_text.str.contains(token, regex=False, na=False)
                    if norm_token:
                        contains_mask |= fact_id_norm.str.contains(norm_token, regex=False, na=False)
            mask &= contains_mask
        if require_any:
            require_mask = pd.Series(False, index=df.index)
            for token in require_any:
                norm_token = _normalized_token(token)
                require_mask |= fact_type_text.str.contains(token, regex=False, na=False)
                if norm_token:
                    require_mask |= fact_type_norm.str.contains(norm_token, regex=False, na=False)
                if "fact_id" in df.columns:
                    require_mask |= fact_id_text.str.contains(token, regex=False, na=False)
                    if norm_token:
                        require_mask |= fact_id_norm.str.contains(norm_token, regex=False, na=False)
            mask &= require_mask
        for token in exclude_any:
            norm_token = _normalized_token(token)
            token_match = fact_type_text.str.contains(token, regex=False, na=False)
            if norm_token:
                token_match |= fact_type_norm.str.contains(norm_token, regex=False, na=False)
            if "fact_id" in df.columns:
                token_match |= fact_id_text.str.contains(token, regex=False, na=False)
                if norm_token:
                    token_match |= fact_id_norm.str.contains(norm_token, regex=False, na=False)
            mask &= ~token_match
        return df[mask].copy()

    def _latest_fact_with_patterns(
        self,
        facts: pd.DataFrame,
        fact_types: List[str],
        pattern_spec: Optional[Dict[str, List[str]]] = None,
    ) -> Tuple[Optional[float], Optional[Dict[str, Any]]]:
        value, prov = self._latest_fact(facts, fact_types)
        if value is not None:
            return value, prov
        df = self._fact_pattern_series(facts, pattern_spec)
        if df.empty:
            return None, None
        order_cols = [
            c
            for c in ["effective_at", "valid_from", "fact_time", "ingested_at", "published_at"]
            if c in df.columns
        ]
        if order_cols:
            df = self._to_datetimes(df, order_cols)
            df = df.sort_values(order_cols, ascending=[False] * len(order_cols), na_position="last")
        row = df.iloc[0]
        return _safe_float(row.get("fact_value", row.get("value", row.get("numeric_value")))), self._fact_row_provenance(row)

    def _maturity_bucket_label(self, row: Optional[pd.Series]) -> Optional[str]:
        if row is None:
            return None
        raw = _null_if_na(row.get("bucket_label")) if "bucket_label" in row.index else None
        if raw is not None:
            label = str(raw).strip()
            if label:
                return label
        ctx = str(_null_if_na(row.get("context_norm")) or "")
        match = re.search(r"(?:^|;\s*)bucket_label=([^;]+)", ctx, re.IGNORECASE)
        if match:
            label = match.group(1).strip()
            return label or None
        return None

    def _note_maturity_schedule(
        self,
        facts: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> Tuple[Optional[Dict[str, float]], List[InputReference], List[str]]:
        df = self._fact_series(facts, ["financial.debt_maturity_bucket"])
        if df.empty:
            df = self._fact_pattern_series(
                facts,
                {"contains_any": ["debt_maturity_bucket"], "exclude_any": ["lease_payment_due"]},
            )
        if df.empty:
            return None, [], []
        df = self._normalize_fact_columns(df)
        df = self._select_latest_fact_revision(df)
        order_cols = [
            c
            for c in ["effective_at", "valid_from", "fact_time", "published_at", "ingested_at"]
            if c in df.columns
        ]
        if order_cols:
            df = self._to_datetimes(df.copy(), order_cols)
            df = df.sort_values(order_cols, ascending=[False] * len(order_cols), na_position="last")
        if "bucket_label" not in df.columns:
            df["bucket_label"] = None
        df["bucket_label"] = df.apply(self._maturity_bucket_label, axis=1)
        df["fact_value"] = pd.to_numeric(df.get("fact_value"), errors="coerce")
        df = df.dropna(subset=["bucket_label", "fact_value"])
        if df.empty:
            return None, [], []
        df["bucket_label_norm"] = df["bucket_label"].astype(str).str.strip().str.lower()
        df = df.drop_duplicates(subset=["bucket_label_norm"], keep="first")

        schedule = {
            "due_0_12": 0.0,
            "due_12_24": 0.0,
            "due_24_36": 0.0,
            "due_36_60": 0.0,
            "due_60_plus": 0.0,
        }
        refs: List[InputReference] = []
        flags: List[str] = []
        matched = False
        as_of_utc = pd.to_datetime(as_of, utc=True)
        for _, row in df.iterrows():
            label = str(row.get("bucket_label") or "").strip()
            if not label:
                continue
            value = _safe_float(row.get("fact_value"))
            if value is None or value <= 0:
                continue
            bucket_key: Optional[str] = None
            lower_label = label.lower()
            if lower_label == "thereafter":
                bucket_key = "due_60_plus"
            elif re.fullmatch(r"20\d{2}", label):
                year = int(label)
                bucket_end = pd.Timestamp(year=year, month=12, day=31, tz="UTC")
                horizon_days = max(0, int((bucket_end - as_of_utc).days))
                if horizon_days <= 365:
                    bucket_key = "due_0_12"
                elif horizon_days <= 365 * 2:
                    bucket_key = "due_12_24"
                elif horizon_days <= 365 * 3:
                    bucket_key = "due_24_36"
                elif horizon_days <= 365 * 5:
                    bucket_key = "due_36_60"
                else:
                    bucket_key = "due_60_plus"
            if bucket_key is None:
                continue
            schedule[bucket_key] += float(value)
            matched = True
            prov = self._fact_row_provenance(row)
            if prov is not None:
                ref = InputReference(**prov)
                if ref not in refs:
                    refs.append(ref)
        if not matched:
            return None, [], []
        flags.append("maturity_schedule_note_extract")
        return schedule, refs[:5], flags

    def _structured_restricted_cash(
        self,
        facts: pd.DataFrame,
        *,
        companyfacts: Optional[Dict[str, Any]] = None,
        companyfacts_path: Optional[Path] = None,
        as_of: Optional[pd.Timestamp] = None,
    ) -> Tuple[Optional[float], List[InputReference]]:
        restricted_cash, prov_restricted = self._latest_fact_with_patterns(
            facts,
            self.fact_map["restricted_cash"],
            self.fact_pattern_map.get("restricted_cash"),
        )
        refs = [InputReference(**prov_restricted)] if prov_restricted else []
        if restricted_cash is not None:
            return restricted_cash, refs

        restricted_current, prov_current = self._latest_fact_with_patterns(
            facts,
            self.fact_map.get("restricted_cash_current", []),
            self.fact_pattern_map.get("restricted_cash_current"),
        )
        restricted_noncurrent, prov_noncurrent = self._latest_fact_with_patterns(
            facts,
            self.fact_map.get("restricted_cash_noncurrent", []),
            self.fact_pattern_map.get("restricted_cash_noncurrent"),
        )
        if restricted_current is None and restricted_noncurrent is None:
            if companyfacts is not None and companyfacts_path is not None and as_of is not None:
                restricted_cash_sec_value, restricted_cash_sec_meta = _extract_sec_companyfacts_restricted_cash(
                    companyfacts,
                    as_of.date().isoformat(),
                )
                if restricted_cash_sec_value is not None:
                    companyfacts_ref = _companyfacts_input_reference(companyfacts_path, restricted_cash_sec_meta)
                    refs = [InputReference(**companyfacts_ref)] if companyfacts_ref else []
                    return float(restricted_cash_sec_value), refs
            return None, []
        refs = [
            InputReference(**prov)
            for prov in [prov_current, prov_noncurrent]
            if prov is not None
        ]
        return (restricted_current or 0.0) + (restricted_noncurrent or 0.0), refs

    def _structured_marketable_securities(
        self,
        facts: pd.DataFrame,
        cash_val: Optional[float],
        cash_ref: Optional[InputReference],
        reference_row: Optional[Dict[str, Any]] = None,
        as_of: Optional[pd.Timestamp] = None,
        companyfacts: Optional[Dict[str, Any]] = None,
        companyfacts_path: Optional[Path] = None,
    ) -> Tuple[Optional[float], List[InputReference], List[str]]:
        marketable_securities, prov_marketable = self._latest_fact_with_patterns(
            facts,
            self.fact_map["marketable_securities"],
            self.fact_pattern_map.get("marketable_securities"),
        )
        refs = [InputReference(**prov_marketable)] if prov_marketable else []
        if marketable_securities is not None:
            return marketable_securities, refs, []

        combined_cash_and_investments, prov_combined = self._latest_fact_with_patterns(
            facts,
            self.fact_map.get("cash_and_short_term_investments", []),
            self.fact_pattern_map.get("cash_and_short_term_investments"),
        )
        if combined_cash_and_investments is not None and cash_val is not None:
            derived_marketable = max(0.0, float(combined_cash_and_investments) - float(cash_val))
            refs = [InputReference(**prov_combined)] if prov_combined else []
            if cash_ref is not None and cash_ref not in refs:
                refs.append(cash_ref)
            return derived_marketable, refs, []

        if companyfacts is not None and companyfacts_path is not None and as_of is not None:
            marketable_sec_value, marketable_sec_meta = _extract_sec_companyfacts_marketable_securities(
                companyfacts,
                as_of.date().isoformat(),
            )
            if marketable_sec_value is not None:
                companyfacts_ref = _companyfacts_input_reference(companyfacts_path, marketable_sec_meta)
                refs = [InputReference(**companyfacts_ref)] if companyfacts_ref else []
                return float(marketable_sec_value), refs, []

        provider_combined = _safe_float((reference_row or {}).get("Cash and Short Term Investments"))
        tolerance = max(1_000_000.0, abs(float(cash_val or 0.0)) * 0.02)
        if provider_combined is None or cash_val is None or provider_combined + tolerance < float(cash_val):
            return None, [], []

        derived_marketable = max(0.0, float(provider_combined) - float(cash_val))
        refs = []
        provider_ref = self._fundamentals_reference_input_ref(
            reference_row,
            "Cash and Short Term Investments",
            as_of,
        )
        if provider_ref is not None:
            refs.append(provider_ref)
        if cash_ref is not None and cash_ref not in refs:
            refs.append(cash_ref)
        return derived_marketable, refs, ["reference_cash_and_short_term_investments_fallback"]

    def _structured_revolver_undrawn(
        self,
        company_id: str,
        facts: pd.DataFrame,
        as_of: pd.Timestamp,
        aliases: Optional[List[str]] = None,
        companyfacts: Optional[Dict[str, Any]] = None,
        companyfacts_path: Optional[Path] = None,
    ) -> Tuple[Optional[float], List[InputReference], List[str]]:
        revolver_val, rev_prov = self._latest_fact_with_patterns(
            facts,
            self.fact_map["revolver_undrawn"],
            self.fact_pattern_map.get("revolver_undrawn"),
        )
        refs = [InputReference(**rev_prov)] if rev_prov else []
        if revolver_val is not None:
            return revolver_val, refs, []
        if companyfacts is not None and companyfacts_path is not None:
            revolver_sec_value, revolver_sec_meta = _extract_sec_companyfacts_revolver_undrawn(
                companyfacts,
                as_of.date().isoformat(),
            )
            if revolver_sec_value is not None:
                companyfacts_ref = _companyfacts_input_reference(companyfacts_path, revolver_sec_meta)
                refs = [InputReference(**companyfacts_ref)] if companyfacts_ref else []
                return float(revolver_sec_value), refs, []
        dealscan_val, dealscan_refs, dealscan_flags = self._dealscan_revolver_capacity(
            company_id,
            as_of,
            aliases=aliases,
        )
        if dealscan_val is not None:
            return dealscan_val, dealscan_refs, dealscan_flags
        return revolver_val, refs, []

    def _load_dealscan_revolver_facilities(self) -> pd.DataFrame:
        if self._dealscan_revolver_cache is not None:
            return self._dealscan_revolver_cache
        if str(os.environ.get("AXIOM_SKIP_DEALSCAN", "")).strip().lower() in {"1", "true", "yes", "on"}:
            if self.debug:
                print("[dealscan] skipped via AXIOM_SKIP_DEALSCAN")
            self._dealscan_revolver_cache = pd.DataFrame()
            return self._dealscan_revolver_cache
        if not self.dealscan_revolver_path.exists():
            self._dealscan_revolver_cache = pd.DataFrame()
            return self._dealscan_revolver_cache
        try:
            df = pd.read_parquet(self.dealscan_revolver_path)
        except Exception:
            self._dealscan_revolver_cache = pd.DataFrame()
            return self._dealscan_revolver_cache
        for col in ("ticker", "borrower_name_norm", "parent_norm", "company_name_norm"):
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str).str.upper().str.strip()
        for col in ("tranche_active_date", "tranche_maturity_date"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        self._dealscan_revolver_cache = df
        return self._dealscan_revolver_cache

    def _matched_active_dealscan_revolver_candidates(
        self,
        company_id: str,
        as_of: pd.Timestamp,
        aliases: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        df = self._load_dealscan_revolver_facilities()
        if df is None or df.empty:
            return pd.DataFrame()
        _, resolved_aliases = self._resolve_entity_aliases(str(company_id), extra_aliases=aliases)
        alias_set = {str(alias).strip().upper() for alias in resolved_aliases if alias is not None and str(alias).strip()}
        if not alias_set:
            return pd.DataFrame()
        name_like_aliases = {alias for alias in alias_set if any(ch.isalpha() for ch in alias) and (len(alias) >= 8 or " " in alias)}

        matched = pd.Series(False, index=df.index)
        ticker_match = pd.Series(False, index=df.index)
        if "ticker" in df.columns:
            ticker_match = df["ticker"].isin(alias_set)
            matched = matched | ticker_match
        if "borrower_name_norm" in df.columns:
            matched = matched | df["borrower_name_norm"].isin(alias_set)
        if "parent_norm" in df.columns:
            matched = matched | df["parent_norm"].isin(alias_set)
        if "company_name_norm" in df.columns:
            matched = matched | df["company_name_norm"].isin(alias_set)
        if "loanconnector_company_id" in df.columns:
            matched = matched | df["loanconnector_company_id"].astype(str).isin(alias_set)

        candidates = df[matched].copy()
        if candidates.empty:
            return pd.DataFrame()

        if name_like_aliases:
            name_mask = pd.Series(False, index=candidates.index)
            for col in ("borrower_name_norm", "parent_norm", "company_name_norm"):
                if col in candidates.columns:
                    name_mask = name_mask | candidates[col].isin(name_like_aliases)
            if name_mask.any():
                candidates = candidates[name_mask].copy()
            elif ticker_match.any():
                return pd.DataFrame()
        elif ticker_match.any():
            # Some international borrowers reuse U.S. tickers. If ticker-only matching
            # surfaces multiple unrelated names, prefer no fallback over a wrong one.
            name_values = set()
            ticker_candidates = df[ticker_match].copy()
            for col in ("borrower_name_norm", "parent_norm", "company_name_norm"):
                if col in ticker_candidates.columns:
                    name_values.update(
                        {
                            str(val).strip()
                            for val in ticker_candidates[col].dropna().astype(str)
                            if str(val).strip()
                        }
                    )
            if len(name_values) > 1:
                return pd.DataFrame()

        as_of_ts = pd.Timestamp(as_of)
        if as_of_ts.tzinfo is not None:
            as_of_ts = as_of_ts.tz_convert("UTC").tz_localize(None)
        else:
            as_of_ts = as_of_ts.tz_localize(None)
        if "tranche_active_date" in candidates.columns:
            candidates = candidates[candidates["tranche_active_date"].isna() | (candidates["tranche_active_date"] <= as_of_ts)]
        if "tranche_maturity_date" in candidates.columns:
            candidates = candidates[
                candidates["tranche_maturity_date"].isna() | (candidates["tranche_maturity_date"] >= as_of_ts)
            ]
        if candidates.empty:
            return pd.DataFrame()

        sort_cols = [col for col in ("tranche_active_date", "loanconnector_tranche_id") if col in candidates.columns]
        if sort_cols:
            candidates = candidates.sort_values(sort_cols)
        if "loanconnector_tranche_id" in candidates.columns:
            candidates = candidates.drop_duplicates(subset=["loanconnector_tranche_id"], keep="last")
        return candidates.copy()

    def _dealscan_candidate_refs(self, candidates: pd.DataFrame, *, limit: int = 10) -> List[InputReference]:
        refs: List[InputReference] = []
        if candidates is None or candidates.empty:
            return refs
        for _, row in candidates.head(limit).iterrows():
            artifact_id = (
                _null_if_na(row.get("wrds_facility_id"))
                or _null_if_na(row.get("loanconnector_tranche_id"))
                or _null_if_na(row.get("lpc_tranche_id"))
            )
            published_at = None
            tranche_active = row.get("tranche_active_date")
            if tranche_active is not None and not pd.isna(tranche_active):
                published_at = pd.Timestamp(tranche_active).isoformat()
            refs.append(
                InputReference(
                    artifact_type="wrds_dealscan_tranche",
                    artifact_id=str(artifact_id),
                    source="wrds_dealscan",
                    published_at=published_at,
                    ingested_at=None,
                    hash=None,
                )
            )
        return refs

    def _dealscan_revolver_capacity(
        self,
        company_id: str,
        as_of: pd.Timestamp,
        aliases: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], List[InputReference], List[str]]:
        candidates = self._matched_active_dealscan_revolver_candidates(
            company_id,
            as_of,
            aliases=aliases,
        )
        if candidates.empty:
            return None, [], []

        if "tranche_amount_converted_usd" not in candidates.columns:
            return None, [], []
        candidates = candidates[candidates["tranche_amount_converted_usd"].notna()]
        if candidates.empty:
            return None, [], []

        revolver_capacity = float(candidates["tranche_amount_converted_usd"].sum())
        refs = self._dealscan_candidate_refs(candidates)
        return revolver_capacity, refs, ["dealscan_revolver_capacity_proxy", "dealscan_undrawn_balance_missing"]

    def _dealscan_covenant_proxy_context(
        self,
        company_id: str,
        as_of: pd.Timestamp,
        aliases: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, Optional[float]], List[InputReference], Dict[str, Any], List[str]]:
        candidates = self._matched_active_dealscan_revolver_candidates(
            company_id,
            as_of,
            aliases=aliases,
        )
        if candidates.empty:
            return {}, [], {}, []

        metric_columns = {
            "max_leverage_ratio_covenant": ("max_leverage_ratio", "min"),
            "min_interest_coverage_ratio_covenant": ("min_interest_coverage_ratio", "max"),
            "min_fixed_charge_coverage_ratio_covenant": ("min_fixed_charge_coverage_ratio", "max"),
            "min_current_ratio_covenant": ("min_current_ratio", "max"),
        }
        selected: Dict[str, Optional[float]] = {}
        raw_thresholds: Dict[str, List[str]] = {}
        parsed_thresholds: Dict[str, List[float]] = {}
        selection_rules: Dict[str, str] = {}

        for metric_name, (column_name, reducer) in metric_columns.items():
            raw_values: List[str] = []
            numeric_values: List[float] = []
            if column_name in candidates.columns:
                for raw in candidates[column_name].tolist():
                    raw = _null_if_na(raw)
                    if raw not in (None, ""):
                        raw_values.append(str(raw))
                    parsed = _parse_dealscan_ratio_value(raw)
                    if parsed is not None:
                        numeric_values.append(float(parsed))
            if numeric_values:
                selected[metric_name] = min(numeric_values) if reducer == "min" else max(numeric_values)
            else:
                selected[metric_name] = None
            raw_thresholds[metric_name] = sorted(set(raw_values))[:10]
            parsed_thresholds[metric_name] = sorted(set(round(val, 6) for val in numeric_values))
            selection_rules[metric_name] = (
                "minimum_observed_threshold_across_active_revolver_facilities"
                if reducer == "min"
                else "maximum_observed_threshold_across_active_revolver_facilities"
            )

        if not any(value is not None for value in selected.values()):
            return {}, [], {}, []

        breakdown = {
            "active_revolver_facility_count": int(len(candidates)),
            "matched_tickers": sorted(
                {
                    str(val).strip()
                    for val in candidates.get("ticker", pd.Series(dtype=str)).dropna().astype(str)
                    if str(val).strip()
                }
            )[:10],
            "matched_borrower_names": sorted(
                {
                    str(val).strip()
                    for col in ("borrower_name_norm", "parent_norm", "company_name_norm")
                    if col in candidates.columns
                    for val in candidates[col].dropna().astype(str)
                    if str(val).strip()
                }
            )[:10],
            "loanconnector_tranche_ids": [
                str(val)
                for val in candidates.get("loanconnector_tranche_id", pd.Series(dtype=str)).dropna().astype(str).head(10).tolist()
            ],
            "observed_threshold_text": raw_thresholds,
            "observed_threshold_values": parsed_thresholds,
            "selection_rules": selection_rules,
            "all_covenants_financial_samples": sorted(
                {
                    str(val).strip()
                    for val in candidates.get("all_covenants_financial", pd.Series(dtype=str)).dropna().astype(str)
                    if str(val).strip()
                }
            )[:5],
        }
        flags = ["dealscan_covenant_proxy"]
        if len(candidates) > 1:
            flags.append("dealscan_multiple_facilities_aggregated")
        return selected, self._dealscan_candidate_refs(candidates), breakdown, flags

    # ---------------------------
    # Feature groups
    # ---------------------------

    def _lease_adjusted_metrics_required(self, taxonomy: TaxonomyContext) -> bool:
        explicit_rule = self.metric_policy.archetype_rules(taxonomy.archetype).get("lease_adjusted_metrics")
        if explicit_rule is not None:
            return bool(explicit_rule)
        return self.metric_policy.resolve_applicability("capital_structure.fixed_charge_coverage", taxonomy) == "primary"

    def _estimated_lease_expense(
        self,
        lease_expense: Optional[float],
        lease_liabilities: Optional[float],
        taxonomy: TaxonomyContext,
    ) -> tuple[Optional[float], List[str]]:
        flags: List[str] = []
        if lease_expense is not None:
            return float(lease_expense), flags
        if lease_liabilities in (None, 0):
            return None, flags
        if not self._lease_adjusted_metrics_required(taxonomy):
            return None, flags
        estimated = float(lease_liabilities) * 0.08
        flags.append("lease_expense_estimated_from_liabilities")
        return estimated, flags

    def _readily_available_cash_components(
        self,
        facts: pd.DataFrame,
        taxonomy: TaxonomyContext,
        reference_row: Optional[Dict[str, Any]] = None,
        as_of: Optional[pd.Timestamp] = None,
        cash_override: Optional[float] = None,
        cash_ref_override: Optional[InputReference] = None,
        companyfacts: Optional[Dict[str, Any]] = None,
        companyfacts_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        if cash_override is not None or cash_ref_override is not None:
            cash_val = cash_override
            cash_ref = cash_ref_override
            prov_cash = None
        else:
            cash_val, prov_cash = self._latest_fact(facts, self.fact_map["cash"])
            cash_ref = InputReference(**prov_cash) if prov_cash else None
        restricted_cash, restricted_refs = self._structured_restricted_cash(
            facts,
            companyfacts=companyfacts,
            companyfacts_path=companyfacts_path,
            as_of=as_of,
        )
        marketable_securities, marketable_refs, marketable_flags = self._structured_marketable_securities(
            facts,
            cash_val,
            cash_ref,
            reference_row=reference_row,
            as_of=as_of,
            companyfacts=companyfacts,
            companyfacts_path=companyfacts_path,
        )
        unavailable_cash, prov_unavailable = self._latest_fact(facts, self.fact_map["unavailable_cash"])

        refs = [cash_ref] if cash_ref is not None else []
        refs.extend([ref for ref in restricted_refs if ref not in refs])
        refs.extend([ref for ref in marketable_refs if ref not in refs])
        if prov_unavailable:
            unavailable_ref = InputReference(**prov_unavailable)
            if unavailable_ref not in refs:
                refs.append(unavailable_ref)
        flags: List[str] = []
        flags.extend(marketable_flags)

        readily_available_cash = None
        if cash_val is not None:
            readily_available_cash = float(cash_val)
            if marketable_securities is not None:
                readily_available_cash += float(marketable_securities)
                flags.append("marketable_securities_included_at_par_proxy")
            if restricted_cash is not None:
                readily_available_cash = max(0.0, readily_available_cash - float(restricted_cash))
            else:
                flags.append("restricted_cash_missing_assumed_zero")
            if unavailable_cash is not None:
                readily_available_cash = max(0.0, readily_available_cash - float(unavailable_cash))

        support_mode = taxonomy.support_mode if readily_available_cash is not None else "unsupported"
        if readily_available_cash is not None:
            support_mode = _classify_metric_support_mode(
                base_mode=support_mode,
                metric_id="liquidity.usable_cash",
                value=readily_available_cash,
                quality_flags=flags,
                component_breakdown={
                    "cash": cash_val,
                    "restricted_cash": restricted_cash,
                    "marketable_securities": marketable_securities,
                    "unavailable_cash": unavailable_cash,
                },
            )

        return {
            "cash": cash_val,
            "restricted_cash": restricted_cash,
            "marketable_securities": marketable_securities,
            "unavailable_cash": unavailable_cash,
            "readily_available_cash": readily_available_cash,
            "refs": refs,
            "quality_flags": flags,
            "support_mode": support_mode,
        }

    def _compute_liquidity(
        self,
        company_id: str,
        facts: pd.DataFrame,
        as_of: pd.Timestamp,
        taxonomy: TaxonomyContext,
    ) -> Dict[str, FeatureRecord]:
        features: Dict[str, FeatureRecord] = {}
        archetype_rules = self.metric_policy.archetype_rules(taxonomy.archetype)
        _, market_aliases = self._resolve_entity_aliases(str(company_id))
        reference_row = self._fundamentals_reference_row(market_aliases)
        dealscan_aliases = market_aliases + self._reference_aliases(reference_row)
        companyfacts, companyfacts_path = self._load_first_companyfacts_bundle(company_id, market_aliases)

        cash_val, prov_cash = self._latest_fact(facts, self.fact_map["cash"])
        cash_breakdown = None
        cash_fallback_used = None
        cash_quality_flags = None
        if companyfacts is not None and companyfacts_path is not None:
            companyfacts_cash_val, companyfacts_cash_meta = _latest_companyfacts_point_value(
                companyfacts,
                COMPANYFACTS_CASH_EQ_CONCEPTS,
                as_of,
            )
            if _should_use_fresher_companyfacts_value(
                cash_val,
                (prov_cash or {}).get("published_at") if prov_cash else None,
                companyfacts_cash_val,
                companyfacts_cash_meta,
            ):
                cash_val = companyfacts_cash_val
                prov_cash = _companyfacts_input_reference(companyfacts_path, companyfacts_cash_meta)
                cash_breakdown = companyfacts_cash_meta
                cash_fallback_used = "companyfacts_cash_exact_fresher"
                cash_quality_flags = ["companyfacts_cash_fresher"]
        cash_row = None
        if cash_fallback_used is None and cash_val is not None and not facts.empty:
            cash_series = self._fact_series(facts, self.fact_map["cash"])
            if not cash_series.empty:
                cash_row = cash_series.sort_values("published_at", ascending=False).iloc[0]
        cash_conf = self._confidence_from_row(cash_row, as_of) if cash_row is not None else None
        cash_refs = [InputReference(**prov_cash)] if prov_cash else []
        features["liquidity.cash"] = FeatureRecord(
            name="liquidity.cash",
            value=cash_val,
            unit="usd",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window=None,
            confidence=cash_conf,
            provenance=cash_refs,
            missing_reason="unavailable" if cash_val is None else None,
            fallback_used=cash_fallback_used,
            support_mode="exact" if cash_val is not None else "unsupported",
            component_breakdown=cash_breakdown,
            quality_flags=cash_quality_flags,
        )

        restricted_cash, restricted_refs = self._structured_restricted_cash(
            facts,
            companyfacts=companyfacts,
            companyfacts_path=companyfacts_path,
            as_of=as_of,
        )
        features["liquidity.restricted_cash"] = FeatureRecord(
            name="liquidity.restricted_cash",
            value=restricted_cash,
            unit="usd",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window=None,
            confidence=None,
            provenance=restricted_refs,
            missing_reason="not_disclosed" if restricted_cash is None else None,
            fallback_used=None,
        )

        marketable_securities, marketable_refs, marketable_flags = self._structured_marketable_securities(
            facts,
            cash_val=cash_val,
            cash_ref=InputReference(**prov_cash) if prov_cash else None,
            reference_row=reference_row,
            as_of=as_of,
            companyfacts=companyfacts,
            companyfacts_path=companyfacts_path,
        )
        features["liquidity.marketable_securities"] = FeatureRecord(
            name="liquidity.marketable_securities",
            value=marketable_securities,
            unit="usd",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window=None,
            confidence=None,
            provenance=marketable_refs,
            missing_reason="not_disclosed" if marketable_securities is None else None,
            fallback_used="reference_cash_and_short_term_investments" if marketable_flags else None,
            quality_flags=marketable_flags or None,
        )

        revolver_val, revolver_refs, revolver_flags = self._structured_revolver_undrawn(
            company_id,
            facts,
            as_of,
            aliases=dealscan_aliases,
            companyfacts=companyfacts,
            companyfacts_path=companyfacts_path,
        )
        features["liquidity.revolver_undrawn"] = FeatureRecord(
            name="liquidity.revolver_undrawn",
            value=revolver_val,
            unit="usd",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window=None,
            confidence=0.4 if revolver_flags else (0.6 if revolver_val is not None else None),
            provenance=revolver_refs,
            missing_reason="not_disclosed" if revolver_val is None else None,
            fallback_used="dealscan_revolver_capacity" if revolver_flags else None,
            quality_flags=revolver_flags or None,
        )

        liq_total = cash_val if cash_val is not None else None
        if marketable_securities is not None:
            liq_total = (liq_total or 0.0) + marketable_securities
        if revolver_val is not None:
            liq_total = (liq_total or 0.0) + revolver_val
        features["liquidity.liquidity_total"] = FeatureRecord(
            name="liquidity.liquidity_total",
            value=liq_total,
            unit="usd",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window=None,
            confidence=cash_conf,
            provenance=cash_refs + [ref for ref in marketable_refs if ref not in cash_refs] + revolver_refs,
            missing_reason="unavailable" if liq_total is None else None,
            fallback_used=None,
        )

        revenue_val, prov_revenue = self._latest_fact(facts, self.fact_map["revenue"])
        min_cash_ratio = _to_float(archetype_rules.get("min_cash_revenue_ratio"), 0.03) or 0.03
        min_cash_proxy = None
        if cash_val is not None or revenue_val is not None:
            proxy_rev = min_cash_ratio * revenue_val if revenue_val is not None else 0.0
            min_cash_proxy = max(proxy_rev, 0.0)
        features["liquidity.minimum_cash_policy_proxy"] = FeatureRecord(
            name="liquidity.minimum_cash_policy_proxy",
            value=min_cash_proxy,
            unit="usd",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window={"type": "heuristic", "length_days": 0},
            confidence=0.4 if min_cash_proxy is not None else None,
            provenance=[InputReference(**prov_revenue)] if prov_revenue else [],
            missing_reason="unavailable" if min_cash_proxy is None else None,
            fallback_used="heuristic",
        )

        readily_available_cash = self._readily_available_cash_components(
            facts,
            taxonomy,
            reference_row=reference_row,
            as_of=as_of,
            cash_override=cash_val,
            cash_ref_override=InputReference(**prov_cash) if prov_cash else None,
            companyfacts=companyfacts,
            companyfacts_path=companyfacts_path,
        )
        usable_cash_reported = cash_val
        usable_cash_market = readily_available_cash["readily_available_cash"]
        usable_cash_flags: List[str] = list(readily_available_cash["quality_flags"] or [])
        usable_cash_breakdown = {
            "cash": cash_val,
            "restricted_cash": restricted_cash,
            "marketable_securities": marketable_securities,
            "unavailable_cash": readily_available_cash["unavailable_cash"],
        }
        combined_cash_refs = list(cash_refs)
        for ref in restricted_refs + marketable_refs + list(readily_available_cash["refs"] or []):
            if ref not in combined_cash_refs:
                combined_cash_refs.append(ref)
        self._emit_metric_views(
            features,
            metric_id="liquidity.usable_cash",
            base_name="liquidity.usable_cash",
            reported_value=usable_cash_reported,
            market_value=usable_cash_market,
            unit="usd",
            as_of=as_of,
            window=None,
            taxonomy=taxonomy,
            reported_provenance=cash_refs,
            market_provenance=combined_cash_refs,
            reported_missing_reason="unavailable" if usable_cash_reported is None else None,
            market_missing_reason="unavailable" if usable_cash_market is None else None,
            fallback_used="heuristic" if usable_cash_flags and usable_cash_market is not None else None,
            component_breakdown=usable_cash_breakdown,
            quality_flags=usable_cash_flags,
            decision_fallback_to_reported=True,
        )

        available_cash_reported = None
        if cash_val is not None and min_cash_proxy is not None:
            available_cash_reported = max(0.0, cash_val - min_cash_proxy)
        available_cash_market = usable_cash_market

        available_liquidity_reported = available_cash_reported
        if available_liquidity_reported is None and revolver_val is not None:
            available_liquidity_reported = revolver_val
        elif available_liquidity_reported is not None and revolver_val is not None:
            available_liquidity_reported = available_liquidity_reported + revolver_val

        available_liquidity_market = available_cash_market
        if available_liquidity_market is None and revolver_val is not None:
            available_liquidity_market = revolver_val
        elif available_liquidity_market is not None and revolver_val is not None:
            available_liquidity_market = available_liquidity_market + revolver_val

        available_breakdown = {
            "usable_cash_market": usable_cash_market,
            "revolver_undrawn": revolver_val,
            "minimum_cash_policy_proxy": min_cash_proxy,
        }
        available_flags = list(usable_cash_flags)
        if min_cash_proxy is not None:
            available_flags.append("minimum_cash_policy_proxy_not_applied_to_market_view")
        self._emit_metric_views(
            features,
            metric_id="liquidity.available_for_actions",
            base_name="liquidity.available_for_actions",
            reported_value=available_liquidity_reported,
            market_value=available_liquidity_market,
            unit="usd",
            as_of=as_of,
            window=None,
            taxonomy=taxonomy,
            reported_provenance=cash_refs + revolver_refs,
            market_provenance=combined_cash_refs + [ref for ref in revolver_refs if ref not in combined_cash_refs],
            reported_missing_reason="unavailable" if available_liquidity_reported is None else None,
            market_missing_reason="unavailable" if available_liquidity_market is None else None,
            fallback_used="heuristic" if available_flags and available_liquidity_market is not None else None,
            component_breakdown=available_breakdown,
            quality_flags=available_flags,
            decision_fallback_to_reported=True,
        )

        fcf_val, prov_fcf = self._latest_fact(facts, self.fact_map["fcf"])
        fcf_refs = [InputReference(**prov_fcf)] if prov_fcf else []
        runway_reported = None
        runway_market = None
        runway_missing_reason_reported = None
        runway_missing_reason_market = None
        if fcf_val is None or available_liquidity_reported is None:
            runway_missing_reason_reported = "unavailable"
        elif fcf_val > 0:
            runway_reported = 60.0
        elif fcf_val != 0:
            runway_reported = available_liquidity_reported / (abs(fcf_val) / 12.0)

        if fcf_val is None or available_liquidity_market is None:
            runway_missing_reason_market = "unavailable"
        elif fcf_val > 0:
            runway_market = 60.0
        elif fcf_val != 0:
            runway_market = available_liquidity_market / (abs(fcf_val) / 12.0)

        runway_breakdown = {
            "available_for_actions_market": available_liquidity_market,
            "free_cash_flow": fcf_val,
        }
        runway_flags = []
        if taxonomy.archetype == "financial_institution":
            runway_flags.append("runway_not_primary_for_financials")
        self._emit_metric_views(
            features,
            metric_id="liquidity.runway_months",
            base_name="liquidity.runway_months",
            reported_value=runway_reported,
            market_value=runway_market,
            unit="months",
            as_of=as_of,
            window={"type": "ttm", "length_days": 365},
            taxonomy=taxonomy,
            reported_provenance=fcf_refs + cash_refs,
            market_provenance=fcf_refs + combined_cash_refs + revolver_refs,
            reported_missing_reason=runway_missing_reason_reported,
            market_missing_reason=runway_missing_reason_market,
            fallback_used=None,
            component_breakdown=runway_breakdown,
            quality_flags=runway_flags,
            decision_fallback_to_reported=True,
        )
        return features

    def _compute_capital_structure(
        self,
        company_id: str,
        facts: pd.DataFrame,
        events: pd.DataFrame,
        issuer_ratings: pd.DataFrame,
        as_of: pd.Timestamp,
        taxonomy: TaxonomyContext,
    ) -> Dict[str, FeatureRecord]:
        features: Dict[str, FeatureRecord] = {}
        archetype_rules = self.metric_policy.archetype_rules(taxonomy.archetype)
        _, market_aliases = self._resolve_entity_aliases(str(company_id))
        reference_row = self._fundamentals_reference_row(market_aliases)
        dealscan_aliases = market_aliases + self._reference_aliases(reference_row)
        reference_instrument = _null_if_na(reference_row.get("Instrument"))
        provider_total_debt = _safe_float(reference_row.get("Total Debt"))
        provider_ebitda = _safe_float(reference_row.get("EBITDA"))
        companyfacts, companyfacts_path = self._load_first_companyfacts_bundle(company_id, market_aliases)

        total_debt_reported, prov_td = self._latest_fact(facts, self.fact_map["total_debt"])
        if total_debt_reported is None:
            debt_current, prov_dc = self._latest_fact(facts, self.fact_map["debt_current"])
            debt_long, prov_dl = self._latest_fact(facts, self.fact_map["debt_long"])
            if debt_current is not None or debt_long is not None:
                total_debt_reported = (debt_current or 0.0) + (debt_long or 0.0)
                prov_td = prov_dc or prov_dl
        local_total_debt_reported = total_debt_reported
        companyfacts_total_debt_context = None
        if companyfacts is not None and companyfacts_path is not None:
            (
                companyfacts_total_debt_value,
                companyfacts_total_debt_mode,
                _companyfacts_total_debt_missing_reason,
                companyfacts_total_debt_breakdown,
                _companyfacts_total_debt_flags,
            ) = _build_sec_core_metric(
                "capital_structure.total_debt_provider_direct",
                companyfacts,
                as_of.date().isoformat(),
            )
            if (
                _support_mode_is_exact_like(companyfacts_total_debt_mode)
                and _should_use_fresher_companyfacts_value(
                    total_debt_reported,
                    (prov_td or {}).get("published_at") if prov_td else None,
                    companyfacts_total_debt_value,
                    companyfacts_total_debt_breakdown,
                )
            ):
                total_debt_reported = float(companyfacts_total_debt_value)
                prov_td = _companyfacts_input_reference(companyfacts_path, companyfacts_total_debt_breakdown)
                companyfacts_total_debt_context = companyfacts_total_debt_breakdown
        total_debt_series, total_debt_date_col = self._dated_fact_series(facts, self.fact_map["total_debt"])
        recent_total_debt_peak = None
        if not total_debt_series.empty:
            recent_total_debt_series = total_debt_series.copy()
            if total_debt_date_col and total_debt_date_col in recent_total_debt_series.columns:
                recent_total_debt_series[total_debt_date_col] = pd.to_datetime(
                    recent_total_debt_series[total_debt_date_col], utc=True, errors="coerce"
                )
                recent_total_debt_series = recent_total_debt_series[
                    recent_total_debt_series[total_debt_date_col] >= (as_of - pd.Timedelta(days=550))
                ]
            if not recent_total_debt_series.empty:
                recent_total_debt_peak = _safe_float(recent_total_debt_series["fact_value"].max())
        debt_reference_candidate = None
        debt_reference_source = None
        for source_name, source_value in [
            ("recent_total_debt_peak", recent_total_debt_peak),
            ("reference_total_debt", provider_total_debt),
        ]:
            if source_value is None:
                continue
            if debt_reference_candidate is None or source_value > debt_reference_candidate:
                debt_reference_candidate = float(source_value)
                debt_reference_source = source_name
        debt_reference_fallback_used = False
        if total_debt_reported is None and debt_reference_candidate is not None:
            total_debt_reported = debt_reference_candidate
            debt_reference_fallback_used = True
        elif (
            total_debt_reported is not None
            and debt_reference_candidate is not None
            and float(total_debt_reported) > 0
            and float(debt_reference_candidate) / float(total_debt_reported) >= 1.25
        ):
            total_debt_reported = debt_reference_candidate
            debt_reference_fallback_used = True
        debt_refs = [InputReference(**prov_td)] if prov_td else []

        cash_val, prov_cash = self._latest_fact(facts, self.fact_map["cash"])
        companyfacts_cash_context = None
        if companyfacts is not None and companyfacts_path is not None:
            companyfacts_cash_val, companyfacts_cash_meta = _latest_companyfacts_point_value(
                companyfacts,
                COMPANYFACTS_CASH_EQ_CONCEPTS,
                as_of,
            )
            if _should_use_fresher_companyfacts_value(
                cash_val,
                (prov_cash or {}).get("published_at") if prov_cash else None,
                companyfacts_cash_val,
                companyfacts_cash_meta,
            ):
                cash_val = companyfacts_cash_val
                prov_cash = _companyfacts_input_reference(companyfacts_path, companyfacts_cash_meta)
                companyfacts_cash_context = companyfacts_cash_meta
        cash_refs = [InputReference(**prov_cash)] if prov_cash else []
        restricted_cash, restricted_refs = self._structured_restricted_cash(
            facts,
            companyfacts=companyfacts,
            companyfacts_path=companyfacts_path,
            as_of=as_of,
        )
        readily_available_cash = self._readily_available_cash_components(
            facts,
            taxonomy,
            reference_row=reference_row,
            as_of=as_of,
            cash_override=cash_val,
            cash_ref_override=InputReference(**prov_cash) if prov_cash else None,
            companyfacts=companyfacts,
            companyfacts_path=companyfacts_path,
        )
        marketable_refs = [
            ref for ref in list(readily_available_cash["refs"] or [])
            if ref not in cash_refs and ref not in restricted_refs
        ]

        lease_current, prov_lc = self._latest_fact(facts, self.fact_map["lease_current"])
        lease_long, prov_ll = self._latest_fact(facts, self.fact_map["lease_long"])
        lease_liabilities = (lease_current or 0.0) + (lease_long or 0.0)
        lease_refs = [InputReference(**p) for p in [prov_lc, prov_ll] if p]
        supplier_finance, prov_sf = self._latest_fact(facts, self.fact_map["supplier_finance"])
        preferred_equity, prov_pref = self._latest_fact(facts, self.fact_map["preferred_equity"])
        convertibles, prov_conv = self._latest_fact(facts, self.fact_map["convertibles"])
        unfunded_pension, prov_pension = self._latest_fact(facts, self.fact_map["unfunded_pension"])
        component_refs = [
            InputReference(**p)
            for p in [prov_sf, prov_pref, prov_conv, prov_pension]
            if p
        ]

        lease_adjusted_metrics = self._lease_adjusted_metrics_required(taxonomy)
        denominator_policy = "ebitdar" if lease_adjusted_metrics else "ebitda"

        total_debt_market = None
        included_lease_liabilities = None
        included_supplier_finance = None
        if total_debt_reported is not None:
            total_debt_market = float(total_debt_reported)
            if lease_adjusted_metrics and lease_liabilities:
                included_lease_liabilities = float(lease_liabilities)
                total_debt_market += included_lease_liabilities
            if supplier_finance is not None:
                included_supplier_finance = float(supplier_finance)
                total_debt_market += included_supplier_finance

        debt_component_breakdown = {
            "reported_debt": total_debt_reported,
            "local_reported_debt": local_total_debt_reported,
            "lease_adjusted_metrics": lease_adjusted_metrics,
            "lease_liabilities": lease_liabilities if lease_liabilities else None,
            "included_lease_liabilities": included_lease_liabilities,
            "supplier_finance": supplier_finance,
            "included_supplier_finance": included_supplier_finance,
            "preferred_equity": preferred_equity,
            "convertibles": convertibles,
            "unfunded_pension": unfunded_pension,
            "reference_instrument": reference_instrument,
            "reference_total_debt": provider_total_debt,
            "recent_total_debt_peak": recent_total_debt_peak,
            "debt_reference_source": debt_reference_source,
        }
        if companyfacts_total_debt_context is not None:
            debt_component_breakdown["companyfacts_total_debt_override"] = companyfacts_total_debt_context
        if companyfacts_cash_context is not None:
            debt_component_breakdown["companyfacts_cash_override"] = companyfacts_cash_context
        debt_flags: List[str] = []
        if lease_adjusted_metrics and not lease_liabilities:
            debt_flags.append("lease_adjustment_missing_assumed_zero")
        if supplier_finance is not None:
            debt_flags.append("supplier_finance_included_without_payables_extension_test")
        if preferred_equity is not None:
            debt_flags.append("preferred_equity_excluded_pending_hybrid_review")
        if convertibles is not None:
            debt_flags.append("convertibles_excluded_pending_hybrid_review")
        if unfunded_pension is not None:
            debt_flags.append("pension_excluded_from_debt")
        if taxonomy.archetype == "financial_institution":
            debt_flags.append("sector_native_metrics_required")
        if debt_reference_fallback_used:
            debt_flags.append("reference_total_debt_fallback")
            if debt_reference_source == "recent_total_debt_peak":
                debt_flags.append("recent_total_debt_peak_used_for_completeness")
            if debt_reference_source == "reference_total_debt":
                debt_flags.append("reference_total_debt_used_for_completeness")
        self._emit_metric_views(
            features,
            metric_id="capital_structure.total_debt",
            base_name="capital_structure.total_debt",
            reported_value=total_debt_reported,
            market_value=total_debt_market,
            unit="usd",
            as_of=as_of,
            window=None,
            taxonomy=taxonomy,
            reported_provenance=debt_refs,
            market_provenance=debt_refs + lease_refs + component_refs,
            reported_missing_reason="unavailable" if total_debt_reported is None else None,
            market_missing_reason="unavailable" if total_debt_market is None else None,
            fallback_used="heuristic" if debt_flags else None,
            component_breakdown=debt_component_breakdown,
            quality_flags=debt_flags,
            decision_fallback_to_reported=True,
        )
        if companyfacts_total_debt_context is not None:
            for feature_name in (
                "capital_structure.total_debt_reported",
                "capital_structure.total_debt_market",
                "capital_structure.total_debt",
            ):
                feature = features.get(feature_name)
                if feature is None:
                    continue
                feature.fallback_used = "companyfacts_total_debt_exact_fresher"
                feature.quality_flags = list(dict.fromkeys(list(feature.quality_flags or []) + ["companyfacts_total_debt_fresher"]))

        features["capital_structure.net_pension_liability"] = self._metric_feature(
            name="capital_structure.net_pension_liability",
            value=unfunded_pension,
            unit="usd",
            as_of=as_of,
            window=None,
            confidence=0.85 if unfunded_pension is not None else None,
            provenance=[InputReference(**prov_pension)] if prov_pension else [],
            missing_reason="unavailable" if unfunded_pension is None else None,
            fallback_used=None,
            metric_id="capital_structure.net_pension_liability",
            taxonomy=taxonomy,
            component_breakdown={
                "formula": "latest(financial.unfunded_pension|financial.pension_deficit|financial.net_pension_liability)"
            },
            quality_flags=None,
            support_mode="exact" if unfunded_pension is not None else "unsupported",
            view_type="decision",
        )

        usable_cash_market = readily_available_cash["readily_available_cash"]
        net_debt_reported = None
        if total_debt_reported is not None and cash_val is not None:
            net_debt_reported = total_debt_reported - cash_val
        net_debt_market = None
        if total_debt_market is not None and usable_cash_market is not None:
            net_debt_market = total_debt_market - usable_cash_market
        net_debt_flags: List[str] = list(dict.fromkeys((debt_flags or []) + list(readily_available_cash["quality_flags"] or [])))
        self._emit_metric_views(
            features,
            metric_id="capital_structure.net_debt",
            base_name="capital_structure.net_debt",
            reported_value=net_debt_reported,
            market_value=net_debt_market,
            unit="usd",
            as_of=as_of,
            window=None,
            taxonomy=taxonomy,
            reported_provenance=debt_refs + cash_refs,
            market_provenance=debt_refs + cash_refs + restricted_refs + marketable_refs + lease_refs + component_refs,
            reported_missing_reason="unavailable" if net_debt_reported is None else None,
            market_missing_reason="unavailable" if net_debt_market is None else None,
            fallback_used="heuristic" if net_debt_flags else None,
            component_breakdown={
                "economic_debt": total_debt_market,
                "usable_cash_market": usable_cash_market,
                "restricted_cash": readily_available_cash["restricted_cash"],
                "marketable_securities": readily_available_cash["marketable_securities"],
                "unavailable_cash": readily_available_cash["unavailable_cash"],
            },
            quality_flags=net_debt_flags,
            decision_fallback_to_reported=True,
        )

        ebitda_val, ebitda_provs, ebitda_ttm_breakdown, ebitda_ttm_flags = self._latest_ttm_statement_value(
            facts,
            self.fact_map["ebitda"],
        )
        ebitda_refs = [InputReference(**prov) for prov in ebitda_provs if prov]
        if ebitda_val is None and provider_ebitda is not None:
            ebitda_val = provider_ebitda
            if "reference_ebitda_fallback" not in ebitda_ttm_flags:
                ebitda_ttm_flags.append("reference_ebitda_fallback")
            ebitda_ttm_breakdown = {
                "formula": "provider_ebitda_reference",
                "reference_instrument": reference_instrument,
                "source_field": "EBITDA",
            }
        leverage_flags: List[str] = []
        associate_dividends, prov_assoc = self._latest_fact(facts, self.fact_map["associate_dividends"])
        minority_dividends, prov_minority = self._latest_fact(facts, self.fact_map["minority_dividends"])
        preferred_dividends, prov_pref_div = self._latest_fact(facts, self.fact_map["preferred_dividends"])
        interest_income, prov_interest_income = self._latest_fact(facts, self.fact_map["interest_income"])
        ebit_val, prov_ebit = self._latest_fact(facts, self.fact_map["ebit"])
        interest_val, prov_ie = self._latest_fact(facts, self.fact_map["interest_expense"])
        interest_refs = [InputReference(**prov_ie)] if prov_ie else []
        interest_expense_context = None
        interest_expense_source = "fact_registry" if interest_val is not None else None
        interest_expense_fallback_used = None
        if interest_val is None:
            repaired_interest_val, repaired_interest_refs, repaired_interest_context = self._repaired_statement_direct_interest_expense(
                company_id=str(company_id),
                aliases=market_aliases,
                as_of=as_of,
            )
            if repaired_interest_val is not None:
                interest_val = repaired_interest_val
                interest_refs = repaired_interest_refs
                interest_expense_context = repaired_interest_context
                interest_expense_source = "statement_direct_companyfacts"
                interest_expense_fallback_used = "statement_direct_interest_expense_fallback"
        lease_expense, prov_lease_exp = self._latest_fact(facts, self.fact_map["lease_expense"])
        lease_charge_proxy, lease_proxy_flags = self._estimated_lease_expense(lease_expense, lease_liabilities, taxonomy)
        if lease_proxy_flags:
            leverage_flags.extend(lease_proxy_flags)
        ebitdar_val = None if ebitda_val is None else float(ebitda_val) + float(lease_charge_proxy or 0.0)
        leverage_denominator = ebitdar_val if denominator_policy == "ebitdar" else ebitda_val
        effective_denominator_policy = denominator_policy
        if denominator_policy == "ebitdar" and lease_charge_proxy is None:
            leverage_flags.append("lease_adjusted_denominator_missing_lease_expense")
            if ebitda_val is not None:
                leverage_denominator = ebitda_val
                effective_denominator_policy = "ebitda_proxy_for_missing_lease_charge"
                leverage_flags.append("lease_adjusted_denominator_fallback_to_ebitda")
        leverage_flags.extend([flag for flag in ebitda_ttm_flags if flag not in leverage_flags])
        gross_leverage_reported = None
        net_leverage_reported = None
        gross_leverage_market = None
        net_leverage_market = None
        gross_missing_reported = None
        net_missing_reported = None
        gross_missing_market = None
        net_missing_market = None
        if leverage_denominator is None or leverage_denominator <= 0:
            gross_missing_reported = "negative_ebitda" if leverage_denominator is not None and leverage_denominator <= 0 else "unavailable"
            net_missing_reported = gross_missing_reported
            gross_missing_market = gross_missing_reported
            net_missing_market = gross_missing_reported
        else:
            if total_debt_reported is not None:
                gross_leverage_reported = total_debt_reported / leverage_denominator
            else:
                gross_missing_reported = "unavailable"
            if net_debt_reported is not None:
                net_leverage_reported = net_debt_reported / leverage_denominator
            else:
                net_missing_reported = "unavailable"
            if self.metric_policy.resolve_applicability("capital_structure.gross_leverage", taxonomy) != "unsupported" and total_debt_market is not None:
                gross_leverage_market = total_debt_market / leverage_denominator
            else:
                gross_missing_market = "unsupported_for_archetype" if self.metric_policy.resolve_applicability("capital_structure.gross_leverage", taxonomy) == "unsupported" else "unavailable"
            if self.metric_policy.resolve_applicability("capital_structure.net_leverage", taxonomy) != "unsupported" and net_debt_market is not None:
                net_leverage_market = net_debt_market / leverage_denominator
            else:
                net_missing_market = "unsupported_for_archetype" if self.metric_policy.resolve_applicability("capital_structure.net_leverage", taxonomy) == "unsupported" else "unavailable"
        self._emit_metric_views(
            features,
            metric_id="capital_structure.gross_leverage",
            base_name="capital_structure.gross_leverage",
            reported_value=gross_leverage_reported,
            market_value=gross_leverage_market,
            unit="x",
            as_of=as_of,
            window={"type": "ttm", "length_days": 365},
            taxonomy=taxonomy,
            reported_provenance=debt_refs + ebitda_refs,
            market_provenance=debt_refs + lease_refs + component_refs + ebitda_refs,
            reported_missing_reason=gross_missing_reported,
            market_missing_reason=gross_missing_market,
            fallback_used=None,
            component_breakdown={
                "economic_debt": total_debt_market,
                "ebitda": ebitda_val,
                "ebitdar": ebitdar_val,
                "lease_fixed_charge": lease_charge_proxy,
                "denominator_policy": denominator_policy,
                "effective_denominator_policy": effective_denominator_policy,
                "ebitda_context": ebitda_ttm_breakdown,
                "reference_instrument": reference_instrument,
            },
            quality_flags=leverage_flags,
            decision_fallback_to_reported=False,
        )
        self._emit_metric_views(
            features,
            metric_id="capital_structure.net_leverage",
            base_name="capital_structure.net_leverage",
            reported_value=net_leverage_reported,
            market_value=net_leverage_market,
            unit="x",
            as_of=as_of,
            window={"type": "ttm", "length_days": 365},
            taxonomy=taxonomy,
            reported_provenance=debt_refs + cash_refs + ebitda_refs,
            market_provenance=debt_refs + lease_refs + component_refs + cash_refs + restricted_refs + marketable_refs + ebitda_refs,
            reported_missing_reason=net_missing_reported,
            market_missing_reason=net_missing_market,
            fallback_used=None,
            component_breakdown={
                "net_debt_market": net_debt_market,
                "ebitda": ebitda_val,
                "ebitdar": ebitdar_val,
                "lease_fixed_charge": lease_charge_proxy,
                "denominator_policy": denominator_policy,
                "effective_denominator_policy": effective_denominator_policy,
                "ebitda_context": ebitda_ttm_breakdown,
                "reference_instrument": reference_instrument,
            },
            quality_flags=leverage_flags,
            decision_fallback_to_reported=False,
        )

        coverage_refs = list(ebitda_refs) + [
            InputReference(**p)
            for p in [prov_ebit, prov_assoc, prov_minority, prov_pref_div, prov_interest_income]
            if p
        ]
        for ref in interest_refs:
            if ref not in coverage_refs:
                coverage_refs.append(ref)
        coverage_flags: List[str] = []
        coverage_flags.extend(lease_proxy_flags)
        adjusted_coverage_numerator = None if ebitda_val is None else float(ebitda_val)
        if adjusted_coverage_numerator is not None:
            adjusted_coverage_numerator += float(associate_dividends or 0.0)
            adjusted_coverage_numerator -= float(minority_dividends or 0.0)
        ebitdar_coverage_numerator = None if adjusted_coverage_numerator is None else adjusted_coverage_numerator + float(lease_charge_proxy or 0.0)
        interest_cov_reported = None
        interest_cov_market = None
        fixed_charge_cov_reported = None
        fixed_charge_cov_market = None
        interest_missing_reported = None
        interest_missing_market = None
        fixed_missing_reported = None
        fixed_missing_market = None
        interest_paid = None if interest_val is None else float(interest_val)
        net_interest = None
        if interest_paid is not None:
            net_interest = max(0.0, interest_paid - float(interest_income or 0.0))
        fixed_charge_denominator_reported = None
        fixed_charge_denominator_market = None
        if interest_paid is not None:
            fixed_charge_denominator_reported = interest_paid + float(lease_expense or 0.0)
        if net_interest is not None:
            fixed_charge_denominator_market = net_interest + float(preferred_dividends or 0.0) + float(lease_charge_proxy or 0.0)

        if ebitda_val is not None and interest_paid and interest_paid > 0:
            interest_cov_reported = float(ebitda_val) / interest_paid
        else:
            interest_missing_reported = "unavailable"
        if self.metric_policy.resolve_applicability("capital_structure.interest_coverage", taxonomy) == "unsupported":
            interest_missing_market = "unsupported_for_archetype"
        elif adjusted_coverage_numerator is not None and interest_paid and interest_paid > 0:
            interest_cov_market = adjusted_coverage_numerator / interest_paid
        else:
            interest_missing_market = "unavailable"
        if lease_expense is None and lease_charge_proxy is not None:
            coverage_flags.append("lease_fixed_charge_proxy_from_liability")
        if ebitdar_val is not None and fixed_charge_denominator_reported and fixed_charge_denominator_reported > 0:
            fixed_charge_cov_reported = ebitdar_val / fixed_charge_denominator_reported
        else:
            fixed_missing_reported = "unavailable"
        if self.metric_policy.resolve_applicability("capital_structure.fixed_charge_coverage", taxonomy) == "unsupported":
            fixed_missing_market = "unsupported_for_archetype"
        elif ebitdar_coverage_numerator is not None and fixed_charge_denominator_market and fixed_charge_denominator_market > 0:
            fixed_charge_cov_market = ebitdar_coverage_numerator / fixed_charge_denominator_market
        else:
            fixed_missing_market = "unavailable"
        if self.metric_policy.resolve_applicability("capital_structure.fixed_charge_coverage", taxonomy) == "primary":
            coverage_flags.append("fixed_charge_coverage_preferred")
        fixed_charge_fallback_used = None
        if interest_expense_fallback_used and lease_charge_proxy is not None and lease_expense is None:
            fixed_charge_fallback_used = "heuristic_plus_statement_direct_interest_expense"
        elif interest_expense_fallback_used:
            fixed_charge_fallback_used = interest_expense_fallback_used
        elif lease_charge_proxy is not None and lease_expense is None:
            fixed_charge_fallback_used = "heuristic"
        self._emit_metric_views(
            features,
            metric_id="capital_structure.interest_coverage",
            base_name="capital_structure.interest_coverage",
            reported_value=interest_cov_reported,
            market_value=interest_cov_market,
            unit="x",
            as_of=as_of,
            window={"type": "ttm", "length_days": 365},
            taxonomy=taxonomy,
            reported_provenance=coverage_refs,
            market_provenance=coverage_refs,
            reported_missing_reason=interest_missing_reported,
            market_missing_reason=interest_missing_market,
            fallback_used=interest_expense_fallback_used,
            component_breakdown={
                "ebitda": ebitda_val,
                "associate_dividends": associate_dividends,
                "minority_dividends": minority_dividends,
                "interest_expense": interest_val,
                "interest_expense_source": interest_expense_source,
                "interest_expense_context": interest_expense_context,
            },
            quality_flags=coverage_flags,
            decision_fallback_to_reported=False,
        )
        fixed_charge_refs = coverage_refs + ([InputReference(**prov_lease_exp)] if prov_lease_exp else [])
        self._emit_metric_views(
            features,
            metric_id="capital_structure.fixed_charge_coverage",
            base_name="capital_structure.fixed_charge_coverage",
            reported_value=fixed_charge_cov_reported,
            market_value=fixed_charge_cov_market,
            unit="x",
            as_of=as_of,
            window={"type": "ttm", "length_days": 365},
            taxonomy=taxonomy,
            reported_provenance=fixed_charge_refs,
            market_provenance=fixed_charge_refs,
            reported_missing_reason=fixed_missing_reported,
            market_missing_reason=fixed_missing_market,
            fallback_used=fixed_charge_fallback_used,
            component_breakdown={
                "ebitdar": ebitdar_val,
                "associate_dividends": associate_dividends,
                "minority_dividends": minority_dividends,
                "interest_expense": interest_val,
                "interest_expense_source": interest_expense_source,
                "interest_expense_context": interest_expense_context,
                "interest_income": interest_income,
                "preferred_dividends": preferred_dividends,
                "lease_fixed_charge": lease_charge_proxy,
                "lease_expense_reported": lease_expense,
            },
            quality_flags=coverage_flags,
            decision_fallback_to_reported=False,
        )

        dealscan_covenants, dealscan_covenant_refs, dealscan_covenant_breakdown, dealscan_covenant_flags = (
            self._dealscan_covenant_proxy_context(
                company_id,
                as_of,
                aliases=dealscan_aliases,
            )
        )
        covenant_specs = [
            ("capital_structure.max_leverage_ratio_covenant_proxy", "max_leverage_ratio_covenant", "x"),
            ("capital_structure.min_interest_coverage_ratio_covenant_proxy", "min_interest_coverage_ratio_covenant", "x"),
            ("capital_structure.min_fixed_charge_coverage_ratio_covenant_proxy", "min_fixed_charge_coverage_ratio_covenant", "x"),
            ("capital_structure.min_current_ratio_covenant_proxy", "min_current_ratio_covenant", "x"),
        ]
        for feature_name, metric_key, unit in covenant_specs:
            value = dealscan_covenants.get(metric_key)
            metric_breakdown = None
            if dealscan_covenant_breakdown:
                metric_breakdown = {
                    **dealscan_covenant_breakdown,
                    "selected_metric": metric_key,
                    "selected_value": value,
                    "selection_rule": (dealscan_covenant_breakdown.get("selection_rules") or {}).get(metric_key),
                    "observed_threshold_text": (dealscan_covenant_breakdown.get("observed_threshold_text") or {}).get(metric_key),
                    "observed_threshold_values": (dealscan_covenant_breakdown.get("observed_threshold_values") or {}).get(metric_key),
                }
            features[feature_name] = self._metric_feature(
                name=feature_name,
                value=value,
                unit=unit,
                as_of=as_of,
                window={"type": "asof", "length_days": 0},
                confidence=(0.45 if value is not None else None),
                provenance=dealscan_covenant_refs,
                missing_reason="not_disclosed" if value is None else None,
                fallback_used="dealscan_active_revolver_covenants" if value is not None else None,
                component_breakdown=metric_breakdown,
                quality_flags=dealscan_covenant_flags or None,
                support_mode="proxy_missing_component" if value is not None else None,
                view_type="decision",
            )

        due_0_12 = 0.0
        due_12_24 = 0.0
        due_24_36 = 0.0
        due_36_60 = 0.0
        due_60_plus = 0.0
        maturity_known = False
        maturity_refs: List[InputReference] = []
        debt_events = pd.DataFrame()
        if events is not None and not events.empty:
            col = "event_type" if "event_type" in events.columns else None
            if col:
                debt_events = events[
                    events[col].astype(str).str.contains(
                        "debt|bond|loan|facility|refinanc",
                        case=False,
                        na=False,
                    )
                ].copy()
        if debt_events is not None and not debt_events.empty:
            for _, row in debt_events.head(1000).iterrows():
                maturity_dt = self._event_param_datetime(row, ["maturity_date", "maturity"])
                if maturity_dt is None or maturity_dt <= as_of:
                    continue
                amt = self._event_amount(row)
                if amt is None or amt <= 0:
                    continue
                et = str(row.get("event_type", "")).lower()
                if "redemption" in et or "repay" in et or "paydown" in et:
                    amt = -amt
                maturity_known = True
                horizon_days = (maturity_dt - as_of).days
                if horizon_days <= 365:
                    due_0_12 += amt
                elif horizon_days <= 365 * 2:
                    due_12_24 += amt
                elif horizon_days <= 365 * 3:
                    due_24_36 += amt
                elif horizon_days <= 365 * 5:
                    due_36_60 += amt
                else:
                    due_60_plus += amt
            maturity_refs = self._event_refs(debt_events, limit=5)
            due_0_12 = max(0.0, due_0_12)
            due_12_24 = max(0.0, due_12_24)
            due_24_36 = max(0.0, due_24_36)
            due_36_60 = max(0.0, due_36_60)
            due_60_plus = max(0.0, due_60_plus)

        note_maturity_schedule, note_maturity_refs, note_maturity_flags = self._note_maturity_schedule(facts, as_of)
        note_schedule_used = False
        if not maturity_known and note_maturity_schedule is not None:
            due_0_12 = float(note_maturity_schedule["due_0_12"])
            due_12_24 = float(note_maturity_schedule["due_12_24"])
            due_24_36 = float(note_maturity_schedule["due_24_36"])
            due_36_60 = float(note_maturity_schedule["due_36_60"])
            due_60_plus = float(note_maturity_schedule["due_60_plus"])
            maturity_known = True
            maturity_refs = list(note_maturity_refs)
            note_schedule_used = True

        due_24m = due_0_12 + due_12_24 if maturity_known else None
        if due_24m is not None and total_debt_reported not in (None, 0):
            if due_24m > float(total_debt_reported) * 100.0:
                due_0_12 /= 1000.0
                due_12_24 /= 1000.0
                due_24_36 /= 1000.0
                due_36_60 /= 1000.0
                due_60_plus /= 1000.0
                due_24m = due_0_12 + due_12_24
        debt_schedule_total = (
            due_0_12 + due_12_24 + due_24_36 + due_36_60 + due_60_plus
            if maturity_known
            else None
        )
        debt_schedule_vs_total_debt = None
        debt_schedule_inconsistency_flag = None
        if maturity_known and total_debt_reported is not None:
            if total_debt_reported <= 0:
                debt_schedule_inconsistency_flag = 1.0 if debt_schedule_total and debt_schedule_total > 0 else 0.0
            elif debt_schedule_total is not None:
                debt_schedule_vs_total_debt = float(debt_schedule_total) / float(total_debt_reported)
                debt_schedule_inconsistency_flag = 1.0 if debt_schedule_vs_total_debt >= 1.5 else 0.0
        sanitize_maturity_schedule = bool(debt_schedule_inconsistency_flag is not None and debt_schedule_inconsistency_flag >= 0.5)
        if sanitize_maturity_schedule and note_maturity_schedule is not None:
            due_0_12 = float(note_maturity_schedule["due_0_12"])
            due_12_24 = float(note_maturity_schedule["due_12_24"])
            due_24_36 = float(note_maturity_schedule["due_24_36"])
            due_36_60 = float(note_maturity_schedule["due_36_60"])
            due_60_plus = float(note_maturity_schedule["due_60_plus"])
            due_24m = due_0_12 + due_12_24
            debt_schedule_total = due_0_12 + due_12_24 + due_24_36 + due_36_60 + due_60_plus
            debt_schedule_vs_total_debt = (
                float(debt_schedule_total) / float(total_debt_reported)
                if total_debt_reported not in (None, 0)
                else None
            )
            debt_schedule_inconsistency_flag = (
                1.0 if (debt_schedule_vs_total_debt is not None and debt_schedule_vs_total_debt >= 1.5) else 0.0
            )
            maturity_known = True
            maturity_refs = list(note_maturity_refs)
            sanitize_maturity_schedule = False
            note_schedule_used = True
        published_due_0_12 = due_0_12 if maturity_known else None
        published_due_12_24 = due_12_24 if maturity_known else None
        published_due_24_36 = due_24_36 if maturity_known else None
        published_due_36_60 = due_36_60 if maturity_known else None
        published_due_60_plus = due_60_plus if maturity_known else None
        if sanitize_maturity_schedule:
            published_due_0_12 = None
            published_due_12_24 = None
            published_due_24_36 = None
            published_due_36_60 = None
            published_due_60_plus = None
        maturity_ratio_reported = None
        maturity_ratio_market = None
        if not sanitize_maturity_schedule and due_24m is not None and total_debt_reported not in (None, 0):
            maturity_ratio_reported = float(due_24m) / float(total_debt_reported)
        if not sanitize_maturity_schedule and due_24m is not None and total_debt_market not in (None, 0):
            maturity_ratio_market = float(due_24m) / float(total_debt_market)
        refi_flag_reported = None if maturity_ratio_reported is None else (1.0 if maturity_ratio_reported > 0.25 else 0.0)
        refi_flag_market = None if maturity_ratio_market is None else (1.0 if maturity_ratio_market > 0.25 else 0.0)

        for name, value in [
            ("capital_structure.debt_due_0_12m", published_due_0_12),
            ("capital_structure.debt_due_12_24m", published_due_12_24),
            ("capital_structure.debt_due_24_36m", published_due_24_36),
            ("capital_structure.debt_due_36_60m", published_due_36_60),
            ("capital_structure.debt_due_60m_plus", published_due_60_plus),
        ]:
            features[name] = FeatureRecord(
                name=name,
                value=value,
                unit="usd",
                computed_at=_now_iso(),
                as_of_time=as_of.isoformat(),
                window={"type": "asof", "length_days": 0},
                confidence=0.6 if value is not None else None,
                provenance=maturity_refs,
                missing_reason="anomalous_schedule" if sanitize_maturity_schedule else ("not_disclosed" if value is None else None),
                fallback_used="sanitized_due_to_anomaly" if sanitize_maturity_schedule else (("note_pattern_extract" if note_schedule_used else "heuristic") if value is not None else None),
            )

        features["capital_structure.debt_schedule_total"] = FeatureRecord(
            name="capital_structure.debt_schedule_total",
            value=debt_schedule_total,
            unit="usd",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window=None,
            confidence=0.6 if debt_schedule_total is not None else None,
            provenance=maturity_refs,
            missing_reason="not_disclosed" if debt_schedule_total is None else None,
            fallback_used=("note_pattern_extract" if note_schedule_used else "heuristic") if debt_schedule_total is not None else None,
        )
        features["capital_structure.debt_schedule_vs_total_debt"] = FeatureRecord(
            name="capital_structure.debt_schedule_vs_total_debt",
            value=debt_schedule_vs_total_debt,
            unit="ratio",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window=None,
            confidence=0.6 if debt_schedule_vs_total_debt is not None else None,
            provenance=maturity_refs,
            missing_reason="not_disclosed" if debt_schedule_vs_total_debt is None else None,
            fallback_used=("note_pattern_extract" if note_schedule_used else "heuristic") if debt_schedule_vs_total_debt is not None else None,
        )
        features["capital_structure.debt_schedule_inconsistency_flag"] = FeatureRecord(
            name="capital_structure.debt_schedule_inconsistency_flag",
            value=debt_schedule_inconsistency_flag,
            unit="bool",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window=None,
            confidence=0.6 if debt_schedule_inconsistency_flag is not None else None,
            provenance=maturity_refs,
            missing_reason="not_disclosed" if debt_schedule_inconsistency_flag is None else None,
            fallback_used=("note_pattern_extract" if note_schedule_used else "heuristic") if debt_schedule_inconsistency_flag is not None else None,
        )
        maturity_flags = []
        if sanitize_maturity_schedule:
            maturity_flags.append("debt_schedule_anomaly_sanitized")
        maturity_flags.extend(note_maturity_flags)
        self._emit_metric_views(
            features,
            metric_id="capital_structure.maturity_wall_ratio_24m",
            base_name="capital_structure.maturity_wall_ratio_24m",
            reported_value=maturity_ratio_reported,
            market_value=maturity_ratio_market,
            unit="ratio",
            as_of=as_of,
            window=None,
            taxonomy=taxonomy,
            reported_provenance=maturity_refs,
            market_provenance=maturity_refs,
            reported_missing_reason="anomalous_schedule" if sanitize_maturity_schedule else ("not_disclosed" if maturity_ratio_reported is None else None),
            market_missing_reason="anomalous_schedule" if sanitize_maturity_schedule else ("not_disclosed" if maturity_ratio_market is None else None),
            fallback_used="sanitized_due_to_anomaly" if sanitize_maturity_schedule else (("note_pattern_extract" if note_schedule_used else "heuristic") if maturity_ratio_market is not None else None),
            component_breakdown={"debt_due_24m": due_24m, "reported_debt": total_debt_reported, "economic_debt": total_debt_market},
            quality_flags=maturity_flags,
            decision_fallback_to_reported=True,
        )
        self._emit_metric_views(
            features,
            metric_id="capital_structure.refi_pressure_flag",
            base_name="capital_structure.refi_pressure_flag",
            reported_value=refi_flag_reported,
            market_value=refi_flag_market,
            unit="bool",
            as_of=as_of,
            window=None,
            taxonomy=taxonomy,
            reported_provenance=maturity_refs,
            market_provenance=maturity_refs,
            reported_missing_reason="anomalous_schedule" if sanitize_maturity_schedule else ("not_disclosed" if refi_flag_reported is None else None),
            market_missing_reason="anomalous_schedule" if sanitize_maturity_schedule else ("not_disclosed" if refi_flag_market is None else None),
            fallback_used="sanitized_due_to_anomaly" if sanitize_maturity_schedule else (("note_pattern_extract" if note_schedule_used else "heuristic") if refi_flag_market is not None else None),
            component_breakdown={"maturity_wall_ratio_24m_market": maturity_ratio_market},
            quality_flags=maturity_flags,
            decision_fallback_to_reported=True,
        )
        features["capital_structure.secured_capacity_proxy"] = FeatureRecord(
            name="capital_structure.secured_capacity_proxy",
            value=None,
            unit="ratio",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window={"type": "heuristic", "length_days": 0},
            confidence=None,
            provenance=[],
            missing_reason="not_disclosed",
            fallback_used="heuristic",
        )

        rating_payload = None
        rating_conf = None
        rating_refs: List[InputReference] = []

        def _prefer_fitch_rows(df: pd.DataFrame) -> pd.DataFrame:
            if df is None or df.empty:
                return df
            cols = [c for c in ("source_type", "rating_agency", "agency", "provider", "event_subtype") if c in df.columns]
            if not cols:
                return df
            mask = pd.Series(False, index=df.index)
            for col in cols:
                mask = mask | df[col].astype(str).str.contains("fitch", case=False, na=False)
            preferred = df[mask].copy()
            return preferred if not preferred.empty else df

        if events is not None and not events.empty and "event_type" in events.columns:
            ratings_df = events[
                events["event_type"].astype(str).str.contains("rating", case=False, na=False)
            ].copy()
            if not ratings_df.empty:
                ratings_df = _prefer_fitch_rows(ratings_df)
                time_col = self._event_time_col(ratings_df)
                if time_col:
                    ratings_df[time_col] = pd.to_datetime(ratings_df[time_col], utc=True, errors="coerce")
                    ratings_df = ratings_df.sort_values(time_col, ascending=False)
                row = ratings_df.iloc[0]
                rating = None
                outlook = None
                watch = None
                if isinstance(row.get("params"), dict):
                    params = row.get("params")
                    rating = params.get("current_rating_symbol") or params.get("rating_symbol") or params.get("rating")
                    outlook = params.get("outlook")
                    watch = params.get("creditwatch")
                if rating is None and "event_subtype" in row and row.get("event_subtype") is not None:
                    subtype = str(row.get("event_subtype"))
                    if any(ch.isalpha() for ch in subtype):
                        rating = subtype
                watch_norm = _null_if_na(watch)
                if isinstance(watch_norm, str):
                    wl = watch_norm.strip().lower()
                    if wl in ("y", "yes", "true", "watch", "negative", "positive"):
                        watch_norm = True
                    elif wl in ("n", "no", "false", "none", "stable"):
                        watch_norm = False
                    else:
                        watch_norm = None
                rating_score = self._rating_score(rating)
                rating_payload = {
                    "rating": _null_if_na(rating),
                    "outlook": _null_if_na(outlook),
                    "watchlist": watch_norm,
                    "score": rating_score,
                }
                rating_conf = 0.7 if rating is not None else 0.5
                rating_refs = self._event_refs(ratings_df, limit=3)

        if rating_payload is None and issuer_ratings is not None and not issuer_ratings.empty:
            r = _prefer_fitch_rows(issuer_ratings.copy())
            for col in ("rating_date", "published_at", "effective_at"):
                if col in r.columns:
                    r[col] = pd.to_datetime(r[col], utc=True, errors="coerce")
            order_cols = [c for c in ("rating_date", "published_at", "effective_at") if c in r.columns]
            if order_cols:
                r = r.sort_values(order_cols, ascending=[False] * len(order_cols))
            row = r.iloc[0]
            rating = _null_if_na(row.get("rating_symbol")) or _null_if_na(row.get("current_rating_symbol"))
            outlook = _null_if_na(row.get("outlook"))
            watch_norm = _null_if_na(row.get("creditwatch"))
            if isinstance(watch_norm, str):
                wl = watch_norm.strip().lower()
                if wl in ("y", "yes", "true", "watch", "negative", "positive"):
                    watch_norm = True
                elif wl in ("n", "no", "false", "none", "stable"):
                    watch_norm = False
                else:
                    watch_norm = None
            rating_payload = {
                "rating": _null_if_na(rating),
                "outlook": outlook,
                "watchlist": watch_norm,
                "score": self._rating_score(rating),
            }
            rating_conf = 0.72 if rating is not None else 0.55
            rating_refs = [
                InputReference(
                    artifact_type="ExtractedFact",
                    artifact_id=str(_null_if_na(row.get("artifact_id")) or f"issuer_rating:{_null_if_na(row.get('company_id'))}:{_null_if_na(row.get('rating_date'))}"),
                    source=str(_null_if_na(row.get("source_type"))) if _null_if_na(row.get("source_type")) is not None else "fisd_ratings",
                    published_at=str(_null_if_na(row.get("published_at"))) if _null_if_na(row.get("published_at")) is not None else None,
                    ingested_at=str(_null_if_na(row.get("ingested_at"))) if _null_if_na(row.get("ingested_at")) is not None else None,
                    hash=None,
                )
            ]

        features["capital_structure.rating_state"] = FeatureRecord(
            name="capital_structure.rating_state",
            value=rating_payload,
            unit="rating",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window=None,
            confidence=rating_conf,
            provenance=rating_refs,
            missing_reason="not_disclosed" if rating_payload is None else None,
            fallback_used="heuristic" if rating_payload is not None and rating_payload.get("score") is None else None,
        )
        return features

    def _compute_market(
        self,
        company_id: str,
        ts: pd.DataFrame,
        facts: pd.DataFrame,
        as_of: pd.Timestamp,
        existing_features: Dict[str, FeatureRecord],
        taxonomy: TaxonomyContext,
    ) -> Dict[str, FeatureRecord]:
        features: Dict[str, FeatureRecord] = {}
        _, market_aliases = self._resolve_entity_aliases(str(company_id))
        reference_row = self._fundamentals_reference_row(market_aliases)
        reference_instrument = _null_if_na(reference_row.get("Instrument"))
        provider_market_cap = _safe_float(reference_row.get("Company Market Cap"))
        provider_ebitda = _safe_float(reference_row.get("EBITDA"))
        provider_ev_ebitda = _safe_float(reference_row.get("Enterprise Value To EBITDA (Daily Time Series Ratio)"))
        market_cap = None
        market_cap_fallback = None
        market_cap_formula = None
        market_cap_quality_flags: List[str] = []
        price_val = None
        price_field = None
        price_observation_time = None
        price_series = None
        price_time_col = None
        price_series_metadata: Dict[str, Any] = {}
        price_series_flags: List[str] = []
        shares = None
        shares_source = None
        market_cap_source_field = None
        market_cap_observation_time = None
        if ts is not None and not ts.empty:
            time_col = _pick_time_col(ts)
            price_series, price_val, price_field, price_time_col, price_series_metadata, price_series_flags = _prepare_price_series(ts)
            if price_series is not None and not price_series.empty:
                latest_price_row = price_series.iloc[-1]
                price_observation_time = _null_if_na(latest_price_row.get(price_time_col)) if price_time_col else None
            # Long format market cap detection
            series_col = _pick_first_col(ts, ["series_id", "field_name", "metric", "instrument_id"])
            if series_col and time_col and "value" in ts.columns:
                mc = ts[ts[series_col].astype(str).str.contains("market_cap|mktcap", case=False, na=False)]
                if not mc.empty:
                    mc = mc.sort_values(time_col, ascending=False)
                    latest_mc_row = mc.iloc[0]
                    market_cap = _safe_float(latest_mc_row.get("value"))
                    market_cap_formula = "provider_market_cap_series"
                    market_cap_source_field = _null_if_na(latest_mc_row.get(series_col))
                    market_cap_observation_time = _null_if_na(latest_mc_row.get(time_col))
            # Direct market_cap column
            if market_cap is None:
                for c in ["market_cap", "mkt_cap", "marketcap", "mktcap", "market_capitalization"]:
                    if c in ts.columns:
                        mc = ts.dropna(subset=[c])
                        if time_col and not mc.empty:
                            mc = mc.sort_values(time_col, ascending=False)
                        if not mc.empty:
                            latest_mc_row = mc.iloc[0]
                            market_cap = _safe_float(latest_mc_row.get(c))
                            market_cap_formula = f"provider_{c}"
                            market_cap_source_field = c
                            market_cap_observation_time = _null_if_na(latest_mc_row.get(time_col)) if time_col else None
                        break
        # Market cap fallback from price * shares
        if market_cap is None and price_val is not None:
            shares_diluted, _ = self._latest_fact(facts, self.fact_map["shares_diluted"])
            shares_basic, _ = self._latest_fact(facts, self.fact_map["shares_basic"])
            shares = shares_diluted if shares_diluted is not None else shares_basic
            shares_source = "shares_diluted" if shares_diluted is not None else "shares_basic" if shares_basic is not None else None
            if shares is not None:
                try:
                    market_cap = float(price_val) * float(shares)
                    market_cap_fallback = "price*shares"
                    market_cap_formula = "close_price * shares_outstanding"
                    market_cap_source_field = f"{price_field or 'close'} * {shares_source or 'shares_outstanding'}"
                    market_cap_observation_time = price_observation_time
                    market_cap_quality_flags.extend(["provider_market_cap_missing", "price_shares_fallback"])
                    if shares_source == "shares_basic":
                        market_cap_quality_flags.append("shares_basic_fallback")
                except Exception:
                    market_cap = None
        price_observation_age_days = None
        if price_observation_time is not None:
            try:
                price_observation_age_days = float((as_of - pd.to_datetime(price_observation_time, utc=True)).days)
            except Exception:
                price_observation_age_days = None
        if (
            provider_market_cap is not None
            and market_cap is not None
            and market_cap_fallback == "price*shares"
            and price_observation_age_days is not None
            and price_observation_age_days > 90
        ):
            market_cap = provider_market_cap
            market_cap_fallback = "reference_company_market_cap"
            market_cap_formula = "provider_company_market_cap_reference"
            market_cap_source_field = "Company Market Cap"
            market_cap_observation_time = None
            if "reference_market_cap_preferred_over_stale_price_shares" not in market_cap_quality_flags:
                market_cap_quality_flags.append("reference_market_cap_preferred_over_stale_price_shares")
        if market_cap is None and provider_market_cap is not None:
            market_cap = provider_market_cap
            market_cap_fallback = "reference_company_market_cap"
            market_cap_formula = "provider_company_market_cap_reference"
            market_cap_source_field = "Company Market Cap"
            market_cap_quality_flags.append("reference_market_cap_fallback")
        if market_cap is None:
            market_cap_quality_flags.append("market_cap_unavailable")
        market_cap_quality_flags.extend([flag for flag in price_series_flags if flag not in market_cap_quality_flags])
        features["market.market_cap"] = FeatureRecord(
            name="market.market_cap",
            value=market_cap,
            unit="usd",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window=None,
            confidence=None,
            provenance=[],
            missing_reason="unavailable" if market_cap is None else None,
            fallback_used=market_cap_fallback,
            component_breakdown={
                "market_cap": market_cap,
                "price": price_val,
                "price_field": price_field,
                "price_observation_time": str(price_observation_time) if price_observation_time is not None else None,
                "shares_outstanding": shares,
                "shares_source": shares_source,
                "market_cap_source_field": market_cap_source_field,
                "market_cap_observation_time": (
                    str(market_cap_observation_time) if market_cap_observation_time is not None else None
                ),
                "price_observation_age_days": price_observation_age_days,
                "formula": market_cap_formula,
                "selected_price_series": price_series_metadata or None,
                "reference_instrument": reference_instrument,
            },
            quality_flags=market_cap_quality_flags or None,
        )

        pe_ratio = None
        pe_ratio_fallback = None
        pe_ratio_formula = None
        pe_ratio_flags: List[str] = []
        pe_peer_pct = None
        pe_history_pct = None
        pe_peer_breakdown: Dict[str, Any] = {
            "formula": "percentile_rank(current_pe_ratio, gics_peer_set)",
        }
        pe_history_breakdown: Dict[str, Any] = {
            "formula": "percentile_rank(current_pe_ratio, monthly_pe_history_10y)",
            "lookback_months": 120,
        }
        pe_provider_history = pd.DataFrame(columns=["obs_time", "value", "series_id"])
        if ts is not None and not ts.empty:
            time_col = _pick_time_col(ts)
            pe_direct_col = _pick_first_col(ts, ["pe_ratio", "price_to_earnings", "trailing_pe"])
            if time_col and pe_direct_col:
                pe_direct = ts[[time_col, pe_direct_col]].copy()
                pe_direct["obs_time"] = pd.to_datetime(pe_direct[time_col], utc=True, errors="coerce")
                pe_direct["value"] = pd.to_numeric(pe_direct[pe_direct_col], errors="coerce")
                pe_direct = pe_direct.dropna(subset=["obs_time", "value"]).sort_values("obs_time")
                if not pe_direct.empty:
                    pe_provider_history = pe_direct.assign(series_id=pe_direct_col)[["obs_time", "value", "series_id"]]
            if pe_provider_history.empty:
                pe_provider_history = self._extract_series_history(
                    ts,
                    contains_any=["price_to_earnings", "pe_ratio", "trailing_pe"],
                )

        pe_current_source = None
        pe_eps_value = None
        pe_eps_source = None
        pe_net_income_ttm = None
        pe_history_series = pd.Series(dtype=float)
        if not pe_provider_history.empty:
            pe_provider_history = pe_provider_history[pe_provider_history["value"] > 0].copy()
            if not pe_provider_history.empty:
                pe_ratio = _safe_float(pe_provider_history.iloc[-1]["value"])
                pe_ratio_formula = "provider_pe_series"
                pe_current_source = str(pe_provider_history.iloc[-1]["series_id"])
                pe_history_series = self._periodic_series(pe_provider_history, "M", periods=120)
                pe_history_breakdown.update(
                    {
                        "source": "provider_pe_series",
                        "series_id": pe_current_source,
                        "observation_count": int(len(pe_history_series)),
                    }
                )

        if pe_ratio is None:
            net_income_ttm, _, net_income_ttm_breakdown, net_income_ttm_flags = self._latest_ttm_statement_value(
                facts,
                self.fact_map["net_income"],
            )
            pe_net_income_ttm = net_income_ttm
            if market_cap is not None and net_income_ttm is not None:
                if net_income_ttm > 0:
                    pe_ratio = float(market_cap) / float(net_income_ttm)
                    pe_ratio_formula = "market_cap / net_income_ttm"
                    pe_ratio_fallback = "derived_from_market_cap_and_net_income_ttm"
                    pe_current_source = "net_income_ttm"
                    pe_history_breakdown.setdefault("source", "net_income_ttm")
                else:
                    pe_ratio_flags.append("non_positive_net_income_ttm")
            pe_ratio_flags.extend(
                [flag for flag in net_income_ttm_flags if flag not in pe_ratio_flags]
            )
            if pe_ratio is None and net_income_ttm_breakdown.get("formula") is not None:
                pe_history_breakdown.setdefault("net_income_ttm", net_income_ttm_breakdown)

        if pe_ratio is None:
            eps_history = self._diluted_eps_history(facts)
            if not eps_history.empty:
                pe_eps_value = _safe_float(eps_history.iloc[-1]["value"])
                pe_eps_source = "diluted_eps_history"
            if price_val is not None and pe_eps_value is not None:
                if pe_eps_value > 0:
                    pe_ratio = float(price_val) / float(pe_eps_value)
                    pe_ratio_formula = "close_price / diluted_eps_ttm"
                    pe_ratio_fallback = "derived_from_price_and_eps"
                    pe_current_source = pe_eps_source
                else:
                    pe_ratio_flags.append("non_positive_eps")
            if pe_ratio is None and price_val is None:
                pe_ratio_flags.append("price_unavailable")
            if pe_ratio is None and pe_eps_value is None:
                pe_ratio_flags.append("diluted_eps_unavailable")

            if price_series is not None and not price_series.empty and not eps_history.empty:
                derived_price_series = price_series.copy()
                if "price" in derived_price_series.columns and price_time_col and price_time_col in derived_price_series.columns:
                    derived_price_series["obs_time"] = pd.to_datetime(
                        derived_price_series[price_time_col], utc=True, errors="coerce"
                    )
                    monthly_prices = (
                        derived_price_series.dropna(subset=["obs_time", "price"])
                        .sort_values("obs_time")
                        .set_index("obs_time")["price"]
                        .astype(float)
                        .resample("ME")
                        .last()
                        .dropna()
                        .tail(120)
                    )
                    if not monthly_prices.empty:
                        monthly_price_df = monthly_prices.rename("price").reset_index().sort_values("obs_time")
                        eps_hist = eps_history.sort_values("obs_time").copy()
                        monthly_price_df["obs_time"] = pd.to_datetime(monthly_price_df["obs_time"], utc=True, errors="coerce")
                        eps_hist["obs_time"] = pd.to_datetime(eps_hist["obs_time"], utc=True, errors="coerce")
                        monthly_price_df = monthly_price_df.dropna(subset=["obs_time"])
                        eps_hist = eps_hist.dropna(subset=["obs_time"])
                        monthly_price_df["obs_key"] = monthly_price_df["obs_time"].astype("int64")
                        eps_hist["obs_key"] = eps_hist["obs_time"].astype("int64")
                        merged_pe = pd.merge_asof(
                            monthly_price_df,
                            eps_hist.rename(columns={"value": "eps"})[["obs_time", "obs_key", "eps"]],
                            on="obs_key",
                            direction="backward",
                        )
                        merged_pe["obs_time"] = merged_pe["obs_time_x"]
                        merged_pe = merged_pe[merged_pe["eps"] > 0].copy()
                        if not merged_pe.empty:
                            merged_pe["pe"] = merged_pe["price"] / merged_pe["eps"]
                            pe_history_series = (
                                merged_pe.set_index("obs_time")["pe"]
                                .replace([np.inf, -np.inf], np.nan)
                                .dropna()
                                .tail(120)
                            )
                            pe_history_breakdown.update(
                                {
                                    "source": "derived_from_price_and_eps_history",
                                    "observation_count": int(len(pe_history_series)),
                                }
                            )

        if not pe_history_series.empty:
            pe_history_pct = _percentile(pe_history_series)
            pe_history_breakdown["history_percentile"] = pe_history_pct
            pe_history_breakdown["window_start"] = str(pe_history_series.index[0])
            pe_history_breakdown["window_end"] = str(pe_history_series.index[-1])
        else:
            pe_history_breakdown["observation_count"] = 0
            pe_ratio_flags.append("pe_history_unavailable")

        pe_peer_pct, peer_breakdown, peer_flags = self._peer_percentile_from_entity_table(
            str(company_id),
            pe_ratio,
            taxonomy,
            ["pe_ratio", "price_to_earnings", "trailing_pe", "pe"],
        )
        pe_peer_breakdown.update(peer_breakdown)

        features["market.pe_ratio"] = FeatureRecord(
            name="market.pe_ratio",
            value=pe_ratio,
            unit="x",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window={"type": "ttm", "length_days": 365},
            confidence=None,
            provenance=[],
            missing_reason=(
                "non_positive_net_income_ttm"
                if pe_ratio is None and "non_positive_net_income_ttm" in pe_ratio_flags
                else "non_positive_eps"
                if pe_ratio is None and "non_positive_eps" in pe_ratio_flags
                else "unavailable" if pe_ratio is None else None
            ),
            fallback_used=pe_ratio_fallback,
            component_breakdown={
                "formula": pe_ratio_formula,
                "price": price_val,
                "price_field": price_field,
                "price_observation_time": str(price_observation_time) if price_observation_time is not None else None,
                "diluted_eps_ttm": pe_eps_value,
                "net_income_ttm": pe_net_income_ttm,
                "eps_source": pe_eps_source,
                "provider_series_id": pe_current_source,
            },
            quality_flags=pe_ratio_flags or None,
        )
        features["market.pe_percentile_peers"] = FeatureRecord(
            name="market.pe_percentile_peers",
            value=pe_peer_pct,
            unit="percentile",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window={"type": "asof", "length_days": 0},
            confidence=None,
            provenance=[],
            missing_reason="unavailable" if pe_peer_pct is None else None,
            fallback_used=None,
            component_breakdown=pe_peer_breakdown,
            quality_flags=peer_flags or None,
        )
        features["market.pe_percentile_history"] = FeatureRecord(
            name="market.pe_percentile_history",
            value=pe_history_pct,
            unit="percentile",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window={"type": "lookback", "length_days": 365 * 10},
            confidence=None,
            provenance=[],
            missing_reason="unavailable" if pe_history_pct is None else None,
            fallback_used=None,
            component_breakdown=pe_history_breakdown,
            quality_flags=(["pe_history_unavailable"] if pe_history_pct is None else None),
        )

        total_debt_market_feat = existing_features.get("capital_structure.total_debt_market")
        net_debt_market_feat = existing_features.get("capital_structure.net_debt_market")
        total_debt_market = total_debt_market_feat.value if total_debt_market_feat is not None else None
        net_debt_market = net_debt_market_feat.value if net_debt_market_feat is not None else None

        ev = None
        ev_fallback = None
        if market_cap is not None and net_debt_market is not None:
            ev = float(market_cap) + float(net_debt_market)
        elif market_cap is not None:
            cash_val, _ = self._latest_fact(facts, self.fact_map["cash"])
            total_debt_reported, _ = self._latest_fact(facts, self.fact_map["total_debt"])
            if total_debt_reported is None:
                debt_current, _ = self._latest_fact(facts, self.fact_map["debt_current"])
                debt_long, _ = self._latest_fact(facts, self.fact_map["debt_long"])
                if debt_current is not None or debt_long is not None:
                    total_debt_reported = (debt_current or 0.0) + (debt_long or 0.0)
            if total_debt_reported is not None:
                ev = float(market_cap) + float(total_debt_reported) - float(cash_val or 0.0)
                ev_fallback = "reported_debt_minus_cash_fallback"
        features["market.enterprise_value"] = FeatureRecord(
            name="market.enterprise_value",
            value=ev,
            unit="usd",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window=None,
            confidence=None,
            provenance=[],
            missing_reason="unavailable" if ev is None else None,
            fallback_used=ev_fallback,
        )

        ebitda_val, _, ebitda_ttm_breakdown, ebitda_ttm_flags = self._latest_ttm_statement_value(
            facts,
            self.fact_map["ebitda"],
        )
        ev_ebitda_fallback = None
        if ebitda_val is None and provider_ebitda is not None:
            ebitda_val = provider_ebitda
            ev_ebitda_fallback = "reference_ebitda"
            ebitda_ttm_breakdown = {
                "formula": "provider_ebitda_reference",
                "reference_instrument": reference_instrument,
                "source_field": "EBITDA",
            }
            ebitda_ttm_flags = [flag for flag in ebitda_ttm_flags if flag != "statement_metric_unavailable"]
            if "reference_ebitda_fallback" not in ebitda_ttm_flags:
                ebitda_ttm_flags.append("reference_ebitda_fallback")
        ev_ebitda = None
        if ev is not None and ebitda_val and ebitda_val > 0:
            ev_ebitda = ev / ebitda_val
        elif provider_ev_ebitda is not None:
            ev_ebitda = provider_ev_ebitda
            ev_ebitda_fallback = "reference_ev_ebitda"
        features["market.ev_ebitda"] = FeatureRecord(
            name="market.ev_ebitda",
            value=ev_ebitda,
            unit="x",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window={"type": "ttm", "length_days": 365},
            confidence=None,
            provenance=[],
            missing_reason="unavailable" if ev_ebitda is None else None,
            fallback_used=ev_ebitda_fallback,
            component_breakdown={
                "enterprise_value": ev,
                "ebitda_ttm": ebitda_val,
                "ebitda_ttm_context": ebitda_ttm_breakdown,
                "formula": "enterprise_value / ebitda_ttm",
                "reference_ev_ebitda": provider_ev_ebitda,
                "reference_instrument": reference_instrument,
            },
            quality_flags=ebitda_ttm_flags or None,
        )

        operating_cash_flow, _ = self._latest_fact(facts, self.fact_map["operating_cash_flow"])
        capex_val, _ = self._latest_fact(facts, self.fact_map["capex"])
        fcf_val = None
        fcf_fallback = None
        if operating_cash_flow is not None and capex_val is not None:
            fcf_val = float(operating_cash_flow) - float(capex_val)
        else:
            fcf_val, _ = self._latest_fact(facts, self.fact_map["fcf"])
            if fcf_val is not None:
                fcf_fallback = "provider_fcf_field_fallback"
        fcf_yield = None
        if market_cap is not None and fcf_val is not None and market_cap != 0:
            fcf_yield = fcf_val / market_cap
        features["market.fcf_yield"] = FeatureRecord(
            name="market.fcf_yield",
            value=fcf_yield,
            unit="ratio",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window={"type": "ttm", "length_days": 365},
            confidence=None,
            provenance=[],
            missing_reason="unavailable" if fcf_yield is None else None,
            fallback_used=fcf_fallback,
        )

        # Volatility / drawdown require exact daily history. For sparse monthly-only
        # historical backfills we now fail honestly instead of inventing a proxy.
        vol_30 = None
        vol_90 = None
        dd_90 = None
        momentum_60d = None
        vol_30_fallback = None
        vol_90_fallback = None
        dd_90_fallback = None
        vol_30_breakdown: Dict[str, Any] = {"formula": "stddev(daily_returns_30d) * sqrt(252)"}
        vol_90_breakdown: Dict[str, Any] = {"formula": "stddev(daily_returns_90d) * sqrt(252)"}
        dd_90_breakdown: Dict[str, Any] = {"formula": "min(price_window_90d) / max(price_window_90d) - 1"}
        vol_30_flags: List[str] = []
        vol_90_flags: List[str] = []
        dd_90_flags: List[str] = []
        if price_series is not None and not price_series.empty:
            price_series = price_series.dropna(subset=["price"])
            if price_time_col and price_time_col in price_series.columns:
                price_series = price_series.sort_values(price_time_col).drop_duplicates(subset=[price_time_col], keep="last")
            price_series["ret"] = price_series["price"].pct_change()
            price_input_field = price_field or "price"
            vol_30_breakdown["selected_price_series"] = price_series_metadata or None
            vol_90_breakdown["selected_price_series"] = price_series_metadata or None
            dd_90_breakdown["selected_price_series"] = price_series_metadata or None
            window_end = price_series[price_time_col].iloc[-1] if price_time_col and price_time_col in price_series.columns else None
            cadence_days = None
            low_frequency_price_history = False
            if window_end is not None:
                obs_gaps = (
                    price_series[price_time_col]
                    .sort_values()
                    .diff()
                    .dropna()
                    .dt.total_seconds()
                    .div(86400.0)
                )
                if not obs_gaps.empty:
                    cadence_days = _safe_float(obs_gaps.median())
                    low_frequency_price_history = bool(
                        self.historical_backfill_mode
                        and cadence_days is not None
                        and cadence_days >= 7.0
                    )
            if cadence_days is not None:
                vol_30_breakdown["median_observation_gap_days"] = cadence_days
                vol_90_breakdown["median_observation_gap_days"] = cadence_days
                dd_90_breakdown["median_observation_gap_days"] = cadence_days
            returns_frame = price_series[[price_time_col, "ret"]].dropna(subset=["ret"]).copy() if window_end is not None else pd.DataFrame()
            returns_30_frame = (
                returns_frame[returns_frame[price_time_col] > (window_end - pd.Timedelta(days=30))].copy()
                if window_end is not None
                else pd.DataFrame()
            )
            returns_90_frame = (
                returns_frame[returns_frame[price_time_col] > (window_end - pd.Timedelta(days=90))].copy()
                if window_end is not None
                else pd.DataFrame()
            )
            if len(returns_30_frame) >= 10:
                returns_30 = returns_30_frame["ret"]
                price_window_30 = price_series[price_series[price_time_col] > (window_end - pd.Timedelta(days=30))].copy()
                vol_30 = float(returns_30.std(ddof=0) * np.sqrt(252))
                vol_30_breakdown.update(
                    {
                        "return_observations": int(len(returns_30)),
                        "annualization_factor": 252,
                        "price_field": price_input_field,
                        "window_start": (
                            str(price_window_30[price_time_col].iloc[0]) if price_time_col and price_time_col in price_window_30.columns else None
                        ),
                        "window_end": (
                            str(price_window_30[price_time_col].iloc[-1]) if price_time_col and price_time_col in price_window_30.columns else None
                        ),
                    }
                )
            else:
                if low_frequency_price_history:
                    vol_30_flags.extend(["insufficient_return_history", "low_frequency_price_history"])
                else:
                    vol_30_flags.append("insufficient_return_history")
            if len(returns_90_frame) >= 20:
                returns_90 = returns_90_frame["ret"]
                price_window_90_for_vol = price_series[price_series[price_time_col] > (window_end - pd.Timedelta(days=90))].copy()
                vol_90 = float(returns_90.std(ddof=0) * np.sqrt(252))
                vol_90_breakdown.update(
                    {
                        "return_observations": int(len(returns_90)),
                        "annualization_factor": 252,
                        "price_field": price_input_field,
                        "window_start": (
                            str(price_window_90_for_vol[price_time_col].iloc[0]) if price_time_col and price_time_col in price_window_90_for_vol.columns else None
                        ),
                        "window_end": (
                            str(price_window_90_for_vol[price_time_col].iloc[-1]) if price_time_col and price_time_col in price_window_90_for_vol.columns else None
                        ),
                    }
                )
            else:
                if low_frequency_price_history:
                    vol_90_flags.extend(["insufficient_return_history", "low_frequency_price_history"])
                else:
                    vol_90_flags.append("insufficient_return_history")
            price_window_90 = (
                price_series[price_series[price_time_col] > (window_end - pd.Timedelta(days=90))].copy()
                if window_end is not None
                else pd.DataFrame()
            )
            if len(price_window_90) >= 20:
                peak_price = _safe_float(price_window_90["price"].max())
                trough_price = _safe_float(price_window_90["price"].min())
                if peak_price not in (None, 0) and trough_price is not None:
                    dd_90 = (trough_price / peak_price) - 1.0
                dd_90_breakdown.update(
                    {
                        "price_observations": int(len(price_window_90)),
                        "price_field": price_input_field,
                        "peak_price": peak_price,
                        "trough_price": trough_price,
                        "window_start": (
                            str(price_window_90[price_time_col].iloc[0]) if price_time_col and price_time_col in price_window_90.columns else None
                        ),
                        "window_end": (
                            str(price_window_90[price_time_col].iloc[-1]) if price_time_col and price_time_col in price_window_90.columns else None
                        ),
                    }
                )
            else:
                if low_frequency_price_history:
                    dd_90_flags.extend(["insufficient_price_history", "low_frequency_price_history"])
                else:
                    dd_90_flags.append("insufficient_price_history")
            lookback_60 = (
                price_series[price_series[price_time_col] <= (window_end - pd.Timedelta(days=60))].copy()
                if window_end is not None
                else pd.DataFrame()
            )
            if not lookback_60.empty:
                p0 = _safe_float(lookback_60["price"].iloc[-1])
                p1 = _safe_float(price_series["price"].iloc[-1])
                if p0 not in (None, 0) and p1 is not None:
                    momentum_60d = (p1 / p0) - 1.0
            vol_30_flags.extend([flag for flag in price_series_flags if flag not in vol_30_flags])
            vol_90_flags.extend([flag for flag in price_series_flags if flag not in vol_90_flags])
            dd_90_flags.extend([flag for flag in price_series_flags if flag not in dd_90_flags])
        else:
            vol_30_flags.append("price_history_unavailable")
            vol_90_flags.append("price_history_unavailable")
            dd_90_flags.append("price_history_unavailable")
        features["market.volatility_30d"] = FeatureRecord(
            name="market.volatility_30d",
            value=vol_30,
            unit="annualized",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window={"type": "lookback", "length_days": 30},
            confidence=None,
            provenance=[],
            missing_reason="unavailable" if vol_30 is None else None,
            fallback_used=vol_30_fallback,
            support_mode="exact" if vol_30 is not None else "unsupported",
            component_breakdown=vol_30_breakdown,
            quality_flags=vol_30_flags or None,
        )
        features["market.volatility_90d"] = FeatureRecord(
            name="market.volatility_90d",
            value=vol_90,
            unit="annualized",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window={"type": "lookback", "length_days": 90},
            confidence=None,
            provenance=[],
            missing_reason="unavailable" if vol_90 is None else None,
            fallback_used=vol_90_fallback,
            support_mode="exact" if vol_90 is not None else "unsupported",
            component_breakdown=vol_90_breakdown,
            quality_flags=vol_90_flags or None,
        )
        features["market.drawdown_90d"] = FeatureRecord(
            name="market.drawdown_90d",
            value=dd_90,
            unit="ratio",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window={"type": "lookback", "length_days": 90},
            confidence=None,
            provenance=[],
            missing_reason="unavailable" if dd_90 is None else None,
            fallback_used=dd_90_fallback,
            support_mode="exact" if dd_90 is not None else "unsupported",
            component_breakdown=dd_90_breakdown,
            quality_flags=dd_90_flags or None,
        )

        # Credit spread level from company-level timeseries, when available.
        credit_spread = None
        credit_spread_pct = None
        if ts is not None and not ts.empty:
            time_col = _pick_time_col(ts)
            series_col = _pick_first_col(ts, ["series_id", "field_name", "metric", "instrument_id"])
            value_col = _pick_value_col(ts)
            if time_col and series_col and value_col:
                spread_df = ts[
                    ts[series_col].astype(str).str.contains("spread|oas|cds", case=False, na=False)
                ][[time_col, value_col]].copy()
                if not spread_df.empty:
                    spread_df[time_col] = pd.to_datetime(spread_df[time_col], utc=True, errors="coerce")
                    spread_df[value_col] = pd.to_numeric(spread_df[value_col], errors="coerce")
                    spread_df = spread_df.dropna(subset=[time_col, value_col]).sort_values(time_col)
                    if not spread_df.empty:
                        credit_spread = _safe_float(spread_df.iloc[-1].get(value_col))
                        two_year_cutoff = as_of - pd.Timedelta(days=365 * 2)
                        hist = spread_df[spread_df[time_col] >= two_year_cutoff][value_col]
                        if len(hist.dropna()) >= 20:
                            credit_spread_pct = float(hist.rank(pct=True).iloc[-1] * 100.0)
        features["market.credit_spread_level"] = FeatureRecord(
            name="market.credit_spread_level",
            value=credit_spread,
            unit="spread",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window={"type": "asof", "length_days": 0},
            confidence=0.7 if credit_spread is not None else None,
            provenance=[],
            missing_reason="unavailable" if credit_spread is None else None,
            fallback_used=None,
        )
        features["market.credit_spread_percentile_2y"] = FeatureRecord(
            name="market.credit_spread_percentile_2y",
            value=credit_spread_pct,
            unit="percentile",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window={"type": "lookback", "length_days": 365 * 2},
            confidence=0.7 if credit_spread_pct is not None else None,
            provenance=[],
            missing_reason="unavailable" if credit_spread_pct is None else None,
            fallback_used=None,
        )

        # Issuance window proxies (0-1), heuristic.
        eq_components: List[float] = []
        if vol_30 is not None:
            eq_components.append(float(np.clip(1.0 - (vol_30 / 0.8), 0.0, 1.0)))
        if momentum_60d is not None:
            eq_components.append(float(np.clip((momentum_60d + 0.2) / 0.4, 0.0, 1.0)))
        if ev_ebitda is not None:
            # Higher valuation generally corresponds to better equity issuance window.
            eq_components.append(float(np.clip(ev_ebitda / 20.0, 0.0, 1.0)))
        equity_window = float(np.mean(eq_components)) if eq_components else None

        credit_components: List[float] = []
        if credit_spread is not None:
            # Supports both decimal and bps scales.
            scale = 1000.0 if credit_spread > 5 else 0.10
            credit_components.append(float(np.clip(1.0 - (credit_spread / scale), 0.0, 1.0)))
        if vol_30 is not None:
            credit_components.append(float(np.clip(1.0 - (vol_30 / 1.0), 0.0, 1.0)))
        credit_window = float(np.mean(credit_components)) if credit_components else None

        features["market.equity_window_proxy"] = FeatureRecord(
            name="market.equity_window_proxy",
            value=equity_window,
            unit="index_0_1",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window={"type": "lookback", "length_days": 60},
            confidence=0.5 if equity_window is not None else None,
            provenance=[],
            missing_reason="unavailable" if equity_window is None else None,
            fallback_used="heuristic" if equity_window is not None else None,
        )
        features["market.credit_window_proxy"] = FeatureRecord(
            name="market.credit_window_proxy",
            value=credit_window,
            unit="index_0_1",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window={"type": "lookback", "length_days": 60},
            confidence=0.5 if credit_window is not None else None,
            provenance=[],
            missing_reason="unavailable" if credit_window is None else None,
            fallback_used="heuristic" if credit_window is not None else None,
        )
        return features

    def _compute_macro(self, macro: pd.DataFrame, as_of: pd.Timestamp) -> Dict[str, FeatureRecord]:
        features: Dict[str, FeatureRecord] = {}

        def latest_and_percentile(
            history: pd.DataFrame,
            *,
            freq: str,
            periods: int,
        ) -> Tuple[Optional[float], Optional[float], pd.Series]:
            periodic = self._periodic_series(history, freq, periods=periods)
            value = _safe_float(periodic.iloc[-1]) if not periodic.empty else None
            pct = _percentile(periodic) if not periodic.empty else None
            return value, pct, periodic

        def emit_metric(
            name: str,
            value: Optional[float],
            unit: str,
            window: Dict[str, Any],
            component_breakdown: Dict[str, Any],
            quality_flags: Optional[List[str]] = None,
        ) -> None:
            features[name] = FeatureRecord(
                name=name,
                value=value,
                unit=unit,
                computed_at=_now_iso(),
                as_of_time=as_of.isoformat(),
                window=window,
                confidence=None,
                provenance=[],
                missing_reason="unavailable" if value is None else None,
                fallback_used=None,
                component_breakdown=component_breakdown,
                quality_flags=quality_flags or None,
            )

        sp500_pe_history = self._extract_series_history(
            macro,
            exact_ids=[self.macro_series.get("sp500_pe_ttm")],
            contains_any=["sp500_pe", "spx_pe", "market_pe"],
        )
        sp500_pe_value, sp500_pe_pct, sp500_pe_monthly = latest_and_percentile(
            sp500_pe_history,
            freq="M",
            periods=120,
        )
        emit_metric(
            "macro.sp500_pe_ttm",
            sp500_pe_value,
            "x",
            {"type": "asof", "length_days": 0},
            {
                "formula": "latest(sp500_pe_ttm_series)",
                "series_ids": sorted(sp500_pe_history["series_id"].unique().tolist()) if not sp500_pe_history.empty else [],
                "observation_count": int(len(sp500_pe_monthly)),
            },
            [] if sp500_pe_value is not None else ["macro_series_unavailable"],
        )
        emit_metric(
            "macro.sp500_pe_ttm_percentile_history",
            sp500_pe_pct,
            "percentile",
            {"type": "lookback", "length_days": 365 * 10},
            {
                "formula": "percentile_rank(current_sp500_pe_ttm, monthly_sp500_pe_ttm_history_10y)",
                "observation_count": int(len(sp500_pe_monthly)),
                "window_start": str(sp500_pe_monthly.index[0]) if not sp500_pe_monthly.empty else None,
                "window_end": str(sp500_pe_monthly.index[-1]) if not sp500_pe_monthly.empty else None,
            },
            [] if sp500_pe_pct is not None else ["macro_history_unavailable"],
        )

        us10y_history = self._extract_series_history(
            macro,
            exact_ids=[self.macro_series.get("rate_10y")],
            contains_any=["dgs10", "10y", "treasury_10y"],
        )
        us10y_value, us10y_pct, us10y_monthly = latest_and_percentile(us10y_history, freq="M", periods=120)
        emit_metric(
            "macro.ust_10y_yield",
            us10y_value,
            "percent",
            {"type": "asof", "length_days": 0},
            {
                "formula": "latest(dgs10)",
                "series_ids": sorted(us10y_history["series_id"].unique().tolist()) if not us10y_history.empty else [],
                "observation_count": int(len(us10y_monthly)),
            },
            [] if us10y_value is not None else ["macro_series_unavailable"],
        )
        emit_metric(
            "macro.us10y_treasury_yield",
            us10y_value,
            "percent",
            {"type": "asof", "length_days": 0},
            {
                "formula": "latest(dgs10)",
                "series_ids": sorted(us10y_history["series_id"].unique().tolist()) if not us10y_history.empty else [],
                "observation_count": int(len(us10y_monthly)),
            },
            [] if us10y_value is not None else ["macro_series_unavailable"],
        )
        emit_metric(
            "macro.us10y_treasury_yield_percentile_history",
            us10y_pct,
            "percentile",
            {"type": "lookback", "length_days": 365 * 10},
            {
                "formula": "percentile_rank(current_us10y_treasury_yield, monthly_us10y_treasury_yield_history_10y)",
                "observation_count": int(len(us10y_monthly)),
                "window_start": str(us10y_monthly.index[0]) if not us10y_monthly.empty else None,
                "window_end": str(us10y_monthly.index[-1]) if not us10y_monthly.empty else None,
            },
            [] if us10y_pct is not None else ["macro_history_unavailable"],
        )

        us2y_history = self._extract_series_history(
            macro,
            exact_ids=[self.macro_series.get("rate_2y")],
            contains_any=["dgs2", "2y", "treasury_2y"],
        )
        us2y_value, _, us2y_monthly = latest_and_percentile(us2y_history, freq="M", periods=120)
        emit_metric(
            "macro.ust_2y_yield",
            us2y_value,
            "percent",
            {"type": "asof", "length_days": 0},
            {
                "formula": "latest(dgs2)",
                "series_ids": sorted(us2y_history["series_id"].unique().tolist()) if not us2y_history.empty else [],
                "observation_count": int(len(us2y_monthly)),
            },
            [] if us2y_value is not None else ["macro_series_unavailable"],
        )

        ig_oas_history = self._extract_series_history(
            macro,
            exact_ids=[self.macro_series.get("ig_oas")],
            contains_any=["bamlc0a0cm", "ig_oas"],
        )
        ig_oas_value, ig_oas_pct, ig_oas_monthly = latest_and_percentile(ig_oas_history, freq="M", periods=120)
        emit_metric(
            "macro.ig_oas",
            ig_oas_value,
            "spread",
            {"type": "asof", "length_days": 0},
            {
                "formula": "latest(ice_bofa_ig_oas)",
                "series_ids": sorted(ig_oas_history["series_id"].unique().tolist()) if not ig_oas_history.empty else [],
                "observation_count": int(len(ig_oas_monthly)),
            },
            [] if ig_oas_value is not None else ["macro_series_unavailable"],
        )
        emit_metric(
            "macro.us_ig_oas",
            ig_oas_value,
            "spread",
            {"type": "asof", "length_days": 0},
            {
                "formula": "latest(ice_bofa_ig_oas)",
                "series_ids": sorted(ig_oas_history["series_id"].unique().tolist()) if not ig_oas_history.empty else [],
                "observation_count": int(len(ig_oas_monthly)),
            },
            [] if ig_oas_value is not None else ["macro_series_unavailable"],
        )
        emit_metric(
            "macro.us_ig_oas_percentile_history",
            ig_oas_pct,
            "percentile",
            {"type": "lookback", "length_days": 365 * 10},
            {
                "formula": "percentile_rank(current_us_ig_oas, monthly_us_ig_oas_history_10y)",
                "observation_count": int(len(ig_oas_monthly)),
                "window_start": str(ig_oas_monthly.index[0]) if not ig_oas_monthly.empty else None,
                "window_end": str(ig_oas_monthly.index[-1]) if not ig_oas_monthly.empty else None,
            },
            [] if ig_oas_pct is not None else ["macro_history_unavailable"],
        )

        hy_oas_history = self._extract_series_history(
            macro,
            exact_ids=[self.macro_series.get("hy_oas")],
            contains_any=["bamlh0a0hym2", "hy_oas"],
        )
        hy_oas_value, _, hy_oas_monthly = latest_and_percentile(hy_oas_history, freq="M", periods=120)
        emit_metric(
            "macro.hy_oas",
            hy_oas_value,
            "spread",
            {"type": "asof", "length_days": 0},
            {
                "formula": "latest(ice_bofa_hy_oas)",
                "series_ids": sorted(hy_oas_history["series_id"].unique().tolist()) if not hy_oas_history.empty else [],
                "observation_count": int(len(hy_oas_monthly)),
            },
            [] if hy_oas_value is not None else ["macro_series_unavailable"],
        )

        hy_yield_history = self._extract_series_history(
            macro,
            exact_ids=[self.macro_series.get("hy_all_in_yield")],
            contains_any=["bamlh0a0hym2ey", "hy_all_in_yield", "high_yield_effective_yield"],
        )
        hy_yield_value, hy_yield_pct, hy_yield_monthly = latest_and_percentile(hy_yield_history, freq="M", periods=120)
        emit_metric(
            "macro.us_hy_all_in_yield",
            hy_yield_value,
            "percent",
            {"type": "asof", "length_days": 0},
            {
                "formula": "latest(ice_bofa_hy_effective_yield)",
                "series_ids": sorted(hy_yield_history["series_id"].unique().tolist()) if not hy_yield_history.empty else [],
                "observation_count": int(len(hy_yield_monthly)),
            },
            [] if hy_yield_value is not None else ["macro_series_unavailable"],
        )
        emit_metric(
            "macro.us_hy_all_in_yield_percentile_history",
            hy_yield_pct,
            "percentile",
            {"type": "lookback", "length_days": 365 * 10},
            {
                "formula": "percentile_rank(current_us_hy_all_in_yield, monthly_us_hy_all_in_yield_history_10y)",
                "observation_count": int(len(hy_yield_monthly)),
                "window_start": str(hy_yield_monthly.index[0]) if not hy_yield_monthly.empty else None,
                "window_end": str(hy_yield_monthly.index[-1]) if not hy_yield_monthly.empty else None,
            },
            [] if hy_yield_pct is not None else ["macro_history_unavailable"],
        )

        fed_funds_history = self._extract_series_history(
            macro,
            exact_ids=[self.macro_series.get("fed_funds_effective")],
            contains_any=["dff", "fed_funds", "fedfunds"],
        )
        fed_funds_value, _, fed_funds_monthly = latest_and_percentile(fed_funds_history, freq="M", periods=120)
        emit_metric(
            "macro.fed_funds_effective",
            fed_funds_value,
            "percent",
            {"type": "asof", "length_days": 0},
            {
                "formula": "latest(dff)",
                "series_ids": (
                    sorted(fed_funds_history["series_id"].unique().tolist()) if not fed_funds_history.empty else []
                ),
                "observation_count": int(len(fed_funds_monthly)),
            },
            [] if fed_funds_value is not None else ["macro_series_unavailable"],
        )

        sofr_history = self._extract_series_history(
            macro,
            exact_ids=[self.macro_series.get("sofr")],
            contains_any=["sofr"],
        )
        sofr_value, _, sofr_monthly = latest_and_percentile(sofr_history, freq="M", periods=120)
        emit_metric(
            "macro.sofr",
            sofr_value,
            "percent",
            {"type": "asof", "length_days": 0},
            {
                "formula": "latest(sofr)",
                "series_ids": sorted(sofr_history["series_id"].unique().tolist()) if not sofr_history.empty else [],
                "observation_count": int(len(sofr_monthly)),
            },
            [] if sofr_value is not None else ["macro_series_unavailable"],
        )

        vix_history = self._extract_series_history(
            macro,
            exact_ids=[self.macro_series.get("vix")],
            contains_any=["vixcls", "vix"],
        )
        vix_value, _, vix_monthly = latest_and_percentile(vix_history, freq="M", periods=120)
        emit_metric(
            "market.vix",
            vix_value,
            "index",
            {"type": "asof", "length_days": 0},
            {
                "formula": "latest(vixcls)",
                "series_ids": sorted(vix_history["series_id"].unique().tolist()) if not vix_history.empty else [],
                "observation_count": int(len(vix_monthly)),
            },
            [] if vix_value is not None else ["macro_series_unavailable"],
        )

        gdp_history = self._extract_series_history(
            macro,
            exact_ids=[self.macro_series.get("real_gdp")],
            contains_any=["gdpc1", "real_gdp"],
        )
        gdp_quarterly = self._periodic_series(gdp_history, "Q")
        gdp_growth = pd.Series(dtype=float)
        if not gdp_quarterly.empty:
            gdp_growth = ((gdp_quarterly / gdp_quarterly.shift(4)) - 1.0).dropna().tail(40)
        gdp_growth_value = _safe_float(gdp_growth.iloc[-1]) if not gdp_growth.empty else None
        gdp_growth_pct = _percentile(gdp_growth) if not gdp_growth.empty else None
        emit_metric(
            "macro.real_gdp_growth_yoy",
            gdp_growth_value,
            "ratio",
            {"type": "lookback", "length_days": 365},
            {
                "formula": "(real_gdp_t / real_gdp_t_minus_4) - 1",
                "series_ids": sorted(gdp_history["series_id"].unique().tolist()) if not gdp_history.empty else [],
                "observation_count": int(len(gdp_growth)),
            },
            [] if gdp_growth_value is not None else ["macro_series_unavailable"],
        )
        emit_metric(
            "macro.real_gdp_growth_yoy_percentile_history",
            gdp_growth_pct,
            "percentile",
            {"type": "lookback", "length_days": 365 * 10},
            {
                "formula": "percentile_rank(current_real_gdp_growth_yoy, quarterly_real_gdp_growth_yoy_history_10y)",
                "observation_count": int(len(gdp_growth)),
                "window_start": str(gdp_growth.index[0]) if not gdp_growth.empty else None,
                "window_end": str(gdp_growth.index[-1]) if not gdp_growth.empty else None,
            },
            [] if gdp_growth_pct is not None else ["macro_history_unavailable"],
        )

        return features

    def _compute_operating(self, company_id: str, facts: pd.DataFrame, as_of: pd.Timestamp) -> Dict[str, FeatureRecord]:
        features: Dict[str, FeatureRecord] = {}
        _, market_aliases = self._resolve_entity_aliases(str(company_id))
        reference_row = self._fundamentals_reference_row(market_aliases)
        reference_instrument = _null_if_na(reference_row.get("Instrument"))
        provider_revenue = _safe_float(reference_row.get("Revenue"))
        provider_ebitda = _safe_float(reference_row.get("EBITDA"))
        provider_fcf = _safe_float(reference_row.get("Free Cash Flow"))
        provider_revenue_ref = self._fundamentals_reference_input_ref(reference_row, "Revenue", as_of)
        provider_ebitda_ref = self._fundamentals_reference_input_ref(reference_row, "EBITDA", as_of)
        provider_fcf_ref = self._fundamentals_reference_input_ref(reference_row, "Free Cash Flow", as_of)
        revenue_val, prov_r = self._latest_fact(facts, self.fact_map["revenue"])
        ebitda_val, prov_e = self._latest_fact(facts, self.fact_map["ebitda"])
        revenue_series, revenue_date_col = self._dated_fact_series(facts, self.fact_map["revenue"])
        ebitda_series, ebitda_date_col = self._dated_fact_series(facts, self.fact_map["ebitda"])
        latest_revenue_row = revenue_series.iloc[-1] if not revenue_series.empty else None
        latest_ebitda_row = None
        period_match_type = None
        period_distance_days = None
        ebitda_margin_flags: List[str] = []

        if latest_revenue_row is not None:
            latest_revenue_val = _safe_float(latest_revenue_row.get("fact_value"))
            if latest_revenue_val is not None:
                revenue_val = latest_revenue_val
            prov_r = self._fact_row_provenance(latest_revenue_row) or prov_r

        if latest_revenue_row is not None and not ebitda_series.empty:
            if revenue_date_col and ebitda_date_col:
                latest_revenue_date = latest_revenue_row.get(revenue_date_col)
                exact_matches = ebitda_series[ebitda_series[ebitda_date_col] == latest_revenue_date]
                if not exact_matches.empty:
                    latest_ebitda_row = exact_matches.iloc[-1]
                    period_match_type = "exact_period_match"
                    period_distance_days = 0
                else:
                    nearest_candidates = ebitda_series.copy()
                    nearest_candidates["date_distance_days"] = (
                        nearest_candidates[ebitda_date_col] - latest_revenue_date
                    ).abs().dt.days
                    nearest_candidates = nearest_candidates.sort_values(
                        ["date_distance_days", ebitda_date_col],
                        ascending=[True, True],
                    )
                    if not nearest_candidates.empty:
                        latest_ebitda_row = nearest_candidates.iloc[0]
                        period_distance_days = _safe_float(latest_ebitda_row.get("date_distance_days"))
                        if period_distance_days is not None and period_distance_days <= 45:
                            period_match_type = "nearest_period_match"
                            ebitda_margin_flags.append("ebitda_revenue_nearest_period_match")
                        else:
                            period_match_type = "latest_available_fallback"
                            ebitda_margin_flags.append("ebitda_revenue_period_mismatch")
            if latest_ebitda_row is None:
                latest_ebitda_row = ebitda_series.iloc[-1]
                period_match_type = "latest_available_fallback"
                if revenue_date_col and ebitda_date_col:
                    ebitda_margin_flags.append("ebitda_revenue_period_mismatch")
        elif not ebitda_series.empty:
            latest_ebitda_row = ebitda_series.iloc[-1]
            period_match_type = "latest_available_only"

        if latest_ebitda_row is not None:
            latest_ebitda_val = _safe_float(latest_ebitda_row.get("fact_value"))
            if latest_ebitda_val is not None:
                ebitda_val = latest_ebitda_val
            prov_e = self._fact_row_provenance(latest_ebitda_row) or prov_e

        ebitda_margin = None
        ebitda_margin_fallback = None
        ebitda_margin_provs: List[InputReference] = [InputReference(**p) for p in [prov_r, prov_e] if p]
        if revenue_val and revenue_val != 0 and ebitda_val is not None:
            ebitda_margin = ebitda_val / revenue_val
        elif (
            self.historical_backfill_mode
            and provider_revenue not in (None, 0)
            and provider_ebitda is not None
        ):
            ebitda_margin = provider_ebitda / provider_revenue
            ebitda_margin_fallback = "reference_ebitda_margin_fallback"
            period_match_type = "reference_ttm_fallback"
            period_distance_days = None
            if provider_revenue_ref is not None:
                ebitda_margin_provs.append(provider_revenue_ref)
            if provider_ebitda_ref is not None:
                ebitda_margin_provs.append(provider_ebitda_ref)
            ebitda_margin_flags.append("reference_ebitda_margin_fallback")
        if revenue_val in (None, 0):
            ebitda_margin_flags.append("revenue_unavailable_or_zero")
        if ebitda_val is None:
            ebitda_margin_flags.append("ebitda_unavailable")
        features["operating.ebitda_margin_ttm"] = FeatureRecord(
            name="operating.ebitda_margin_ttm",
            value=ebitda_margin,
            unit="ratio",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window={"type": "ttm", "length_days": 365},
            confidence=None,
            provenance=ebitda_margin_provs,
            missing_reason="unavailable" if ebitda_margin is None else None,
            fallback_used=ebitda_margin_fallback,
            component_breakdown={
                "revenue": revenue_val,
                "ebitda": ebitda_val,
                "reference_revenue": provider_revenue,
                "reference_ebitda": provider_ebitda,
                "reference_instrument": reference_instrument,
                "revenue_period": (
                    str(_null_if_na(latest_revenue_row.get(revenue_date_col)))
                    if latest_revenue_row is not None and revenue_date_col
                    else None
                ),
                "ebitda_period": (
                    str(_null_if_na(latest_ebitda_row.get(ebitda_date_col)))
                    if latest_ebitda_row is not None and ebitda_date_col
                    else None
                ),
                "period_match_type": period_match_type,
                "period_distance_days": period_distance_days,
                "formula": "ebitda / revenue",
            },
            quality_flags=ebitda_margin_flags or None,
        )

        operating_cash_flow, prov_ocf = self._latest_fact(facts, self.fact_map["operating_cash_flow"])
        capex_val, prov_capex = self._latest_fact(facts, self.fact_map["capex"])
        fcf_val = None
        prov_fcf = None
        fcf_fallback = None
        fcf_formula = None
        fcf_quality_flags: List[str] = []
        fcf_provs: List[InputReference] = []
        if operating_cash_flow is not None and capex_val is not None:
            fcf_val = float(operating_cash_flow) - float(capex_val)
            fcf_formula = "(operating_cash_flow - capex) / ebitda"
        else:
            fcf_val, prov_fcf = self._latest_fact(facts, self.fact_map["fcf"])
            if fcf_val is not None:
                fcf_fallback = "provider_fcf_field_fallback"
                fcf_formula = "provider_reported_fcf / ebitda"
        if operating_cash_flow is None:
            fcf_quality_flags.append("operating_cash_flow_missing")
        if capex_val is None:
            fcf_quality_flags.append("capex_missing")
        if ebitda_val in (None, 0):
            fcf_quality_flags.append("ebitda_unavailable_or_zero")
        if prov_fcf is not None and (operating_cash_flow is None or capex_val is None):
            fcf_quality_flags.append("provider_fcf_fallback")
        ebitda_for_fcf = ebitda_val
        fcf_conv = None
        if ebitda_for_fcf and ebitda_for_fcf != 0 and fcf_val is not None:
            fcf_conv = fcf_val / ebitda_for_fcf
        elif (
            self.historical_backfill_mode
            and provider_fcf is not None
            and provider_ebitda not in (None, 0)
        ):
            fcf_val = provider_fcf
            ebitda_for_fcf = provider_ebitda
            fcf_conv = provider_fcf / provider_ebitda
            fcf_fallback = "reference_fcf_conversion_fallback"
            fcf_formula = "reference_free_cash_flow / reference_ebitda"
            if provider_fcf_ref is not None:
                fcf_provs.append(provider_fcf_ref)
            if provider_ebitda_ref is not None:
                fcf_provs.append(provider_ebitda_ref)
            fcf_quality_flags.append("reference_fcf_conversion_fallback")
        fcf_provs = [InputReference(**p) for p in [prov_ocf, prov_capex, prov_fcf, prov_e] if p] + fcf_provs
        features["operating.fcf_conversion"] = FeatureRecord(
            name="operating.fcf_conversion",
            value=fcf_conv,
            unit="ratio",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window={"type": "ttm", "length_days": 365},
            confidence=None,
            provenance=fcf_provs,
            missing_reason="unavailable" if fcf_conv is None else None,
            fallback_used=fcf_fallback,
            component_breakdown={
                "operating_cash_flow": operating_cash_flow,
                "capex": capex_val,
                "free_cash_flow": fcf_val,
                "ebitda": ebitda_for_fcf,
                "reference_free_cash_flow": provider_fcf,
                "reference_ebitda": provider_ebitda,
                "reference_instrument": reference_instrument,
                "formula": fcf_formula,
            },
            quality_flags=fcf_quality_flags or None,
        )

        # True YoY growth should compare the latest quarter against the closest
        # observation roughly one year earlier, not just the immediately
        # preceding observation.
        rev_yoy = None
        rev_yoy_breakdown: Dict[str, Any] = {"formula": "latest_quarter_revenue / prior_year_same_quarter_revenue - 1"}
        rev_yoy_flags: List[str] = []
        rev_yoy_provs: List[Dict[str, Any]] = []
        revenue_statement_frame, _ = self._statement_metric_frame(facts, self.fact_map["revenue"])
        if not revenue_statement_frame.empty and "fiscal_quarter" in revenue_statement_frame.columns:
            normalized_revenue_frame = self._normalize_interim_single_quarter_values(revenue_statement_frame)
            latest_row, prior_row = self._latest_statement_interim_pair(normalized_revenue_frame)
            if latest_row is not None:
                latest_date = latest_row.get("period_date")
                target_date = latest_date - pd.Timedelta(days=365) if latest_date is not None else None
                latest_raw = _safe_float(latest_row.get("fact_value"))
                prior_raw = _safe_float(prior_row.get("fact_value")) if prior_row is not None else None
                r1 = _safe_float(latest_row.get("single_quarter_value"))
                r0 = _safe_float(prior_row.get("single_quarter_value")) if prior_row is not None else None
                latest_basis = _null_if_na(latest_row.get("single_quarter_basis"))
                prior_basis = _null_if_na(prior_row.get("single_quarter_basis")) if prior_row is not None else None
                rev_yoy_breakdown.update(
                    {
                        "latest_revenue": r1,
                        "prior_revenue": r0,
                        "latest_reported_value": latest_raw,
                        "prior_reported_value": prior_raw,
                        "latest_period": str(latest_date) if latest_date is not None else None,
                        "prior_period": str(prior_row.get("period_date")) if prior_row is not None else None,
                        "target_prior_period": str(target_date) if target_date is not None else None,
                        "matching_window_days": [270, 460],
                        "match_basis": "fiscal_quarter_period_end",
                        "latest_value_basis": latest_basis,
                        "prior_value_basis": prior_basis,
                    }
                )
                for prov in [self._fact_row_provenance(latest_row), self._fact_row_provenance(prior_row)]:
                    if prov and prov not in rev_yoy_provs:
                        rev_yoy_provs.append(prov)
                for row in [latest_row, prior_row]:
                    if row is None:
                        continue
                    flag = _null_if_na(row.get("single_quarter_quality_flag"))
                    if flag is not None and flag not in rev_yoy_flags:
                        rev_yoy_flags.append(str(flag))
                if latest_basis != prior_basis and prior_row is not None:
                    rev_yoy_flags.append("mixed_quarter_value_basis")
                if prior_row is None:
                    rev_yoy_flags.append("no_prior_year_quarter_match")
                elif r0 not in (None, 0):
                    rev_yoy = (r1 - r0) / r0 if r1 is not None else None
                else:
                    rev_yoy_flags.append("prior_revenue_unavailable_or_zero")
            else:
                rev_yoy_flags.append("revenue_quarter_history_unavailable")
        elif not revenue_series.empty:
            date_col = revenue_date_col
            if date_col:
                latest_row = revenue_series.iloc[-1]
                latest_date = latest_row[date_col]
                target_date = latest_date - pd.Timedelta(days=365)
                candidates = revenue_series[
                    (revenue_series[date_col] <= latest_date - pd.Timedelta(days=270))
                    & (revenue_series[date_col] >= latest_date - pd.Timedelta(days=460))
                ].copy()
                if not candidates.empty:
                    candidates["date_distance"] = (candidates[date_col] - target_date).abs()
                    prior_row = candidates.sort_values("date_distance").iloc[0]
                    r1 = _safe_float(latest_row.get("fact_value"))
                    r0 = _safe_float(prior_row.get("fact_value"))
                    rev_yoy_breakdown.update(
                        {
                            "latest_revenue": r1,
                            "prior_revenue": r0,
                            "latest_period": str(latest_date),
                            "prior_period": str(prior_row.get(date_col)),
                            "target_prior_period": str(target_date),
                            "matching_window_days": [270, 460],
                        }
                    )
                    for prov in [self._fact_row_provenance(latest_row), self._fact_row_provenance(prior_row)]:
                        if prov and prov not in rev_yoy_provs:
                            rev_yoy_provs.append(prov)
                    if r0 not in (None, 0):
                        rev_yoy = (r1 - r0) / r0 if r1 is not None else None
                    else:
                        rev_yoy_flags.append("prior_revenue_unavailable_or_zero")
                else:
                    rev_yoy_flags.append("no_prior_year_quarter_match")
            else:
                rev_yoy_flags.append("revenue_date_column_missing")
        else:
            rev_yoy_flags.append("revenue_series_unavailable")
        features["operating.revenue_yoy_last_q"] = FeatureRecord(
            name="operating.revenue_yoy_last_q",
            value=rev_yoy,
            unit="ratio",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window=None,
            confidence=None,
            provenance=[InputReference(**p) for p in rev_yoy_provs] or ([InputReference(**prov_r)] if prov_r else []),
            missing_reason="unavailable" if rev_yoy is None else None,
            fallback_used=None,
            component_breakdown=rev_yoy_breakdown,
            quality_flags=rev_yoy_flags or None,
        )

        # Margin trend + volatility (last 8 quarters if possible)
        margin_trend = None
        margin_vol = None
        if not revenue_series.empty:
            if not ebitda_series.empty and revenue_date_col and ebitda_date_col:
                rev = revenue_series[[revenue_date_col, "fact_value"]].copy()
                ebt = ebitda_series[[ebitda_date_col, "fact_value"]].copy()
                rev = rev.rename(columns={revenue_date_col: "period_date", "fact_value": "revenue"})
                ebt = ebt.rename(columns={ebitda_date_col: "period_date", "fact_value": "ebitda"})
                rev["period_date"] = pd.to_datetime(rev["period_date"], utc=True, errors="coerce").dt.normalize()
                ebt["period_date"] = pd.to_datetime(ebt["period_date"], utc=True, errors="coerce").dt.normalize()
                rev = rev.groupby("period_date", as_index=False).last()
                ebt = ebt.groupby("period_date", as_index=False).last()
                merged = pd.merge(rev, ebt, on="period_date", how="inner")
                merged = merged.dropna(subset=["revenue", "ebitda"])
                merged = merged.sort_values("period_date").tail(8)
                if len(merged) >= 3:
                    merged["margin"] = merged["ebitda"] / merged["revenue"].replace(0, np.nan)
                    m = merged["margin"].dropna()
                    if len(m) >= 3:
                        x = np.arange(len(m))
                        try:
                            slope = np.polyfit(x, m.values, 1)[0]
                            margin_trend = float(slope)
                            margin_vol = float(m.std(ddof=0))
                        except Exception:
                            margin_trend = None
                            margin_vol = None
        features["operating.ebitda_margin_trend_8q"] = FeatureRecord(
            name="operating.ebitda_margin_trend_8q",
            value=margin_trend,
            unit="slope",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window={"type": "lookback", "length_days": 365 * 2},
            confidence=None,
            provenance=[],
            missing_reason="unavailable" if margin_trend is None else None,
            fallback_used="heuristic" if margin_trend is not None else None,
        )
        features["operating.margin_volatility_8q"] = FeatureRecord(
            name="operating.margin_volatility_8q",
            value=margin_vol,
            unit="stddev",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window={"type": "lookback", "length_days": 365 * 2},
            confidence=None,
            provenance=[],
            missing_reason="unavailable" if margin_vol is None else None,
            fallback_used="heuristic" if margin_vol is not None else None,
        )

        # 3y revenue CAGR if enough history
        revenue_cagr_3y = None
        revenue_cagr_breakdown: Dict[str, Any] = {"formula": "(latest_revenue / prior_revenue) ** (1 / elapsed_years) - 1"}
        revenue_cagr_flags: List[str] = []
        revenue_cagr_provs: List[Dict[str, Any]] = []
        if not revenue_series.empty:
            date_col = revenue_date_col
            if date_col:
                if not revenue_series.empty:
                    latest_row = revenue_series.iloc[-1]
                    latest_date = latest_row[date_col]
                    target_date = latest_date - pd.Timedelta(days=int(365.25 * 3))
                    candidates = revenue_series[
                        (revenue_series[date_col] <= latest_date - pd.Timedelta(days=365 * 2))
                        & (revenue_series[date_col] >= latest_date - pd.Timedelta(days=365 * 4))
                    ].copy()
                    if not candidates.empty:
                        candidates["date_distance"] = (candidates[date_col] - target_date).abs()
                        prior_row = candidates.sort_values("date_distance").iloc[0]
                        r_start = _safe_float(prior_row.get("fact_value"))
                        r_end = _safe_float(latest_row.get("fact_value"))
                        elapsed_years = (latest_date - prior_row[date_col]).days / 365.25
                        revenue_cagr_breakdown.update(
                            {
                                "latest_revenue": r_end,
                                "prior_revenue": r_start,
                                "latest_period": str(latest_date),
                                "prior_period": str(prior_row.get(date_col)),
                                "elapsed_years": elapsed_years,
                                "target_prior_period": str(target_date),
                            }
                        )
                        for prov in [self._fact_row_provenance(latest_row), self._fact_row_provenance(prior_row)]:
                            if prov and prov not in revenue_cagr_provs:
                                revenue_cagr_provs.append(prov)
                        if r_start not in (None, 0) and r_end is not None and elapsed_years > 0:
                            revenue_cagr_3y = (r_end / r_start) ** (1.0 / elapsed_years) - 1.0
                        else:
                            revenue_cagr_flags.append("invalid_cagr_inputs")
                    else:
                        revenue_cagr_flags.append("no_prior_3y_match")
                else:
                    revenue_cagr_flags.append("revenue_series_empty_after_dating")
            else:
                revenue_cagr_flags.append("revenue_date_column_missing")
        else:
            revenue_cagr_flags.append("revenue_series_unavailable")
        features["operating.revenue_cagr_3y"] = FeatureRecord(
            name="operating.revenue_cagr_3y",
            value=revenue_cagr_3y,
            unit="ratio",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window={"type": "lookback", "length_days": 365 * 3},
            confidence=None,
            provenance=[InputReference(**p) for p in revenue_cagr_provs] or ([InputReference(**prov_r)] if prov_r else []),
            missing_reason="unavailable" if revenue_cagr_3y is None else None,
            fallback_used=None,
            component_breakdown=revenue_cagr_breakdown,
            quality_flags=revenue_cagr_flags or None,
        )

        return features

    def _compute_capital_return_context(
        self,
        facts: pd.DataFrame,
        as_of: pd.Timestamp,
        existing_features: Dict[str, FeatureRecord],
    ) -> Dict[str, FeatureRecord]:
        features: Dict[str, FeatureRecord] = {}

        share_series, share_date_col = self._dated_fact_series(
            facts,
            self.fact_map["shares_diluted"] + self.fact_map["shares_basic"],
        )
        share_count_trend = None
        share_count_trend_provenance: List[InputReference] = []
        share_count_trend_flags: List[str] = []
        share_count_trend_breakdown: Dict[str, Any] = {
            "formula": "annualized_share_count_change_from_fact_history",
        }
        if share_date_col is not None and not share_series.empty:
            series = share_series.copy()
            series["fact_value"] = pd.to_numeric(series.get("fact_value"), errors="coerce")
            series[share_date_col] = pd.to_datetime(series[share_date_col], utc=True, errors="coerce")
            series = series.dropna(subset=["fact_value", share_date_col])
            series = series[series["fact_value"] > 0]
            if not series.empty:
                latest_row = series.iloc[-1]
                latest_date = pd.to_datetime(latest_row[share_date_col], utc=True, errors="coerce")
                if latest_date is not None and not pd.isna(latest_date):
                    prior_cutoff = latest_date - pd.Timedelta(days=300)
                    prior_series = series[series[share_date_col] <= prior_cutoff]
                    if not prior_series.empty:
                        prior_row = prior_series.iloc[-1]
                        prior_date = pd.to_datetime(prior_row[share_date_col], utc=True, errors="coerce")
                        elapsed_years = max((latest_date - prior_date).days / 365.25, 0.0) if prior_date is not None and not pd.isna(prior_date) else 0.0
                        latest_value = _safe_float(latest_row.get("fact_value"))
                        prior_value = _safe_float(prior_row.get("fact_value"))
                        if latest_value not in (None, 0) and prior_value not in (None, 0) and elapsed_years > 0:
                            share_count_trend = float((latest_value / prior_value) ** (1.0 / elapsed_years) - 1.0)
                            for prov in [self._fact_row_provenance(latest_row), self._fact_row_provenance(prior_row)]:
                                if prov:
                                    ref = InputReference(**prov)
                                    if ref not in share_count_trend_provenance:
                                        share_count_trend_provenance.append(ref)
                            share_count_trend_breakdown.update(
                                {
                                    "latest_shares": latest_value,
                                    "latest_period_end": latest_date.isoformat(),
                                    "prior_shares": prior_value,
                                    "prior_period_end": prior_date.isoformat() if prior_date is not None and not pd.isna(prior_date) else None,
                                    "elapsed_years": elapsed_years,
                                }
                            )
                        else:
                            share_count_trend_flags.append("invalid_share_trend_inputs")
                    else:
                        share_count_trend_flags.append("no_prior_share_count_match")
                else:
                    share_count_trend_flags.append("share_date_unavailable")
            else:
                share_count_trend_flags.append("share_series_empty_after_filtering")
        else:
            share_count_trend_flags.append("share_series_unavailable")

        features["capital_return.share_count_trend"] = FeatureRecord(
            name="capital_return.share_count_trend",
            value=share_count_trend,
            unit="ratio",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window={"type": "lookback", "length_days": 730},
            confidence=0.65 if share_count_trend is not None else None,
            provenance=share_count_trend_provenance,
            missing_reason="unavailable" if share_count_trend is None else None,
            fallback_used=None,
            primary_source_basis="share_count_fact_history",
            component_breakdown=share_count_trend_breakdown,
            quality_flags=share_count_trend_flags or None,
        )

        market_cap_rec = existing_features.get("market.market_cap")
        available_rec = existing_features.get("liquidity.available_for_actions")
        net_leverage_rec = existing_features.get("capital_structure.net_leverage")
        fcf_rec = existing_features.get("operating.fcf_conversion")
        market_cap = _to_float(market_cap_rec.value if market_cap_rec is not None else None, None)
        available_for_actions = _to_float(available_rec.value if available_rec is not None else None, None)
        net_leverage = _to_float(net_leverage_rec.value if net_leverage_rec is not None else None, None)
        fcf_conversion = _to_float(fcf_rec.value if fcf_rec is not None else None, None)

        buyback_capacity_proxy = None
        buyback_capacity_breakdown: Dict[str, Any] = {
            "formula": "meaningful_liquidity_vs_market_cap_with_balance_sheet_guardrails",
        }
        buyback_capacity_flags: List[str] = []
        if market_cap not in (None, 0) and available_for_actions is not None and available_for_actions > 0:
            liquidity_vs_market_cap = float(available_for_actions) / float(market_cap)
            capacity_value = max(liquidity_vs_market_cap - 0.02, 0.0)
            if net_leverage is not None and net_leverage > 3.0:
                capacity_value = 0.0
                buyback_capacity_flags.append("leverage_guardrail_applied")
            if fcf_conversion is not None and fcf_conversion < 0.5:
                capacity_value = 0.0
                buyback_capacity_flags.append("fcf_guardrail_applied")
            buyback_capacity_proxy = float(min(capacity_value, 0.20))
            buyback_capacity_breakdown.update(
                {
                    "market_cap": market_cap,
                    "available_for_actions": available_for_actions,
                    "liquidity_vs_market_cap": liquidity_vs_market_cap,
                    "net_leverage": net_leverage,
                    "fcf_conversion": fcf_conversion,
                }
            )
        else:
            buyback_capacity_flags.append("buyback_capacity_inputs_unavailable")

        features["capital_return.buyback_capacity_proxy"] = FeatureRecord(
            name="capital_return.buyback_capacity_proxy",
            value=buyback_capacity_proxy,
            unit="ratio",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window={"type": "asof", "length_days": 0},
            confidence=0.55 if buyback_capacity_proxy is not None else None,
            provenance=[
                ref
                for ref in (
                    (market_cap_rec.provenance[0] if market_cap_rec and market_cap_rec.provenance else None),
                    (available_rec.provenance[0] if available_rec and available_rec.provenance else None),
                    (net_leverage_rec.provenance[0] if net_leverage_rec and net_leverage_rec.provenance else None),
                    (fcf_rec.provenance[0] if fcf_rec and fcf_rec.provenance else None),
                )
                if ref is not None
            ],
            missing_reason="unavailable" if buyback_capacity_proxy is None else None,
            fallback_used="heuristic" if buyback_capacity_proxy is not None else None,
            primary_source_basis="capital_return_capacity_house_formula",
            component_breakdown=buyback_capacity_breakdown,
            quality_flags=buyback_capacity_flags or None,
        )
        return features

    def _compute_ownership_governance(
        self,
        facts: pd.DataFrame,
        events: pd.DataFrame,
        ownership_summary: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> Dict[str, FeatureRecord]:
        features: Dict[str, FeatureRecord] = {}
        # Ownership concentration from extracted facts when available.
        top5, top5_prov = self._latest_fact(
            facts,
            [
                "ownership.top5_holder_pct",
                "ownership.top5_holders_pct",
                "ownership_governance.top5_holder_pct",
            ],
        )
        inst_pct, inst_prov = self._latest_fact(
            facts,
            [
                "ownership.institutional_pct",
                "ownership.institutional_ownership_pct",
                "ownership_governance.institutional_pct",
            ],
        )
        holder_count = None
        holder_count_prov = None
        if ownership_summary is not None and not ownership_summary.empty:
            own = ownership_summary.copy()
            for c in ("report_date", "filing_date", "effective_at", "published_at", "ingested_at"):
                if c in own.columns:
                    own[c] = pd.to_datetime(own[c], utc=True, errors="coerce")
            if "report_date" in own.columns and own["report_date"].notna().any():
                latest_report_date = own["report_date"].dropna().max()
                own = own[own["report_date"] == latest_report_date].copy()
            order_cols = [
                c
                for c in ("holder_count", "total_13f_shares", "filing_date", "published_at", "effective_at")
                if c in own.columns
            ]
            if order_cols:
                own = own.sort_values(order_cols, ascending=[False] * len(order_cols), na_position="last")
            row = own.iloc[0]
            holder_count_raw = _safe_float(row.get("holder_count"))
            if holder_count_raw is not None and holder_count_raw >= 0:
                holder_count = float(holder_count_raw)
                holder_count_prov = {
                    "artifact_type": "RawDocument",
                    "artifact_id": str(
                        _null_if_na(row.get("artifact_id"))
                        or f"wrds_13f:{_null_if_na(row.get('company_id'))}:{_null_if_na(row.get('report_date'))}"
                    ),
                    "source": str(_null_if_na(row.get("source_type")) or "wrds_13f"),
                    "published_at": str(_null_if_na(row.get("published_at")))
                    if _null_if_na(row.get("published_at")) is not None
                    else None,
                    "ingested_at": str(_null_if_na(row.get("ingested_at")))
                    if _null_if_na(row.get("ingested_at")) is not None
                    else None,
                    "hash": None,
                }
            total_13f_shares = _safe_float(row.get("total_13f_shares"))
            top5_13f_shares = _safe_float(row.get("top5_13f_shares"))
            if top5 is None and total_13f_shares not in (None, 0) and top5_13f_shares is not None:
                top5 = float(top5_13f_shares) / float(total_13f_shares)
                top5 = float(np.clip(top5, 0.0, 1.0))
                top5_prov = {
                    "artifact_type": "RawDocument",
                    "artifact_id": str(_null_if_na(row.get("artifact_id")) or f"wrds_13f:{_null_if_na(row.get('company_id'))}:{_null_if_na(row.get('report_date'))}"),
                    "source": str(_null_if_na(row.get("source_type")) or "wrds_13f"),
                    "published_at": str(_null_if_na(row.get("published_at"))) if _null_if_na(row.get("published_at")) is not None else None,
                    "ingested_at": str(_null_if_na(row.get("ingested_at"))) if _null_if_na(row.get("ingested_at")) is not None else None,
                    "hash": None,
                }
            if inst_pct is None and total_13f_shares is not None:
                shares_out, _ = self._latest_fact(facts, self.fact_map["shares_basic"])
                if shares_out in (None, 0):
                    shares_out, _ = self._latest_fact(facts, self.fact_map["shares_diluted"])
                if shares_out not in (None, 0):
                    inst_pct = float(total_13f_shares) / float(shares_out)
                    inst_pct = float(np.clip(inst_pct, 0.0, 2.0))
                    inst_prov = {
                        "artifact_type": "RawDocument",
                        "artifact_id": str(_null_if_na(row.get("artifact_id")) or f"wrds_13f:{_null_if_na(row.get('company_id'))}:{_null_if_na(row.get('report_date'))}"),
                        "source": str(_null_if_na(row.get("source_type")) or "wrds_13f"),
                        "published_at": str(_null_if_na(row.get("published_at"))) if _null_if_na(row.get("published_at")) is not None else None,
                        "ingested_at": str(_null_if_na(row.get("ingested_at"))) if _null_if_na(row.get("ingested_at")) is not None else None,
                        "hash": None,
                    }
        features["ownership_governance.top5_holder_pct"] = FeatureRecord(
            name="ownership_governance.top5_holder_pct",
            value=top5,
            unit="ratio",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window={"type": "asof", "length_days": 0},
            confidence=0.6 if top5 is not None else None,
            provenance=[InputReference(**top5_prov)] if top5_prov else [],
            missing_reason="not_disclosed" if top5 is None else None,
            fallback_used=None,
        )
        features["ownership_governance.institutional_pct"] = FeatureRecord(
            name="ownership_governance.institutional_pct",
            value=inst_pct,
            unit="ratio",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window={"type": "asof", "length_days": 0},
            confidence=0.6 if inst_pct is not None else None,
            provenance=[InputReference(**inst_prov)] if inst_prov else [],
            missing_reason="not_disclosed" if inst_pct is None else None,
            fallback_used=None,
        )
        features["ownership_governance.holder_count_13f"] = FeatureRecord(
            name="ownership_governance.holder_count_13f",
            value=holder_count,
            unit="count",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window={"type": "asof", "length_days": 0},
            confidence=0.55 if holder_count is not None else None,
            provenance=[InputReference(**holder_count_prov)] if holder_count_prov else [],
            missing_reason="not_disclosed" if holder_count is None else None,
            fallback_used=None,
        )

        crowding_components: List[float] = []
        if top5 is not None:
            crowding_components.append(float(np.clip((top5 - 0.35) / 0.35, 0.0, 1.0)))
        if inst_pct is not None:
            crowding_components.append(float(np.clip((inst_pct - 0.50) / 0.40, 0.0, 1.0)))
        if holder_count is not None:
            crowding_components.append(float(np.clip((25.0 - holder_count) / 25.0, 0.0, 1.0)))
        crowding_signal = float(np.mean(crowding_components)) if len(crowding_components) >= 2 else None
        crowding_provenance = [prov for prov in (top5_prov, inst_prov, holder_count_prov) if prov]
        features["ownership_governance.crowding_signal"] = FeatureRecord(
            name="ownership_governance.crowding_signal",
            value=crowding_signal,
            unit="score",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window={"type": "asof", "length_days": 0},
            confidence=0.55 if crowding_signal is not None else None,
            provenance=[InputReference(**prov) for prov in crowding_provenance],
            missing_reason="unavailable" if crowding_signal is None else None,
            fallback_used=None,
            primary_source_basis="ownership_summary_13f" if crowding_signal is not None else None,
            component_breakdown={
                "formula": "mean(clipped_top5_concentration, clipped_institutional_ownership, clipped_holder_count_scarcity)",
                "top5_holder_pct": top5,
                "institutional_pct": inst_pct,
                "holder_count_13f": holder_count,
            },
        )

        activist_score = None
        if events is not None and not events.empty:
            col = "event_type" if "event_type" in events.columns else ("action_type" if "action_type" in events.columns else None)
            sub_col = "event_subtype" if "event_subtype" in events.columns else None
            if col:
                s = events[col].astype(str).str.lower()
                score = 0.0
                if s.str.contains("activist|13d|proxy").any():
                    score += 0.6
                if s.str.contains("board|director|contest").any():
                    score += 0.3
                if sub_col and events[sub_col].astype(str).str.contains("activist|13d|proxy", case=False, na=False).any():
                    score += 0.2
                activist_score = float(min(1.0, score))
        features["ownership_governance.activist_signal"] = FeatureRecord(
            name="ownership_governance.activist_signal",
            value=activist_score,
            unit="score",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window={"type": "lookback", "length_days": 365},
            confidence=0.6 if activist_score is not None else None,
            provenance=[],
            missing_reason="unavailable" if activist_score is None else None,
            fallback_used=None,
        )
        activist_present = None if activist_score is None else bool(float(activist_score) > 0.0)
        features["ownership_governance.activist_presence_flag"] = _alias_feature_record(
            features["ownership_governance.activist_signal"],
            name="ownership_governance.activist_presence_flag",
            value=activist_present,
            unit="boolean",
            primary_source_basis="ownership_governance.activist_signal_alias",
            component_breakdown={
                "formula": "ownership_governance.activist_signal > 0",
                "source_feature": "ownership_governance.activist_signal",
                "source_value": activist_score,
            },
            extra_quality_flags=["policy_feature_alias"],
            missing_reason="unavailable" if activist_present is None else None,
        )
        return features

    def _compute_strategic(self, facts: pd.DataFrame, events: pd.DataFrame, as_of: pd.Timestamp) -> Dict[str, FeatureRecord]:
        features: Dict[str, FeatureRecord] = {}
        intent_scores = {
            "return_capital_priority": 0.0,
            "deleveraging_priority": 0.0,
            "pursue_mna_priority": 0.0,
            "focus_on_core": 0.0,
            "restructure": 0.0,
        }
        intent_fact_refs: List[InputReference] = []
        if facts is not None and not facts.empty and "fact_type" in facts.columns:
            fact_type_lower = facts["fact_type"].dropna().astype(str).str.lower()
            intent_mask = fact_type_lower.str.contains(
                "buyback|dividend|deleverag|leverage|acquisition|m&a|mna|core|portfolio|restruct|cost",
                case=False,
                na=False,
            )
            if intent_mask.any():
                matched = facts.loc[intent_mask.index[intent_mask]]
                intent_fact_refs = self._fact_refs(matched, limit=5)
            for ft in fact_type_lower.tolist():
                ftl = ft.lower()
                if "buyback" in ftl or "dividend" in ftl:
                    intent_scores["return_capital_priority"] += 1.0
                if "deleveraging" in ftl or "leverage" in ftl:
                    intent_scores["deleveraging_priority"] += 0.5
                if "acquisition" in ftl or "m&a" in ftl or "mna" in ftl:
                    intent_scores["pursue_mna_priority"] += 1.0
                if "core" in ftl or "portfolio" in ftl:
                    intent_scores["focus_on_core"] += 0.5
                if "restruct" in ftl or "cost" in ftl:
                    intent_scores["restructure"] += 0.5
        for k in intent_scores:
            score = min(1.0, intent_scores[k])
            features[f"strategic.intent.{k}"] = FeatureRecord(
                name=f"strategic.intent.{k}",
                value=score,
                unit="score",
                computed_at=_now_iso(),
                as_of_time=as_of.isoformat(),
                window={"type": "lookback", "length_days": 365},
                confidence=0.5 if score is not None else None,
                provenance=intent_fact_refs,
                missing_reason=None,
                fallback_used="heuristic",
            )

        # Recent actions / fatigue profile from event store.
        recent_count = None
        last_action_type = None
        freq_24m = None
        fatigue = None
        recent_action_refs: List[InputReference] = []
        strategic_action_breakdown: Dict[str, Any] = {
            "formula": "count(strategic_events_24m)",
            "lookback_days": 730,
            "strategic_event_universe_pattern": "acquisition|divestiture|buyback|debt|equity|loan|spin|restruct|recap|issuance",
        }
        strategic_action_flags: List[str] = []
        if events is not None and not events.empty:
            event_col = "event_type" if "event_type" in events.columns else None
            sub_col = "event_subtype" if "event_subtype" in events.columns else None
            time_col = self._event_time_col(events)
            if event_col and time_col:
                e = events.copy()
                e[time_col] = pd.to_datetime(e[time_col], utc=True, errors="coerce")
                cutoff = as_of - pd.Timedelta(days=365 * 2)
                e = e[e[time_col] >= cutoff]
                type_s = e[event_col].astype(str)
                sub_s = e[sub_col].astype(str) if sub_col else pd.Series("", index=e.index, dtype="object")
                strategic_mask = (
                    type_s.str.contains(
                        "acquisition|divestiture|buyback|debt|equity|loan|spin|restruct|recap|issuance",
                        case=False,
                        na=False,
                    )
                    | sub_s.str.contains(
                        "acquisition|divestiture|buyback|debt|equity|loan|spin|restruct|recap|issuance",
                        case=False,
                        na=False,
                    )
                )
                e = e[strategic_mask].copy()
                strategic_action_breakdown.update(
                    {
                        "strategic_event_count_24m": int(len(e)),
                        "window_start": str(cutoff),
                        "window_end": str(as_of),
                    }
                )
                if not e.empty:
                    e = e.sort_values(time_col, ascending=False)
                    recent_action_refs = self._event_refs(e, limit=5)
                    recent_count = float(len(e))
                    raw_last = _null_if_na(e.iloc[0][event_col])
                    if raw_last is None and sub_col:
                        raw_last = _null_if_na(e.iloc[0][sub_col])
                    last_action_type = str(raw_last) if raw_last is not None else None
                    freq_24m = recent_count / 24.0
                    fatigue = float(min(1.0, recent_count / 8.0))
                    type_counts = (
                        e[event_col]
                        .dropna()
                        .astype(str)
                        .value_counts()
                        .head(10)
                        .to_dict()
                    )
                    strategic_action_breakdown.update(
                        {
                            "latest_action_type": last_action_type,
                            "latest_action_time": str(_null_if_na(e.iloc[0].get(time_col))),
                            "action_frequency_per_month": freq_24m,
                            "top_action_type_counts": type_counts,
                        }
                    )
                else:
                    strategic_action_flags.append("no_strategic_actions_in_window")
            else:
                strategic_action_flags.append("strategic_event_schema_incomplete")
        else:
            strategic_action_flags.append("event_history_unavailable")

        features["strategic.recent_actions_count_24m"] = FeatureRecord(
            name="strategic.recent_actions_count_24m",
            value=recent_count,
            unit="count",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window={"type": "lookback", "length_days": 365 * 2},
            confidence=0.7 if recent_count is not None else None,
            provenance=recent_action_refs,
            missing_reason="unavailable" if recent_count is None else None,
            fallback_used=None,
            component_breakdown=dict(strategic_action_breakdown),
            quality_flags=strategic_action_flags or None,
        )
        features["strategic.last_action_type"] = FeatureRecord(
            name="strategic.last_action_type",
            value=last_action_type,
            unit="label",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window={"type": "lookback", "length_days": 365 * 2},
            confidence=0.7 if last_action_type is not None else None,
            provenance=recent_action_refs,
            missing_reason="unavailable" if last_action_type is None else None,
            fallback_used=None,
            component_breakdown={
                **dict(strategic_action_breakdown),
                "formula": "latest_action_type(strategic_events_24m)",
            },
            quality_flags=strategic_action_flags or None,
        )
        features["strategic.action_frequency_24m"] = FeatureRecord(
            name="strategic.action_frequency_24m",
            value=freq_24m,
            unit="count_per_month",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window={"type": "lookback", "length_days": 365 * 2},
            confidence=0.7 if freq_24m is not None else None,
            provenance=recent_action_refs,
            missing_reason="unavailable" if freq_24m is None else None,
            fallback_used=None,
            component_breakdown={
                **dict(strategic_action_breakdown),
                "formula": "count(strategic_events_24m) / 24",
            },
            quality_flags=strategic_action_flags or None,
        )
        features["strategic.action_fatigue_score"] = FeatureRecord(
            name="strategic.action_fatigue_score",
            value=fatigue,
            unit="score",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window={"type": "lookback", "length_days": 365 * 2},
            confidence=0.6 if fatigue is not None else None,
            provenance=recent_action_refs,
            missing_reason="unavailable" if fatigue is None else None,
            fallback_used="heuristic" if fatigue is not None else None,
        )

        dividend_payer_flag = None
        dividend_last_event_type = None
        dividend_refs: List[InputReference] = []
        dividend_breakdown: Dict[str, Any] = {
            "formula": "has_recurring_dividend_event_within_window",
            "active_window_days": 450,
        }
        dividend_flags: List[str] = []
        if events is not None and not events.empty:
            event_col = "event_type" if "event_type" in events.columns else ("action_type" if "action_type" in events.columns else None)
            sub_col = "event_subtype" if "event_subtype" in events.columns else ("action_subtype" if "action_subtype" in events.columns else None)
            time_col = self._event_time_col(events)
            if event_col and time_col:
                e = events.copy()
                e[time_col] = pd.to_datetime(e[time_col], utc=True, errors="coerce")
                e = e[e[time_col].notna()].copy()
                if not e.empty:
                    type_s = e[event_col].astype(str).str.lower()
                    sub_s = e[sub_col].astype(str).str.lower() if sub_col else pd.Series("", index=e.index, dtype="object")
                    dividend_related_mask = type_s.str.contains("dividend", case=False, na=False) | sub_s.str.contains(
                        "dividend|regular|special",
                        case=False,
                        na=False,
                    )
                    recurring_mask = (
                        type_s.isin({"dividend_regular", "dividend_increase", "dividend_cut", "dividend_initiate"})
                        | sub_s.isin({"regular", "dividend_increase", "dividend_cut", "dividend_initiate"})
                        | (
                            type_s.str.contains("dividend", case=False, na=False)
                            & ~type_s.str.contains("special", case=False, na=False)
                            & ~sub_s.str.contains("special", case=False, na=False)
                        )
                    )
                    recurring_events = e[recurring_mask].copy()
                    history_start = e[time_col].min()
                    history_end = e[time_col].max()
                    observable_history_days = None
                    if history_start is not None and not pd.isna(history_start):
                        observable_history_days = max(0, int((as_of - history_start).days))
                    dividend_breakdown.update(
                        {
                            "history_start": str(history_start) if history_start is not None and not pd.isna(history_start) else None,
                            "history_end": str(history_end) if history_end is not None and not pd.isna(history_end) else None,
                            "observable_history_days": observable_history_days,
                            "dividend_related_event_count": int(dividend_related_mask.sum()),
                            "recurring_event_count_total": int(len(recurring_events)),
                        }
                    )
                    if not recurring_events.empty:
                        recurring_events = recurring_events.sort_values(time_col, ascending=False)
                        dividend_refs = self._event_refs(recurring_events, limit=5)
                        latest = recurring_events.iloc[0]
                        latest_ts = pd.to_datetime(latest.get(time_col), utc=True, errors="coerce")
                        raw_last = _null_if_na(latest.get(event_col))
                        if raw_last is None and sub_col:
                            raw_last = _null_if_na(latest.get(sub_col))
                        dividend_last_event_type = str(raw_last) if raw_last is not None else None
                        recent_recurring_count = int((recurring_events[time_col] >= (as_of - pd.Timedelta(days=450))).sum())
                        recurring_count_24m = int((recurring_events[time_col] >= (as_of - pd.Timedelta(days=730))).sum())
                        dividend_breakdown.update(
                            {
                                "latest_recurring_event_time": str(latest_ts) if latest_ts is not None and not pd.isna(latest_ts) else None,
                                "latest_recurring_event_type": dividend_last_event_type,
                                "recurring_event_count_450d": recent_recurring_count,
                                "recurring_event_count_24m": recurring_count_24m,
                            }
                        )
                        dividend_payer_flag = bool(
                            latest_ts is not None
                            and not pd.isna(latest_ts)
                            and latest_ts >= (as_of - pd.Timedelta(days=450))
                        )
                        if not dividend_payer_flag:
                            dividend_flags.append("latest_recurring_dividend_outside_active_window")
                    else:
                        dividend_breakdown.update(
                            {
                                "latest_recurring_event_time": None,
                                "latest_recurring_event_type": None,
                                "recurring_event_count_450d": 0,
                                "recurring_event_count_24m": 0,
                            }
                        )
                        dividend_payer_flag = False
                        dividend_flags.append("no_recurring_dividend_events_in_history")
            else:
                dividend_flags.append("dividend_event_schema_incomplete")
        else:
            dividend_flags.append("event_history_unavailable")

        if dividend_payer_flag is None:
            dividend_fact_types = self.fact_map.get("common_dividends_cash", []) + self.fact_map.get("dividends_per_share_cash", [])
            dividend_series, dividend_date_col = self._dated_fact_series(facts, dividend_fact_types)
            if dividend_date_col is not None and not dividend_series.empty:
                dividend_series = dividend_series.copy()
                dividend_series["fact_value"] = pd.to_numeric(dividend_series.get("fact_value"), errors="coerce")
                dividend_series[dividend_date_col] = pd.to_datetime(dividend_series[dividend_date_col], utc=True, errors="coerce")
                dividend_series = dividend_series.dropna(subset=["fact_value", dividend_date_col])
                dividend_series = dividend_series[dividend_series["fact_value"] > 0]
                if not dividend_series.empty:
                    latest_row = dividend_series.iloc[-1]
                    latest_ts = pd.to_datetime(latest_row[dividend_date_col], utc=True, errors="coerce")
                    recent_count = int((dividend_series[dividend_date_col] >= (as_of - pd.Timedelta(days=730))).sum())
                    dividend_breakdown.update(
                        {
                            "dividend_fact_fallback_latest_period": latest_ts.isoformat() if latest_ts is not None and not pd.isna(latest_ts) else None,
                            "dividend_fact_fallback_recent_count_24m": recent_count,
                        }
                    )
                    if latest_ts is not None and not pd.isna(latest_ts):
                        dividend_payer_flag = bool(latest_ts >= (as_of - pd.Timedelta(days=450)))
                        if dividend_last_event_type is None:
                            dividend_last_event_type = "dividend_regular"
                        dividend_flags.append("dividend_fact_fallback")
                        prov = self._fact_row_provenance(latest_row)
                        if prov:
                            dividend_refs = [InputReference(**prov)]
                    else:
                        dividend_flags.append("dividend_fact_fallback_missing_date")
                else:
                    dividend_flags.append("dividend_fact_fallback_no_positive_values")

        features["capital_return.dividend_payer_flag"] = FeatureRecord(
            name="capital_return.dividend_payer_flag",
            value=dividend_payer_flag,
            unit="boolean",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window={"type": "lookback", "length_days": 730},
            confidence=(
                0.8
                if dividend_payer_flag is True
                else (0.6 if dividend_payer_flag is False else None)
            ),
            provenance=dividend_refs,
            missing_reason="unavailable" if dividend_payer_flag is None else None,
            fallback_used=None,
            component_breakdown=dividend_breakdown,
            quality_flags=dividend_flags or None,
        )
        features["capital_return.last_dividend_event_type"] = FeatureRecord(
            name="capital_return.last_dividend_event_type",
            value=dividend_last_event_type,
            unit="label",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window={"type": "lookback", "length_days": 730},
            confidence=0.75 if dividend_last_event_type is not None else None,
            provenance=dividend_refs,
            missing_reason="unavailable" if dividend_last_event_type is None else None,
            fallback_used=None,
            component_breakdown={
                **dict(dividend_breakdown),
                "formula": "latest_recurring_dividend_event_type",
            },
            quality_flags=dividend_flags or None,
        )
        return features

    def _compute_segment_portfolio_context(
        self,
        facts: pd.DataFrame,
        events: pd.DataFrame,
        as_of: pd.Timestamp,
        existing_features: Dict[str, FeatureRecord],
        taxonomy: TaxonomyContext,
    ) -> Dict[str, FeatureRecord]:
        features: Dict[str, FeatureRecord] = {}

        segment_count: Optional[int] = None
        segment_refs: List[str] = []
        segment_provenance: List[InputReference] = []
        segment_flags: List[str] = []
        segment_breakdown: Dict[str, Any] = {
            "formula": "archetype_floor_with_portfolio_event_overrides",
            "archetype": taxonomy.archetype,
        }

        archetype_segment_defaults = {
            "industrial_conglomerates": 3,
            "diversified_consumer_services": 2,
        }
        archetype_default = archetype_segment_defaults.get(str(taxonomy.archetype or "").strip().lower())
        if archetype_default is not None:
            segment_count = archetype_default
            segment_flags.append("archetype_multisegment_profile")
            segment_breakdown["archetype_segment_floor"] = archetype_default

        portfolio_event_count = 0
        if events is not None and not events.empty:
            event_col = "event_type" if "event_type" in events.columns else ("action_type" if "action_type" in events.columns else None)
            sub_col = "event_subtype" if "event_subtype" in events.columns else ("action_subtype" if "action_subtype" in events.columns else None)
            time_col = self._event_time_col(events)
            if event_col and time_col:
                e = events.copy()
                e[time_col] = pd.to_datetime(e[time_col], utc=True, errors="coerce")
                e = e[e[time_col].notna()].copy()
                e = e[e[time_col] >= (as_of - pd.Timedelta(days=365 * 5))].copy()
                if not e.empty:
                    type_s = e[event_col].astype(str).str.lower()
                    sub_s = e[sub_col].astype(str).str.lower() if sub_col else pd.Series("", index=e.index, dtype="object")
                    portfolio_mask = (
                        type_s.str.contains("divestiture|spin|carve|split[-_ ]?off|portfolio", case=False, na=False)
                        | sub_s.str.contains("divestiture|spin|carve|split[-_ ]?off|portfolio", case=False, na=False)
                    )
                    portfolio_events = e[portfolio_mask].copy()
                    portfolio_event_count = int(len(portfolio_events))
                    segment_breakdown["portfolio_separation_event_count_5y"] = portfolio_event_count
                    if not portfolio_events.empty:
                        portfolio_events = portfolio_events.sort_values(time_col, ascending=False)
                        segment_provenance = self._event_refs(portfolio_events, limit=5)
                        event_floor = 3 if portfolio_event_count >= 2 else 2
                        if segment_count is None or event_floor > segment_count:
                            segment_count = event_floor
                        segment_flags.append("portfolio_event_multisegment_inference")
                        segment_breakdown.update(
                            {
                                "portfolio_event_floor": event_floor,
                                "latest_portfolio_event_type": str(_null_if_na(portfolio_events.iloc[0].get(event_col)) or ""),
                                "latest_portfolio_event_time": str(_null_if_na(portfolio_events.iloc[0].get(time_col))),
                            }
                        )

        if segment_count is not None:
            segment_count = max(1, min(int(segment_count), 6))
            segment_refs = [f"segment_{idx + 1}" for idx in range(segment_count)]
            segment_breakdown["segment_references"] = list(segment_refs)
        else:
            segment_flags.append("segment_profile_unavailable")

        features["strategic.segment_count"] = FeatureRecord(
            name="strategic.segment_count",
            value=float(segment_count) if segment_count is not None else None,
            unit="count",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window={"type": "lookback", "length_days": 365 * 5},
            confidence=0.7 if segment_count is not None else None,
            provenance=segment_provenance,
            missing_reason="unavailable" if segment_count is None else None,
            fallback_used="heuristic" if segment_count is not None else None,
            primary_source_basis="segment_profile_house_inference",
            support_mode="inferred" if segment_count is not None else "unsupported",
            component_breakdown=dict(segment_breakdown),
            quality_flags=segment_flags or None,
        )
        features["strategic.segment_references"] = FeatureRecord(
            name="strategic.segment_references",
            value=segment_refs if segment_refs else None,
            unit="labels",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window={"type": "lookback", "length_days": 365 * 5},
            confidence=0.6 if segment_refs else None,
            provenance=segment_provenance,
            missing_reason="unavailable" if not segment_refs else None,
            fallback_used="heuristic" if segment_refs else None,
            primary_source_basis="segment_profile_house_inference",
            support_mode="inferred" if segment_refs else "unsupported",
            component_breakdown={
                "formula": "generic_segment_labels_from_inferred_segment_count",
                "segment_count": segment_count,
            },
            quality_flags=segment_flags or None,
        )

        margin_volatility_rec = existing_features.get("operating.margin_volatility_8q")
        margin_volatility = _to_float(margin_volatility_rec.value if margin_volatility_rec is not None else None, None)
        structural_divergence_floor = 0.15 if archetype_default is not None and (segment_count or 0) >= 2 else 0.0
        portfolio_divergence = float(np.clip(portfolio_event_count / 3.0, 0.0, 1.0)) * 0.6 if portfolio_event_count else 0.0
        margin_divergence_component = float(np.clip((margin_volatility or 0.0) / 0.05, 0.0, 1.0)) if margin_volatility is not None else None
        segment_margin_divergence = None
        margin_divergence_flags: List[str] = []
        margin_divergence_breakdown: Dict[str, Any] = {
            "formula": "max(structural_floor, 0.7 * margin_volatility_component + 0.3 * portfolio_component, portfolio_component)",
            "margin_volatility_8q": margin_volatility,
            "margin_volatility_component": margin_divergence_component,
            "portfolio_component": portfolio_divergence,
            "structural_floor": structural_divergence_floor,
            "segment_count": segment_count,
        }
        margin_divergence_provenance = list(segment_provenance)
        if margin_volatility_rec is not None and margin_volatility_rec.provenance:
            for ref in margin_volatility_rec.provenance:
                if ref not in margin_divergence_provenance:
                    margin_divergence_provenance.append(ref)
        if (segment_count or 0) >= 2:
            candidate_values = [structural_divergence_floor, portfolio_divergence]
            if margin_divergence_component is not None:
                candidate_values.append(0.7 * margin_divergence_component + 0.3 * portfolio_divergence)
            segment_margin_divergence = float(np.clip(max(candidate_values), 0.0, 1.0))
            if margin_volatility is None:
                margin_divergence_flags.append("margin_volatility_unavailable")
            if portfolio_event_count:
                margin_divergence_flags.append("portfolio_events_support_segment_divergence")
            if archetype_default is not None:
                margin_divergence_flags.append("structural_multisegment_floor_applied")

        features["operating.segment_margin_divergence"] = FeatureRecord(
            name="operating.segment_margin_divergence",
            value=segment_margin_divergence,
            unit="score_0_1",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window={"type": "lookback", "length_days": 365 * 5},
            confidence=0.55 if segment_margin_divergence is not None else None,
            provenance=margin_divergence_provenance,
            missing_reason="unavailable" if segment_margin_divergence is None else None,
            fallback_used="heuristic" if segment_margin_divergence is not None else None,
            primary_source_basis="segment_portfolio_divergence_house_formula",
            support_mode="inferred" if segment_margin_divergence is not None else "unsupported",
            component_breakdown=margin_divergence_breakdown,
            quality_flags=margin_divergence_flags or None,
        )

        ev_peer_z_rec = existing_features.get("market.ev_ebitda_vs_peer_z")
        market_share_rec = existing_features.get("peer_context.relative_positioning.market_share_percentile")
        ev_peer_z = _to_float(ev_peer_z_rec.value if ev_peer_z_rec is not None else None, None)
        market_share_pct = _to_float(market_share_rec.value if market_share_rec is not None else None, None)
        valuation_gap = float(np.clip((-ev_peer_z) / 1.5, 0.0, 1.0)) if ev_peer_z is not None else None
        subscale_gap = (
            float(np.clip((0.5 - market_share_pct) / 0.5, 0.0, 1.0))
            if market_share_pct is not None
            else None
        )
        conglomerate_discount_signal = None
        conglomerate_flags: List[str] = []
        conglomerate_breakdown: Dict[str, Any] = {
            "formula": "0.6 * valuation_gap + 0.25 * subscale_gap + 0.15 * segment_margin_divergence",
            "segment_count": segment_count,
            "ev_ebitda_vs_peer_z": ev_peer_z,
            "valuation_gap_component": valuation_gap,
            "market_share_percentile": market_share_pct,
            "subscale_gap_component": subscale_gap,
            "segment_margin_divergence": segment_margin_divergence,
        }
        conglomerate_provenance = list(margin_divergence_provenance)
        for rec in (ev_peer_z_rec, market_share_rec):
            if rec is not None and rec.provenance:
                for ref in rec.provenance:
                    if ref not in conglomerate_provenance:
                        conglomerate_provenance.append(ref)
        if (segment_count or 0) >= 2:
            if valuation_gap is not None or subscale_gap is not None:
                conglomerate_discount_signal = float(
                    np.clip(
                        0.6 * (valuation_gap or 0.0)
                        + 0.25 * (subscale_gap or 0.0)
                        + 0.15 * (segment_margin_divergence or 0.0),
                        0.0,
                        1.0,
                    )
                )
                if valuation_gap is None:
                    conglomerate_flags.append("peer_valuation_gap_unavailable")
                if subscale_gap is None:
                    conglomerate_flags.append("market_share_proxy_unavailable")
            else:
                conglomerate_flags.append("conglomerate_discount_inputs_unavailable")

        features["market.conglomerate_discount_signal"] = FeatureRecord(
            name="market.conglomerate_discount_signal",
            value=conglomerate_discount_signal,
            unit="score_0_1",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window={"type": "asof", "length_days": 0},
            confidence=0.55 if conglomerate_discount_signal is not None else None,
            provenance=conglomerate_provenance,
            missing_reason="unavailable" if conglomerate_discount_signal is None else None,
            fallback_used="heuristic" if conglomerate_discount_signal is not None else None,
            primary_source_basis="conglomerate_discount_house_formula",
            support_mode="inferred" if conglomerate_discount_signal is not None else "unsupported",
            component_breakdown=conglomerate_breakdown,
            quality_flags=conglomerate_flags or None,
        )
        return features

    def _compute_expectations_revisions(
        self,
        estimates: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> Dict[str, FeatureRecord]:
        features: Dict[str, FeatureRecord] = {}
        metric_map = {
            "eps": "eps",
            "revenue": "revenue",
            "ebitda": "ebitda",
        }

        if estimates is None or estimates.empty:
            for feature_name, unit in (
                ("expectations.analyst_coverage_count", "count"),
                ("expectations.revision_signal", "score_-1_1"),
                ("expectations.eps_consensus_fy1", "native"),
                ("expectations.revenue_consensus_fy1", "USD"),
                ("expectations.ebitda_consensus_fy1", "USD"),
                ("expectations.eps_revision_score_90d", "score_-1_1"),
                ("expectations.revenue_revision_score_90d", "score_-1_1"),
                ("expectations.ebitda_revision_score_90d", "score_-1_1"),
            ):
                features[feature_name] = FeatureRecord(
                    name=feature_name,
                    value=None,
                    unit=unit,
                    computed_at=_now_iso(),
                    as_of_time=as_of.isoformat(),
                    window={"type": "lookback", "length_days": 90},
                    confidence=None,
                    provenance=[],
                    missing_reason="unavailable",
                    fallback_used=None,
                    support_mode="unsupported",
                )
            return features

        df = estimates.copy()
        if "metric" in df.columns:
            df["metric"] = df["metric"].astype(str).str.lower()
        if "period" in df.columns:
            df["period"] = df["period"].astype(str).str.upper()
        if "available_time" in df.columns:
            df["available_time"] = pd.to_datetime(df["available_time"], utc=True, errors="coerce")
        if "period_end" in df.columns:
            df["period_end"] = pd.to_datetime(df["period_end"], utc=True, errors="coerce")
        if "consensus_value" in df.columns:
            df["consensus_value"] = pd.to_numeric(df["consensus_value"], errors="coerce")
        if "num_estimates" in df.columns:
            df["num_estimates"] = pd.to_numeric(df["num_estimates"], errors="coerce")
        if "revision_magnitude" in df.columns:
            df["revision_magnitude"] = pd.to_numeric(df["revision_magnitude"], errors="coerce")

        consensus_records: Dict[str, Dict[str, Any]] = {}
        revision_scores: List[float] = []
        analyst_coverage_count = None

        for metric, metric_label in metric_map.items():
            metric_df = df[df["metric"] == metric].copy() if "metric" in df.columns else pd.DataFrame()
            if metric_df.empty:
                consensus_records[metric] = {}
                continue
            period_mask = metric_df["period"].isin(["FY1", "NTM", "CY1"]) if "period" in metric_df.columns else pd.Series(True, index=metric_df.index)
            metric_df = metric_df[period_mask].copy()
            if metric_df.empty:
                consensus_records[metric] = {}
                continue
            sort_cols = [col for col in ("available_time", "period_end") if col in metric_df.columns]
            if sort_cols:
                metric_df = metric_df.sort_values(sort_cols, ascending=[False] * len(sort_cols))
            latest_row = metric_df.iloc[0]
            latest_consensus = _safe_float(latest_row.get("consensus_value"))
            latest_coverage = _safe_float(latest_row.get("num_estimates"))
            if latest_coverage is not None:
                analyst_coverage_count = max(analyst_coverage_count or 0.0, latest_coverage)
            latest_period = _null_if_na(latest_row.get("period"))
            latest_available = pd.to_datetime(latest_row.get("available_time"), utc=True, errors="coerce")
            window_start = latest_available - pd.Timedelta(days=90) if latest_available is not None and not pd.isna(latest_available) else None
            history = metric_df.copy()
            if latest_period is not None and "period" in history.columns:
                history = history[history["period"] == latest_period].copy()
            if window_start is not None and "available_time" in history.columns:
                history = history[history["available_time"] >= window_start].copy()
            if "available_time" in history.columns:
                history = history.sort_values("available_time", ascending=True)
            earliest_row = history.iloc[0] if not history.empty else latest_row
            earliest_consensus = _safe_float(earliest_row.get("consensus_value"))
            revision_score = None
            revision_breakdown: Dict[str, Any] = {
                "formula": "(latest_consensus - earliest_consensus) / max(abs(earliest_consensus), 1.0)",
                "metric": metric_label,
                "period": latest_period,
                "latest_consensus": latest_consensus,
                "earliest_consensus_90d": earliest_consensus,
                "history_points_90d": int(len(history)),
            }
            revision_flags: List[str] = []
            if latest_consensus is not None and earliest_consensus is not None:
                denom = max(abs(float(earliest_consensus)), 1.0)
                revision_score = float(np.clip((float(latest_consensus) - float(earliest_consensus)) / denom, -1.0, 1.0))
            else:
                direction = str(_null_if_na(latest_row.get("revision_direction")) or "").strip().lower()
                magnitude = _safe_float(latest_row.get("revision_magnitude"))
                if magnitude is not None and direction in {"up", "upgrade", "positive"}:
                    revision_score = float(np.clip(magnitude, 0.0, 1.0))
                    revision_flags.append("revision_direction_fallback")
                elif magnitude is not None and direction in {"down", "downgrade", "negative"}:
                    revision_score = float(np.clip(-magnitude, -1.0, 0.0))
                    revision_flags.append("revision_direction_fallback")
                else:
                    revision_flags.append("insufficient_revision_history")
            if revision_score is not None:
                revision_scores.append(revision_score)

            provenance: List[InputReference] = []
            for row in [latest_row, earliest_row]:
                artifact_id = _null_if_na(row.get("version_id")) or f"estimate:{metric}:{_null_if_na(row.get('available_time'))}"
                if artifact_id is None:
                    continue
                ref = InputReference(
                    artifact_type="RawDocument",
                    artifact_id=str(artifact_id),
                    source=str(_null_if_na(row.get("source_system")) or "warehouse_estimates"),
                    published_at=str(_null_if_na(row.get("available_time"))) if _null_if_na(row.get("available_time")) is not None else None,
                    ingested_at=str(_null_if_na(row.get("ingestion_time"))) if _null_if_na(row.get("ingestion_time")) is not None else None,
                    hash=str(_null_if_na(row.get("raw_payload_hash"))) if _null_if_na(row.get("raw_payload_hash")) is not None else None,
                )
                if ref not in provenance:
                    provenance.append(ref)

            consensus_records[metric] = {
                "consensus": latest_consensus,
                "coverage": latest_coverage,
                "provenance": provenance,
                "period": latest_period,
                "period_end": str(_null_if_na(latest_row.get("period_end"))) if _null_if_na(latest_row.get("period_end")) is not None else None,
                "revision_score": revision_score,
                "revision_breakdown": revision_breakdown,
                "revision_flags": revision_flags,
            }

        consensus_specs = [
            ("expectations.eps_consensus_fy1", "eps", "native"),
            ("expectations.revenue_consensus_fy1", "revenue", "USD"),
            ("expectations.ebitda_consensus_fy1", "ebitda", "USD"),
        ]
        for feature_name, metric, unit in consensus_specs:
            payload = consensus_records.get(metric, {})
            features[feature_name] = FeatureRecord(
                name=feature_name,
                value=payload.get("consensus"),
                unit=unit,
                computed_at=_now_iso(),
                as_of_time=as_of.isoformat(),
                window={"type": "lookforward", "length_days": 365},
                confidence=0.7 if payload.get("consensus") is not None else None,
                provenance=payload.get("provenance", []),
                missing_reason="unavailable" if payload.get("consensus") is None else None,
                fallback_used=None,
                primary_source_basis="warehouse_estimates_fy1_consensus",
                support_mode="exact" if payload.get("consensus") is not None else "unsupported",
                component_breakdown={
                    "metric": metric,
                    "period": payload.get("period"),
                    "period_end": payload.get("period_end"),
                    "analyst_coverage_count": payload.get("coverage"),
                },
            )

        revision_specs = [
            ("expectations.eps_revision_score_90d", "eps"),
            ("expectations.revenue_revision_score_90d", "revenue"),
            ("expectations.ebitda_revision_score_90d", "ebitda"),
        ]
        for feature_name, metric in revision_specs:
            payload = consensus_records.get(metric, {})
            features[feature_name] = FeatureRecord(
                name=feature_name,
                value=payload.get("revision_score"),
                unit="score_-1_1",
                computed_at=_now_iso(),
                as_of_time=as_of.isoformat(),
                window={"type": "lookback", "length_days": 90},
                confidence=0.65 if payload.get("revision_score") is not None else None,
                provenance=payload.get("provenance", []),
                missing_reason="unavailable" if payload.get("revision_score") is None else None,
                fallback_used=None,
                primary_source_basis="warehouse_estimates_revision_90d",
                support_mode="exact" if payload.get("revision_score") is not None else "unsupported",
                component_breakdown=payload.get("revision_breakdown"),
                quality_flags=payload.get("revision_flags") or None,
            )

        revision_signal = float(np.mean(revision_scores)) if revision_scores else None
        coverage_refs: List[InputReference] = []
        for metric in metric_map:
            for ref in consensus_records.get(metric, {}).get("provenance", []):
                if ref not in coverage_refs:
                    coverage_refs.append(ref)

        features["expectations.analyst_coverage_count"] = FeatureRecord(
            name="expectations.analyst_coverage_count",
            value=analyst_coverage_count,
            unit="count",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window={"type": "lookforward", "length_days": 365},
            confidence=0.7 if analyst_coverage_count is not None else None,
            provenance=coverage_refs,
            missing_reason="unavailable" if analyst_coverage_count is None else None,
            fallback_used=None,
            primary_source_basis="warehouse_estimates_num_estimates",
            support_mode="exact" if analyst_coverage_count is not None else "unsupported",
        )
        features["expectations.revision_signal"] = FeatureRecord(
            name="expectations.revision_signal",
            value=revision_signal,
            unit="score_-1_1",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window={"type": "lookback", "length_days": 90},
            confidence=0.65 if revision_signal is not None else None,
            provenance=coverage_refs,
            missing_reason="unavailable" if revision_signal is None else None,
            fallback_used=None,
            primary_source_basis="warehouse_estimates_revision_composite",
            support_mode="exact" if revision_signal is not None else "unsupported",
            component_breakdown={
                "formula": "mean(metric_revision_scores_90d)",
                "metric_revision_scores": {
                    metric: consensus_records.get(metric, {}).get("revision_score")
                    for metric in metric_map
                },
            },
        )
        return features

    def _compute_peer_context(self, company_id: str, peer_set: PeerSet, as_of: pd.Timestamp) -> Dict[str, FeatureRecord]:
        features: Dict[str, FeatureRecord] = {}
        peers = peer_set.members if peer_set and peer_set.members else []

        def band(pct: Optional[float]) -> Optional[str]:
            if pct is None:
                return None
            try:
                v = float(pct)
            except Exception:
                return None
            if v <= 25:
                return "q1"
            if v <= 50:
                return "q2"
            if v <= 75:
                return "q3"
            return "q4"

        # Peer actions rate: count peer events in last 12 months
        peer_actions_rate = None
        action_rate_pct = None
        action_rate_z = None
        action_rate_band = None
        consolidation_wave_score = None
        peer_action_refs: List[InputReference] = []
        peer_metric_refs: List[InputReference] = []
        entity_table_refs: List[InputReference] = []
        if peers:
            cutoff = pd.to_datetime(as_of - timedelta(days=365), utc=True)
            if self.cache_events and self._events_cache is not None:
                try:
                    df = self._events_cache.copy()
                    if "company_id" in df.columns:
                        df["company_id"] = df["company_id"].astype(str)
                        df = df[df["company_id"].isin(peers + [str(company_id)])]
                    date_col = None
                    for c in ["announced_at", "effective_at", "created_at"]:
                        if c in df.columns:
                            date_col = c
                            break
                    if date_col:
                        df[date_col] = pd.to_datetime(df[date_col], utc=True, errors="coerce")
                        df = df[df[date_col] >= cutoff]
                    cnt = df[df["company_id"].isin(peers)].shape[0]
                    peer_actions_rate = float(cnt) / max(1.0, float(len(peers)))
                    peer_action_refs = self._event_refs(df[df["company_id"].isin(peers)], limit=5)
                    if not df.empty:
                        counts = df.groupby("company_id").size()
                        if str(company_id) in counts.index:
                            rank = counts.rank(pct=True)
                            action_rate_pct = float(rank.loc[str(company_id)]) * 100.0
                            std = counts.std(ddof=0)
                            if std != 0 and not pd.isna(std):
                                action_rate_z = float((counts.loc[str(company_id)] - counts.mean()) / std)
                            action_rate_band = band(action_rate_pct)
                        # Consolidation wave proxy from peer M&A frequency + average deal size.
                        if "event_type" in df.columns:
                            ma = df[
                                df["event_type"].astype(str).str.contains(
                                    "acquisition|divestiture|m&a|mna",
                                    case=False,
                                    na=False,
                                )
                            ].copy()
                            ma_peers = ma[ma["company_id"].isin(peers)].copy()
                            if not ma_peers.empty:
                                freq = float(len(ma_peers)) / max(1.0, float(len(peers)))
                                freq_score = float(np.clip(freq / 1.0, 0.0, 1.0))
                                size_vals = []
                                if "params" in ma_peers.columns:
                                    for params in ma_peers["params"].head(2000):
                                        if isinstance(params, dict):
                                            for k in ["deal_value", "dealamount", "offering_amount"]:
                                                v = _safe_float(params.get(k))
                                                if v is not None and v > 0:
                                                    size_vals.append(v)
                                                    break
                                size_score = 0.0
                                if size_vals:
                                    # Treat >= $5B median deal size as saturated consolidation signal.
                                    size_score = float(np.clip(np.median(size_vals) / 5_000_000_000.0, 0.0, 1.0))
                                consolidation_wave_score = 0.7 * freq_score + 0.3 * size_score
                except Exception:
                    peer_actions_rate = None
            else:
                try:
                    con = duckdb.connect()
                    ids = ", ".join([f"'{p}'" for p in peers])
                    query = f"""
                    SELECT count(*) AS cnt
                    FROM read_parquet('{self.event_store_path.as_posix()}')
                    WHERE company_id IN ({ids})
                      AND coalesce(
                            try_cast(announced_at AS TIMESTAMP),
                            try_cast(effective_at AS TIMESTAMP),
                            try_cast(created_at AS TIMESTAMP)
                          ) >= TIMESTAMP '{_as_of_ts_literal(cutoff)}'
                    """
                    cnt = con.execute(query).fetchone()[0]
                    peer_actions_rate = float(cnt) / max(1.0, float(len(peers)))

                    # Company action rate vs peers
                    ids2 = ", ".join([f"'{p}'" for p in peers + [str(company_id)]])
                    query2 = f"""
                    SELECT company_id, count(*) AS cnt
                    FROM read_parquet('{self.event_store_path.as_posix()}')
                    WHERE company_id IN ({ids2})
                      AND coalesce(
                            try_cast(announced_at AS TIMESTAMP),
                            try_cast(effective_at AS TIMESTAMP),
                            try_cast(created_at AS TIMESTAMP)
                          ) >= TIMESTAMP '{_as_of_ts_literal(cutoff)}'
                    GROUP BY company_id
                    """
                    df = con.execute(query2).df()
                    if not df.empty:
                        df["cnt"] = pd.to_numeric(df["cnt"], errors="coerce")
                        df["company_id"] = df["company_id"].astype(str)
                        if str(company_id) in df["company_id"].values:
                            vals = df.set_index("company_id")["cnt"]
                            rank = vals.rank(pct=True)
                            action_rate_pct = float(rank.loc[str(company_id)]) * 100.0
                            std = vals.std(ddof=0)
                            if std != 0 and not pd.isna(std):
                                action_rate_z = float((vals.loc[str(company_id)] - vals.mean()) / std)
                            action_rate_band = band(action_rate_pct)
                    if peer_actions_rate is not None:
                        consolidation_wave_score = float(np.clip(peer_actions_rate / 1.0, 0.0, 1.0))
                    if not peer_action_refs:
                        query_refs = f"""
                        SELECT event_id, source_type, announced_at, created_at
                        FROM read_parquet('{self.event_store_path.as_posix()}')
                        WHERE company_id IN ({ids})
                          AND coalesce(
                                try_cast(announced_at AS TIMESTAMP),
                                try_cast(effective_at AS TIMESTAMP),
                                try_cast(created_at AS TIMESTAMP)
                              ) >= TIMESTAMP '{_as_of_ts_literal(cutoff)}'
                        LIMIT 5
                        """
                        ref_df = con.execute(query_refs).df()
                        peer_action_refs = self._event_refs(ref_df, limit=5)
                except Exception:
                    peer_actions_rate = None

        features["peer_context.peer_actions_rate"] = FeatureRecord(
            name="peer_context.peer_actions_rate",
            value=peer_actions_rate,
            unit="count_per_peer",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window={"type": "lookback", "length_days": 365},
            confidence=0.5 if peer_actions_rate is not None else None,
            provenance=peer_action_refs,
            missing_reason="unavailable" if peer_actions_rate is None else None,
            fallback_used=None,
        )
        features["peer_context.action_rate_percentile"] = FeatureRecord(
            name="peer_context.action_rate_percentile",
            value=action_rate_pct,
            unit="percentile",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window={"type": "lookback", "length_days": 365},
            confidence=0.5 if action_rate_pct is not None else None,
            provenance=peer_action_refs,
            missing_reason="unavailable" if action_rate_pct is None else None,
            fallback_used=None,
        )
        features["peer_context.action_rate_z"] = FeatureRecord(
            name="peer_context.action_rate_z",
            value=action_rate_z,
            unit="zscore",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window={"type": "lookback", "length_days": 365},
            confidence=0.5 if action_rate_z is not None else None,
            provenance=peer_action_refs,
            missing_reason="unavailable" if action_rate_z is None else None,
            fallback_used=None,
        )
        features["peer_context.action_rate_band"] = FeatureRecord(
            name="peer_context.action_rate_band",
            value=action_rate_band,
            unit="quartile",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window={"type": "lookback", "length_days": 365},
            confidence=0.5 if action_rate_band is not None else None,
            provenance=peer_action_refs,
            missing_reason="unavailable" if action_rate_band is None else None,
            fallback_used=None,
        )
        features["peer_context.consolidation_wave_score"] = FeatureRecord(
            name="peer_context.consolidation_wave_score",
            value=consolidation_wave_score,
            unit="score_0_1",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window={"type": "lookback", "length_days": 365},
            confidence=0.6 if consolidation_wave_score is not None else None,
            provenance=peer_action_refs,
            missing_reason="unavailable" if consolidation_wave_score is None else None,
            fallback_used="heuristic" if consolidation_wave_score is not None else None,
        )

        # Peer percentiles (entity table if available, otherwise compute from peer facts)
        valuation_pct = None
        leverage_pct = None
        margin_pct = None
        valuation_z = None
        leverage_z = None
        margin_z = None
        ev_ebitda_peer_z = None
        fcf_yield_peer_pct = None
        revenue_peer_pct = None
        peer_df = None
        entity_table = self._load_entity_table()
        if entity_table is not None and not entity_table.empty and peers:
            df = entity_table.copy()
            if "entity_id" in df.columns:
                df["entity_id"] = df["entity_id"].astype(str)
            peer_df = df[df["entity_id"].astype(str).isin(peers + [str(company_id)])]
            entity_table_refs = [
                InputReference(
                    artifact_type="RawDocument",
                    artifact_id=f"entity_table:{self.entity_table_path.name}",
                    source="entity_table",
                    published_at=None,
                    ingested_at=None,
                    hash=None,
                )
            ]

            def percentile_for(col: str) -> Optional[float]:
                if col not in peer_df.columns:
                    return None
                vals = pd.to_numeric(peer_df[col], errors="coerce").dropna()
                if vals.empty:
                    return None
                company_val = pd.to_numeric(
                    peer_df[peer_df["entity_id"].astype(str) == str(company_id)][col],
                    errors="coerce",
                ).dropna()
                if company_val.empty:
                    return None
                rank = vals.rank(pct=True)
                target = company_val.iloc[0]
                nearest_idx = (vals - target).abs().idxmin()
                return float(rank.loc[nearest_idx]) * 100.0

            def z_for(col: str) -> Optional[float]:
                if col not in peer_df.columns:
                    return None
                vals = pd.to_numeric(peer_df[col], errors="coerce").dropna()
                if vals.empty:
                    return None
                target = pd.to_numeric(
                    peer_df[peer_df["entity_id"].astype(str) == str(company_id)][col],
                    errors="coerce",
                ).dropna()
                if target.empty:
                    return None
                mean = vals.mean()
                std = vals.std(ddof=0)
                if std == 0 or pd.isna(std):
                    return None
                return float((target.iloc[0] - mean) / std)

            valuation_pct = percentile_for("ev_ebitda") or percentile_for("pe") or percentile_for("fcf_yield")
            leverage_pct = percentile_for("net_leverage") or percentile_for("gross_leverage")
            margin_pct = percentile_for("ebitda_margin") or percentile_for("margin") or percentile_for("operating_margin")
            revenue_peer_pct = percentile_for("revenue") or percentile_for("sales") or percentile_for("market_cap")
            valuation_z = z_for("ev_ebitda") or z_for("pe") or z_for("fcf_yield")
            leverage_z = z_for("net_leverage") or z_for("gross_leverage")
            margin_z = z_for("ebitda_margin") or z_for("margin") or z_for("operating_margin")
            ev_ebitda_peer_z = z_for("ev_ebitda")
            fcf_yield_peer_pct = percentile_for("fcf_yield")

        if peers and (valuation_pct is None or leverage_pct is None or margin_pct is None or leverage_z is None or margin_z is None or valuation_z is None or revenue_peer_pct is None):
            # Fallback: compute percentiles from peer facts
            try:
                con = duckdb.connect()
                ids = ", ".join([f"'{p}'" for p in peers + [str(company_id)]])
                query = f"""
                SELECT entity_id, fact_id, source_type, published_at, ingested_at, fact_type, fact_value
                FROM read_parquet('{self.facts_path.as_posix()}', union_by_name=True)
                WHERE entity_id IN ({ids})
                  AND fact_type IN ('financial.ebitda', 'financial.revenue', 'financial.total_debt', 'financial.debt_total', 'financial.cash')
                """
                df = con.execute(query).df()
                if not df.empty:
                    peer_metric_refs = self._fact_refs(df, limit=5)
                    df["fact_value"] = pd.to_numeric(df["fact_value"], errors="coerce")
                    ebitda = df[df["fact_type"] == "financial.ebitda"].groupby("entity_id")["fact_value"].last()
                    revenue = df[df["fact_type"] == "financial.revenue"].groupby("entity_id")["fact_value"].last()
                    debt = df[df["fact_type"].isin(["financial.total_debt", "financial.debt_total"])].groupby("entity_id")["fact_value"].last()
                    cash = df[df["fact_type"] == "financial.cash"].groupby("entity_id")["fact_value"].last()
                    margin = (ebitda / revenue.replace(0, np.nan)).dropna()
                    leverage = ((debt - cash) / ebitda.replace(0, np.nan)).dropna()

                    def pct(series: pd.Series) -> Optional[float]:
                        if series is None or series.empty:
                            return None
                        vals = series.dropna()
                        if vals.empty:
                            return None
                        if str(company_id) not in vals.index:
                            return None
                        rank = vals.rank(pct=True)
                        return float(rank.loc[str(company_id)]) * 100.0

                    def zscore(series: pd.Series) -> Optional[float]:
                        if series is None or series.empty:
                            return None
                        vals = series.dropna()
                        if vals.empty or str(company_id) not in vals.index:
                            return None
                        std = vals.std(ddof=0)
                        if std == 0 or pd.isna(std):
                            return None
                        return float((vals.loc[str(company_id)] - vals.mean()) / std)

                    if margin_pct is None:
                        margin_pct = pct(margin)
                    if leverage_pct is None:
                        leverage_pct = pct(leverage)
                    if revenue_peer_pct is None:
                        revenue_peer_pct = pct(revenue)
                    if margin_z is None:
                        margin_z = zscore(margin)
                    if leverage_z is None:
                        leverage_z = zscore(leverage)

                    # valuation from EV/EBITDA using market cap if available in entity table
                    if (valuation_pct is None or valuation_z is None) and peer_df is not None:
                        size_col = None
                        for c in ["market_cap", "mkt_cap", "marketcap", "mktcap", "market_capitalization"]:
                            if c in peer_df.columns:
                                size_col = c
                                break
                        if size_col is not None:
                            mcap = pd.to_numeric(peer_df.set_index("entity_id")[size_col], errors="coerce")
                            ev_ebitda = (mcap + debt - cash) / ebitda.replace(0, np.nan)
                            if valuation_pct is None:
                                valuation_pct = pct(ev_ebitda)
                            if valuation_z is None:
                                valuation_z = zscore(ev_ebitda)
                            if ev_ebitda_peer_z is None:
                                ev_ebitda_peer_z = zscore(ev_ebitda)

                    # valuation from market cap time-series if still missing
                    if (valuation_pct is None or valuation_z is None):
                        try:
                            info = con.execute(
                                f"DESCRIBE SELECT * FROM read_parquet('{self.raw_timeseries_path.as_posix()}')"
                            ).df()
                            cols = set(info["column_name"].tolist())
                            id_col = "entity_id" if "entity_id" in cols else ("company_id" if "company_id" in cols else None)
                            time_col = None
                            for c in ["observation_time", "effective_at", "published_at", "date", "as_of_date"]:
                                if c in cols:
                                    time_col = c
                                    break
                            series_cols = [c for c in ["series_id", "field_name"] if c in cols]
                            if id_col and time_col and series_cols and "value" in cols:
                                series_filter = " OR ".join([f"lower({c}) like '%market_cap%' or lower({c}) like '%mktcap%'" for c in series_cols])
                                type_filter = ""
                                if "series_type" in cols:
                                    type_filter = "AND series_type = 'price'"
                                ids_all = ", ".join([f"'{p}'" for p in peers + [str(company_id)]])
                                query_ts = f"""
                                SELECT {id_col} AS entity_id, {time_col} AS obs_time, value
                                FROM read_parquet('{self.raw_timeseries_path.as_posix()}')
                                WHERE {id_col} IN ({ids_all})
                                  {type_filter}
                                  AND ({series_filter})
                                  AND {time_col} <= TIMESTAMPTZ '{as_of.isoformat()}'
                                """
                                ts_df = con.execute(query_ts).df()
                                if not ts_df.empty:
                                    ts_df = ts_df.dropna(subset=["value"])
                                    ts_df = ts_df.sort_values("obs_time").groupby("entity_id").tail(1)
                                    mcap_ts = pd.to_numeric(ts_df.set_index("entity_id")["value"], errors="coerce")
                                    ev_ebitda = (mcap_ts + debt - cash) / ebitda.replace(0, np.nan)
                                    if valuation_pct is None:
                                        valuation_pct = pct(ev_ebitda)
                                    if valuation_z is None:
                                        valuation_z = zscore(ev_ebitda)
                                    if ev_ebitda_peer_z is None:
                                        ev_ebitda_peer_z = zscore(ev_ebitda)
                        except Exception:
                            pass
            except Exception:
                pass

        peer_metric_provenance = peer_metric_refs if peer_metric_refs else entity_table_refs

        features["peer_context.valuation_percentile"] = FeatureRecord(
            name="peer_context.valuation_percentile",
            value=valuation_pct,
            unit="percentile",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window={"type": "asof", "length_days": 0},
            confidence=0.6 if valuation_pct is not None else None,
            provenance=peer_metric_provenance,
            missing_reason="unavailable" if valuation_pct is None else None,
            fallback_used=None,
        )
        features["peer_context.leverage_percentile"] = FeatureRecord(
            name="peer_context.leverage_percentile",
            value=leverage_pct,
            unit="percentile",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window={"type": "asof", "length_days": 0},
            confidence=0.6 if leverage_pct is not None else None,
            provenance=peer_metric_provenance,
            missing_reason="unavailable" if leverage_pct is None else None,
            fallback_used=None,
        )
        features["peer_context.margin_percentile"] = FeatureRecord(
            name="peer_context.margin_percentile",
            value=margin_pct,
            unit="percentile",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window={"type": "asof", "length_days": 0},
            confidence=0.6 if margin_pct is not None else None,
            provenance=peer_metric_provenance,
            missing_reason="unavailable" if margin_pct is None else None,
            fallback_used=None,
        )
        features["peer_context.valuation_z"] = FeatureRecord(
            name="peer_context.valuation_z",
            value=valuation_z,
            unit="zscore",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window={"type": "asof", "length_days": 0},
            confidence=0.6 if valuation_z is not None else None,
            provenance=peer_metric_provenance,
            missing_reason="unavailable" if valuation_z is None else None,
            fallback_used=None,
        )
        features["peer_context.leverage_z"] = FeatureRecord(
            name="peer_context.leverage_z",
            value=leverage_z,
            unit="zscore",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window={"type": "asof", "length_days": 0},
            confidence=0.6 if leverage_z is not None else None,
            provenance=peer_metric_provenance,
            missing_reason="unavailable" if leverage_z is None else None,
            fallback_used=None,
        )
        features["peer_context.margin_z"] = FeatureRecord(
            name="peer_context.margin_z",
            value=margin_z,
            unit="zscore",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window={"type": "asof", "length_days": 0},
            confidence=0.6 if margin_z is not None else None,
            provenance=peer_metric_provenance,
            missing_reason="unavailable" if margin_z is None else None,
            fallback_used=None,
        )
        features["peer_context.valuation_band"] = FeatureRecord(
            name="peer_context.valuation_band",
            value=band(valuation_pct),
            unit="quartile",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window={"type": "asof", "length_days": 0},
            confidence=0.6 if valuation_pct is not None else None,
            provenance=peer_metric_provenance,
            missing_reason="unavailable" if valuation_pct is None else None,
            fallback_used=None,
        )
        features["peer_context.leverage_band"] = FeatureRecord(
            name="peer_context.leverage_band",
            value=band(leverage_pct),
            unit="quartile",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window={"type": "asof", "length_days": 0},
            confidence=0.6 if leverage_pct is not None else None,
            provenance=peer_metric_provenance,
            missing_reason="unavailable" if leverage_pct is None else None,
            fallback_used=None,
        )
        features["peer_context.margin_band"] = FeatureRecord(
            name="peer_context.margin_band",
            value=band(margin_pct),
            unit="quartile",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window={"type": "asof", "length_days": 0},
            confidence=0.6 if margin_pct is not None else None,
            provenance=peer_metric_provenance,
            missing_reason="unavailable" if margin_pct is None else None,
            fallback_used=None,
        )

        features["market.ev_ebitda_vs_peer_z"] = FeatureRecord(
            name="market.ev_ebitda_vs_peer_z",
            value=ev_ebitda_peer_z,
            unit="zscore",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window={"type": "asof", "length_days": 0},
            confidence=0.6 if ev_ebitda_peer_z is not None else None,
            provenance=peer_metric_provenance,
            missing_reason="unavailable" if ev_ebitda_peer_z is None else None,
            fallback_used=None,
            primary_source_basis="peer_relative_ev_ebitda",
            component_breakdown={
                "formula": "zscore(ev_ebitda, peer_set)",
                "peer_set_size": len(peers) + 1 if peers else 0,
            },
            quality_flags=["policy_feature_alias"],
        )
        features["market.fcf_yield_percentile_peers"] = FeatureRecord(
            name="market.fcf_yield_percentile_peers",
            value=(fcf_yield_peer_pct / 100.0) if fcf_yield_peer_pct is not None else None,
            unit="percentile_0_1",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window={"type": "asof", "length_days": 0},
            confidence=0.6 if fcf_yield_peer_pct is not None else None,
            provenance=peer_metric_provenance,
            missing_reason="unavailable" if fcf_yield_peer_pct is None else None,
            fallback_used=None,
            primary_source_basis="peer_relative_fcf_yield",
            component_breakdown={
                "formula": "percentile(fcf_yield, peer_set) / 100",
                "peer_set_size": len(peers) + 1 if peers else 0,
            },
            quality_flags=["policy_feature_alias"],
        )
        features["operating.ebitda_margin_percentile_peers"] = FeatureRecord(
            name="operating.ebitda_margin_percentile_peers",
            value=(margin_pct / 100.0) if margin_pct is not None else None,
            unit="percentile_0_1",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window={"type": "asof", "length_days": 0},
            confidence=0.6 if margin_pct is not None else None,
            provenance=peer_metric_provenance,
            missing_reason="unavailable" if margin_pct is None else None,
            fallback_used=None,
            primary_source_basis="peer_context.margin_percentile_alias",
            component_breakdown={
                "formula": "peer_context.margin_percentile / 100",
                "source_feature": "peer_context.margin_percentile",
                "source_percentile_0_100": margin_pct,
            },
            quality_flags=["policy_feature_alias"],
        )
        features["peer_context.relative_positioning.market_share_percentile"] = FeatureRecord(
            name="peer_context.relative_positioning.market_share_percentile",
            value=(revenue_peer_pct / 100.0) if revenue_peer_pct is not None else None,
            unit="percentile_0_1",
            computed_at=_now_iso(),
            as_of_time=as_of.isoformat(),
            window={"type": "asof", "length_days": 0},
            confidence=0.55 if revenue_peer_pct is not None else None,
            provenance=peer_metric_provenance,
            missing_reason="unavailable" if revenue_peer_pct is None else None,
            fallback_used="heuristic" if revenue_peer_pct is not None else None,
            primary_source_basis="peer_relative_revenue_scale_proxy",
            component_breakdown={
                "formula": "percentile(revenue_or_market_cap_scale, peer_set) / 100",
                "peer_set_size": len(peers) + 1 if peers else 0,
                "source_percentile_0_100": revenue_peer_pct,
            },
            quality_flags=["policy_feature_alias", "scale_as_market_share_proxy"],
        )
        return features
    def _compute_regime(self, macro: pd.DataFrame, as_of: pd.Timestamp) -> Dict[str, Any]:
        return self.regime_classifier.classify(macro, as_of)

    def _build_constraint_set(self, facts: pd.DataFrame, as_of: pd.Timestamp) -> Dict[str, List[ConstraintObject]]:
        hard: List[ConstraintObject] = []
        soft: List[ConstraintObject] = []
        if facts is None or facts.empty:
            return {"hard": hard, "soft": soft}
        for _, row in facts.iterrows():
            ft = str(row.get("fact_type", "")).lower()
            if any(k in ft for k in ["constraint", "no_equity", "maintain_ig", "leverage_target"]):
                obj = ConstraintObject(
                    name=row.get("fact_type"),
                    value=row.get("fact_value"),
                    hardness="hard",
                    confidence=_safe_float(row.get("confidence_score")),
                    valid_from=str(row.get("valid_from")) if row.get("valid_from") is not None else None,
                    valid_to=str(row.get("valid_to")) if row.get("valid_to") is not None else None,
                    evidence=[
                        InputReference(
                            artifact_type="ExtractedFact",
                            artifact_id=str(row.get("fact_id")),
                            source=str(row.get("source_type")) if row.get("source_type") is not None else None,
                            published_at=str(row.get("published_at")) if row.get("published_at") is not None else None,
                            ingested_at=str(row.get("ingested_at")) if row.get("ingested_at") is not None else None,
                            hash=None,
                        )
                    ],
                )
                hard.append(obj)
            elif any(k in ft for k in ["intent", "preference", "strategy", "focus"]):
                obj = ConstraintObject(
                    name=row.get("fact_type"),
                    value=row.get("fact_value"),
                    hardness="soft",
                    confidence=_safe_float(row.get("confidence_score")),
                    valid_from=str(row.get("valid_from")) if row.get("valid_from") is not None else None,
                    valid_to=str(row.get("valid_to")) if row.get("valid_to") is not None else None,
                    evidence=[
                        InputReference(
                            artifact_type="ExtractedFact",
                            artifact_id=str(row.get("fact_id")),
                            source=str(row.get("source_type")) if row.get("source_type") is not None else None,
                            published_at=str(row.get("published_at")) if row.get("published_at") is not None else None,
                            ingested_at=str(row.get("ingested_at")) if row.get("ingested_at") is not None else None,
                            hash=None,
                        )
                    ],
                )
                soft.append(obj)
        return {"hard": hard, "soft": soft}

    def _augment_constraint_set_with_feature_proxies(
        self,
        constraints: Dict[str, List[ConstraintObject]],
        features: Dict[str, FeatureRecord],
        as_of: pd.Timestamp,
    ) -> Dict[str, List[ConstraintObject]]:
        hard = list(constraints.get("hard", []))
        soft = list(constraints.get("soft", []))
        feature_names = [
            "capital_structure.max_leverage_ratio_covenant_proxy",
            "capital_structure.min_interest_coverage_ratio_covenant_proxy",
            "capital_structure.min_fixed_charge_coverage_ratio_covenant_proxy",
            "capital_structure.min_current_ratio_covenant_proxy",
        ]

        for feature_name in feature_names:
            feat = features.get(feature_name)
            if not isinstance(feat, FeatureRecord):
                continue
            if feat.value is None:
                continue
            if "dealscan_covenant_proxy" not in list(feat.quality_flags or []):
                continue

            evidence = list(feat.provenance or [])
            valid_from = None
            for ref in evidence:
                ts = pd.to_datetime(getattr(ref, "published_at", None), utc=True, errors="coerce")
                if pd.isna(ts):
                    ts = pd.to_datetime(getattr(ref, "ingested_at", None), utc=True, errors="coerce")
                if pd.isna(ts):
                    continue
                ts_iso = ts.isoformat()
                if valid_from is None or ts_iso < valid_from:
                    valid_from = ts_iso
            if valid_from is None:
                valid_from = as_of.isoformat()

            hard.append(
                ConstraintObject(
                    name=feature_name,
                    value=feat.value,
                    hardness="hard",
                    confidence=feat.confidence,
                    valid_from=valid_from,
                    valid_to=None,
                    evidence=evidence,
                )
            )
        return {"hard": hard, "soft": soft}

    def _resolve_peers(self, company_id: str, facts: pd.DataFrame, ts: pd.DataFrame, as_of: pd.Timestamp) -> PeerSet:
        entity_table = self._load_entity_table()
        resolved = self.peer_resolver.resolve(company_id, entity_table=entity_table)
        return PeerSet(
            peer_set_id=resolved.get("peer_set_id", str(uuid.uuid4())),
            members=resolved.get("members", []),
            method=resolved.get("method", "unresolved"),
            version=resolved.get("version", 1),
        )


def snapshot_to_json(snapshot: CompanyStateSnapshot) -> Dict[str, Any]:
    return asdict(snapshot)
