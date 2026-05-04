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


