from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import pandas as pd


MetricLoader = Callable[[Path], Optional[dict]]
MetricBuilder = Callable[[str, dict, str], Tuple[Optional[float], str, Optional[str], Optional[Dict[str, Any]], Optional[list[str]]]]

_COMPANYFACTS_METRICS: tuple[tuple[str, str], ...] = (
    ("operating.revenue_ttm_provider_direct", "usd"),
    ("operating.revenue_ttm_lag_1y", "usd"),
    ("liquidity.cash_and_short_term_investments_provider_direct", "usd"),
    ("capital_structure.total_debt_provider_direct", "usd"),
)
_EXACTISH_SUPPORT_MODES = {"exact", "exact_not_applicable", "exact_structural_zero"}
_MAX_SAFE_PRICE_STALENESS_DAYS = 21.0
_MAX_SAFE_REFERENCE_EV_RATIO = 2.0
_MAX_DAILY_ANCHOR_GAP_DAYS = 7


def _sec_metric_builders() -> tuple[MetricLoader, MetricBuilder]:
    try:
        from scripts.backfill_input_layer_v1_metrics import _build_sec_core_metric, _load_companyfacts
    except Exception:
        from backfill_input_layer_v1_metrics import _build_sec_core_metric, _load_companyfacts
    return _load_companyfacts, _build_sec_core_metric


def _price_history_loaders():
    try:
        from scripts.backfill_market_macro_input_layer_v1 import (
            _load_crsp_daily_from_repo,
            _load_crsp_market_cache,
        )
    except Exception:
        from backfill_market_macro_input_layer_v1 import (
            _load_crsp_daily_from_repo,
            _load_crsp_market_cache,
        )
    return _load_crsp_daily_from_repo, _load_crsp_market_cache


def _feature_record_needs_enrichment(raw: Any) -> bool:
    if raw is None:
        return True
    if not isinstance(raw, dict):
        return raw is None
    if raw.get("value") is None:
        return True
    support_mode = str(raw.get("support_mode") or "").strip().lower()
    return support_mode == "unsupported"


def _feature_value(raw: Any) -> Any:
    if isinstance(raw, dict):
        return raw.get("value")
    return raw


def _safe_float(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except Exception:
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _support_mode(raw: Any) -> str | None:
    if not isinstance(raw, dict):
        return None
    value = raw.get("support_mode")
    return str(value).strip().lower() if value is not None else None


def _is_exactish_support_mode(mode: str | None) -> bool:
    return str(mode or "").strip().lower() in _EXACTISH_SUPPORT_MODES


def _quality_flags(raw: Any) -> list[str]:
    if not isinstance(raw, dict):
        return []
    return [str(flag) for flag in (raw.get("quality_flags") or []) if flag is not None]


@lru_cache(maxsize=8)
def _permno_lookup(entity_identifier_path: str) -> Dict[str, str]:
    ids = pd.read_parquet(entity_identifier_path, columns=["entity_id", "identifier_type", "identifier_value"])
    ids = ids[ids["identifier_type"].astype(str).str.lower() == "permno"].copy()
    ids["entity_id"] = ids["entity_id"].astype(str)
    ids["permno"] = ids["identifier_value"].astype(str).str.strip()
    ids = ids.drop_duplicates(subset=["entity_id"], keep="last")
    return dict(zip(ids["entity_id"], ids["permno"]))


def _dedupe_provenance(*records: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        for item in list(record.get("provenance") or []):
            if not isinstance(item, dict):
                continue
            key = json.dumps(item, sort_keys=True, default=str)
            if key in seen:
                continue
            seen.add(key)
            out.append(deepcopy(item))
    return out


def _companyfacts_path(companyfacts_root: Path, company_id: str) -> Path:
    normalized = str(company_id or "").strip()
    if normalized.startswith("CIK"):
        normalized = normalized.removeprefix("CIK")
    normalized = normalized.zfill(10)
    return companyfacts_root / f"CIK{normalized}.json"


def _metric_record(
    *,
    metric_name: str,
    unit: str,
    value: Any,
    support_mode: str,
    missing_reason: str | None,
    component_breakdown: Optional[Dict[str, Any]],
    quality_flags: Optional[list[str]],
    as_of_time: str,
    artifact_id: str,
) -> Dict[str, Any]:
    flags = list(dict.fromkeys([*(quality_flags or []), "replay_snapshot_growth_enrichment"]))
    return {
        "name": metric_name,
        "value": value,
        "unit": unit,
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "as_of_time": as_of_time,
        "window": None,
        "confidence": None,
        "provenance": [
            {
                "artifact_type": "sec_companyfacts",
                "artifact_id": artifact_id,
                "source": "sec_companyfacts",
                "published_at": None,
                "ingested_at": None,
                "hash": None,
            }
        ],
        "missing_reason": missing_reason if value is None else None,
        "fallback_used": None,
        "metric_policy_id": None,
        "market_owner": None,
        "primary_source_basis": None,
        "methodology_registry_id": None,
        "methodology_metric_id": None,
        "canonical_owner_id": None,
        "canonical_owner_name": None,
        "canonical_classification": None,
        "market_layer_status": None,
        "current_alignment_status": None,
        "primary_source_document_id": None,
        "recommended_metric_name": None,
        "input_source_registry_id": "replay_snapshot_growth_enrichment_v1",
        "input_source_owner_id": "sec_companyfacts",
        "input_source_owner_name": "SEC companyfacts",
        "input_source_classification": "external_raw_plus_deterministic_formula",
        "input_source_formula_basis": (component_breakdown or {}).get("formula"),
        "input_source_alignment_status": "point_in_time_asof_safe",
        "input_source_document_ids": ["sec_companyfacts"],
        "definition_requirement": None,
        "definition_requirement_reason": None,
        "methodology_execution_decision": "adopt_exact_external_methodology",
        "methodology_execution_reason": "Backfilled from SEC companyfacts using the same point-in-time logic as input-layer metric materialization.",
        "input_layer_bucket": "strict_market_defined",
        "input_layer_bucket_reason": "Backfilled from SEC companyfacts using deterministic point-in-time methodology.",
        "strict_market_defined": True,
        "archetype": None,
        "sector": None,
        "subsector": None,
        "override_level_applied": None,
        "support_mode": support_mode,
        "applicability_status": None,
        "component_breakdown": component_breakdown or {},
        "quality_flags": flags or None,
        "view_type": None,
    }


def _clone_metric_record(
    source_record: dict[str, Any],
    *,
    metric_name: str,
    value: Any = None,
    support_mode: str | None = None,
    as_of_time: str,
    extra_quality_flags: Optional[Iterable[str]] = None,
    fallback_used: str | None = None,
    component_breakdown_updates: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    out = deepcopy(dict(source_record or {}))
    out["name"] = metric_name
    out["computed_at"] = datetime.now(timezone.utc).isoformat()
    out["as_of_time"] = as_of_time
    if value is not None or "value" not in out:
        out["value"] = value
    if support_mode is not None:
        out["support_mode"] = support_mode
    if fallback_used is not None:
        out["fallback_used"] = fallback_used
    breakdown = dict(out.get("component_breakdown") or {})
    breakdown["replay_snapshot_matching_enrichment"] = {
        "source_metric": source_record.get("name"),
        "target_metric": metric_name,
    }
    if component_breakdown_updates:
        breakdown.update(component_breakdown_updates)
    out["component_breakdown"] = breakdown
    flags = list(
        dict.fromkeys(
            [
                *(_quality_flags(source_record)),
                *(list(extra_quality_flags or [])),
                "replay_snapshot_matching_enrichment",
            ]
        )
    )
    out["quality_flags"] = flags or None
    return out


