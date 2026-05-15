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


def _derived_metric_record(
    *,
    metric_name: str,
    unit: str,
    value: Any,
    support_mode: str,
    as_of_time: str,
    formula: str,
    source_records: Iterable[dict[str, Any] | None] = (),
    component_values: Optional[Dict[str, Any]] = None,
    fallback_used: str | None = None,
    quality_flags: Optional[Iterable[str]] = None,
    missing_reason: str | None = None,
) -> Dict[str, Any]:
    component_breakdown = dict(component_values or {})
    component_breakdown["formula"] = formula
    flags = list(dict.fromkeys([*(list(quality_flags or [])), "replay_snapshot_matching_enrichment"]))
    return {
        "name": metric_name,
        "value": value,
        "unit": unit,
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "as_of_time": as_of_time,
        "window": None,
        "confidence": None,
        "provenance": _dedupe_provenance(*list(source_records or ())),
        "missing_reason": missing_reason if value is None else None,
        "fallback_used": fallback_used,
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
        "input_source_registry_id": "replay_snapshot_matching_enrichment_v1",
        "input_source_owner_id": "axiom_replay_snapshot_enrichment",
        "input_source_owner_name": "Axiom replay snapshot enrichment",
        "input_source_classification": "internal_derived",
        "input_source_formula_basis": formula,
        "input_source_alignment_status": "point_in_time_asof_safe",
        "input_source_document_ids": None,
        "definition_requirement": None,
        "definition_requirement_reason": None,
        "methodology_execution_decision": "retain_internal_inference",
        "methodology_execution_reason": "Derived from already-available point-in-time snapshot inputs without using future information.",
        "input_layer_bucket": "internal_inference",
        "input_layer_bucket_reason": "As-of-safe enrichment from replay snapshot inputs.",
        "strict_market_defined": False,
        "archetype": None,
        "sector": None,
        "subsector": None,
        "override_level_applied": None,
        "support_mode": support_mode,
        "applicability_status": None,
        "component_breakdown": component_breakdown,
        "quality_flags": flags or None,
        "view_type": None,
    }


def _record_value(features: Dict[str, Any], *keys: str) -> Optional[float]:
    return _safe_float(_feature_value(_record(features, *keys)))


def _record(features: Dict[str, Any], *keys: str) -> Optional[dict]:
    for key in keys:
        record = features.get(key)
        if record is not None:
            return record
    return None


def _needs_exact_price_history_repair(raw: Any) -> bool:
    if raw is None:
        return True
    if not isinstance(raw, dict):
        return True
    if raw.get("value") is None:
        return True
    if _support_mode(raw) != "exact":
        return True
    fallback_used = str(raw.get("fallback_used") or "")
    quality_flags = set(_quality_flags(raw))
    breakdown = dict(raw.get("component_breakdown") or {})
    formula = str(breakdown.get("formula") or "")
    source_kind = str(breakdown.get("source_kind") or "")
    selected_price_series = breakdown.get("selected_price_series") or {}
    selected_source_kind = str(selected_price_series.get("source_kind") or "")
    median_gap = _safe_float(
        breakdown.get("median_observation_gap_days")
        or selected_price_series.get("median_observation_gap_days")
    )
    if fallback_used == "monthly_price_history_proxy":
        return True
    if "monthly_returns" in formula or "monthly_price_window" in formula:
        return True
    if {"low_frequency_price_history", "insufficient_return_history", "insufficient_price_history"} & quality_flags:
        return True
    if source_kind and source_kind not in {"crsp_market_cache", "crsp_daily_root"}:
        return True
    if selected_source_kind and selected_source_kind not in {"crsp_market_cache", "crsp_daily_root"}:
        return True
    if median_gap is not None and median_gap >= 7.0:
        return True
    return False


def _load_exact_price_history(
    *,
    company_id: str,
    as_of_time: str,
    entity_identifier_path: Path | None,
    crsp_market_cache_path: Path | None,
    crsp_daily_root: Path | None,
) -> tuple[str | None, str | None, str | None, pd.DataFrame]:
    if entity_identifier_path is None or not entity_identifier_path.exists():
        return None, None, None, pd.DataFrame()
    permno = _permno_lookup(str(entity_identifier_path)).get(str(company_id))
    if not permno:
        return None, None, None, pd.DataFrame()

    as_of_date = pd.Timestamp(as_of_time).tz_convert("UTC").normalize()
    load_crsp_daily_from_repo, load_crsp_market_cache = _price_history_loaders()
    if crsp_market_cache_path is not None and crsp_market_cache_path.exists():
        price_history = load_crsp_market_cache(crsp_market_cache_path, [permno])
        source_kind = "crsp_market_cache"
        source_path = str(crsp_market_cache_path)
    elif crsp_daily_root is not None and crsp_daily_root.exists():
        price_history = load_crsp_daily_from_repo(
            crsp_daily_root,
            [permno],
            min_asof_date=as_of_date,
            max_asof_date=as_of_date,
        )
        source_kind = "crsp_daily_root"
        source_path = str(crsp_daily_root)
    else:
        return permno, None, None, pd.DataFrame()

    if price_history is None or price_history.empty:
        return permno, source_kind, source_path, pd.DataFrame()
    frame = price_history[price_history["permno"].astype(str) == permno].copy()
    if frame.empty:
        return permno, source_kind, source_path, pd.DataFrame()
    price_col = "price_proxy" if "price_proxy" in frame.columns else "close_price"
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], utc=True).dt.normalize()
    frame["price"] = pd.to_numeric(frame[price_col], errors="coerce")
    frame = frame.dropna(subset=["trade_date", "price"])
    frame = frame[(frame["price"] > 0.0) & (frame["trade_date"] <= as_of_date)].copy()
    frame = frame.sort_values("trade_date").drop_duplicates(subset=["trade_date"], keep="last")
    return permno, source_kind, source_path, frame[["trade_date", "price"]].reset_index(drop=True)


def _compute_exact_price_metrics(
    price_history: pd.DataFrame,
    *,
    as_of_time: str,
    source_kind: str,
) -> Dict[str, Dict[str, Any]]:
    if price_history is None or price_history.empty:
        return {}
    frame = price_history.sort_values("trade_date").drop_duplicates(subset=["trade_date"], keep="last").copy()
    if frame.empty:
        return {}
    as_of_date = pd.Timestamp(as_of_time).tz_convert("UTC").normalize()
    latest_trade_date = frame["trade_date"].iloc[-1]
    if not (0 <= int((as_of_date - latest_trade_date).days) <= _MAX_DAILY_ANCHOR_GAP_DAYS):
        return {}

    frame["ret"] = frame["price"].pct_change()
    results: Dict[str, Dict[str, Any]] = {}
    for days, min_obs, metric_name in ((30, 10, "market.volatility_30d"), (90, 20, "market.volatility_90d")):
        returns_window = frame.loc[frame["trade_date"] > (latest_trade_date - pd.Timedelta(days=days)), ["trade_date", "ret"]].dropna()
        if len(returns_window) < min_obs:
            continue
        results[metric_name] = {
            "value": float(returns_window["ret"].std(ddof=0) * math.sqrt(252)),
            "unit": "annualized",
            "formula": f"stddev(daily_returns_{days}d) * sqrt(252)",
            "component_values": {
                "return_observations": int(len(returns_window)),
                "annualization_factor": 252,
                "window_start": str(returns_window["trade_date"].iloc[0]),
                "window_end": str(returns_window["trade_date"].iloc[-1]),
                "source_kind": source_kind,
            },
        }

    price_window_90 = frame.loc[frame["trade_date"] > (latest_trade_date - pd.Timedelta(days=90)), ["trade_date", "price"]].copy()
    if len(price_window_90) >= 20:
        peak_price = float(price_window_90["price"].max())
        trough_price = float(price_window_90["price"].min())
        if peak_price != 0.0:
            results["market.drawdown_90d"] = {
                "value": (trough_price / peak_price) - 1.0,
                "unit": "ratio",
                "formula": "min(price_window_90d) / max(price_window_90d) - 1",
                "component_values": {
                    "price_observations": int(len(price_window_90)),
                    "peak_price": peak_price,
                    "trough_price": trough_price,
                    "window_start": str(price_window_90["trade_date"].iloc[0]),
                    "window_end": str(price_window_90["trade_date"].iloc[-1]),
                    "source_kind": source_kind,
                },
            }
    return results


def _copy_metric_if_missing(
    features: Dict[str, Any],
    *,
    target_key: str,
    source_keys: tuple[str, ...],
    as_of_time: str,
    summary: Dict[str, Any],
) -> bool:
    previous = features.get(target_key)
    if not _feature_record_needs_enrichment(previous):
        return False
    source_record = _record(features, *source_keys)
    if not isinstance(source_record, dict) or _safe_float(_feature_value(source_record)) is None:
        return False
    record = _clone_metric_record(
        source_record,
        metric_name=target_key,
        as_of_time=as_of_time,
        extra_quality_flags=["copied_from_same_date_snapshot_metric"],
    )
    features[target_key] = record
    summary["metrics"][target_key] = {
        "value": record.get("value"),
        "support_mode": record.get("support_mode"),
        "missing_reason": None,
        "changed": previous != record,
    }
    return previous != record


def _write_metric(
    features: Dict[str, Any],
    *,
    metric_name: str,
    record: Dict[str, Any],
    summary: Dict[str, Any],
) -> bool:
    previous = features.get(metric_name)
    features[metric_name] = record
    summary["metrics"][metric_name] = {
        "value": record.get("value"),
        "support_mode": record.get("support_mode"),
        "missing_reason": record.get("missing_reason"),
        "changed": previous != record,
    }
    return previous != record


def _supports_are_exactish(*records: Any) -> bool:
    return all(_is_exactish_support_mode(_support_mode(record)) for record in records if record is not None)


def _market_cap_looks_suspicious(features: Dict[str, Any], *, reference_enterprise_value: Optional[float]) -> bool:
    market_cap_record = _record(features, "market.market_cap_provider_direct", "market.market_cap")
    if not isinstance(market_cap_record, dict):
        return False
    fallback_used = str(market_cap_record.get("fallback_used") or "").strip().lower()
    component_breakdown = dict(market_cap_record.get("component_breakdown") or {})
    age_days = _safe_float(component_breakdown.get("price_observation_age_days"))
    confidence = _safe_float(market_cap_record.get("confidence"))
    suspicious = False
    if fallback_used == "price*shares" and (
        (age_days is not None and age_days > _MAX_SAFE_PRICE_STALENESS_DAYS)
        or (confidence is not None and confidence < 0.35)
    ):
        suspicious = True
    current_enterprise_value = _record_value(features, "market.enterprise_value")
    if (
        not suspicious
        and reference_enterprise_value is not None
        and current_enterprise_value is not None
        and reference_enterprise_value > 0.0
    ):
        ratio = current_enterprise_value / reference_enterprise_value
        if ratio > _MAX_SAFE_REFERENCE_EV_RATIO or ratio < (1.0 / _MAX_SAFE_REFERENCE_EV_RATIO):
            suspicious = True
    return suspicious


def _derive_reference_ebitda_record(features: Dict[str, Any], *, as_of_time: str) -> Optional[Dict[str, Any]]:
    ev_record = _record(features, "market.ev_ebitda")
    if not isinstance(ev_record, dict):
        return None
    breakdown = dict(ev_record.get("component_breakdown") or {})
    reference_ebitda = _safe_float(breakdown.get("ebitda_ttm"))
    if reference_ebitda is None or reference_ebitda <= 0.0:
        return None
    return _derived_metric_record(
        metric_name="operating.ebitda_ltm_provider_direct",
        unit="usd",
        value=reference_ebitda,
        support_mode="proxy_missing_component",
        as_of_time=as_of_time,
        formula="reference_ebitda_from_market_ev_ebitda_component_breakdown",
        source_records=[ev_record],
        component_values={
            "reference_ebitda_ttm": reference_ebitda,
            "reference_ev_ebitda": _safe_float(breakdown.get("reference_ev_ebitda")),
            "reference_instrument": breakdown.get("reference_instrument"),
        },
        fallback_used="reference_ebitda_snapshot_fallback",
        quality_flags=["ebitda_derived_from_reference_market_metric"],
    )


def _derive_margin_ebitda_record(features: Dict[str, Any], *, as_of_time: str) -> Optional[Dict[str, Any]]:
    revenue_record = _record(features, "operating.revenue_ttm_provider_direct", "operating.revenue_ttm")
    margin_record = _record(features, "operating.ebitda_margin_ttm")
    revenue = _safe_float(_feature_value(revenue_record))
    margin = _safe_float(_feature_value(margin_record))
    if revenue is None or revenue <= 0.0 or margin is None:
        return None
    support_mode = "exact" if _supports_are_exactish(revenue_record, margin_record) else "proxy_missing_component"
    return _derived_metric_record(
        metric_name="operating.ebitda_ltm_provider_direct",
        unit="usd",
        value=revenue * margin,
        support_mode=support_mode,
        as_of_time=as_of_time,
        formula="revenue_ttm * ebitda_margin_ttm",
        source_records=[revenue_record, margin_record],
        component_values={
            "revenue_ttm": revenue,
            "ebitda_margin_ttm": margin,
        },
        fallback_used="revenue_times_margin",
        quality_flags=["ebitda_derived_from_margin_and_revenue"],
    )


def enrich_snapshot_with_revenue_growth_inputs(
    snapshot: Dict[str, Any],
    *,
    companyfacts_root: str | Path | None,
    entity_identifier_path: str | Path | None = None,
    crsp_market_cache_path: str | Path | None = None,
    crsp_daily_root: str | Path | None = None,
    company_id: str | None = None,
    as_of_time: str | None = None,
    loaders: Optional[tuple[MetricLoader, MetricBuilder]] = None,
) -> tuple[Dict[str, Any], bool, Dict[str, Any]]:
    payload = deepcopy(dict(snapshot or {}))
    features = dict(payload.get("features") or {})
    payload["features"] = features
    resolved_company_id = str(company_id or payload.get("company_id") or "").strip()
    resolved_as_of_time = str(as_of_time or payload.get("as_of_time") or "").strip()
    root = Path(companyfacts_root) if companyfacts_root else None
    summary: Dict[str, Any] = {
        "company_id": resolved_company_id,
        "as_of_time": resolved_as_of_time,
        "companyfacts_path": None,
        "loaded_companyfacts": False,
        "price_history_source": None,
        "price_history_permno": None,
        "metrics": {},
    }
    if not resolved_company_id or not resolved_as_of_time:
        return payload, False, summary

    pending = [metric_name for metric_name, _ in _COMPANYFACTS_METRICS if _feature_record_needs_enrichment(features.get(metric_name))]
    changed = False
    if pending and root is not None:
        companyfacts_path = _companyfacts_path(root, resolved_company_id)
        summary["companyfacts_path"] = str(companyfacts_path)
        if companyfacts_path.exists():
            load_companyfacts, build_metric = loaders or _sec_metric_builders()
            companyfacts = load_companyfacts(companyfacts_path)
            summary["loaded_companyfacts"] = companyfacts is not None
            if companyfacts:
                artifact_id = companyfacts_path.stem
                as_of_date = resolved_as_of_time[:10]
                for metric_name, unit in _COMPANYFACTS_METRICS:
                    previous = features.get(metric_name)
                    if not _feature_record_needs_enrichment(previous):
                        continue
                    value, support_mode, missing_reason, component_breakdown, quality_flags = build_metric(metric_name, companyfacts, as_of_date)
                    record = _metric_record(
                        metric_name=metric_name,
                        unit=unit,
                        value=value,
                        support_mode=support_mode,
                        missing_reason=missing_reason,
                        component_breakdown=component_breakdown,
                        quality_flags=quality_flags,
                        as_of_time=resolved_as_of_time,
                        artifact_id=artifact_id,
                    )
                    summary["metrics"][metric_name] = {
                        "value": value,
                        "support_mode": support_mode,
                        "missing_reason": missing_reason,
                        "changed": previous != record,
                    }
                    if previous != record:
                        features[metric_name] = record
                        changed = True

    if _feature_record_needs_enrichment(features.get("liquidity.cash")):
        cash_and_sti_record = _record(features, "liquidity.cash_and_short_term_investments_provider_direct")
        cash_and_sti_value = _safe_float(_feature_value(cash_and_sti_record))
        if cash_and_sti_value is not None:
            cash_record = _clone_metric_record(
                cash_and_sti_record,
                metric_name="liquidity.cash",
                value=cash_and_sti_value,
                support_mode="proxy_missing_component",
                as_of_time=resolved_as_of_time,
                fallback_used="cash_and_short_term_investments_provider_direct",
                extra_quality_flags=["cash_proxy_from_cash_and_short_term_investments"],
            )
            changed |= _write_metric(
                features,
                metric_name="liquidity.cash",
                record=cash_record,
                summary=summary,
            )

    if _feature_record_needs_enrichment(features.get("operating.ebitda_ltm_provider_direct")):
        ebitda_record = _derive_reference_ebitda_record(features, as_of_time=resolved_as_of_time)
        if ebitda_record is None:
            ebitda_record = _derive_margin_ebitda_record(features, as_of_time=resolved_as_of_time)
        if ebitda_record is not None:
            changed |= _write_metric(
                features,
                metric_name="operating.ebitda_ltm_provider_direct",
                record=ebitda_record,
                summary=summary,
            )

    changed |= _copy_metric_if_missing(
        features,
        target_key="capital_structure.total_debt_provider_direct",
        source_keys=("capital_structure.total_debt", "capital_structure.total_debt_reported"),
        as_of_time=resolved_as_of_time,
        summary=summary,
    )

    marketable_record = features.get("liquidity.marketable_securities_sec_exact")
    if _feature_record_needs_enrichment(marketable_record):
        available_liquidity_record = _record(features, "liquidity.available_liquidity_normalized")
        available_breakdown = dict((available_liquidity_record or {}).get("component_breakdown") or {})
        marketable_value = _safe_float(available_breakdown.get("marketable_securities_sec_exact"))
        if marketable_value is not None:
            derived_marketable = _derived_metric_record(
                metric_name="liquidity.marketable_securities_sec_exact",
                unit="usd",
                value=marketable_value,
                support_mode="proxy_missing_component",
                as_of_time=resolved_as_of_time,
                formula="available_liquidity_normalized.component_breakdown.marketable_securities_sec_exact",
                source_records=[available_liquidity_record],
                component_values={"marketable_securities_sec_exact": marketable_value},
                fallback_used="available_liquidity_component_breakdown",
                quality_flags=["marketable_securities_derived_from_available_liquidity"],
            )
            changed |= _write_metric(
                features,
                metric_name="liquidity.marketable_securities_sec_exact",
                record=derived_marketable,
                summary=summary,
            )

    if _feature_record_needs_enrichment(features.get("liquidity.cash_and_short_term_investments_provider_direct")):
        cash_record = _record(features, "liquidity.cash")
        cash_value = _safe_float(_feature_value(cash_record))
        marketable_value = _record_value(features, "liquidity.marketable_securities_sec_exact")
        if cash_value is not None:
            cash_and_sti_record = _derived_metric_record(
                metric_name="liquidity.cash_and_short_term_investments_provider_direct",
                unit="usd",
                value=cash_value + (marketable_value or 0.0),
                support_mode=(
                    "exact"
                    if _supports_are_exactish(cash_record) and (marketable_value in (None, 0.0) or _supports_are_exactish(features.get("liquidity.marketable_securities_sec_exact")))
                    else "proxy_missing_component"
                ),
                as_of_time=resolved_as_of_time,
                formula="cash + marketable_securities",
                source_records=[cash_record, features.get("liquidity.marketable_securities_sec_exact")],
                component_values={
                    "cash": cash_value,
                    "marketable_securities": marketable_value,
                },
                fallback_used="cash_plus_marketable_securities_snapshot_fallback",
                quality_flags=["cash_and_short_term_investments_derived_from_snapshot_inputs"],
            )
            changed |= _write_metric(
                features,
                metric_name="liquidity.cash_and_short_term_investments_provider_direct",
                record=cash_and_sti_record,
                summary=summary,
            )

    if _feature_record_needs_enrichment(features.get("liquidity.usable_cash")):
        cash_record = _record(
            features,
            "liquidity.cash",
            "liquidity.cash_and_short_term_investments_provider_direct",
        )
        cash_value = _safe_float(_feature_value(cash_record))
        if cash_value is not None:
            usable_cash_record = _derived_metric_record(
                metric_name="liquidity.usable_cash",
                unit="usd",
                value=max(0.0, cash_value),
                support_mode="proxy_missing_component",
                as_of_time=resolved_as_of_time,
                formula="max(0, cash_proxy)",
                source_records=[cash_record],
                component_values={"cash_proxy": cash_value},
                fallback_used="cash_proxy_no_restriction_adjustment",
                quality_flags=["usable_cash_derived_from_cash_proxy"],
            )
            changed |= _write_metric(
                features,
                metric_name="liquidity.usable_cash",
                record=usable_cash_record,
                summary=summary,
            )

    if _feature_record_needs_enrichment(features.get("liquidity.available_liquidity_normalized")):
        cash_and_sti_record = _record(features, "liquidity.cash_and_short_term_investments_provider_direct")
        revolver_record = _record(features, "liquidity.revolver_undrawn")
        cash_and_sti_value = _safe_float(_feature_value(cash_and_sti_record))
        revolver_value = _safe_float(_feature_value(revolver_record))
        if cash_and_sti_value is not None or revolver_value is not None:
            available_liquidity_value = max(0.0, (cash_and_sti_value or 0.0) + (revolver_value or 0.0))
            available_liquidity_record = _derived_metric_record(
                metric_name="liquidity.available_liquidity_normalized",
                unit="usd",
                value=available_liquidity_value,
                support_mode="proxy_missing_component",
                as_of_time=resolved_as_of_time,
                formula="cash_and_short_term_investments_provider_direct + revolver_undrawn",
                source_records=[cash_and_sti_record, revolver_record],
                component_values={
                    "cash_and_short_term_investments_provider_direct": cash_and_sti_value,
                    "revolver_undrawn": revolver_value,
                },
                fallback_used="cash_and_short_term_plus_revolver",
                quality_flags=["available_liquidity_derived_from_cash_and_revolver"],
            )
            changed |= _write_metric(
                features,
                metric_name="liquidity.available_liquidity_normalized",
                record=available_liquidity_record,
                summary=summary,
            )

    if _feature_record_needs_enrichment(features.get("liquidity.available_for_actions")):
        usable_cash_record = _record(
            features,
            "liquidity.available_liquidity_normalized",
            "liquidity.usable_cash",
            "liquidity.cash_and_short_term_investments_provider_direct",
        )
        usable_cash_value = _safe_float(_feature_value(usable_cash_record))
        revolver_record = _record(features, "liquidity.revolver_undrawn")
        revolver_value = _safe_float(_feature_value(revolver_record))
        minimum_cash_record = _record(features, "liquidity.minimum_cash_policy_proxy")
        minimum_cash_value = _safe_float(_feature_value(minimum_cash_record))
        if usable_cash_value is not None or revolver_value is not None:
            deployable_cash = usable_cash_value
            fallback_used = "available_liquidity_proxy"
            quality_flags = ["available_for_actions_derived_from_proxy_liquidity"]
            component_values = {
                "usable_cash_proxy": usable_cash_value,
                "revolver_undrawn": revolver_value,
                "minimum_cash_policy_proxy": minimum_cash_value,
            }
            if deployable_cash is not None and minimum_cash_value is not None:
                deployable_cash = max(0.0, deployable_cash - minimum_cash_value)
                fallback_used = "proxy_liquidity_minus_minimum_cash_policy"
            available_for_actions_value = max(0.0, (deployable_cash or 0.0) + (revolver_value or 0.0))
            available_for_actions_record = _derived_metric_record(
                metric_name="liquidity.available_for_actions",
                unit="usd",
                value=available_for_actions_value,
                support_mode="proxy_missing_component",
                as_of_time=resolved_as_of_time,
                formula="max(0, usable_cash_proxy - minimum_cash_policy_proxy) + revolver_undrawn",
                source_records=[usable_cash_record, minimum_cash_record, revolver_record],
                component_values=component_values,
                fallback_used=fallback_used,
                quality_flags=quality_flags,
            )
            changed |= _write_metric(
                features,
                metric_name="liquidity.available_for_actions",
                record=available_for_actions_record,
                summary=summary,
            )

    if _feature_record_needs_enrichment(features.get("capital_structure.net_debt")):
        total_debt_record = _record(
            features,
            "capital_structure.total_debt_provider_direct",
            "capital_structure.total_debt",
        )
        total_debt_value = _safe_float(_feature_value(total_debt_record))
        cash_and_sti_record = _record(features, "liquidity.cash_and_short_term_investments_provider_direct")
        cash_and_sti_value = _safe_float(_feature_value(cash_and_sti_record))
        if total_debt_value is not None and cash_and_sti_value is not None:
            net_debt_record = _derived_metric_record(
                metric_name="capital_structure.net_debt",
                unit="usd",
                value=total_debt_value - cash_and_sti_value,
                support_mode="proxy_missing_component",
                as_of_time=resolved_as_of_time,
                formula="total_debt_provider_direct - cash_and_short_term_investments_provider_direct",
                source_records=[total_debt_record, cash_and_sti_record],
                component_values={
                    "total_debt_provider_direct": total_debt_value,
                    "cash_and_short_term_investments_provider_direct": cash_and_sti_value,
                },
                fallback_used="total_debt_minus_cash_and_short_term_investments",
                quality_flags=["net_debt_derived_from_companyfacts_proxies"],
            )
            changed |= _write_metric(
                features,
                metric_name="capital_structure.net_debt",
                record=net_debt_record,
                summary=summary,
            )

    if _feature_record_needs_enrichment(features.get("capital_structure.net_leverage")):
        net_debt_record = _record(features, "capital_structure.net_debt")
        net_debt_value = _safe_float(_feature_value(net_debt_record))
        ebitda_record = _record(features, "operating.ebitda_ltm_provider_direct")
        ebitda_value = _safe_float(_feature_value(ebitda_record))
        if net_debt_value is not None and ebitda_value is not None and ebitda_value > 0:
            net_leverage_record = _derived_metric_record(
                metric_name="capital_structure.net_leverage",
                unit="x",
                value=net_debt_value / ebitda_value,
                support_mode="proxy_missing_component",
                as_of_time=resolved_as_of_time,
                formula="net_debt / ebitda_ltm_provider_direct",
                source_records=[net_debt_record, ebitda_record],
                component_values={
                    "net_debt": net_debt_value,
                    "ebitda_ltm_provider_direct": ebitda_value,
                },
                fallback_used="net_debt_divided_by_ebitda",
                quality_flags=["net_leverage_derived_from_proxy_net_debt"],
            )
            changed |= _write_metric(
                features,
                metric_name="capital_structure.net_leverage",
                record=net_leverage_record,
                summary=summary,
            )

    if _feature_record_needs_enrichment(features.get("capital_structure.debt_due_next_24m")):
        debt_0_12_record = _record(features, "capital_structure.debt_due_0_12m")
        debt_12_24_record = _record(features, "capital_structure.debt_due_12_24m")
        debt_0_12 = _safe_float(_feature_value(debt_0_12_record))
        debt_12_24 = _safe_float(_feature_value(debt_12_24_record))
        if debt_0_12 is not None or debt_12_24 is not None:
            debt_next_24m = max(0.0, (debt_0_12 or 0.0) + (debt_12_24 or 0.0))
            debt_due_next_record = _derived_metric_record(
                metric_name="capital_structure.debt_due_next_24m",
                unit="usd",
                value=debt_next_24m,
                support_mode=(
                    "exact"
                    if _supports_are_exactish(debt_0_12_record, debt_12_24_record)
                    else "proxy_missing_component"
                ),
                as_of_time=resolved_as_of_time,
                formula="debt_due_0_12m + debt_due_12_24m",
                source_records=[debt_0_12_record, debt_12_24_record],
                component_values={
                    "debt_due_0_12m": debt_0_12,
                    "debt_due_12_24m": debt_12_24,
                },
                fallback_used="debt_due_bucket_sum",
                quality_flags=["debt_due_next_24m_derived_from_maturity_buckets"],
            )
            changed |= _write_metric(
                features,
                metric_name="capital_structure.debt_due_next_24m",
                record=debt_due_next_record,
                summary=summary,
            )

    market_ev_record = _record(features, "market.ev_ebitda")
    market_ev_breakdown = dict((market_ev_record or {}).get("component_breakdown") or {})
    reference_ev_multiple = _safe_float(market_ev_breakdown.get("reference_ev_ebitda"))
    ebitda_value = _record_value(features, "operating.ebitda_ltm_provider_direct")
    if ebitda_value is None:
        ebitda_value = _safe_float(market_ev_breakdown.get("ebitda_ttm"))
    reference_enterprise_value = None
    if reference_ev_multiple is not None and ebitda_value is not None and ebitda_value > 0.0:
        reference_enterprise_value = reference_ev_multiple * ebitda_value
    suspicious_market_cap = _market_cap_looks_suspicious(
        features,
        reference_enterprise_value=reference_enterprise_value,
    )
    net_debt_value = _record_value(
        features,
        "capital_structure.net_debt_normalized",
    )
    if net_debt_value is None:
        total_debt_value = _record_value(features, "capital_structure.total_debt_provider_direct", "capital_structure.total_debt")
        available_liquidity_value = _record_value(features, "liquidity.available_liquidity_normalized")
        if total_debt_value is not None and available_liquidity_value is not None:
            net_debt_value = total_debt_value - available_liquidity_value

    if reference_enterprise_value is not None and net_debt_value is not None:
        safe_market_cap = reference_enterprise_value - net_debt_value
        if safe_market_cap > 0.0 and (
            suspicious_market_cap or _feature_record_needs_enrichment(features.get("market.market_cap_provider_direct"))
        ):
            market_cap_record = _derived_metric_record(
                metric_name="market.market_cap_provider_direct",
                unit="usd",
                value=safe_market_cap,
                support_mode="proxy_missing_component",
                as_of_time=resolved_as_of_time,
                formula="reference_enterprise_value - net_debt_normalized",
                source_records=[market_ev_record, features.get("capital_structure.net_debt_normalized")],
                component_values={
                    "reference_enterprise_value": reference_enterprise_value,
                    "net_debt_normalized": net_debt_value,
                    "reference_ev_ebitda": reference_ev_multiple,
                    "ebitda_ttm": ebitda_value,
                },
                fallback_used="reference_ev_ebitda_minus_net_debt",
                quality_flags=(
                    ["replaced_suspicious_price_shares_market_cap"]
                    if suspicious_market_cap
                    else ["market_cap_reconstructed_from_reference_ev_ebitda"]
                ),
            )
            changed |= _write_metric(
                features,
                metric_name="market.market_cap_provider_direct",
                record=market_cap_record,
                summary=summary,
            )

            enterprise_value_record = _derived_metric_record(
                metric_name="market.enterprise_value",
                unit="usd",
                value=reference_enterprise_value,
                support_mode="proxy_missing_component",
                as_of_time=resolved_as_of_time,
                formula="reference_ev_ebitda * ebitda_ttm",
                source_records=[market_ev_record, features.get("operating.ebitda_ltm_provider_direct")],
                component_values={
                    "reference_ev_ebitda": reference_ev_multiple,
                    "ebitda_ttm": ebitda_value,
                },
                fallback_used="reference_ev_ebitda_times_ebitda",
                quality_flags=(
                    ["replaced_suspicious_enterprise_value"]
                    if suspicious_market_cap
                    else ["enterprise_value_reconstructed_from_reference_ev_ebitda"]
                ),
            )
            changed |= _write_metric(
                features,
                metric_name="market.enterprise_value",
                record=enterprise_value_record,
                summary=summary,
            )

            ev_ebitda_record = _derived_metric_record(
                metric_name="market.ev_ebitda",
                unit="x",
                value=reference_ev_multiple,
                support_mode="proxy_missing_component",
                as_of_time=resolved_as_of_time,
                formula="reference_ev_ebitda",
                source_records=[market_ev_record, features.get("operating.ebitda_ltm_provider_direct")],
                component_values={
                    "reference_ev_ebitda": reference_ev_multiple,
                    "ebitda_ttm": ebitda_value,
                    "reference_enterprise_value": reference_enterprise_value,
                },
                fallback_used="reference_ev_ebitda",
                quality_flags=(
                    ["replaced_suspicious_ev_ebitda"]
                    if suspicious_market_cap
                    else ["ev_ebitda_reconstructed_from_reference_metric"]
                ),
            )
            changed |= _write_metric(
                features,
                metric_name="market.ev_ebitda",
                record=ev_ebitda_record,
                summary=summary,
            )
    elif _feature_record_needs_enrichment(features.get("market.market_cap_provider_direct")):
        changed |= _copy_metric_if_missing(
            features,
            target_key="market.market_cap_provider_direct",
            source_keys=("market.market_cap",),
            as_of_time=resolved_as_of_time,
            summary=summary,
        )

    if _feature_record_needs_enrichment(features.get("cash_flow.free_cash_flow_ttm")):
        fcf_conversion_record = _record(features, "operating.fcf_conversion")
        fcf_conversion_value = _safe_float(_feature_value(fcf_conversion_record))
        if fcf_conversion_value is not None and ebitda_value is not None and ebitda_value > 0.0:
            free_cash_flow_record = _derived_metric_record(
                metric_name="cash_flow.free_cash_flow_ttm",
                unit="usd",
                value=fcf_conversion_value * ebitda_value,
                support_mode="proxy_missing_component",
                as_of_time=resolved_as_of_time,
                formula="fcf_conversion * ebitda_ttm",
                source_records=[fcf_conversion_record, features.get("operating.ebitda_ltm_provider_direct")],
                component_values={
                    "fcf_conversion": fcf_conversion_value,
                    "ebitda_ttm": ebitda_value,
                },
                fallback_used="fcf_conversion_times_ebitda",
                quality_flags=["free_cash_flow_derived_from_fcf_conversion"],
            )
            changed |= _write_metric(
                features,
                metric_name="cash_flow.free_cash_flow_ttm",
                record=free_cash_flow_record,
                summary=summary,
            )

    if _feature_record_needs_enrichment(features.get("market.fcf_yield")):
        free_cash_flow_value = _record_value(features, "cash_flow.free_cash_flow_ttm")
        market_cap_value = _record_value(features, "market.market_cap_provider_direct", "market.market_cap")
        if free_cash_flow_value is not None and market_cap_value is not None and market_cap_value > 0.0:
            fcf_yield_record = _derived_metric_record(
                metric_name="market.fcf_yield",
                unit="ratio",
                value=free_cash_flow_value / market_cap_value,
                support_mode="proxy_missing_component",
                as_of_time=resolved_as_of_time,
                formula="free_cash_flow_ttm / market_cap",
                source_records=[features.get("cash_flow.free_cash_flow_ttm"), _record(features, "market.market_cap_provider_direct", "market.market_cap")],
                component_values={
                    "free_cash_flow_ttm": free_cash_flow_value,
                    "equity_market_cap": market_cap_value,
                },
                fallback_used="free_cash_flow_over_market_cap",
                quality_flags=["fcf_yield_derived_from_snapshot_inputs"],
            )
            changed |= _write_metric(
                features,
                metric_name="market.fcf_yield",
                record=fcf_yield_record,
                summary=summary,
            )

    entity_identifier = Path(entity_identifier_path) if entity_identifier_path else None
    market_cache_path = Path(crsp_market_cache_path) if crsp_market_cache_path else None
    daily_root = Path(crsp_daily_root) if crsp_daily_root else None
    permno, source_kind, source_path, price_history = _load_exact_price_history(
        company_id=resolved_company_id,
        as_of_time=resolved_as_of_time,
        entity_identifier_path=entity_identifier,
        crsp_market_cache_path=market_cache_path,
        crsp_daily_root=daily_root,
    )
    summary["price_history_permno"] = permno
    summary["price_history_source"] = source_path
    if source_kind and not price_history.empty:
        exact_price_metrics = _compute_exact_price_metrics(
            price_history,
            as_of_time=resolved_as_of_time,
            source_kind=source_kind,
        )
        for metric_name, metric_payload in exact_price_metrics.items():
            previous = features.get(metric_name)
            if not _needs_exact_price_history_repair(previous):
                continue
            previous_flags = set(_quality_flags(previous))
            repair_flags = {"repaired_exact_price_history_from_crsp"}
            if "low_frequency_price_history" in previous_flags:
                repair_flags.add("replaced_low_frequency_price_history")
            repaired_record = _derived_metric_record(
                metric_name=metric_name,
                unit=str(metric_payload["unit"]),
                value=metric_payload["value"],
                support_mode="exact",
                as_of_time=resolved_as_of_time,
                formula=str(metric_payload["formula"]),
                source_records=[previous] if isinstance(previous, dict) else [],
                component_values={
                    **dict(metric_payload["component_values"]),
                    "selected_price_series": {
                        "source_kind": source_kind,
                        "group_field": "permno",
                        "group_value": permno,
                        "price_field": "price_proxy",
                        "time_field": "trade_date",
                    },
                },
                fallback_used=f"{source_kind}_price_history",
                quality_flags=sorted(repair_flags),
            )
            repaired_record["provenance"] = _dedupe_provenance(
                previous,
                {
                    "provenance": [
                        {
                            "artifact_type": "MarketTimeseries",
                            "artifact_id": f"{source_kind}:{permno}",
                            "source": source_path,
                            "published_at": resolved_as_of_time,
                            "ingested_at": resolved_as_of_time,
                            "hash": None,
                        }
                    ]
                },
            )
            changed |= _write_metric(
                features,
                metric_name=metric_name,
                record=repaired_record,
                summary=summary,
            )
    return payload, changed, summary
