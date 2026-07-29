"""Step 6 Candidate Generation Engine.

Deterministic, schema-constrained candidate generation under a frozen RecommendationRun.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import uuid

from .model_feature_bundle import feature_view_from_snapshot
from .action_ontology import ActionSchemaRegistry
from .recommendation_run import RecommendationRun
from .runtime_feature_adapter import resolve_feature_record


_RELATION_MAP = {
    ">": "greater_than",
    ">=": "greater_than",
    "<": "less_than",
    "<=": "less_than",
    "==": "equal",
    "=": "equal",
    "between": "within_range",
    "in_range": "within_range",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _round_num(v: float) -> float:
    return float(round(float(v), 12))


def _json_safe(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _json_safe(v[k]) for k in sorted(v.keys(), key=str)}
    if isinstance(v, list):
        return [_json_safe(x) for x in v]
    if isinstance(v, tuple):
        return [_json_safe(x) for x in v]
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return _round_num(v)
    return v


def _candidate_signature(action_id: str, parameters: Dict[str, Any]) -> str:
    norm = _json_safe(parameters)
    key = f"{action_id}|{json.dumps(norm, sort_keys=True, separators=(',', ':'), ensure_ascii=True)}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _feature_record(features: Dict[str, Any], feature_name: str) -> Optional[Dict[str, Any]]:
    if not isinstance(features, dict):
        return None
    raw = features.get(feature_name)
    return raw if isinstance(raw, dict) else None


def _feature_is_hard_blocked(raw: Any) -> bool:
    if not isinstance(raw, dict):
        return False
    support_mode = str(raw.get("support_mode") or "").strip().lower()
    applicability_status = str(raw.get("applicability_status") or "").strip().lower()
    quality_flags = {
        str(flag).strip().lower()
        for flag in (raw.get("quality_flags") or [])
        if flag is not None
    }
    if support_mode == "unsupported":
        return True
    if applicability_status in {"unsupported", "diagnostic"}:
        return True
    if "unsupported_metric" in quality_flags or "sector_native_metrics_required" in quality_flags:
        return True
    return False


def _feature_replacement_value(features: Dict[str, Any], feature_name: str, raw: Any) -> Any:
    if feature_name != "capital_structure.interest_coverage":
        return None
    replacement = _feature_record(features, "capital_structure.fixed_charge_coverage")
    if not isinstance(replacement, dict) or _feature_is_hard_blocked(replacement):
        return None
    replacement_applicability = str(replacement.get("applicability_status") or "").strip().lower()
    current_applicability = str(raw.get("applicability_status") or "").strip().lower() if isinstance(raw, dict) else ""
    if replacement_applicability != "primary":
        return None
    if _feature_is_hard_blocked(raw) or current_applicability in {"secondary", "diagnostic", "unsupported"}:
        return replacement.get("value")
    return None


def _feature_value(features: Dict[str, Any], feature_name: str, default: Any = None) -> Any:
    if not isinstance(features, dict):
        return default

    raw = resolve_feature_record(features, feature_name)
    if raw is not None:
        replacement_value = _feature_replacement_value(features, feature_name, raw)
        if replacement_value is not None:
            return replacement_value
        if _feature_is_hard_blocked(raw):
            return default
        if isinstance(raw, dict) and "value" in raw:
            return raw.get("value")
        return raw

    # Nested fallback: allow key lookups against parent feature values.
    parts = feature_name.split(".")
    for i in range(len(parts) - 1, 0, -1):
        prefix = ".".join(parts[:i])
        suffix = parts[i:]
        if prefix not in features:
            continue
        raw = features.get(prefix)
        if _feature_is_hard_blocked(raw):
            continue
        if isinstance(raw, dict) and "value" in raw:
            raw = raw.get("value")
        cur = raw
        ok = True
        for tok in suffix:
            if isinstance(cur, dict) and tok in cur:
                cur = cur[tok]
            else:
                ok = False
                break
        if ok:
            return cur

    return default


def _to_float(v: Any, default: Optional[float] = None) -> Optional[float]:
    if v is None:
        return default
    try:
        return float(v)
    except Exception:
        return default


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _default_funding_mixes() -> List[Dict[str, float]]:
    return [
        {"cash": 1.0, "debt": 0.0, "equity": 0.0},
        {"cash": 0.7, "debt": 0.3, "equity": 0.0},
        {"cash": 0.5, "debt": 0.5, "equity": 0.0},
    ]


def _is_explicit_false(value: Any) -> bool:
    if value is False:
        return True
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) == 0.0
    if isinstance(value, str):
        return value.strip().lower() in {"false", "0", "no"}
    return False


def _is_explicit_true(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) != 0.0
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return False


def _dividend_initiation_nonpayer_signal(features: Dict[str, Any]) -> bool:
    payer_value = _feature_value(features, "capital_return.dividend_payer_flag")
    if _is_explicit_true(payer_value):
        return False
    if _is_explicit_false(payer_value):
        return True

    last_dividend_event = str(_feature_value(features, "capital_return.last_dividend_event_type") or "").strip().lower()
    if last_dividend_event:
        return False

    payer_record = _feature_record(features, "capital_return.dividend_payer_flag") or {}
    missing_reason = str(payer_record.get("missing_reason") or "").strip().lower()
    quality_flags = {
        str(flag).strip().lower()
        for flag in (payer_record.get("quality_flags") or [])
        if flag is not None
    }
    return bool(
        missing_reason == "unavailable"
        or "event_history_unavailable" in quality_flags
        or "dividend_event_schema_incomplete" in quality_flags
        or "dividend_fact_fallback_no_positive_values" in quality_flags
        or "dividend_fact_fallback_missing_date" in quality_flags
    )


def _product_params(spec: Dict[str, Any], max_variants: int = 96) -> List[Dict[str, Any]]:
    if not spec:
        return [{}]
    keys = sorted(spec.keys())
    values: List[List[Any]] = []
    for k in keys:
        v = spec[k]
        if isinstance(v, list):
            values.append(v if v else [None])
        else:
            values.append([v])
    out: List[Dict[str, Any]] = []
    for combo in itertools.product(*values):
        out.append({k: combo[i] for i, k in enumerate(keys) if combo[i] is not None})
        if len(out) >= max_variants:
            break
    return out


def _numeric_value_grid(parameter_name: str, parameter_def: Dict[str, Any], *, max_anchor: float) -> List[float]:
    minimum = _to_float(parameter_def.get("min"), 0.0) or 0.0
    maximum = _to_float(parameter_def.get("max"), None)
    unit = str(parameter_def.get("unit", "") or "").lower()

    if unit == "years" or parameter_name in {"tenor_years", "new_tenor_years"}:
        anchors = [3.0, 5.0, 7.0]
    elif unit == "months" or parameter_name.endswith("_months"):
        anchors = [6.0, 12.0, 24.0]
    elif parameter_name == "leverage_post_close":
        anchors = [2.0, 2.5, 3.0]
    else:
        anchors = [max(minimum, max_anchor * 0.1), max(minimum, max_anchor * 0.2), max(minimum, max_anchor * 0.35)]

    out: List[float] = []
    for anchor in anchors:
        value = max(minimum, float(anchor))
        if maximum is not None:
            value = min(value, maximum)
        out.append(_round_num(value))
    return sorted(set(out))


def _candidate_variant_sort_key(candidate: Dict[str, Any]) -> Any:
    params = dict(candidate.get("parameters", {}) or {})
    funding_mix = params.get("funding_mix")
    debt_share = None
    if isinstance(funding_mix, dict):
        debt_share = _to_float(funding_mix.get("debt"), 0.0)
    generation_confidence = _to_float(candidate.get("generation_confidence"), 0.0) or 0.0

    size_pct = _to_float(params.get("size_pct_market_cap"))
    size_abs = _to_float(params.get("size_absolute_usd"))
    initial_yield = _to_float(params.get("initial_yield_pct"))
    percent_change = _to_float(params.get("percent_change"))

    return (
        str(candidate.get("action_id", "")),
        -generation_confidence,
        0 if size_pct is not None else 1,
        float(size_pct) if size_pct is not None else float("inf"),
        0 if size_abs is not None else 1,
        float(size_abs) if size_abs is not None else float("inf"),
        0 if initial_yield is not None else 1,
        float(initial_yield) if initial_yield is not None else float("inf"),
        0 if percent_change is not None else 1,
        abs(float(percent_change)) if percent_change is not None else float("inf"),
        float(debt_share) if debt_share is not None else 0.0,
        str(candidate.get("generation_source", "")),
        str(candidate.get("candidate_signature", "")),
    )


@dataclass
class Precondition:
    feature_name: str
    assumed_relation: str
    value: Any
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RationaleReference:
    reference_type: str
    reference_id: str
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ActionCandidateDraft:
    candidate_id: str
    run_id: str
    action_type: str
    action_subtype: str
    action_id: str
    parameters: Dict[str, Any]
    assumed_preconditions: List[Precondition]
    generation_source: str
    rationale_refs: List[RationaleReference]
    generation_confidence: float
    created_at: str
    candidate_signature: str
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["assumed_preconditions"] = [p.to_dict() for p in self.assumed_preconditions]
        out["rationale_refs"] = [r.to_dict() for r in self.rationale_refs]
        if not out["params"]:
            out["params"] = dict(self.parameters)
        return out


