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


