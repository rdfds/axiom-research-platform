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


