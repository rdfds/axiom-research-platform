from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .precedent_brain import (
    _STATE_VECTOR_CORE_CRITICAL_FEATURES,
    _STATE_VECTOR_GROUPS,
    _STATE_VECTOR_MATCHING_COLS,
    _STATE_VECTOR_V2_DEFAULT_BLEND_WEIGHTS,
    _STATE_VECTOR_V2_DEFAULT_FEATURE_RELATIVE_WEIGHTS,
    _STATE_VECTOR_V2_DEFAULT_GATES,
    _STATE_VECTOR_V2_DEFAULT_GROUP_MULTIPLIERS,
    _STATE_VECTOR_V2_DEFAULT_GROUP_WEIGHTS,
    _STATE_VECTOR_V2_DEFAULT_PENALTIES,
    _WEIGHTED_DISTANCE_V2_VERSION,
)


def load_precedent_distance_v2_objective(path: str | Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("Objective config must be a JSON object")
    return payload


def _scope_multipliers(scope_key: str) -> Dict[str, float]:
    scope = str(scope_key or "").strip().lower()
    if scope.startswith("capital_return.dividend"):
        return dict(_STATE_VECTOR_V2_DEFAULT_GROUP_MULTIPLIERS.get("capital_return.dividend", {}))
    if scope in {
        "capital_return.open_market_buyback",
        "capital_return.accelerated_share_repurchase",
        "capital_return.buyback",
    }:
        return dict(_STATE_VECTOR_V2_DEFAULT_GROUP_MULTIPLIERS.get("capital_return.buyback", {}))
    return dict(_STATE_VECTOR_V2_DEFAULT_GROUP_MULTIPLIERS.get(scope, {}))


def default_scope_configuration(scope_key: str) -> Dict[str, Any]:
    scope = str(scope_key or "").strip().lower()
    group_weights = dict(_STATE_VECTOR_V2_DEFAULT_GROUP_WEIGHTS)
    for group_name, multiplier in _scope_multipliers(scope).items():
        group_weights[group_name] = float(group_weights.get(group_name, 1.0)) * float(multiplier)
    feature_relative_weights = dict(_STATE_VECTOR_V2_DEFAULT_FEATURE_RELATIVE_WEIGHTS)
    gates = dict(_STATE_VECTOR_V2_DEFAULT_GATES)
    penalties = dict(_STATE_VECTOR_V2_DEFAULT_PENALTIES)
    critical = set(_STATE_VECTOR_CORE_CRITICAL_FEATURES)
    subtype_text = scope.split(".", 1)[1] if "." in scope else scope
    if scope.startswith("capital_structure.") or "debt" in scope or "refinanc" in subtype_text:
        gates["max_size_gap"] = 1.30
        penalties["sector_penalty_weight"] = 0.22
        critical.update({"state_vector_v1.market_access", "state_vector_v1.credit_spread"})
    elif scope.startswith("capital_return.dividend") or subtype_text.startswith("dividend"):
        gates["max_size_gap"] = 1.05
        gates["soft_burden_gap"] = 1.10
        critical.update({"state_vector_v1.cash_generation"})
    elif "buyback" in scope or "repurchase" in scope or "buyback" in subtype_text:
        feature_relative_weights["state_vector_v1.valuation_multiple"] = 1.35
        feature_relative_weights["state_vector_v1.cash_generation"] = 1.20
        gates["max_size_gap"] = 1.15
    return {
        "scope_key": scope,
        "group_weights": group_weights,
        "feature_relative_weights": feature_relative_weights,
        "gates": gates,
        "penalties": penalties,
        "blend_weights": dict(_STATE_VECTOR_V2_DEFAULT_BLEND_WEIGHTS),
        "critical_features": list(critical),
        "use_in_runtime": True,
    }


def build_precedent_distance_v2_payload(
    *,
    scopes: Dict[str, Dict[str, Any]],
    objective_config: Optional[Dict[str, Any]] = None,
    benchmark_key: str = "",
    notes: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "version": "precedent_distance_weights_v2",
        "state_distance_version": _WEIGHTED_DISTANCE_V2_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_key": str(benchmark_key or ""),
        "objective": objective_config or {},
        "notes": notes or {},
        "scopes": scopes,
    }


