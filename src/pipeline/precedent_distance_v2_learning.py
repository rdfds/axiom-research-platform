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


