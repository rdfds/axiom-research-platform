from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .causal_impact_model import (
    DEFAULT_CAUSAL_IMPACT_MODEL_ARTIFACT,
    DEFAULT_CAUSAL_ROUTING_CONFIG_PATH,
)

DEFAULT_PRECEDENT_RETRIEVAL_VERSION = "precedent_retrieval_state_vector_v1"

_RUNTIME_ENV_KEYS = (
    "AXIOM_DATA_ROOT",
    "AXIOM_COMPANYFACTS_ROOT",
    "AXIOM_ENABLE_RUNTIME_FEATURE_ADAPTER",
    "AXIOM_RUNTIME_FEATURE_ADAPTER_PROFILE",
    "AXIOM_RUNTIME_FEATURE_ADAPTER_RULES",
    "CAUSAL_IMPACT_MODEL_PATH",
    "CAUSAL_ROUTING_CONFIG_PATH",
    "CAUSAL_IMPACT_MODE",
    "CAUSAL_ACTION_BLOCKLIST",
    "CAUSAL_ACTION_DENYLIST",
    "CAUSAL_ACTION_BLOCKLIST_PATH",
    "CAUSAL_MIN_OBJECTIVE_OOS_R2",
    "CAUSAL_STRICT_QUALITY_FLOOR",
    "CAUSAL_STRICT_SUPPORT_FLOOR",
    "CAUSAL_STRICT_MIN_TRAIN_ROWS",
    "CAUSAL_STRICT_MIN_OOS_R2",
    "CAUSAL_STRICT_MIN_TREATED_ROWS",
    "CAUSAL_STRICT_MIN_CONTROL_ROWS",
    "MECHANISM_MODEL_VERSION",
    "PRECEDENT_RETRIEVAL_VERSION",
    "RECO_PRECEDENT_WORKERS",
    "RECOMMENDATION_RUN_TMP_DIR",
)


def _normalize_env_value(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    value = str(raw).strip()
    if not value:
        return None
    if value.lower() in {"none", "null"}:
        return None
    return value


def _safe_float(raw: Any) -> Optional[float]:
    try:
        out = float(raw)
    except Exception:
        return None
    if out != out:
        return None
    return float(out)


def _safe_int(raw: Any) -> Optional[int]:
    val = _safe_float(raw)
    if val is None:
        return None
    return int(val)


