"""Causal-style impact model utilities for Mechanism Brain.

This module provides:
- Offline training data serialization format
- Fast deterministic inference for objective impact distributions
- Action-id -> legacy action-type normalization for outcomes data alignment
"""

from __future__ import annotations

import json
import math
import os
import pickle
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

from .causal_feature_contract import (
    INFERENCE_SOURCE_KEYS,
    build_contract_feature_map,
    canonicalize_feature_name,
)
from .model_feature_bundle import build_model_feature_bundle, feature_view_from_snapshot, get_bundle_value
from .runtime_feature_adapter import adapt_snapshot, resolve_feature_value


_Z = {
    "p10": -1.2815515655446004,
    "p25": -0.6744897501960817,
    "median": 0.0,
    "p75": 0.6744897501960817,
    "p90": 1.2815515655446004,
}

DEFAULT_CAUSAL_IMPACT_MODEL_ARTIFACT = Path(
    "data/models/causal_impact_model_v6_bundle_contract_hgb_actiontype.json"
)
DEFAULT_CAUSAL_ROUTING_CONFIG_PATH = Path("configs/causal_capital_routing_prod_dividend_v2.json")

# Backward-compatible feature normalization defaults for legacy artifacts
# that predate explicit transform metadata.
_DEFAULT_USD_MILLIONS_FEATURES = {
    "base_market_cap",
    "base_net_debt",
    "base_revenue_ttm",
    "action_size",
}
_DEFAULT_RATE_PERCENT_FEATURES = {
    "macro_rate_10y",
    "macro_rate_2y",
    "macro_sofr",
}
_DEFAULT_OAS_PERCENT_FEATURES = {
    "macro_ig_oas",
    "macro_hy_oas",
}


class _RidgePredictor:
    """Compatibility wrapper for legacy pickled ridge predictors."""

    def __init__(self, beta: Any = None) -> None:
        self.beta = beta

    def predict(self, X: Any) -> list[float]:  # noqa: N803
        beta_raw = self.beta
        if beta_raw is None:
            return [0.0 for _ in list(X or [])]
        try:
            beta = [float(v) for v in list(beta_raw)]
        except Exception:
            return [0.0 for _ in list(X or [])]
        if not beta:
            return [0.0 for _ in list(X or [])]
        out: list[float] = []
        for row_raw in list(X or []):
            try:
                row = [float(v) for v in list(row_raw)]
            except Exception:
                row = []
            y = beta[0]
            width = min(len(row), max(0, len(beta) - 1))
            for idx in range(width):
                y += beta[idx + 1] * row[idx]
            out.append(float(y))
        return out


class _BundleUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        # Legacy training artifacts may pickle _RidgePredictor under __main__
        # when the trainer is executed as a script path.
        if module == "__main__" and name == "_RidgePredictor":
            return _RidgePredictor
        return super().find_class(module, name)


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    qq = _clip(float(q), 0.0, 1.0)
    pos = qq * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    w = pos - lo
    return xs[lo] * (1.0 - w) + xs[hi] * w


def _scalar_float(value: Any) -> Optional[float]:
    raw = _to_float(value)
    if raw is not None:
        return float(raw)
    try:
        seq = list(value)  # type: ignore[arg-type]
    except Exception:
        return None
    if not seq:
        return None
    return _scalar_float(seq[0])


def _to_float(v: Any, default: Optional[float] = None) -> Optional[float]:
    if v is None:
        return default
    if isinstance(v, bool):
        return float(v)
    try:
        out = float(v)
    except Exception:
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def _feature_value(raw: Any) -> Any:
    if isinstance(raw, dict):
        return raw.get("value")
    return raw


def _extract_feature(features: Dict[str, Any], name: str, default: Any = None) -> Any:
    return resolve_feature_value(features, name, default=default)


