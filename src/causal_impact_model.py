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


