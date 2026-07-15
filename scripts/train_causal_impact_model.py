#!/usr/bin/env python
"""Train a lightweight causal-style impact model for Mechanism Brain.

Outputs a JSON artifact consumed at runtime by `src/causal_impact_model.py`.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime, timezone
from fnmatch import fnmatchcase
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.causal_feature_contract import (
    CONTRACT_VERSION as CAUSAL_FEATURE_CONTRACT_VERSION,
    FEATURE_ALIASES as CAUSAL_FEATURE_ALIASES,
    FEATURE_ORDER as CAUSAL_FEATURE_ORDER,
    OAS_PERCENT_FEATURES as CONTRACT_OAS_PERCENT_FEATURES,
    RATE_PERCENT_FEATURES as CONTRACT_RATE_PERCENT_FEATURES,
    SIGNED_LOG1P_FEATURES as CONTRACT_SIGNED_LOG1P_FEATURES,
    USD_MILLIONS_FEATURES as CONTRACT_USD_MILLIONS_FEATURES,
)

FEATURE_ORDER = list(CAUSAL_FEATURE_ORDER)
_DEFAULT_OUTCOMES_CANDIDATES: Tuple[Path, ...] = (
    _REPO_ROOT / "data" / "curated" / "action_outcomes_with_credit_ratings.normalized_full.rich_contract_v3.parquet",
    _REPO_ROOT / "data" / "curated" / "action_outcomes_with_credit_ratings.normalized_full.rich_contract_v2.parquet",
    _REPO_ROOT / "data" / "curated" / "action_outcomes_with_credit_ratings.normalized_full.rich_contract_v1.parquet",
    _REPO_ROOT / "data" / "curated" / "action_outcomes_with_credit_ratings.normalized_full.parquet",
    _REPO_ROOT / "data" / "curated" / "action_outcomes_with_credit_ratings.parquet",
    _REPO_ROOT / "data" / "curated" / "action_outcomes.parquet",
)

# Canonical feature normalization rules shared with runtime inference.
USD_MILLIONS_FEATURES = {
    *CONTRACT_USD_MILLIONS_FEATURES,
}
RATE_PERCENT_FEATURES = {
    *CONTRACT_RATE_PERCENT_FEATURES,
}
OAS_PERCENT_FEATURES = {
    *CONTRACT_OAS_PERCENT_FEATURES,
}
SIGNED_LOG1P_FEATURES = {
    *CONTRACT_SIGNED_LOG1P_FEATURES,
}

OBJECTIVES = [
    "value_creation",
    "risk_reduction",
    "growth",
    "rating_preservation",
    "optionality",
    "growth_v2",
    "optionality_v2",
]


class _RidgePredictor:
    """Pickle-friendly ridge predictor wrapper with sklearn-like API."""

    def __init__(self, beta: np.ndarray) -> None:
        self.beta = np.asarray(beta, dtype=float)

    def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        arr = np.asarray(X, dtype=float)
        return _linear_predict(self.beta, arr)


# Prefer serializing under runtime module when importable; otherwise keep
# __main__ and let runtime fallback unpickler handle legacy/main-module objects.
try:
    from src import causal_impact_model as _runtime_causal_impact_model

    setattr(_runtime_causal_impact_model, "_RidgePredictor", _RidgePredictor)
    _RidgePredictor.__module__ = "src.causal_impact_model"
except Exception:
    pass


def _log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[train_causal] {ts} {msg}", flush=True)


