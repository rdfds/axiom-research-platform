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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train causal impact model artifact.")
    p.add_argument(
        "--outcomes-path",
        default="",
        help=(
            "Optional path to outcomes dataset parquet. If omitted, resolves the richest available "
            "normalized outcomes artifact before falling back to action_outcomes.parquet."
        ),
    )
    p.add_argument(
        "--out-path",
        default="data/models/causal_impact_model_v1.json",
        help="Output model artifact path",
    )
    p.add_argument("--min-rows-per-action", type=int, default=300)
    p.add_argument("--ridge-alpha", type=float, default=2.0)
    p.add_argument("--winsor-pct", type=float, default=0.01)
    p.add_argument("--validation-fraction", type=float, default=0.20)
    p.add_argument("--min-validation-rows", type=int, default=100)
    p.add_argument("--crossfit-folds", type=int, default=2)
    p.add_argument("--dr-min-treated-rows", type=int, default=250)
    p.add_argument("--dr-min-control-rows", type=int, default=500)
    p.add_argument("--propensity-clip", type=float, default=0.05)
    p.add_argument(
        "--model-family",
        choices=["linear", "hgb"],
        default="hgb",
        help="Estimator family for DR CATE models. `hgb` uses nonlinear gradient-boosted trees.",
    )
    p.add_argument(
        "--cell-level",
        choices=["action_type", "action_subtype"],
        default="action_subtype",
        help="Train DR models per action_type or per action_type::action_subtype cell.",
    )
    p.add_argument(
        "--gate-min-oos-r2",
        type=float,
        default=0.0,
        help="Enable a causal cell only when OOS R2 is at least this threshold.",
    )
    p.add_argument("--gate-min-train-rows", type=int, default=3000)
    p.add_argument("--gate-min-treated-rows", type=int, default=1000)
    p.add_argument("--gate-min-control-rows", type=int, default=5000)
    p.add_argument(
        "--subtype-target-normalize",
        action="store_true",
        help="Normalize objective labels within action subtype buckets (robust z-score).",
    )
    p.add_argument(
        "--model-card-out",
        default="",
        help="Optional path for model card JSON. If omitted, model card is embedded in model artifact only.",
    )
    p.add_argument(
        "--skip-bundle-write",
        action="store_true",
        help="For HGB models, skip writing the pickle bundle. Useful for model-card benchmarking in low-disk environments.",
    )
    p.add_argument(
        "--train-end-date",
        default="",
        help="Optional YYYY-MM-DD cutoff; rows with action_date <= cutoff are train.",
    )
    p.add_argument(
        "--action-id-allowlist",
        default="",
        help="Optional comma-delimited action_id allowlist for targeted rescue training.",
    )
    p.add_argument(
        "--action-id-allowlist-file",
        default="",
        help="Optional newline-delimited action_id allowlist file for targeted rescue training.",
    )
    p.add_argument(
        "--cell-allowlist",
        default="",
        help="Optional comma-delimited action_cell allowlist (e.g. loan_issuance::all,loan_issuance::revolver_*).",
    )
    p.add_argument(
        "--cell-allowlist-file",
        default="",
        help="Optional newline-delimited action_cell allowlist file.",
    )
    p.add_argument(
        "--objective-allowlist",
        default="",
        help="Optional comma-delimited objective allowlist (e.g. value_creation,risk_reduction).",
    )
    p.add_argument(
        "--capital-routing-config-path",
        default="configs/causal_capital_routing_v1.json",
        help="Routing config path used for capital-only causal training presets.",
    )
    p.add_argument(
        "--capital-phase1-only",
        action="store_true",
        help="Restrict training to phase-1 capital actions and their configured objective allowlists.",
    )
    p.add_argument(
        "--dr-control-scope",
        choices=["global", "action_family"],
        default="global",
        help=(
            "Control pool scope for doubly-robust training. "
            "`action_family` compares an action subtype only against alternatives in the same top-level family."
        ),
    )
    p.add_argument(
        "--validation-start-date",
        default="",
        help="Optional YYYY-MM-DD start date for validation rows (overrides validation-fraction if set).",
    )
    p.add_argument(
        "--progress-every-cells",
        type=int,
        default=1,
        help="Progress log frequency for subtype-cell training (1 = every cell, 0 = no per-cell logs).",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce progress logging (final JSON output is still printed).",
    )
    return p.parse_args()


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _numeric_series_or_nan(df: pd.DataFrame, column_name: str) -> pd.Series:
    if column_name in df.columns:
        return _to_num(df[column_name])
    return pd.Series(np.nan, index=df.index, dtype=float)


def _resolve_outcomes_path(raw_path: str) -> Path:
    raw_path_str = str(raw_path or "").strip()
    if raw_path_str:
        candidate = Path(raw_path_str)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"outcomes dataset not found: {candidate}")
    for path in _DEFAULT_OUTCOMES_CANDIDATES:
        if path.exists():
            return path
    return _DEFAULT_OUTCOMES_CANDIDATES[0]


def _validate_action_allowlist_coverage(df: pd.DataFrame, action_id_allowlist: List[str]) -> None:
    if not action_id_allowlist:
        return
    action_ids = set(df.get("action_id_key", pd.Series(dtype=str)).dropna().astype(str).tolist())
    explicit_ids = [
        action_id
        for action_id in action_id_allowlist
        if action_id and not re.search(r"[*?[]", str(action_id))
    ]
    missing = [action_id for action_id in explicit_ids if action_id not in action_ids]
    if missing:
        raise ValueError(
            "action-id allowlist includes actions missing from outcomes dataset after filtering: "
            + ", ".join(sorted(missing))
        )


def _canonical_subtype(raw: Any) -> str:
    s = str(raw or "").strip().lower()
    if not s:
        return "unknown"
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def _canonical_action_id(raw: Any) -> str:
    s = str(raw or "").strip().lower()
    s = re.sub(r"\s+", "", s)
    return s


def _parse_action_id_allowlist(raw: str, file_path: str) -> List[str]:
    tokens: List[str] = []

    def _append_text(text: str) -> None:
        for piece in str(text or "").replace("\n", ",").split(","):
            value = _canonical_action_id(piece)
            if value and value not in tokens:
                tokens.append(value)

    if str(raw or "").strip():
        _append_text(str(raw))
    if str(file_path or "").strip():
        _append_text(Path(str(file_path)).read_text())
    return tokens


def _matches_action_allowlist(action_id: str, allowlist: List[str]) -> bool:
    if not allowlist:
        return True
    aid = _canonical_action_id(action_id)
    if not aid:
        return False
    return any(fnmatchcase(aid, token) for token in allowlist)


def _parse_cell_allowlist(raw: str, file_path: str) -> List[str]:
    tokens: List[str] = []

    def _append_text(text: str) -> None:
        for piece in str(text or "").replace("\n", ",").split(","):
            value = str(piece or "").strip().lower()
            if value and value not in tokens:
                tokens.append(value)

    if str(raw or "").strip():
        _append_text(str(raw))
    if str(file_path or "").strip():
        _append_text(Path(str(file_path)).read_text())
    return tokens


