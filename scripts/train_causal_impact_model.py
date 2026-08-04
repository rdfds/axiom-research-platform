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


def _matches_cell_allowlist(cell_key: str, allowlist: List[str]) -> bool:
    if not allowlist:
        return True
    key = str(cell_key or "").strip().lower()
    if not key:
        return False
    return any(fnmatchcase(key, token) for token in allowlist)


def _with_action_cells(df: pd.DataFrame, cell_level: str) -> pd.DataFrame:
    out = df.copy()
    normalized_family = out.get("normalized_action_family", pd.Series("", index=out.index)).astype(str).str.strip().str.lower()
    normalized_subfamily = out.get("normalized_action_subfamily", pd.Series("", index=out.index)).map(_canonical_subtype)
    raw_action_type = out.get("action_type", pd.Series("", index=out.index)).astype(str).str.strip().str.lower()
    raw_subtype = out.get("action_subtype", pd.Series("", index=out.index)).map(_canonical_subtype)
    action_type = normalized_family.where(normalized_family != "", raw_action_type)
    subtype = normalized_subfamily.where(normalized_subfamily != "", raw_subtype)
    normalized_action_id = out.get("normalized_action_id", pd.Series("", index=out.index)).map(_canonical_action_id)
    existing_action_id = out.get("action_id", pd.Series("", index=out.index)).map(_canonical_action_id)
    effective_action_id = normalized_action_id.where(normalized_action_id != "", existing_action_id)
    out["action_type_key"] = action_type
    out["action_subtype_key"] = subtype
    out["action_id_key"] = effective_action_id.where(effective_action_id != "", action_type + "." + subtype)
    if str(cell_level) == "action_subtype":
        out["action_cell"] = action_type + "::" + subtype
    else:
        out["action_cell"] = action_type + "::all"
    return out


def _parse_objective_allowlist(raw: str) -> List[str]:
    out: List[str] = []
    for piece in str(raw or "").replace("\n", ",").split(","):
        value = str(piece or "").strip()
        if value and value in OBJECTIVES and value not in out:
            out.append(value)
    return out


def _load_capital_routing_config(path_value: str) -> Dict[str, Any]:
    path = Path(str(path_value or "").strip())
    if not str(path) or not path.exists() or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _capital_phase1_defaults(config: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    actions = dict(config.get("actions", {}) or {})
    allow_actions: List[str] = []
    allow_objectives: List[str] = []
    for action_id, spec_raw in actions.items():
        spec = dict(spec_raw or {})
        status = str(spec.get("status", "") or "").strip().lower()
        if status not in {"enabled", "weak_prior_only"}:
            continue
        normalized_action = _canonical_action_id(action_id)
        if normalized_action and normalized_action not in allow_actions:
            allow_actions.append(normalized_action)
        for objective in list(spec.get("objective_allowlist", []) or []):
            objective_name = str(objective or "").strip()
            if objective_name in OBJECTIVES and objective_name not in allow_objectives:
                allow_objectives.append(objective_name)
    return allow_actions, allow_objectives


def _winsorize(y: pd.Series, p: float) -> pd.Series:
    if y.dropna().empty:
        return y
    lo = float(y.quantile(p))
    hi = float(y.quantile(1.0 - p))
    return y.clip(lower=lo, upper=hi)


def _robust_subtype_normalize(y: pd.Series, subtype: pd.Series) -> pd.Series:
    ys = _to_num(y).astype(float)
    st = subtype.astype(str).fillna("unknown")
    out = pd.Series(np.nan, index=ys.index, dtype=float)
    global_med = float(ys.median()) if ys.notna().any() else 0.0
    global_mad = float((ys - global_med).abs().median()) if ys.notna().any() else 1.0
    global_scale = max(1e-6, 1.4826 * global_mad)

    for key, idx in st.groupby(st).groups.items():
        grp = ys.loc[idx]
        ok = grp.dropna()
        if len(ok) < 200:
            out.loc[idx] = (grp - global_med) / global_scale
            continue
        med = float(ok.median())
        mad = float((ok - med).abs().median())
        scale = max(1e-6, 1.4826 * mad)
        out.loc[idx] = (grp - med) / scale
    return out.clip(-6.0, 6.0)


def _signed_log1p_series(y: pd.Series) -> pd.Series:
    ys = _to_num(y).astype(float)
    return np.sign(ys) * np.log1p(np.abs(ys))


def _robust_component_standardize(y: pd.Series) -> pd.Series:
    ys = _to_num(y).astype(float)
    ok = ys.dropna()
    if ok.empty:
        return ys
    med = float(ok.median())
    mad = float((ok - med).abs().median())
    scale = max(1e-6, 1.4826 * mad)
    return ((ys - med) / scale).clip(-6.0, 6.0)


def _build_targets(
    df: pd.DataFrame,
    normalize_by_subtype: bool = False,
    subtype_col: Optional[pd.Series] = None,
) -> Dict[str, pd.Series]:
    pe_6m = _numeric_series_or_nan(df, "outcome_pe_6m")
    ev_6m = _numeric_series_or_nan(df, "outcome_ev_ebitda_6m")
    pe_12m = _numeric_series_or_nan(df, "outcome_pe_12m")
    ev_12m = _numeric_series_or_nan(df, "outcome_ev_ebitda_12m")
    val_6m = pd.concat([pe_6m, ev_6m], axis=1).mean(axis=1, skipna=True)
    val_12m = pd.concat([pe_12m, ev_12m], axis=1).mean(axis=1, skipna=True)
    # Blend medium-horizon and long-horizon valuation signals for better stability.
    val = 0.65 * val_12m.fillna(val_6m) + 0.35 * val_6m.fillna(val_12m)

    leverage_delta = _numeric_series_or_nan(df, "leverage_delta")
    revenue_delta = _numeric_series_or_nan(df, "revenue_delta")
    margin_delta = _numeric_series_or_nan(df, "margin_delta")
    eps_delta = _numeric_series_or_nan(df, "eps_delta")
    roic_delta = _numeric_series_or_nan(df, "roic_delta")
    fcf_margin_delta = _numeric_series_or_nan(df, "fcf_margin_delta")
    spread_6m = _numeric_series_or_nan(df, "credit_spread_change_6m")
    spread_12m = _numeric_series_or_nan(df, "credit_spread_change_12m")
    rating_6m = _numeric_series_or_nan(df, "rating_migration_6m")
    rating_12m = _numeric_series_or_nan(df, "rating_migration_12m")

    # Rating migration: positive means upgrade and should improve rating_preservation.
    rating_signal = pd.concat(
        [
            -0.45 * leverage_delta,
            0.15 * fcf_margin_delta,
            -0.15 * spread_6m,
            -0.25 * spread_12m,
            0.20 * rating_6m,
            0.45 * rating_12m,
        ],
        axis=1,
    ).mean(axis=1, skipna=True)

    optionality_signal = pd.concat(
        [
            fcf_margin_delta,
            -0.25 * leverage_delta,
            -0.15 * spread_12m,
            0.20 * val_6m,
        ],
        axis=1,
    ).mean(axis=1, skipna=True)

    growth_signal = pd.concat(
        [
            revenue_delta,
            0.6 * margin_delta,
            0.8 * eps_delta,
            0.6 * roic_delta,
            0.4 * fcf_margin_delta,
        ],
        axis=1,
    ).mean(axis=1, skipna=True)

    # Experimental label variants for action-specific rescue work:
    # - growth_v2 dampens heavy-tailed revenue / EPS swings so the target is
    #   not dominated by a small number of corporate-action outliers.
    # - optionality_v2 drops sparse spread data and instead focuses on the
    #   more consistently observed mix of cash-generation, leverage relief,
    #   and near-term market confidence.
    growth_signal_v2 = pd.concat(
        [
            _robust_component_standardize(_signed_log1p_series(revenue_delta)),
            0.5 * _robust_component_standardize(_signed_log1p_series(eps_delta)),
            0.75 * _robust_component_standardize(margin_delta),
            0.75 * _robust_component_standardize(roic_delta),
            0.5 * _robust_component_standardize(fcf_margin_delta),
        ],
        axis=1,
    ).mean(axis=1, skipna=True)

    risk_signal = pd.concat(
        [
            -0.50 * leverage_delta,
            -0.20 * spread_6m,
            -0.25 * spread_12m,
            0.15 * rating_6m,
            0.25 * rating_12m,
            0.10 * fcf_margin_delta,
        ],
        axis=1,
    ).mean(axis=1, skipna=True)

    optionality_signal_v2 = pd.concat(
        [
            0.8 * _robust_component_standardize(fcf_margin_delta),
            0.8 * _robust_component_standardize(-1.0 * leverage_delta),
            0.4 * _robust_component_standardize(val_6m),
        ],
        axis=1,
    ).mean(axis=1, skipna=True)

    out = {
        "value_creation": val,
        "risk_reduction": risk_signal,
        "growth": growth_signal,
        "rating_preservation": rating_signal,
        "optionality": optionality_signal,
        "growth_v2": growth_signal_v2,
        "optionality_v2": optionality_signal_v2,
    }
    if normalize_by_subtype and subtype_col is not None:
        for k, y in list(out.items()):
            out[k] = _robust_subtype_normalize(y, subtype_col)
    return out


def _cell_scope_mask(
    action_type_series: pd.Series,
    action_type_key: str,
    subtype_key: str,
    dr_control_scope: str,
) -> pd.Series:
    scope = str(dr_control_scope or "global").strip().lower()
    if scope != "action_family":
        return pd.Series(True, index=action_type_series.index, dtype=bool)
    # Family-level "all" cells need the global pool; otherwise there is no control set.
    if str(subtype_key or "").strip().lower() == "all":
        return pd.Series(True, index=action_type_series.index, dtype=bool)
    return action_type_series.astype(str).eq(str(action_type_key))


def _resolve_dr_control_scope(
    requested_scope: str,
    capital_phase1_only: bool,
    argv: Optional[List[str]] = None,
) -> str:
    scope = str(requested_scope or "global").strip().lower() or "global"
    if not capital_phase1_only:
        return scope
    argv_tokens = list(argv if argv is not None else sys.argv[1:])
    scope_explicit = any(
        token == "--dr-control-scope" or str(token).startswith("--dr-control-scope=")
        for token in argv_tokens
    )
    if scope_explicit:
        return scope
    if scope == "global":
        return "action_family"
    return scope


def _ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    for f in FEATURE_ORDER:
        aliases = list(CAUSAL_FEATURE_ALIASES.get(f, (f,)))
        series = None
        for alias in aliases:
            if alias in x.columns:
                series = _to_num(x[alias])
                break
        if series is None:
            series = pd.Series(np.nan, index=x.index, dtype=float)
        s = series
        # Unit harmonization: inference snapshots may contain dollars / decimals / bps.
        if f in USD_MILLIONS_FEATURES:
            # If values look like raw dollars, convert to USD millions.
            s = s.where(s.abs() < 1e7, s / 1e6)
        if f in RATE_PERCENT_FEATURES:
            # Convert decimal rates (e.g., 0.045) into percent units (4.5).
            s = s.where(s.abs() > 1.0, s * 100.0)
        if f in OAS_PERCENT_FEATURES:
            # Convert bps (e.g., 120) into percent-like units (1.2).
            s = s.where(s.abs() < 50.0, s / 100.0)

        # Heavy-tailed financial features are modeled in signed log space.
        if f in SIGNED_LOG1P_FEATURES:
            s = np.sign(s) * np.log1p(np.abs(s))

        x[f] = s
    return x


def _feature_stats(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for f in FEATURE_ORDER:
        s = _to_num(df[f])
        med = float(s.median()) if not s.dropna().empty else 0.0
        mean = float(s.mean()) if not s.dropna().empty else med
        std = float(s.std(ddof=0)) if not s.dropna().empty else 1.0
        if not np.isfinite(std) or std <= 1e-12:
            std = 1.0
        out[f] = {"mean": mean, "std": std, "median": med}
    return out


def _standardize(df: pd.DataFrame, stats: Dict[str, Dict[str, float]]) -> np.ndarray:
    cols = []
    for f in FEATURE_ORDER:
        st = stats[f]
        s = _to_num(df[f]).fillna(float(st["median"]))
        cols.append(((s - float(st["mean"])) / float(st["std"])).to_numpy(dtype=float))
    return np.column_stack(cols)


def _fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> Tuple[np.ndarray, float]:
    n, p = X.shape
    Xt = np.column_stack([np.ones(n), X])
    eye = np.eye(p + 1, dtype=float)
    eye[0, 0] = 0.0  # no penalty on intercept
    beta = np.linalg.solve(Xt.T @ Xt + alpha * eye, Xt.T @ y)
    preds = Xt @ beta
    resid = y - preds
    resid_std = float(np.sqrt(np.mean(np.square(resid)))) if len(resid) else 0.0
    return beta, resid_std


def _linear_predict(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(X)), X]) @ beta


def _sigmoid(z: np.ndarray) -> np.ndarray:
    zc = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-zc))


def _fit_propensity_ridge(X: np.ndarray, t: np.ndarray, alpha: float) -> np.ndarray:
    # Ridge on binary labels, then calibrated through sigmoid at prediction time.
    beta, _ = _fit_ridge(X, t.astype(float), alpha)
    return beta


def _resolve_split_masks(
    df: pd.DataFrame,
    validation_fraction: float,
    train_end_date: str,
    validation_start_date: str,
) -> Tuple[pd.Series, pd.Series, Dict[str, Any]]:
    frac = max(0.05, min(0.40, float(validation_fraction)))
    n = len(df)
    if n == 0:
        empty = pd.Series([], dtype=bool)
        return empty, empty, {"method": "empty"}

    train_end = pd.to_datetime(train_end_date, errors="coerce", utc=True) if train_end_date else pd.NaT
    val_start = pd.to_datetime(validation_start_date, errors="coerce", utc=True) if validation_start_date else pd.NaT
    dates = pd.to_datetime(df.get("action_date"), errors="coerce", utc=True)
    has_date = dates.notna()
    meta: Dict[str, Any] = {"validation_fraction": frac}

    if has_date.sum() >= 100:
        if pd.notna(val_start):
            train_mask = (dates < val_start) | (~has_date)
            valid_mask = (dates >= val_start) & has_date
            meta.update({"method": "calendar_start", "validation_start": val_start.isoformat()})
        else:
            if pd.notna(train_end):
                cutoff = train_end
                meta["method"] = "calendar_end"
            else:
                cutoff = dates[has_date].quantile(1.0 - frac)
                meta["method"] = "date_quantile"
            train_mask = (dates <= cutoff) | (~has_date)
            valid_mask = (dates > cutoff) & has_date
            meta["split_cutoff"] = cutoff.isoformat()

        if int(train_mask.sum()) >= 200 and int(valid_mask.sum()) >= 50:
            meta["train_rows"] = int(train_mask.sum())
            meta["validation_rows"] = int(valid_mask.sum())
            return train_mask.astype(bool), valid_mask.astype(bool), meta

    # Deterministic fallback when dates are too sparse.
    cut = int(round((1.0 - frac) * n))
    cut = max(1, min(n - 1, cut))
    idx = np.arange(n)
    train_mask = pd.Series(idx < cut, index=df.index)
    valid_mask = pd.Series(idx >= cut, index=df.index)
    meta.update(
        {
            "method": "index_fallback",
            "split_index": int(cut),
            "train_rows": int(train_mask.sum()),
            "validation_rows": int(valid_mask.sum()),
        }
    )
    return train_mask.astype(bool), valid_mask.astype(bool), meta


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    denom = float(np.sum((y - np.mean(y)) ** 2))
    if denom <= 1e-12:
        return 0.0
    num = float(np.sum((y - yhat) ** 2))
    r2 = 1.0 - (num / denom)
    if not np.isfinite(r2):
        return 0.0
    return float(max(-1.0, min(1.0, r2)))


def _temporal_fold_ids(dates: pd.Series, folds: int) -> np.ndarray:
    n = len(dates)
    if n == 0:
        return np.array([], dtype=int)
    f = max(2, int(folds))
    idx = np.arange(n, dtype=int)
    d = pd.to_datetime(dates, errors="coerce", utc=True)
    # Stable fallback for missing dates: keep original order.
    sort_key = d.fillna(pd.Timestamp("1970-01-01", tz="UTC"))
    order = np.argsort(sort_key.to_numpy())
    fold_ids = np.zeros(n, dtype=int)
    for fold, part in enumerate(np.array_split(order, f)):
        fold_ids[part] = fold
    return fold_ids


def _fit_hgb_regressor(
    X: np.ndarray,
    y: np.ndarray,
    variant: str = "balanced",
) -> HistGradientBoostingRegressor:
    v = str(variant or "balanced").strip().lower()
    if v == "conservative":
        cfg = {
            "learning_rate": 0.035,
            "max_depth": 3,
            "max_leaf_nodes": 21,
            "min_samples_leaf": 80,
            "max_iter": 220,
            "l2_regularization": 0.4,
        }
    elif v == "expressive":
        cfg = {
            "learning_rate": 0.06,
            "max_depth": 6,
            "max_leaf_nodes": 63,
            "min_samples_leaf": 20,
            "max_iter": 380,
            "l2_regularization": 0.05,
        }
    else:
        cfg = {
            "learning_rate": 0.05,
            "max_depth": 4,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 30,
            "max_iter": 300,
            "l2_regularization": 0.1,
        }
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        random_state=42,
        **cfg,
    )
    model.fit(X, y)
    return model


def _fit_hgb_classifier(X: np.ndarray, y: np.ndarray) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=4,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        max_iter=250,
        l2_regularization=0.1,
        random_state=42,
    )
    model.fit(X, y.astype(int))
    return model


def _fit_dr_models_for_target_hgb(
    df: pd.DataFrame,
    y: pd.Series,
    stats: Dict[str, Dict[str, float]],
    train_mask: pd.Series,
    valid_mask: pd.Series,
    crossfit_folds: int,
    dr_min_treated_rows: int,
    dr_min_control_rows: int,
    min_validation_rows: int,
    propensity_clip: float,
    gate_min_oos_r2: float,
    gate_min_train_rows: int,
    gate_min_treated_rows: int,
    gate_min_control_rows: int,
    cell_level: str,
    cell_allowlist: List[str],
    dr_control_scope: str,
    progress_every_cells: int = 1,
    log_prefix: str = "",
    quiet: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    out: Dict[str, Any] = {"dr_models": {}}
    bundle: Dict[str, Any] = {}

    base = _ensure_features(df)
    yy = _to_num(y)
    train_ok = yy.notna() & train_mask
    valid_ok = yy.notna() & valid_mask
    if int(train_ok.sum()) < max(2000, dr_min_treated_rows + dr_min_control_rows):
        return out, bundle

    X_all = _standardize(base, stats)
    y_all = yy.to_numpy(dtype=float)
    action_type_series = df.get("action_type_key", pd.Series("", index=df.index)).astype(str)
    cell_series = df.get("action_cell", pd.Series("", index=df.index)).astype(str)
    date_series = pd.to_datetime(df.get("action_date"), errors="coerce", utc=True)

    train_indices = np.where(train_ok.to_numpy(dtype=bool))[0]
    valid_indices = np.where(valid_ok.to_numpy(dtype=bool))[0]
    if len(train_indices) == 0:
        return out, bundle

    candidate_cells = sorted(set(cell_series.loc[train_ok].dropna().tolist()))
    if str(cell_level) == "action_subtype":
        candidate_cells.extend(f"{t}::all" for t in sorted(set(action_type_series.loc[train_ok].dropna().tolist())))
    candidate_cells = sorted(set(candidate_cells))
    if cell_allowlist:
        candidate_cells = [c for c in candidate_cells if _matches_cell_allowlist(c, cell_allowlist)]
        if not quiet:
            _log(f"{log_prefix} filtered candidate_cells={len(candidate_cells)} via cell allowlist")

    clip = min(0.20, max(0.01, float(propensity_clip)))
    folds = max(2, int(crossfit_folds))

    if not quiet:
        _log(f"{log_prefix} candidate_cells={len(candidate_cells)}")
    progress_n = max(0, int(progress_every_cells))

    for idx, cell_key in enumerate(candidate_cells, start=1):
        if not quiet and progress_n > 0 and (idx == 1 or idx % progress_n == 0 or idx == len(candidate_cells)):
            _log(f"{log_prefix} processing cell {idx}/{len(candidate_cells)} key={cell_key}")
        if "::" not in str(cell_key):
            continue
        action_type_key, subtype_key = str(cell_key).split("::", 1)
        if subtype_key == "all":
            t_all = (action_type_series.to_numpy(dtype=str) == str(action_type_key)).astype(float)
        else:
            t_all = (cell_series.to_numpy(dtype=str) == str(cell_key)).astype(float)

        scope_mask = _cell_scope_mask(
            action_type_series=action_type_series,
            action_type_key=str(action_type_key),
            subtype_key=str(subtype_key),
            dr_control_scope=str(dr_control_scope),
        )
        train_scope = train_ok & scope_mask
        valid_scope = valid_ok & scope_mask

        train_indices_scoped = np.where(train_scope.to_numpy(dtype=bool))[0]
        valid_indices_scoped = np.where(valid_scope.to_numpy(dtype=bool))[0]
        if len(train_indices_scoped) == 0:
            continue

        t_train = t_all[train_indices_scoped]
        treated_rows = int(np.sum(t_train == 1.0))
        control_rows = int(np.sum(t_train == 0.0))
        if treated_rows < int(dr_min_treated_rows) or control_rows < int(dr_min_control_rows):
            continue

        X_train = X_all[train_indices_scoped]
        y_train = y_all[train_indices_scoped]
        n_train = len(train_indices_scoped)
        fold_ids = _temporal_fold_ids(date_series.iloc[train_indices_scoped], folds)
        psi = np.zeros(n_train, dtype=float)
        usable = np.ones(n_train, dtype=bool)

        for fold in range(folds):
            hold = fold_ids == fold
            if not np.any(hold):
                continue
            hold_dates = pd.to_datetime(date_series.iloc[train_indices_scoped][hold], errors="coerce", utc=True)
            cutoff = hold_dates.min() if hold_dates.notna().any() else pd.NaT
            if pd.notna(cutoff):
                fit = pd.to_datetime(date_series.iloc[train_indices_scoped], errors="coerce", utc=True) < cutoff
                fit = np.asarray(fit, dtype=bool)
            else:
                fit = ~hold
            if np.sum(fit) < 300:
                fit = ~hold
            if np.sum(fit) < 300:
                usable[hold] = False
                continue

            X_fit = X_train[fit]
            y_fit = y_train[fit]
            t_fit = t_train[fit]
            treat_fit = t_fit == 1.0
            ctrl_fit = t_fit == 0.0
            if int(np.sum(treat_fit)) < 80 or int(np.sum(ctrl_fit)) < 80:
                usable[hold] = False
                continue

            e_model = _fit_hgb_classifier(X_fit, t_fit)
            e_hat = e_model.predict_proba(X_train[hold])[:, 1]
            e_hat = np.clip(e_hat, clip, 1.0 - clip)

            m1_model = _fit_hgb_regressor(X_fit[treat_fit], y_fit[treat_fit])
            m0_model = _fit_hgb_regressor(X_fit[ctrl_fit], y_fit[ctrl_fit])
            m1 = m1_model.predict(X_train[hold])
            m0 = m0_model.predict(X_train[hold])

            yh = y_train[hold]
            th = t_train[hold]
            psi_hold = m1 - m0 + th * (yh - m1) / e_hat - (1.0 - th) * (yh - m0) / (1.0 - e_hat)
            psi[hold] = psi_hold

        if int(np.sum(usable)) < max(500, int(0.5 * n_train)):
            continue

        X_tau = X_train[usable]
        psi_tau = psi[usable]
        # OOS metric uses treated rows in validation period.
        n_valid = 0
        valid_treated = np.array([], dtype=int)
        m0_all = None
        if len(valid_indices_scoped) > 0:
            valid_treated = valid_indices_scoped[t_all[valid_indices_scoped] == 1.0]
            n_valid = int(len(valid_treated))
            if n_valid >= int(min_validation_rows):
                control_fit_all = t_train == 0.0
                if int(np.sum(control_fit_all)) >= 100:
                    m0_all = _fit_hgb_regressor(X_train[control_fit_all], y_train[control_fit_all])
        X_valid_t = X_all[valid_treated] if len(valid_treated) else np.zeros((0, X_all.shape[1]), dtype=float)
        y_valid_t = y_all[valid_treated] if len(valid_treated) else np.zeros((0,), dtype=float)

        challengers: List[Tuple[str, Any, str]] = []
        for hgb_variant in ("conservative", "balanced", "expressive"):
            challengers.append(
                (
                    f"hgb_{hgb_variant}",
                    _fit_hgb_regressor(X_tau, psi_tau, variant=hgb_variant),
                    "hgb",
                )
            )
        ridge_beta, _ = _fit_ridge(X_tau, psi_tau, alpha=2.0)
        challengers.append(("ridge", _RidgePredictor(ridge_beta), "ridge"))

        best_name = ""
        best_kind = ""
        best_model = None
        best_train_r2 = -1e9
        best_oos_r2 = None
        best_resid_std = None
        best_score = -1e9

        for cand_name, cand_model, cand_kind in challengers:
            tau_hat_train = np.asarray(cand_model.predict(X_train), dtype=float)
            train_r2 = float(_r2(psi_tau, np.asarray(cand_model.predict(X_tau), dtype=float)))
            tau_resid = psi - tau_hat_train
            resid_std_tau = float(np.sqrt(np.mean(np.square(tau_resid[usable])))) if np.any(usable) else 0.2
            resid_std_tau = float(max(1e-6, resid_std_tau))

            oos_r2 = None
            if m0_all is not None and len(valid_treated) >= int(min_validation_rows):
                yhat_valid_t = np.asarray(m0_all.predict(X_valid_t), dtype=float) + np.asarray(
                    cand_model.predict(X_valid_t), dtype=float
                )
                oos_r2 = float(_r2(y_valid_t, yhat_valid_t))

            # Prefer true OOS ranking when available; fallback to train fit.
            rank_score = float(oos_r2) if oos_r2 is not None else (float(train_r2) - 2.0)
            if rank_score > best_score:
                best_score = rank_score
                best_name = str(cand_name)
                best_kind = str(cand_kind)
                best_model = cand_model
                best_train_r2 = float(train_r2)
                best_oos_r2 = float(oos_r2) if oos_r2 is not None else None
                best_resid_std = float(resid_std_tau)

        if best_model is None:
            continue

        enabled = (
            best_oos_r2 is not None
            and float(best_oos_r2) >= float(gate_min_oos_r2)
            and int(n_train) >= int(gate_min_train_rows)
            and int(treated_rows) >= int(gate_min_treated_rows)
            and int(control_rows) >= int(gate_min_control_rows)
        )
        fail_reasons = []
        if best_oos_r2 is None:
            fail_reasons.append("oos_unavailable")
        elif float(best_oos_r2) < float(gate_min_oos_r2):
            fail_reasons.append(f"oos_r2<{float(gate_min_oos_r2):.3f}")
        if int(n_train) < int(gate_min_train_rows):
            fail_reasons.append(f"n_train<{int(gate_min_train_rows)}")
        if int(treated_rows) < int(gate_min_treated_rows):
            fail_reasons.append(f"treated<{int(gate_min_treated_rows)}")
        if int(control_rows) < int(gate_min_control_rows):
            fail_reasons.append(f"control<{int(gate_min_control_rows)}")

        bundle_key = str(cell_key)
        bundle[bundle_key] = best_model
        out["dr_models"][str(cell_key)] = {
            "method": "dr_aipw_hgb_v1",
            "model_family": "hgb",
            "bundle_key": bundle_key,
            "challenger_selected": str(best_name),
            "challenger_kind": str(best_kind),
            "residual_std": float(best_resid_std or 0.2),
            "n_train": int(n_train),
            "n_valid": int(n_valid),
            "treated_rows": int(treated_rows),
            "control_rows": int(control_rows),
            "propensity_clip": float(clip),
            "crossfit_folds": int(folds),
            "r2": float(best_train_r2),
            "oos_r2": float(best_oos_r2) if best_oos_r2 is not None else None,
            "enabled": bool(enabled),
            "gate_reason": "pass" if enabled else "|".join(fail_reasons),
            "action_type_key": str(action_type_key),
            "action_subtype_key": str(subtype_key),
        }
    if not quiet:
        enabled_count = sum(1 for m in (out.get("dr_models") or {}).values() if bool(m.get("enabled", True)))
        _log(f"{log_prefix} completed cells={len(out.get('dr_models', {}))} enabled={enabled_count}")
    return out, bundle

def _fit_models_for_target(
    df: pd.DataFrame,
    y: pd.Series,
    stats: Dict[str, Dict[str, float]],
    min_rows_per_action: int,
    alpha: float,
    train_mask: pd.Series,
    valid_mask: pd.Series,
    min_validation_rows: int,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"models": {}}
    base = _ensure_features(df)
    yy = _to_num(y)

    # Global model
    train_ok = yy.notna() & train_mask
    valid_ok = yy.notna() & valid_mask
    dfg = base.loc[train_ok]
    yg = yy.loc[train_ok].to_numpy(dtype=float)
    if len(dfg) >= 200:
        Xg = _standardize(dfg, stats)
        beta, resid_std = _fit_ridge(Xg, yg, alpha)
        yhat = np.column_stack([np.ones(len(Xg)), Xg]) @ beta
        oos_r2 = None
        if int(valid_ok.sum()) >= int(min_validation_rows):
            Xv = _standardize(base.loc[valid_ok], stats)
            yv = yy.loc[valid_ok].to_numpy(dtype=float)
            yhat_v = np.column_stack([np.ones(len(Xv)), Xv]) @ beta
            oos_r2 = float(_r2(yv, yhat_v))
        out["models"]["__global__"] = {
            "intercept": float(beta[0]),
            "coefficients": {f: float(beta[i + 1]) for i, f in enumerate(FEATURE_ORDER)},
            "residual_std": float(max(1e-6, resid_std)),
            "n_train": int(len(yg)),
            "n_valid": int(valid_ok.sum()),
            "r2": float(_r2(yg, yhat)),
            "oos_r2": float(oos_r2) if oos_r2 is not None else None,
        }

    # Action-specific models
    action_types = sorted(set(str(v) for v in df.get("action_type", pd.Series(dtype=str)).dropna().tolist()))
    for action_type in action_types:
        m_train = train_ok & (df["action_type"].astype(str) == action_type)
        m_valid = valid_ok & (df["action_type"].astype(str) == action_type)
        dfa = base.loc[m_train]
        ya = yy.loc[m_train].to_numpy(dtype=float)
        if len(dfa) < int(min_rows_per_action):
            continue
        Xa = _standardize(dfa, stats)
        beta, resid_std = _fit_ridge(Xa, ya, alpha)
        yhat = np.column_stack([np.ones(len(Xa)), Xa]) @ beta
        oos_r2 = None
        if int(m_valid.sum()) >= int(min_validation_rows):
            Xv = _standardize(base.loc[m_valid], stats)
            yv = yy.loc[m_valid].to_numpy(dtype=float)
            yhat_v = np.column_stack([np.ones(len(Xv)), Xv]) @ beta
            oos_r2 = float(_r2(yv, yhat_v))
        out["models"][action_type] = {
            "intercept": float(beta[0]),
            "coefficients": {f: float(beta[i + 1]) for i, f in enumerate(FEATURE_ORDER)},
            "residual_std": float(max(1e-6, resid_std)),
            "n_train": int(len(ya)),
            "n_valid": int(m_valid.sum()),
            "r2": float(_r2(ya, yhat)),
            "oos_r2": float(oos_r2) if oos_r2 is not None else None,
        }
    return out


def _fit_dr_models_for_target(
    df: pd.DataFrame,
    y: pd.Series,
    stats: Dict[str, Dict[str, float]],
    train_mask: pd.Series,
    valid_mask: pd.Series,
    alpha: float,
    crossfit_folds: int,
    dr_min_treated_rows: int,
    dr_min_control_rows: int,
    min_validation_rows: int,
    propensity_clip: float,
    dr_control_scope: str,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"dr_models": {}}
    base = _ensure_features(df)
    yy = _to_num(y)
    action_series = df.get("action_type", pd.Series(dtype=str)).astype(str)

    train_ok = yy.notna() & train_mask
    valid_ok = yy.notna() & valid_mask
    if int(train_ok.sum()) < max(1000, dr_min_treated_rows + dr_min_control_rows):
        return out

    X_all = _standardize(base, stats)
    y_all = yy.to_numpy(dtype=float)
    train_indices = np.where(train_ok.to_numpy(dtype=bool))[0]
    valid_indices = np.where(valid_ok.to_numpy(dtype=bool))[0]
    if len(train_indices) == 0:
        return out

    action_types = sorted(set(action_series.loc[train_ok].dropna().tolist()))
    folds = max(2, int(crossfit_folds))
    clip = min(0.2, max(0.01, float(propensity_clip)))

    for action_type in action_types:
        t_all = (action_series.to_numpy(dtype=str) == str(action_type)).astype(float)
        scope_mask = _cell_scope_mask(
            action_type_series=action_series,
            action_type_key=str(action_type),
            subtype_key="all",
            dr_control_scope=str(dr_control_scope),
        )
        train_scope = train_ok & scope_mask
        valid_scope = valid_ok & scope_mask
        train_indices_scoped = np.where(train_scope.to_numpy(dtype=bool))[0]
        valid_indices_scoped = np.where(valid_scope.to_numpy(dtype=bool))[0]
        if len(train_indices_scoped) == 0:
            continue
        t_train = t_all[train_indices_scoped]
        treated_rows = int(np.sum(t_train == 1.0))
        control_rows = int(np.sum(t_train == 0.0))
        if treated_rows < int(dr_min_treated_rows) or control_rows < int(dr_min_control_rows):
            continue

        X_train = X_all[train_indices_scoped]
        y_train = y_all[train_indices_scoped]
        n_train = len(train_indices_scoped)
        fold_ids = np.arange(n_train) % folds
        psi = np.zeros(n_train, dtype=float)
        usable = np.ones(n_train, dtype=bool)

        for fold in range(folds):
            hold = fold_ids == fold
            fit = ~hold
            if not np.any(hold) or np.sum(fit) < 100:
                continue

            X_fit = X_train[fit]
            y_fit = y_train[fit]
            t_fit = t_train[fit]

            treat_fit = t_fit == 1.0
            ctrl_fit = t_fit == 0.0
            if int(np.sum(treat_fit)) < 50 or int(np.sum(ctrl_fit)) < 50:
                usable[hold] = False
                continue

            beta_e = _fit_propensity_ridge(X_fit, t_fit, alpha=max(0.1, alpha))
            e_hat = _sigmoid(_linear_predict(beta_e, X_train[hold]))
            e_hat = np.clip(e_hat, clip, 1.0 - clip)

            beta_m1, _ = _fit_ridge(X_fit[treat_fit], y_fit[treat_fit], alpha)
            beta_m0, _ = _fit_ridge(X_fit[ctrl_fit], y_fit[ctrl_fit], alpha)
            m1 = _linear_predict(beta_m1, X_train[hold])
            m0 = _linear_predict(beta_m0, X_train[hold])

            yh = y_train[hold]
            th = t_train[hold]
            psi_hold = m1 - m0 + th * (yh - m1) / e_hat - (1.0 - th) * (yh - m0) / (1.0 - e_hat)
            psi[hold] = psi_hold

        if int(np.sum(usable)) < max(200, int(0.5 * n_train)):
            continue

        X_tau = X_train[usable]
        psi_tau = psi[usable]
        beta_tau, resid_std = _fit_ridge(X_tau, psi_tau, alpha)
        tau_hat_train = _linear_predict(beta_tau, X_train)
        tau_resid = psi - tau_hat_train
        resid_std_tau = float(np.sqrt(np.mean(np.square(tau_resid[usable])))) if np.any(usable) else float(resid_std)
        resid_std_tau = float(max(1e-6, resid_std_tau))

        # Out-of-sample treated-only calibration metric.
        oos_r2 = None
        n_valid = 0
        if len(valid_indices_scoped) > 0:
            valid_treated = valid_indices_scoped[t_all[valid_indices_scoped] == 1.0]
            n_valid = int(len(valid_treated))
            if n_valid >= int(min_validation_rows):
                X_fit_all = X_train
                y_fit_all = y_train
                t_fit_all = t_train
                treat_all = t_fit_all == 1.0
                ctrl_all = t_fit_all == 0.0
                if int(np.sum(treat_all)) >= 100 and int(np.sum(ctrl_all)) >= 100:
                    beta_m0_all, _ = _fit_ridge(X_fit_all[ctrl_all], y_fit_all[ctrl_all], alpha)
                    X_valid_t = X_all[valid_treated]
                    y_valid_t = y_all[valid_treated]
                    yhat_valid_t = _linear_predict(beta_m0_all, X_valid_t) + _linear_predict(beta_tau, X_valid_t)
                    oos_r2 = float(_r2(y_valid_t, yhat_valid_t))

        out["dr_models"][str(action_type)] = {
            "method": "dr_aipw_ridge_v1",
            "intercept": float(beta_tau[0]),
            "coefficients": {f: float(beta_tau[i + 1]) for i, f in enumerate(FEATURE_ORDER)},
            "residual_std": float(resid_std_tau),
            "n_train": int(n_train),
            "n_valid": int(n_valid),
            "treated_rows": int(treated_rows),
            "control_rows": int(control_rows),
            "propensity_clip": float(clip),
            "crossfit_folds": int(folds),
            "r2": float(_r2(psi_tau, _linear_predict(beta_tau, X_tau))),
            "oos_r2": float(oos_r2) if oos_r2 is not None else None,
            "ate_train": float(np.mean(tau_hat_train)),
        }

    return out


def main() -> None:
    args = _parse_args()
    outcomes_path = _resolve_outcomes_path(args.outcomes_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    quiet = bool(args.quiet)

    if not quiet:
        _log(f"loading outcomes from {outcomes_path}")
    raw = pd.read_parquet(outcomes_path)
    if raw.empty:
        raise ValueError("outcomes dataset is empty")
    if not quiet:
        _log(f"loaded rows={len(raw)}")

    raw = _with_action_cells(raw, cell_level=str(args.cell_level))
    action_id_allowlist = _parse_action_id_allowlist(args.action_id_allowlist, args.action_id_allowlist_file)
    cell_allowlist = _parse_cell_allowlist(args.cell_allowlist, args.cell_allowlist_file)
    objective_allowlist = _parse_objective_allowlist(args.objective_allowlist)
    dr_control_scope = _resolve_dr_control_scope(
        requested_scope=str(args.dr_control_scope or "global"),
        capital_phase1_only=bool(args.capital_phase1_only),
    )
    if bool(args.capital_phase1_only):
        routing_config = _load_capital_routing_config(args.capital_routing_config_path)
        phase1_actions, phase1_objectives = _capital_phase1_defaults(routing_config)
        if phase1_actions and not action_id_allowlist:
            action_id_allowlist = list(phase1_actions)
        if phase1_objectives and not objective_allowlist:
            objective_allowlist = list(phase1_objectives)
    selected_objectives = list(objective_allowlist or OBJECTIVES)
    if action_id_allowlist:
        mask = raw.get("action_id_key", pd.Series("", index=raw.index)).map(
            lambda x: _matches_action_allowlist(str(x), action_id_allowlist)
        )
        raw = raw[mask].reset_index(drop=True)
        if raw.empty:
            raise ValueError("action-id allowlist removed all rows from outcomes dataset")
        _validate_action_allowlist_coverage(raw, action_id_allowlist)
        if not quiet:
            _log(
                f"applied action-id allowlist entries={len(action_id_allowlist)} rows={len(raw)} "
                f"cells={int(raw.get('action_cell', pd.Series(dtype=str)).nunique())}"
            )
    df = _ensure_features(raw)
    if not quiet:
        distinct_cells = int(raw.get("action_cell", pd.Series(dtype=str)).nunique())
        _log(f"prepared features, distinct action cells={distinct_cells}")
    train_mask, valid_mask, split_meta = _resolve_split_masks(
        raw,
        validation_fraction=float(args.validation_fraction),
        train_end_date=str(args.train_end_date or "").strip(),
        validation_start_date=str(args.validation_start_date or "").strip(),
    )
    if not quiet:
        _log(
            "split resolved "
            f"method={split_meta.get('method')} train_rows={int(train_mask.sum())} valid_rows={int(valid_mask.sum())}"
        )
    normalize_by_subtype = bool(args.subtype_target_normalize) or str(args.cell_level) == "action_subtype"
    targets = _build_targets(
        raw,
        normalize_by_subtype=normalize_by_subtype,
        subtype_col=raw.get("action_subtype_key"),
    )
    for k, y in list(targets.items()):
        targets[k] = _winsorize(_to_num(y), float(args.winsor_pct))

    stats = _feature_stats(df.loc[train_mask])
    bundle_models: Dict[str, Any] = {}

    payload: Dict[str, Any] = {
        "version": "causal_impact_model_v3_bundle_contract",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_rows": int(len(df)),
        "resolved_outcomes_path": str(outcomes_path),
        "training_split": split_meta,
        "model_family": str(args.model_family),
        "cell_level": str(args.cell_level),
        "action_id_allowlist": list(action_id_allowlist),
        "objective_allowlist": list(selected_objectives),
        "dr_control_scope": str(dr_control_scope),
        "subtype_target_normalize": bool(normalize_by_subtype),
        "feature_order": FEATURE_ORDER,
        "feature_contract": {
            "version": CAUSAL_FEATURE_CONTRACT_VERSION,
            "aliases": {key: list(value) for key, value in CAUSAL_FEATURE_ALIASES.items()},
        },
        "feature_transform_spec": {
            "usd_millions_features": sorted(USD_MILLIONS_FEATURES),
            "rate_percent_features": sorted(RATE_PERCENT_FEATURES),
            "oas_percent_features": sorted(OAS_PERCENT_FEATURES),
            "signed_log1p_features": sorted(SIGNED_LOG1P_FEATURES),
        },
        "feature_stats": stats,
        "objectives": {},
    }
    model_card: Dict[str, Any] = {
        "version": "causal_impact_model_v3_bundle_contract",
        "trained_at": payload["trained_at"],
        "dataset_rows": int(len(df)),
        "resolved_outcomes_path": str(outcomes_path),
        "training_split": split_meta,
        "model_family": str(args.model_family),
        "cell_level": str(args.cell_level),
        "action_id_allowlist": list(action_id_allowlist),
        "cell_allowlist": list(cell_allowlist),
        "objective_allowlist": list(selected_objectives),
        "dr_control_scope": str(dr_control_scope),
        "feature_contract": dict(payload["feature_contract"]),
        "feature_transform_spec": dict(payload["feature_transform_spec"]),
        "objectives": {},
    }

    for objective in selected_objectives:
        if not quiet:
            _log(f"objective={objective} training start family={args.model_family}")
        if str(args.model_family) == "linear":
            baseline_models = _fit_models_for_target(
                df=df,
                y=targets[objective],
                stats=stats,
                min_rows_per_action=int(args.min_rows_per_action),
                alpha=float(args.ridge_alpha),
                train_mask=train_mask,
                valid_mask=valid_mask,
                min_validation_rows=int(args.min_validation_rows),
            )
            dr_models = _fit_dr_models_for_target(
                df=df,
                y=targets[objective],
                stats=stats,
                train_mask=train_mask,
                valid_mask=valid_mask,
                alpha=float(args.ridge_alpha),
                crossfit_folds=int(args.crossfit_folds),
                dr_min_treated_rows=int(args.dr_min_treated_rows),
                dr_min_control_rows=int(args.dr_min_control_rows),
                min_validation_rows=int(args.min_validation_rows),
                propensity_clip=float(args.propensity_clip),
                dr_control_scope=str(dr_control_scope),
            )
            merged = dict(baseline_models or {})
            merged.update(dr_models or {})
        else:
            dr_models, objective_bundle = _fit_dr_models_for_target_hgb(
                df=raw,
                y=targets[objective],
                stats=stats,
                train_mask=train_mask,
                valid_mask=valid_mask,
                crossfit_folds=int(args.crossfit_folds),
                dr_min_treated_rows=int(args.dr_min_treated_rows),
                dr_min_control_rows=int(args.dr_min_control_rows),
                min_validation_rows=int(args.min_validation_rows),
                propensity_clip=float(args.propensity_clip),
                gate_min_oos_r2=float(args.gate_min_oos_r2),
                gate_min_train_rows=int(args.gate_min_train_rows),
                gate_min_treated_rows=int(args.gate_min_treated_rows),
                gate_min_control_rows=int(args.gate_min_control_rows),
                cell_level=str(args.cell_level),
                cell_allowlist=list(cell_allowlist),
                dr_control_scope=str(dr_control_scope),
                progress_every_cells=int(args.progress_every_cells),
                log_prefix=f"objective={objective}",
                quiet=quiet,
            )
            merged = {"models": {}, **dict(dr_models or {})}
            for k, v in dict(objective_bundle or {}).items():
                bundle_models[f"{objective}::{k}"] = v
                if k in (merged.get("dr_models") or {}):
                    merged["dr_models"][k]["bundle_key"] = f"{objective}::{k}"

        payload["objectives"][objective] = merged

        objective_card = {"actions": {}, "enabled_actions": 0}
        for action_name, model in dict((merged.get("dr_models") or {})).items():
            enabled = bool(model.get("enabled", True))
            if enabled:
                objective_card["enabled_actions"] += 1
            objective_card["actions"][action_name] = {
                "method": str(model.get("method", "")),
                "n_train": int(model.get("n_train", 0) or 0),
                "n_valid": int(model.get("n_valid", 0) or 0),
                "treated_rows": int(model.get("treated_rows", 0) or 0),
                "control_rows": int(model.get("control_rows", 0) or 0),
                "oos_r2": model.get("oos_r2"),
                "residual_std": model.get("residual_std"),
                "enabled": enabled,
                "gate_reason": str(model.get("gate_reason", "")),
            }
        model_card["objectives"][objective] = objective_card
        if not quiet:
            _log(
                f"objective={objective} done cells={len(objective_card['actions'])} "
                f"enabled={int(objective_card['enabled_actions'])}"
            )

    if str(args.model_family) == "hgb" and not bool(args.skip_bundle_write):
        bundle_path = out_path.with_suffix(".bundle.pkl")
        with open(bundle_path, "wb") as fh:
            pickle.dump(bundle_models, fh, protocol=pickle.HIGHEST_PROTOCOL)
        payload["model_bundle_path"] = str(bundle_path.name)
        if not quiet:
            _log(f"wrote HGB bundle models={len(bundle_models)} to {bundle_path}")
    elif str(args.model_family) == "hgb" and bool(args.skip_bundle_write) and not quiet:
        _log("skipping HGB bundle write per --skip-bundle-write")

    payload["model_card"] = model_card

    out_path.write_text(json.dumps(payload, indent=2))
    if not quiet:
        _log(f"wrote model artifact to {out_path}")
    if str(args.model_card_out or "").strip():
        card_path = Path(str(args.model_card_out).strip())
        card_path.parent.mkdir(parents=True, exist_ok=True)
        card_path.write_text(json.dumps(model_card, indent=2))
        if not quiet:
            _log(f"wrote model card to {card_path}")
    model_count = sum(
        len((payload["objectives"][o] or {}).get("models", {}))
        + len((payload["objectives"][o] or {}).get("dr_models", {}))
        for o in selected_objectives
    )
    print(
        json.dumps(
            {
                "ok": True,
                "out_path": str(out_path),
                "training_rows": int(len(df)),
                "models": int(model_count),
                "dr_models": int(
                    sum(len((payload["objectives"][o] or {}).get("dr_models", {})) for o in selected_objectives)
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
