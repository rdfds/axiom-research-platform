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


