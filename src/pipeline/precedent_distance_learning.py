from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .precedent_brain import (
    _STATE_VECTOR_BASE_WEIGHTS,
    _STATE_VECTOR_MATCHING_COLS,
    augment_precedent_state_vector_columns,
)


_OUTCOME_SPECS: Dict[str, Dict[str, float]] = {
    "ALL": {
        "outcome_pe_6m": 0.18,
        "outcome_pe_12m": 0.22,
        "outcome_ev_ebitda_6m": 0.14,
        "outcome_ev_ebitda_12m": 0.18,
        "credit_spread_change_6m": 0.10,
        "credit_spread_change_12m": 0.10,
        "rating_migration_6m": 0.03,
        "rating_migration_12m": 0.03,
        "leverage_delta": 0.12,
        "fcf_margin_delta": 0.10,
    },
    "capital_return": {
        "outcome_pe_6m": 0.22,
        "outcome_pe_12m": 0.28,
        "outcome_ev_ebitda_6m": 0.16,
        "outcome_ev_ebitda_12m": 0.18,
        "leverage_delta": 0.06,
        "fcf_margin_delta": 0.10,
    },
    "capital_structure": {
        "credit_spread_change_6m": 0.20,
        "credit_spread_change_12m": 0.20,
        "rating_migration_6m": 0.10,
        "rating_migration_12m": 0.10,
        "leverage_delta": 0.20,
        "fcf_margin_delta": 0.08,
        "outcome_ev_ebitda_12m": 0.07,
        "outcome_pe_12m": 0.05,
    },
    "mna": {
        "outcome_pe_6m": 0.18,
        "outcome_pe_12m": 0.24,
        "outcome_ev_ebitda_6m": 0.14,
        "outcome_ev_ebitda_12m": 0.18,
        "leverage_delta": 0.10,
        "fcf_margin_delta": 0.16,
    },
    "portfolio": {
        "outcome_pe_6m": 0.18,
        "outcome_pe_12m": 0.24,
        "outcome_ev_ebitda_6m": 0.14,
        "outcome_ev_ebitda_12m": 0.18,
        "leverage_delta": 0.12,
        "fcf_margin_delta": 0.14,
    },
}


def _clean_scope_key(value: Any) -> str:
    return str(value or "").strip().lower()


def _state_feature_names() -> Tuple[str, ...]:
    return tuple(_STATE_VECTOR_MATCHING_COLS)


def _prior_weight_vector(scope_key: str) -> np.ndarray:
    scope = _clean_scope_key(scope_key)
    weights = dict(_STATE_VECTOR_BASE_WEIGHTS)
    if scope == "capital_return":
        weights["state_vector_v1.net_obligation_burden"] *= 1.20
        weights["state_vector_v1.liquidity_flexibility"] *= 1.20
        weights["state_vector_v1.interest_coverage"] *= 1.15
        weights["state_vector_v1.cash_generation"] *= 1.30
        weights["state_vector_v1.valuation_multiple"] *= 1.10
    elif scope == "capital_structure":
        weights["state_vector_v1.gross_obligation_burden"] *= 1.30
        weights["state_vector_v1.net_obligation_burden"] *= 1.20
        weights["state_vector_v1.liquidity_flexibility"] *= 1.30
        weights["state_vector_v1.interest_coverage"] *= 1.15
        weights["state_vector_v1.market_access"] *= 1.30
        weights["state_vector_v1.credit_spread"] *= 1.20
        weights["state_vector_v1.valuation_multiple"] *= 0.85
    elif scope == "mna":
        weights["state_vector_v1.growth"] *= 1.10
        weights["state_vector_v1.valuation_multiple"] *= 1.15
        weights["state_vector_v1.market_access"] *= 1.10
        weights["state_vector_v1.market_stress"] *= 1.10
    elif scope == "portfolio":
        weights["state_vector_v1.growth"] *= 1.10
        weights["state_vector_v1.cash_generation"] *= 1.10
        weights["state_vector_v1.valuation_multiple"] *= 1.05
    arr = np.array([float(weights.get(col, 1.0)) for col in _state_feature_names()], dtype=float)
    mean = float(np.nanmean(arr)) if arr.size else 1.0
    if mean > 1e-12:
        arr = arr / mean
    return arr


def _selected_outcome_weights(scope_key: str, df: pd.DataFrame, min_non_null: int) -> Dict[str, float]:
    preferred = dict(_OUTCOME_SPECS.get(_clean_scope_key(scope_key), _OUTCOME_SPECS["ALL"]))
    usable: Dict[str, float] = {}
    for col, weight in preferred.items():
        if col not in df.columns:
            continue
        non_null = int(pd.to_numeric(df[col], errors="coerce").notna().sum())
        if non_null >= int(min_non_null):
            usable[col] = float(weight)
    if not usable:
        fallback = dict(_OUTCOME_SPECS["ALL"])
        for col, weight in fallback.items():
            if col not in df.columns:
                continue
            non_null = int(pd.to_numeric(df[col], errors="coerce").notna().sum())
            if non_null >= int(min_non_null):
                usable[col] = float(weight)
    total = float(sum(usable.values()))
    if total > 1e-12:
        usable = {k: float(v / total) for k, v in usable.items()}
    return usable


def _robust_standardize_frame(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in cols:
        s = pd.to_numeric(df[col], errors="coerce")
        valid = s.dropna()
        if valid.empty:
            out[col] = np.nan
            continue
        med = float(valid.median())
        q25 = float(valid.quantile(0.25))
        q75 = float(valid.quantile(0.75))
        scale = (q75 - q25) / 1.349
        if (not np.isfinite(scale)) or scale <= 1e-9:
            scale = float(valid.std())
        if (not np.isfinite(scale)) or scale <= 1e-9:
            scale = 1.0
        out[col] = (s - med) / scale
    return out


def _sample_pair_indices(n_rows: int, n_pairs: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n_rows, size=int(n_pairs), endpoint=False)
    j = rng.integers(0, n_rows, size=int(n_pairs), endpoint=False)
    same = i == j
    if bool(np.any(same)):
        j[same] = (j[same] + 1) % max(1, n_rows)
    swap = i > j
    if bool(np.any(swap)):
        ii = i.copy()
        i[swap] = j[swap]
        j[swap] = ii[swap]
    return i.astype(np.int64), j.astype(np.int64)


def _pairwise_dataset(
    state_df: pd.DataFrame,
    outcome_df: pd.DataFrame,
    outcome_weights: Dict[str, float],
    *,
    max_pairs: int,
    min_state_coverage: float,
    min_outcome_coverage: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    state_values = state_df.to_numpy(dtype=float)
    outcome_cols = list(outcome_weights.keys())
    outcome_values = outcome_df[outcome_cols].to_numpy(dtype=float)
    feature_weights = np.array([float(_STATE_VECTOR_BASE_WEIGHTS.get(col, 1.0)) for col in state_df.columns], dtype=float)
    outcome_weight_arr = np.array([float(outcome_weights[col]) for col in outcome_cols], dtype=float)
    outcome_weight_arr = outcome_weight_arr / max(1e-12, float(outcome_weight_arr.sum()))
    n_rows = int(state_values.shape[0])
    if n_rows < 2:
        return np.empty((0, state_values.shape[1])), np.empty(0), {"n_pairs": 0}
    n_pairs = min(int(max_pairs), max(1, n_rows * 6))
    left, right = _sample_pair_indices(n_rows, n_pairs, seed)

    state_left = state_values[left]
    state_right = state_values[right]
    state_ok = np.isfinite(state_left) & np.isfinite(state_right)
    overlap_weight = np.sum(state_ok * feature_weights.reshape(1, -1), axis=1)
    state_total = float(np.sum(feature_weights))
    state_coverage = overlap_weight / max(1e-12, state_total)

    out_left = outcome_values[left]
    out_right = outcome_values[right]
    out_ok = np.isfinite(out_left) & np.isfinite(out_right)
    outcome_overlap = np.sum(out_ok * outcome_weight_arr.reshape(1, -1), axis=1)
    keep = (state_coverage >= float(min_state_coverage)) & (outcome_overlap >= float(min_outcome_coverage))
    if not bool(np.any(keep)):
        return np.empty((0, state_values.shape[1])), np.empty(0), {"n_pairs": 0}

    state_diff_sq = np.square(np.where(state_ok[keep], state_left[keep] - state_right[keep], 0.0))
    outcome_diff_sq = np.square(np.where(out_ok[keep], out_left[keep] - out_right[keep], 0.0))
    outcome_distance = np.divide(
        np.sum(outcome_diff_sq * outcome_weight_arr.reshape(1, -1), axis=1),
        np.maximum(outcome_overlap[keep], 1e-12),
    )
    return state_diff_sq.astype(float), outcome_distance.astype(float), {
        "n_pairs": int(np.count_nonzero(keep)),
        "sampled_pairs": int(n_pairs),
        "mean_state_coverage": float(np.nanmean(state_coverage[keep])) if bool(np.any(keep)) else 0.0,
        "mean_outcome_coverage": float(np.nanmean(outcome_overlap[keep])) if bool(np.any(keep)) else 0.0,
    }


def _ridge_to_prior(X: np.ndarray, y: np.ndarray, prior: np.ndarray, lam: float) -> np.ndarray:
    n_features = int(X.shape[1])
    xtx = X.T @ X
    ridge = xtx + float(lam) * np.eye(n_features)
    xty = X.T @ y
    rhs = xty + float(lam) * prior
    return np.linalg.solve(ridge, rhs)


def _corr(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if a.size < 5 or b.size < 5:
        return None
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    ok = np.isfinite(aa) & np.isfinite(bb)
    if int(np.count_nonzero(ok)) < 5:
        return None
    aa = aa[ok]
    bb = bb[ok]
    if float(np.nanstd(aa)) <= 1e-12 or float(np.nanstd(bb)) <= 1e-12:
        return None
    return float(np.corrcoef(aa, bb)[0, 1])


def _normalize_weight_vector(beta: np.ndarray, prior: np.ndarray) -> np.ndarray:
    out = np.array(beta, dtype=float, copy=True)
    out = np.where(np.isfinite(out), out, 0.0)
    out = np.clip(out, 0.0, None)
    if float(out.sum()) <= 1e-12:
        out = np.array(prior, dtype=float, copy=True)
    positive = out[out > 0]
    if positive.size:
        out = out / float(np.mean(positive))
    else:
        out = np.array(prior, dtype=float, copy=True)
    out = np.clip(out, 0.25, 4.0)
    return out


def learn_scope_weights(
    df: pd.DataFrame,
    *,
    scope_key: str,
    scope_col: str,
    max_pairs: int = 25000,
    min_rows: int = 1500,
    min_state_coverage: float = 0.60,
    min_outcome_coverage: float = 0.50,
    min_outcome_non_null: int = 800,
    ridge_lambda: float = 30.0,
    holdout_frac: float = 0.20,
    seed: int = 7,
) -> Optional[Dict[str, Any]]:
    subset = df.loc[df[scope_col].astype(str).str.lower().eq(_clean_scope_key(scope_key))].copy()
    if int(len(subset)) < int(min_rows):
        return None
    feature_cols = list(_state_feature_names())
    feature_df = _robust_standardize_frame(subset, feature_cols)
    outcome_weights = _selected_outcome_weights(scope_key, subset, min_non_null=min_outcome_non_null)
    if not outcome_weights:
        return None
    outcome_df = _robust_standardize_frame(subset, list(outcome_weights.keys()))
    X, y, pair_meta = _pairwise_dataset(
        feature_df,
        outcome_df,
        outcome_weights,
        max_pairs=max_pairs,
        min_state_coverage=min_state_coverage,
        min_outcome_coverage=min_outcome_coverage,
        seed=seed,
    )
    if X.shape[0] < 200:
        return None

    rng = np.random.default_rng(seed)
    order = rng.permutation(X.shape[0])
    X = X[order]
    y = y[order]
    holdout_n = max(50, int(round(X.shape[0] * float(holdout_frac))))
    holdout_n = min(holdout_n, max(0, X.shape[0] - 100))
    train_n = X.shape[0] - holdout_n
    if train_n < 100:
        return None

    X_train = X[:train_n]
    y_train = y[:train_n]
    X_holdout = X[train_n:] if holdout_n > 0 else np.empty((0, X.shape[1]))
    y_holdout = y[train_n:] if holdout_n > 0 else np.empty(0)

    prior = _prior_weight_vector(scope_key)
    beta = _ridge_to_prior(X_train, y_train, prior, float(ridge_lambda))
    weights = _normalize_weight_vector(beta, prior)
    pred_train = X_train @ weights
    pred_holdout = X_holdout @ weights if holdout_n > 0 else np.empty(0)
    pred_holdout_prior = X_holdout @ prior if holdout_n > 0 else np.empty(0)
    holdout_corr = _corr(pred_holdout, y_holdout)
    holdout_prior_corr = _corr(pred_holdout_prior, y_holdout)
    improvement = None
    if holdout_corr is not None and holdout_prior_corr is not None:
        improvement = float(holdout_corr - holdout_prior_corr)
    use_in_runtime = bool(
        holdout_corr is not None
        and holdout_prior_corr is not None
        and holdout_corr >= 0.05
        and float(holdout_corr - holdout_prior_corr) >= 0.01
    )

    return {
        "scope_key": _clean_scope_key(scope_key),
        "n_rows": int(len(subset)),
        "n_pairs": int(X.shape[0]),
        "n_pairs_train": int(train_n),
        "n_pairs_holdout": int(holdout_n),
        "feature_order": feature_cols,
        "weights": {col: float(weights[idx]) for idx, col in enumerate(feature_cols)},
        "prior_weights": {col: float(prior[idx]) for idx, col in enumerate(feature_cols)},
        "outcome_weights": {k: float(v) for k, v in outcome_weights.items()},
        "pairwise_meta": pair_meta,
        "train_pair_correlation": _corr(pred_train, y_train),
        "holdout_pair_correlation": holdout_corr,
        "holdout_prior_pair_correlation": holdout_prior_corr,
        "holdout_pair_correlation_improvement": improvement,
        "use_in_runtime": use_in_runtime,
        "ridge_lambda": float(ridge_lambda),
        "min_state_coverage": float(min_state_coverage),
        "min_outcome_coverage": float(min_outcome_coverage),
        "min_outcome_non_null": int(min_outcome_non_null),
        "max_pairs": int(max_pairs),
    }


def learn_precedent_distance_weights(
    outcomes_path: Path,
    *,
    max_pairs: int = 25000,
    min_rows: int = 1500,
    min_state_coverage: float = 0.60,
    min_outcome_coverage: float = 0.50,
    min_outcome_non_null: int = 800,
    ridge_lambda: float = 30.0,
    holdout_frac: float = 0.20,
    seed: int = 7,
) -> Dict[str, Any]:
    raw = pd.read_parquet(outcomes_path)
    df = augment_precedent_state_vector_columns(raw)
    scopes: Dict[str, Any] = {}
    all_scope = learn_scope_weights(
        df.assign(_all_scope="ALL"),
        scope_key="ALL",
        scope_col="_all_scope",
        max_pairs=max_pairs,
        min_rows=min_rows,
        min_state_coverage=min_state_coverage,
        min_outcome_coverage=min_outcome_coverage,
        min_outcome_non_null=min_outcome_non_null,
        ridge_lambda=ridge_lambda,
        holdout_frac=holdout_frac,
        seed=seed,
    )
    if all_scope:
        scopes["ALL"] = all_scope

    if "normalized_action_family" in df.columns:
        families = sorted({_clean_scope_key(x) for x in df["normalized_action_family"].dropna().tolist() if _clean_scope_key(x)})
        for family in families:
            learned = learn_scope_weights(
                df,
                scope_key=family,
                scope_col="normalized_action_family",
                max_pairs=max_pairs,
                min_rows=min_rows,
                min_state_coverage=min_state_coverage,
                min_outcome_coverage=min_outcome_coverage,
                min_outcome_non_null=min_outcome_non_null,
                ridge_lambda=ridge_lambda,
                holdout_frac=holdout_frac,
                seed=seed,
            )
            if learned:
                scopes[family] = learned

    return {
        "version": "precedent_distance_weights_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_outcomes_path": str(outcomes_path),
        "feature_order": list(_state_feature_names()),
        "scopes": scopes,
    }


def write_precedent_distance_weights(payload: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
