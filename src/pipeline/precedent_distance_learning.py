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


def _sample_pair_indices(n_rows: int, n_pairs: int, seed: int) :
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


