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


