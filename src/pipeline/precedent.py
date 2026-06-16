from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .types import ImpactDistribution, PrecedentPack


_COMPACT_PROFILE_FEATURES = [
    "state_vector_v1.size_log_revenue",
    "state_vector_v1.profitability",
    "state_vector_v1.growth",
    "state_vector_v1.net_obligation_burden",
    "state_vector_v1.liquidity_flexibility",
    "state_vector_v1.interest_coverage",
    "state_vector_v1.valuation_multiple",
    "state_vector_v1.cash_generation",
    "state_vector_v1.market_stress",
    "state_vector_v1.market_access",
    "state_vector_v1.rates_level",
    "state_vector_v1.credit_spread",
]


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if np.isnan(out):
        return None
    return out


def _safe_log10(value: Optional[float]) -> Optional[float]:
    if value is None or value <= 0:
        return None
    return float(np.log10(value))


def _weighted_average(parts: List[Tuple[Optional[float], float]]) -> Optional[float]:
    numer = 0.0
    denom = 0.0
    for value, weight in parts:
        if value is None:
            continue
        numer += float(weight) * float(value)
        denom += float(weight)
    if denom <= 0:
        return None
    return numer / denom


def _state_vector_baseline_value(baseline: Dict[str, Any], key: str) -> Optional[float]:
    if key == "state_vector_v1.size_log_revenue":
        return _safe_float(baseline.get(key)) or _safe_log10(_safe_float(baseline.get("revenue_ttm")))
    if key == "state_vector_v1.profitability":
        return _safe_float(baseline.get(key)) or _safe_float(baseline.get("ebitda_margin"))
    if key == "state_vector_v1.growth":
        return (
            _safe_float(baseline.get(key))
            or _safe_float(baseline.get("revenue_yoy_last_q"))
            or _safe_float(baseline.get("revenue_yoy"))
        )
    if key == "state_vector_v1.gross_obligation_burden":
        return (
            _safe_float(baseline.get(key))
            or _safe_float(baseline.get("gross_leverage_including_retirement"))
            or _safe_float(baseline.get("gross_obligation_burden"))
        )
    if key == "state_vector_v1.net_obligation_burden":
        return (
            _safe_float(baseline.get(key))
            or _safe_float(baseline.get("net_leverage_including_retirement"))
            or _safe_float(baseline.get("leverage_net_debt_ebitda"))
        )
    if key == "state_vector_v1.liquidity_flexibility":
        direct = _safe_float(baseline.get(key))
        if direct is not None:
            return direct
        numer = (
            _safe_float(baseline.get("available_liquidity_normalized"))
            or _safe_float(baseline.get("available_for_actions"))
            or _safe_float(baseline.get("cash"))
        )
        denom = (
            _safe_float(baseline.get("debt_due_next_24m"))
            or _safe_float(baseline.get("debt_due_0_12m"))
            or _safe_float(baseline.get("current_debt"))
        )
        if numer is None or denom is None or denom <= 0:
            return None
        return numer / denom
    if key == "state_vector_v1.interest_coverage":
        direct = _safe_float(baseline.get(key)) or _safe_float(baseline.get("interest_coverage"))
        if direct is not None:
            return direct
        ebitda = _safe_float(baseline.get("ebitda_ttm")) or _safe_float(baseline.get("ebitda_ltm"))
        interest = _safe_float(baseline.get("interest_expense"))
        if ebitda is None or interest is None or interest <= 0:
            return None
        return ebitda / interest
    if key == "state_vector_v1.valuation_multiple":
        return (
            _safe_float(baseline.get(key))
            or _safe_float(baseline.get("ev_ebitda"))
            or _safe_float(baseline.get("base_ev_ebitda"))
        )
    if key == "state_vector_v1.cash_generation":
        return (
            _safe_float(baseline.get(key))
            or _safe_float(baseline.get("fcf_yield"))
            or _safe_float(baseline.get("fcf_margin"))
        )
    if key == "state_vector_v1.market_stress":
        direct = _safe_float(baseline.get(key))
        if direct is not None:
            return direct
        vol = _safe_float(baseline.get("volatility_90d"))
        draw = _safe_float(baseline.get("drawdown_90d"))
        return _weighted_average([(vol, 0.6), (abs(draw) if draw is not None else None, 0.4)])
    if key == "state_vector_v1.market_access":
        direct = _safe_float(baseline.get(key))
        if direct is not None:
            return direct
        spread = _safe_float(baseline.get("credit_spread_level"))
        spread_access = None if spread is None else max(0.0, min(1.0, 1.0 - spread / 0.08))
        return _weighted_average(
            [
                (_safe_float(baseline.get("credit_window_proxy")), 0.4),
                (_safe_float(baseline.get("equity_window_proxy")), 0.4),
                (spread_access, 0.2),
            ]
        )
    if key == "state_vector_v1.rates_level":
        return (
            _safe_float(baseline.get(key))
            or _safe_float(baseline.get("fed_funds_effective"))
            or _safe_float(baseline.get("macro_rate_10y"))
        )
    if key == "state_vector_v1.credit_spread":
        return (
            _safe_float(baseline.get(key))
            or _safe_float(baseline.get("hy_oas"))
            or _safe_float(baseline.get("macro_hy_oas"))
        )
    return None


def _first_present_column(df: pd.DataFrame, columns: List[str]) -> Optional[str]:
    for col in columns:
        if col in df.columns and _safe_numeric(df[col]).notna().any():
            return col
    return None


def learn_feature_weights(df: pd.DataFrame, feature_cols: List[str], target_col: str) -> pd.Series:
    """
    Learn feature weights using absolute correlation with the target outcome.
    Falls back to uniform weights when correlations are undefined.
    """
    weights = {}
    for col in feature_cols:
        if col not in df.columns:
            continue
        x = _safe_numeric(df[col])
        y = _safe_numeric(df[target_col]) if target_col in df.columns else None
        if y is None or y.dropna().empty or x.dropna().empty:
            weights[col] = 1.0
            continue
        corr = x.corr(y)
        if corr is None or np.isnan(corr):
            weights[col] = 1.0
        else:
            weights[col] = abs(corr)
    if not weights:
        return pd.Series(dtype=float)
    w = pd.Series(weights)
    if w.sum() == 0:
        w[:] = 1.0
    return w / w.sum()


