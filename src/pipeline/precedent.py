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


