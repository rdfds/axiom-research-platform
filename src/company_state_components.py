"""
CompanyState components: RegimeClassifier, PeerSetResolver, ProvenanceTracker.
These are lightweight, auditable building blocks used by CompanyStateBuilder.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class InputReference:
    artifact_type: str
    artifact_id: str
    source: Optional[str] = None
    published_at: Optional[str] = None
    ingested_at: Optional[str] = None
    hash: Optional[str] = None


def _zscore(series: pd.Series) -> Optional[float]:
    if series is None or series.empty:
        return None
    s = series.dropna().astype(float)
    if len(s) < 10:
        return None
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0:
        return None
    return float((s.iloc[-1] - mu) / sd)


def _percentile(series: pd.Series) -> Optional[float]:
    if series is None or series.empty:
        return None
    s = series.dropna().astype(float)
    if len(s) < 10:
        return None
    return float((s.rank(pct=True).iloc[-1]) * 100.0)


def _pick_first_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


