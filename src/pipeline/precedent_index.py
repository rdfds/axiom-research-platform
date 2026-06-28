from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


_HORIZON_KEYS = ["horizon_1m", "horizon_6m", "horizon_12m", "horizon_24m"]
_METRIC_KEYS = [
    "valuation_multiple_change",
    "equity_return_vs_sector",
    "credit_spread_change",
    "rating_migration",
    "leverage_change",
    "fcf_change",
    "volatility_change",
]
INDEX_VERSION = "v4_calibrated_query"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _norm_str(v: Any) -> str:
    return str(v or "").strip()


def _norm_lower(v: Any) -> str:
    return _norm_str(v).lower()


def _to_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _query_rank_score(
    *,
    precedent_confidence: float,
    sample_size: int,
    out_of_sample_flag: bool,
    source: str,
    retrieval_tier: str,
    low_precedent_coverage: bool,
    exact_support_ratio: float,
    top_similarity_mean: float,
    top_action_match_score: float,
) -> float:
    sample_factor = min(1.0, math.log1p(max(0, int(sample_size))) / math.log(51.0))
    oos_penalty = 0.82 if bool(out_of_sample_flag) else 1.0
    source_penalty = 1.0 if str(source or "") == "overall" else 0.96
    tier = str(retrieval_tier or "")
    if tier == "exact":
        tier_factor = 1.0
    elif tier == "sibling_type":
        tier_factor = 0.93
    else:
        tier_factor = 0.78
    coverage_factor = 0.88 + 0.12 * max(0.0, min(1.0, float(exact_support_ratio)))
    if bool(low_precedent_coverage) and tier == "global":
        coverage_factor *= 0.88
    similarity_factor = 0.85 + 0.15 * max(0.0, min(1.0, float(top_similarity_mean)))
    action_factor = 0.90 + 0.10 * max(0.0, min(1.0, float(top_action_match_score)))
    return round(
        float(precedent_confidence)
        * sample_factor
        * oos_penalty
        * source_penalty
        * tier_factor
        * coverage_factor
        * similarity_factor
        * action_factor,
        6,
    )


