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


