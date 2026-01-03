#!/usr/bin/env python
"""Aggregate causal + precedent + latency diagnostics for recommendation runs."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


_CAUSAL_DRIVER_NAMES = {
    "causal_model_blend_weight",
    "causal_model_quality",
    "causal_model_support_score",
    "causal_model_mode",
}


def _to_float(v: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        out = float(v)
    except Exception:
        return default
    if out != out:  # NaN
        return default
    return float(out)


def _safe_ratio(num: float, den: float) -> float:
    den_f = float(den)
    if den_f <= 0.0:
        return 0.0
    return float(num) / den_f


