"""Causal model risk diagnostics for run-level governance.

This module summarizes causal coverage, support, quality, OOS rates, and
fallback behavior from FeasibilityResults action-candidate payloads.
"""

from __future__ import annotations

from datetime import datetime, timezone
import os
from typing import Any, Dict, List, Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_float(v: Any, default: Optional[float] = None) -> Optional[float]:
    if v is None:
        return default
    if isinstance(v, bool):
        return float(v)
    try:
        out = float(v)
    except Exception:
        return default
    if out != out:  # nan
        return default
    return out


def _driver_map(action_candidate: Dict[str, Any]) -> Dict[str, float]:
    impact = dict(action_candidate.get("impact_distribution", {}) or {})
    drivers = list(impact.get("key_drivers", []) or [])
    out: Dict[str, float] = {}
    for d in drivers:
        if not isinstance(d, dict):
            continue
        name = str(d.get("driver_name", "")).strip()
        if not name:
            continue
        val = _to_float(d.get("contribution"))
        if val is None:
            continue
        out[name] = float(val)
    return out


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    qq = max(0.0, min(1.0, float(q)))
    pos = qq * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    w = pos - lo
    return float(xs[lo] * (1.0 - w) + xs[hi] * w)


def _safe_ratio(num: int, den: int) -> float:
    return float(num) / float(den) if den > 0 else 0.0


