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


