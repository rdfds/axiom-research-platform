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


