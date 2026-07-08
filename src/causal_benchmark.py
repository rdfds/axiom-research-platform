"""Offline benchmark helpers for causal model card evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


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


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    qq = max(0.0, min(1.0, float(q)))
    pos = qq * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    w = pos - lo
    return float(xs[lo] * (1.0 - w) + xs[hi] * w)


def load_model_card(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    return dict(json.loads(p.read_text()) or {})


