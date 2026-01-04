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


def _load_json(path: Path) -> Dict[str, Any]:
    return dict(json.loads(path.read_text()) or {})


def _driver_map(action_candidate: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    drivers = (
        ((action_candidate.get("impact_distribution", {}) or {}).get("key_drivers"))
        or []
    )
    out: Dict[str, Dict[str, Any]] = {}
    for row in drivers:
        if not isinstance(row, dict):
            continue
        name = str(row.get("driver_name", "")).strip()
        if not name:
            continue
        out[name] = row
    return out


def _parse_dt(v: str) -> Optional[datetime]:
    raw = str(v or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return None


def _audit_ts(audit_log: List[Dict[str, Any]], event_type: str) -> Optional[datetime]:
    for row in audit_log:
        if str(row.get("event_type", "")) != event_type:
            continue
        ts = _parse_dt(str(row.get("timestamp", "")))
        if ts is not None:
            return ts
    return None


def _duration_seconds(
    audit_log: List[Dict[str, Any]],
    event_start: str,
    event_end: str,
) -> Optional[float]:
    start = _audit_ts(audit_log, event_start)
    end = _audit_ts(audit_log, event_end)
    if start is None or end is None:
        return None
    return round((end - start).total_seconds(), 6)


def _quantile(xs: List[float], q: float) :
    if not xs:
        return None
    ordered = sorted(float(x) for x in xs)
    idx = int(round((len(ordered) - 1) * float(q)))
    idx = min(max(idx, 0), len(ordered) - 1)
    return float(ordered[idx])


def _summary_stats(rows: List[float]) -> Dict[str, Optional[float]]:
    vals = [float(x) for x in rows]
    if not vals:
        return {"mean": None, "p10": None, "p50": None, "p90": None, "min": None, "max": None}
    return {
        "mean": round(sum(vals) / len(vals), 6),
        "p10": round(float(_quantile(vals, 0.10) or 0.0), 6),
        "p50": round(float(_quantile(vals, 0.50) or 0.0), 6),
        "p90": round(float(_quantile(vals, 0.90) or 0.0), 6),
        "min": round(min(vals), 6),
        "max": round(max(vals), 6),
    }


def _read_run_ids(path_value: str) -> set[str]:
    run_ids: set[str] = set()
    raw = str(path_value or "").strip()
    if not raw:
        return run_ids
    p = Path(raw)
    if not p.exists():
        return run_ids
    for line in p.read_text().splitlines():
        parts = [x for x in str(line).strip().split() if x]
        if not parts:
            continue
        run_ids.add(parts[-1])
    return run_ids


