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


def summarize_model_card(card: Dict[str, Any]) -> Dict[str, Any]:
    objectives = dict(card.get("objectives", {}) or {})
    by_objective: Dict[str, Any] = {}

    enabled_total = 0
    total_cells = 0
    enabled_oos_vals_all: List[float] = []

    for objective_name, objective_payload in objectives.items():
        actions = dict((objective_payload or {}).get("actions", {}) or {})
        total = len(actions)
        enabled = 0
        enabled_oos_vals: List[float] = []

        for model in actions.values():
            if not isinstance(model, dict):
                continue
            is_enabled = bool(model.get("enabled", True))
            if not is_enabled:
                continue
            enabled += 1
            oos = _to_float(model.get("oos_r2"))
            if oos is not None:
                enabled_oos_vals.append(float(oos))

        total_cells += total
        enabled_total += enabled
        enabled_oos_vals_all.extend(enabled_oos_vals)

        by_objective[objective_name] = {
            "total_cells": int(total),
            "enabled_cells": int(enabled),
            "enabled_rate": round(float(enabled / total), 6) if total > 0 else 0.0,
            "enabled_oos_r2_mean": round(float(sum(enabled_oos_vals) / len(enabled_oos_vals)), 6)
            if enabled_oos_vals
            else None,
            "enabled_oos_r2_p50": round(_quantile(enabled_oos_vals, 0.50), 6) if enabled_oos_vals else None,
            "enabled_oos_r2_min": round(min(enabled_oos_vals), 6) if enabled_oos_vals else None,
            "enabled_oos_r2_max": round(max(enabled_oos_vals), 6) if enabled_oos_vals else None,
            "enabled_oos_positive_count": int(sum(1 for v in enabled_oos_vals if v > 0.0)),
        }

    summary = {
        "model_version": str(card.get("version", "")),
        "trained_at": str(card.get("trained_at", "")),
        "dataset_rows": int(_to_float(card.get("dataset_rows"), 0.0) or 0.0),
        "model_family": str(card.get("model_family", "")),
        "cell_level": str(card.get("cell_level", "")),
        "totals": {
            "total_cells": int(total_cells),
            "enabled_cells": int(enabled_total),
            "enabled_rate": round(float(enabled_total / total_cells), 6) if total_cells > 0 else 0.0,
            "enabled_oos_r2_mean": round(float(sum(enabled_oos_vals_all) / len(enabled_oos_vals_all)), 6)
            if enabled_oos_vals_all
            else None,
            "enabled_oos_r2_p50": round(_quantile(enabled_oos_vals_all, 0.50), 6) if enabled_oos_vals_all else None,
        },
        "objectives": by_objective,
    }
    return summary


