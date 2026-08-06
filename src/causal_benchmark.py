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


def evaluate_summary_thresholds(
    summary: Dict[str, Any],
    min_enabled_cells: int = 10,
    min_enabled_rate: float = 0.10,
    min_enabled_oos_r2_mean: float = 0.05,
    required_objectives: Optional[List[str]] = None,
    required_objective_min_enabled: int = 1,
    required_objective_min_oos_r2_mean: float = 0.0,
) -> Dict[str, Any]:
    req_objs = list(required_objectives or [])
    totals = dict(summary.get("totals", {}) or {})
    objectives = dict(summary.get("objectives", {}) or {})
    failures: List[str] = []

    enabled_cells = int(_to_float(totals.get("enabled_cells"), 0.0) or 0.0)
    enabled_rate = float(_to_float(totals.get("enabled_rate"), 0.0) or 0.0)
    enabled_oos_mean = _to_float(totals.get("enabled_oos_r2_mean"))

    if enabled_cells < int(min_enabled_cells):
        failures.append(f"enabled_cells<{int(min_enabled_cells)}")
    if enabled_rate < float(min_enabled_rate):
        failures.append(f"enabled_rate<{float(min_enabled_rate):.3f}")
    if enabled_oos_mean is None:
        failures.append("enabled_oos_r2_mean_unavailable")
    elif float(enabled_oos_mean) < float(min_enabled_oos_r2_mean):
        failures.append(f"enabled_oos_r2_mean<{float(min_enabled_oos_r2_mean):.3f}")

    for objective_name in req_objs:
        obj = dict(objectives.get(objective_name, {}) or {})
        obj_enabled = int(_to_float(obj.get("enabled_cells"), 0.0) or 0.0)
        obj_oos_mean = _to_float(obj.get("enabled_oos_r2_mean"))
        if obj_enabled < int(required_objective_min_enabled):
            failures.append(f"{objective_name}.enabled<{int(required_objective_min_enabled)}")
        if obj_oos_mean is None:
            failures.append(f"{objective_name}.enabled_oos_r2_mean_unavailable")
        elif float(obj_oos_mean) < float(required_objective_min_oos_r2_mean):
            failures.append(f"{objective_name}.enabled_oos_r2_mean<{float(required_objective_min_oos_r2_mean):.3f}")

    return {
        "pass": len(failures) == 0,
        "failures": failures,
    }


def compare_summaries(champion: Dict[str, Any], challenger: Dict[str, Any]) -> Dict[str, Any]:
    champ_tot = dict(champion.get("totals", {}) or {})
    chall_tot = dict(challenger.get("totals", {}) or {})
    champ_obj = dict(champion.get("objectives", {}) or {})
    chall_obj = dict(challenger.get("objectives", {}) or {})
    all_obj = sorted(set(champ_obj.keys()) | set(chall_obj.keys()))

    by_objective: Dict[str, Any] = {}
    better = 0
    comparable = 0
    for obj in all_obj:
        c = dict(champ_obj.get(obj, {}) or {})
        d = dict(chall_obj.get(obj, {}) or {})
        c_mean = _to_float(c.get("enabled_oos_r2_mean"))
        d_mean = _to_float(d.get("enabled_oos_r2_mean"))
        if c_mean is not None and d_mean is not None:
            comparable += 1
            if d_mean > c_mean:
                better += 1
        by_objective[obj] = {
            "delta_enabled_cells": int(_to_float(d.get("enabled_cells"), 0.0) or 0.0)
            - int(_to_float(c.get("enabled_cells"), 0.0) or 0.0),
            "delta_enabled_oos_r2_mean": round(float((d_mean or 0.0) - (c_mean or 0.0)), 6)
            if c_mean is not None and d_mean is not None
            else None,
        }

    return {
        "totals": {
            "delta_enabled_cells": int(_to_float(chall_tot.get("enabled_cells"), 0.0) or 0.0)
            - int(_to_float(champ_tot.get("enabled_cells"), 0.0) or 0.0),
            "delta_enabled_rate": round(
                float(_to_float(chall_tot.get("enabled_rate"), 0.0) or 0.0)
                - float(_to_float(champ_tot.get("enabled_rate"), 0.0) or 0.0),
                6,
            ),
            "delta_enabled_oos_r2_mean": round(
                float(_to_float(chall_tot.get("enabled_oos_r2_mean"), 0.0) or 0.0)
                - float(_to_float(champ_tot.get("enabled_oos_r2_mean"), 0.0) or 0.0),
                6,
            )
            if _to_float(champ_tot.get("enabled_oos_r2_mean")) is not None
            and _to_float(chall_tot.get("enabled_oos_r2_mean")) is not None
            else None,
        },
        "objectives": by_objective,
        "challenger_better_objective_fraction": round(float(better / comparable), 6) if comparable > 0 else None,
    }


__all__ = [
    "compare_summaries",
    "evaluate_summary_thresholds",
    "load_model_card",
    "summarize_model_card",
]

