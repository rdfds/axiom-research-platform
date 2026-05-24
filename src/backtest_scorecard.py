from __future__ import annotations

import math
from statistics import median
from typing import Any, Dict, Iterable, List, Optional

from .backtest_costs import TransactionCostModel
from .backtest_protocol import BacktestProtocol


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        out = float(value)
    except Exception:
        return None
    if math.isnan(out):
        return None
    return float(out)


def _mean(values: Iterable[float]) -> Optional[float]:
    xs = [float(value) for value in values]
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def _stdev(values: Iterable[float]) :
    xs = [float(value) for value in values]
    if len(xs) < 2:
        return None
    mean_val = sum(xs) / len(xs)
    variance = sum((value - mean_val) ** 2 for value in xs) / (len(xs) - 1)
    return math.sqrt(max(0.0, variance))


def _nested_lookup(payload: Dict[str, Any], dotted_path: str) -> Any:
    current: Any = payload
    for token in str(dotted_path or "").split("."):
        if not token:
            continue
        if not isinstance(current, dict):
            return None
        current = current.get(token)
    return current


def _case_primary_family(case: Dict[str, Any]) -> str:
    action_ids = list(case.get("top_action_ids", []) or [])
    if action_ids:
        primary = str(action_ids[0] or "")
        if "." in primary:
            return primary.split(".", 1)[0]
    return str(case.get("anchor_action_family") or "")


