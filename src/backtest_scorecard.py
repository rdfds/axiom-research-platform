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


