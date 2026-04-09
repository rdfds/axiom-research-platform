#!/usr/bin/env python3
"""Add explicit policy/macro and pension-inclusive metrics to an existing artifact."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from backfill_market_macro_input_layer_v1 import _build_macro_metrics, _load_macro_history
from backfill_smart_normalized_metrics_v1 import (
    _effective_net_pension_liability_value,
    _load_companyfacts,
    _registry_metric,
    _smart_value_node,
)


NEW_METRICS = [
    "macro.fed_funds_effective",
    "macro.sofr",
    "macro.real_gdp_growth_yoy",
    "capital_structure.net_pension_liability",
    "capital_structure.debt_like_obligations_including_pension",
    "capital_structure.net_debt_including_pension",
    "capital_structure.gross_leverage_including_pension",
    "capital_structure.net_leverage_including_pension",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-path", required=True, help="Input artifact JSONL path")
    parser.add_argument("--raw-timeseries-path", required=True, help="Local raw_timeseries parquet")
    parser.add_argument("--metric-registry-path", required=True, help="Smart metric registry JSON")
    parser.add_argument("--companyfacts-root", required=True, help="SEC companyfacts root")
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument("--summary-out", help="Optional summary JSON path")
    parser.add_argument("--validation-out", help="Optional validation JSON path")
    return parser.parse_args()


def iter_rows(path: Path):
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _value(node: dict[str, Any] | None) -> float | None:
    if not node:
        return None
    value = node.get("value")
    return None if value is None else float(value)


def _support(node: dict[str, Any] | None) -> str:
    if not node:
        return "unsupported"
    return str(node.get("support_mode") or "unsupported")


def _approx_equal(left: float | None, right: float | None, tolerance: float = 1e-6) :
    if left is None or right is None:
        return left is right
    scale = max(1.0, abs(left), abs(right))
    return abs(left - right) <= tolerance * scale


