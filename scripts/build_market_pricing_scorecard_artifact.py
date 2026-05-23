#!/usr/bin/env python3
"""Build a comp-oriented market-pricing scorecard on top of a company-state artifact."""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


SCORE_METRICS = [
    "market.value_score",
    "market.quality_score",
    "market.balance_sheet_score",
    "market.risk_score",
    "market.comp_overall_score",
    "market.valuation_gap_score",
]

RAW_SCORE_COMPONENTS = {
    "market.value_score": [
        ("market.ev_ebitda", "lower"),
        ("market.fcf_yield", "higher"),
    ],
    "market.quality_score": [
        ("operating.ebitda_margin_ttm", "higher"),
        ("operating.revenue_yoy_last_q", "higher"),
        ("operating.revenue_cagr_3y", "higher"),
        ("operating.ebitda_margin_trend_8q", "higher"),
        ("operating.margin_volatility_8q", "lower"),
        ("operating.fcf_conversion", "higher"),
    ],
    "market.balance_sheet_score": [
        ("capital_structure.net_leverage_normalized", "lower"),
        ("derived.liquidity_coverage_ratio", "higher"),
        ("capital_structure.maturity_wall_ratio_24m", "lower"),
    ],
    "market.risk_score": [
        ("market.volatility_90d", "lower"),
        ("market.drawdown_90d", "higher"),
        ("market.credit_window_proxy", "higher"),
    ],
}

OVERALL_WEIGHTS = {
    "market.value_score": 0.35,
    "market.quality_score": 0.25,
    "market.balance_sheet_score": 0.20,
    "market.risk_score": 0.20,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-path", required=True, help="Input company-state JSONL artifact")
    parser.add_argument("--out", required=True, help="Output artifact with scorecard features")
    parser.add_argument("--summary-out", help="Optional summary JSON path")
    parser.add_argument("--leaderboard-out", help="Optional leaderboard JSON path")
    return parser.parse_args()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def iter_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _node_value(node: Dict[str, Any] | None) -> float | None:
    if not node:
        return None
    value = node.get("value")
    if value is None:
        return None
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return None


def _node_support(node: Dict[str, Any] | None) -> str:
    if not node:
        return "unsupported"
    return str(node.get("support_mode") or "unsupported")


def _union_provenance(*nodes: Dict[str, Any] | None) -> list[Dict[str, Any]]:
    merged: list[Dict[str, Any]] = []
    seen = set()
    for node in nodes:
        for prov in (node or {}).get("provenance") or []:
            key = json.dumps(prov, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            merged.append(copy.deepcopy(prov))
    return merged


def _base_score_node(
    *,
    name: str,
    value: float | None,
    computed_at: str,
    as_of_time: str,
    support_mode: str,
    fallback_used: str | None,
    provenance: list[Dict[str, Any]],
    component_breakdown: Dict[str, Any] | None,
    quality_flags: list[str] | None = None,
) -> Dict[str, Any]:
    return {
        "name": name,
        "value": value,
        "unit": "score",
        "computed_at": computed_at,
        "as_of_time": as_of_time,
        "window": {"type": "cross_sectional", "length_days": 0},
        "confidence": 0.7 if value is not None else None,
        "provenance": provenance,
        "missing_reason": None if value is not None else "insufficient_components",
        "fallback_used": fallback_used,
        "support_mode": support_mode if value is not None else "unsupported",
        "component_breakdown": component_breakdown,
        "quality_flags": quality_flags,
    }


