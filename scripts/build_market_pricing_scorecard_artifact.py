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


def _percentile_map(values_by_company: Dict[str, float], direction: str) -> Dict[str, float]:
    if not values_by_company:
        return {}
    ordered = sorted(values_by_company.items(), key=lambda item: (item[1], item[0]))
    count = len(ordered)
    scores: Dict[str, float] = {}
    if count == 1:
        company_id = ordered[0][0]
        return {company_id: 50.0}
    for idx, (company_id, _) in enumerate(ordered):
        pct = idx / (count - 1)
        if direction == "lower":
            pct = 1.0 - pct
        scores[company_id] = pct * 100.0
    return scores


def _derived_metric_values(features: Dict[str, Any]) -> Dict[str, float | None]:
    liquidity = _node_value(features.get("liquidity.available_liquidity_normalized"))
    debt_like = _node_value(features.get("capital_structure.debt_like_obligations_normalized"))
    liquidity_coverage_ratio = None
    if liquidity is not None and debt_like not in (None, 0):
        liquidity_coverage_ratio = liquidity / debt_like
    return {"derived.liquidity_coverage_ratio": liquidity_coverage_ratio}


def _collect_cross_section(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    values: Dict[str, Dict[str, float]] = {}
    for components in RAW_SCORE_COMPONENTS.values():
        for metric_name, _ in components:
            values.setdefault(metric_name, {})
    for row in rows:
        company_id = str(row.get("company_id") or "")
        features = row.get("features") or {}
        derived = _derived_metric_values(features)
        for metric_name in values:
            value = derived.get(metric_name)
            if value is None:
                value = _node_value(features.get(metric_name))
            if value is None or math.isnan(value) or math.isinf(value):
                continue
            values[metric_name][company_id] = float(value)
    percentile_maps: Dict[str, Dict[str, float]] = {}
    for score_metric, components in RAW_SCORE_COMPONENTS.items():
        for metric_name, direction in components:
            if metric_name in percentile_maps:
                continue
            percentile_maps[metric_name] = _percentile_map(values[metric_name], direction)
    return percentile_maps


def _component_detail(
    metric_name: str,
    *,
    row: Dict[str, Any],
    percentile_maps: Dict[str, Dict[str, float]],
) -> Dict[str, Any] | None:
    company_id = str(row.get("company_id") or "")
    features = row.get("features") or {}
    derived = _derived_metric_values(features)
    value = derived.get(metric_name)
    source_node = features.get(metric_name)
    support_mode = _node_support(source_node)
    if value is None:
        value = _node_value(source_node)
    if value is None:
        return None
    return {
        "metric": metric_name,
        "value": float(value),
        "percentile": percentile_maps.get(metric_name, {}).get(company_id),
        "support_mode": "exact" if metric_name.startswith("derived.") else support_mode,
    }


