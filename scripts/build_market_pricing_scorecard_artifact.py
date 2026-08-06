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


def _score_from_components(
    score_metric: str,
    *,
    row: Dict[str, Any],
    percentile_maps: Dict[str, Dict[str, float]],
    computed_at: str,
) -> Dict[str, Any]:
    company_id = str(row.get("company_id") or "")
    as_of_time = str(row.get("as_of_time") or "")
    features = row.get("features") or {}
    component_nodes: List[Dict[str, Any] | None] = []
    details: List[Dict[str, Any]] = []
    total_components = len(RAW_SCORE_COMPONENTS[score_metric])
    exact_like = True
    for metric_name, _ in RAW_SCORE_COMPONENTS[score_metric]:
        detail = _component_detail(metric_name, row=row, percentile_maps=percentile_maps)
        component_nodes.append(features.get(metric_name))
        if detail is None or detail.get("percentile") is None:
            exact_like = False
            continue
        if detail.get("support_mode") != "exact":
            exact_like = False
        details.append(detail)
    if not details:
        return _base_score_node(
            name=score_metric,
            value=None,
            computed_at=computed_at,
            as_of_time=as_of_time,
            support_mode="unsupported",
            fallback_used=None,
            provenance=_union_provenance(*component_nodes),
            component_breakdown=None,
        )

    score_value = sum(d["percentile"] for d in details) / len(details)
    support_mode = "exact" if exact_like and len(details) == total_components else "proxy_missing_component"
    quality_flags = None
    if len(details) < total_components:
        quality_flags = ["partial_component_coverage"]
    return _base_score_node(
        name=score_metric,
        value=score_value,
        computed_at=computed_at,
        as_of_time=as_of_time,
        support_mode=support_mode,
        fallback_used="cross_sectional_percentile_scorecard",
        provenance=_union_provenance(*component_nodes),
        component_breakdown={
            "component_count_used": len(details),
            "component_count_total": total_components,
            "formula": "mean(component_percentiles)",
            "components": details,
        },
        quality_flags=quality_flags,
    )


def _overall_score_node(
    *,
    row: Dict[str, Any],
    computed_at: str,
) -> Dict[str, Any]:
    features = row.get("features") or {}
    as_of_time = str(row.get("as_of_time") or "")
    available = []
    provenance_nodes = []
    total_weight = 0.0
    weighted_sum = 0.0
    exact_like = True
    for metric_name, weight in OVERALL_WEIGHTS.items():
        node = features.get(metric_name)
        provenance_nodes.append(node)
        value = _node_value(node)
        if value is None:
            exact_like = False
            continue
        weighted_sum += value * weight
        total_weight += weight
        available.append({"metric": metric_name, "value": value, "weight": weight, "support_mode": _node_support(node)})
        if _node_support(node) != "exact":
            exact_like = False
    if total_weight <= 0:
        return _base_score_node(
            name="market.comp_overall_score",
            value=None,
            computed_at=computed_at,
            as_of_time=as_of_time,
            support_mode="unsupported",
            fallback_used=None,
            provenance=_union_provenance(*provenance_nodes),
            component_breakdown=None,
        )
    support_mode = "exact" if exact_like and len(available) == len(OVERALL_WEIGHTS) else "proxy_missing_component"
    quality_flags = None if len(available) == len(OVERALL_WEIGHTS) else ["partial_subscore_coverage"]
    return _base_score_node(
        name="market.comp_overall_score",
        value=weighted_sum / total_weight,
        computed_at=computed_at,
        as_of_time=as_of_time,
        support_mode=support_mode,
        fallback_used="weighted_market_pricing_scorecard",
        provenance=_union_provenance(*provenance_nodes),
        component_breakdown={
            "subscores": available,
            "total_weight_used": total_weight,
            "formula": "weighted_mean(value_score, quality_score, balance_sheet_score, risk_score)",
        },
        quality_flags=quality_flags,
    )


def _valuation_gap_node(
    *,
    row: Dict[str, Any],
    computed_at: str,
) -> Dict[str, Any]:
    features = row.get("features") or {}
    as_of_time = str(row.get("as_of_time") or "")
    value_node = features.get("market.value_score")
    quality_node = features.get("market.quality_score")
    balance_node = features.get("market.balance_sheet_score")
    risk_node = features.get("market.risk_score")
    value = _node_value(value_node)
    quality = _node_value(quality_node)
    balance = _node_value(balance_node)
    risk = _node_value(risk_node)
    if value is None or quality is None:
        return _base_score_node(
            name="market.valuation_gap_score",
            value=None,
            computed_at=computed_at,
            as_of_time=as_of_time,
            support_mode="unsupported",
            fallback_used=None,
            provenance=_union_provenance(value_node, quality_node, balance_node, risk_node),
            component_breakdown=None,
        )
    support_stack = [n for n in [value_node, quality_node, balance_node, risk_node] if n is not None]
    optional_values = [v for v in [balance, risk] if v is not None]
    fundamental_anchor = (quality + sum(optional_values)) / (1 + len(optional_values))
    gap = fundamental_anchor - value
    exact_like = all(_node_support(node) == "exact" for node in support_stack if _node_value(node) is not None)
    support_mode = "exact" if exact_like and balance is not None and risk is not None else "proxy_missing_component"
    quality_flags = None if balance is not None and risk is not None else ["partial_anchor_coverage"]
    return _base_score_node(
        name="market.valuation_gap_score",
        value=gap,
        computed_at=computed_at,
        as_of_time=as_of_time,
        support_mode=support_mode,
        fallback_used="fundamental_anchor_minus_value_score",
        provenance=_union_provenance(value_node, quality_node, balance_node, risk_node),
        component_breakdown={
            "value_score": value,
            "quality_score": quality,
            "balance_sheet_score": balance,
            "risk_score": risk,
            "fundamental_anchor": fundamental_anchor,
            "formula": "mean(quality, balance_sheet?, risk?) - value_score",
        },
        quality_flags=quality_flags,
    )


def build_summary(path: Path) -> Dict[str, Dict[str, int]]:
    counters: Dict[str, Counter[str]] = {metric: Counter() for metric in SCORE_METRICS}
    for row in iter_rows(path):
        features = row.get("features") or {}
        for metric in SCORE_METRICS:
            node = features.get(metric) or {}
            mode = str(node.get("support_mode") or "unsupported")
            if node.get("value") is None:
                mode = "unsupported"
            counters[metric][mode] += 1
    summary: Dict[str, Dict[str, int]] = {}
    for metric, counter in counters.items():
        summary[metric] = {
            "exact": counter["exact"],
            "proxy_missing_component": counter["proxy_missing_component"],
            "unsupported": counter["unsupported"],
        }
    return summary


def build_leaderboard(path: Path, *, limit: int = 20) -> Dict[str, Any]:
    rows = list(iter_rows(path))
    candidates = []
    for row in rows:
        features = row.get("features") or {}
        overall = _node_value(features.get("market.comp_overall_score"))
        if overall is None:
            continue
        candidates.append(
            {
                "company_id": str(row.get("company_id") or ""),
                "overall_score": overall,
                "value_score": _node_value(features.get("market.value_score")),
                "quality_score": _node_value(features.get("market.quality_score")),
                "balance_sheet_score": _node_value(features.get("market.balance_sheet_score")),
                "risk_score": _node_value(features.get("market.risk_score")),
                "valuation_gap_score": _node_value(features.get("market.valuation_gap_score")),
            }
        )
    overall_top = sorted(candidates, key=lambda row: (row["overall_score"], row["valuation_gap_score"] or -999.0), reverse=True)[
        :limit
    ]
    valuation_gap_top = sorted(
        [row for row in candidates if row["valuation_gap_score"] is not None],
        key=lambda row: row["valuation_gap_score"],
        reverse=True,
    )[:limit]
    return {"top_overall": overall_top, "top_valuation_gap": valuation_gap_top}


def main() -> None:
    args = parse_args()
    artifact_path = Path(args.artifact_path)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(iter_rows(artifact_path))
    percentile_maps = _collect_cross_section(rows)
    computed_at = _now_iso()

    with out_path.open("w") as out_handle:
        for row in rows:
            features = row.get("features") or {}
            for score_metric in RAW_SCORE_COMPONENTS:
                features[score_metric] = _score_from_components(
                    score_metric,
                    row=row,
                    percentile_maps=percentile_maps,
                    computed_at=computed_at,
                )
            features["market.comp_overall_score"] = _overall_score_node(row=row, computed_at=computed_at)
            features["market.valuation_gap_score"] = _valuation_gap_node(row=row, computed_at=computed_at)
            out_handle.write(json.dumps(row) + "\n")

    if args.summary_out:
        Path(args.summary_out).write_text(json.dumps(build_summary(out_path), indent=2))
    if args.leaderboard_out:
        Path(args.leaderboard_out).write_text(json.dumps(build_leaderboard(out_path), indent=2))

    print(f"Built market-pricing scorecard -> {out_path}")


if __name__ == "__main__":
    main()
