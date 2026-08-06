#!/usr/bin/env python3
"""Repair market-pricing metrics in a materialized company-state artifact.

This is a narrow repair pass for metrics that are conceptually simple but can be
missing in the built artifact even when the underlying normalized inputs are
already present.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


REPAIR_METRICS = [
    "market.enterprise_value",
    "market.ev_ebitda",
    "market.pe_ratio",
    "operating.ebitda_margin_ttm",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-path", required=True, help="Input company-state JSONL artifact")
    parser.add_argument("--out", required=True, help="Output repaired JSONL artifact")
    parser.add_argument("--summary-out", help="Optional summary JSON")
    return parser.parse_args()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def iter_rows(path: Path):
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
    return float(value)


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


def _base_repaired_node(node: Dict[str, Any], *, computed_at: str) -> Dict[str, Any]:
    repaired = copy.deepcopy(node)
    repaired["computed_at"] = computed_at
    repaired["missing_reason"] = None
    repaired["quality_flags"] = repaired.get("quality_flags") or None
    return repaired


def repair_enterprise_value(*, features: Dict[str, Any], computed_at: str) -> bool:
    target = features.get("market.enterprise_value")
    if not target:
        return False

    market_cap_node = features.get("market.market_cap_provider_direct") or features.get("market.market_cap")
    debt_node = features.get("capital_structure.total_debt_provider_direct")
    cash_grouped_node = features.get("liquidity.cash_and_short_term_investments_provider_direct")
    cash_exact_node = features.get("liquidity.cash_and_equivalents_statement_direct")

    market_cap = _node_value(market_cap_node)
    debt = _node_value(debt_node)
    cash = _node_value(cash_grouped_node)
    cash_source_metric = "liquidity.cash_and_short_term_investments_provider_direct"
    if cash is None:
        cash = _node_value(cash_exact_node)
        cash_source_metric = "liquidity.cash_and_equivalents_statement_direct"

    if market_cap is None or debt is None or cash is None:
        return False

    repaired = _base_repaired_node(target, computed_at=computed_at)
    repaired["value"] = market_cap + debt - cash
    repaired["fallback_used"] = "input_layer_market_cap_plus_total_debt_minus_cash"
    repaired["support_mode"] = (
        "exact"
        if all(
            _node_support(node) == "exact"
            for node in (market_cap_node, debt_node, cash_grouped_node if cash_source_metric.endswith("provider_direct") else cash_exact_node)
        )
        else "proxy_missing_component"
    )
    repaired["provenance"] = _union_provenance(market_cap_node, debt_node, cash_grouped_node, cash_exact_node)
    repaired["component_breakdown"] = {
        "market_cap": market_cap,
        "total_debt": debt,
        "cash": cash,
        "cash_source_metric": cash_source_metric,
        "formula": "market_cap_provider_direct + total_debt_provider_direct - cash",
    }
    repaired["quality_flags"] = ["input_layer_ev_repair"]
    features["market.enterprise_value"] = repaired
    return True


def repair_ebitda_margin_ttm(*, features: Dict[str, Any], computed_at: str) -> bool:
    target = features.get("operating.ebitda_margin_ttm")
    if not target:
        return False

    revenue_node = features.get("operating.revenue_ttm_provider_direct")
    ebitda_node = features.get("operating.ebitda_ltm_provider_direct")
    normalized_node = features.get("operating.operating_earnings_normalized")

    revenue = _node_value(revenue_node)
    ebitda = _node_value(ebitda_node)
    ebitda_source_metric = "operating.ebitda_ltm_provider_direct"
    fallback_used = "provider_direct_revenue_and_ebitda"
    if ebitda is None:
        ebitda = _node_value(normalized_node)
        ebitda_source_metric = "operating.operating_earnings_normalized"
        fallback_used = "provider_revenue_plus_normalized_operating_earnings"

    if revenue in (None, 0) or ebitda is None:
        return False

    repaired = _base_repaired_node(target, computed_at=computed_at)
    repaired["value"] = ebitda / revenue
    repaired["fallback_used"] = fallback_used
    repaired["support_mode"] = (
        "exact"
        if _node_support(revenue_node) == "exact"
        and (
            (_node_support(ebitda_node) == "exact" and ebitda_source_metric == "operating.ebitda_ltm_provider_direct")
            or (_node_support(normalized_node) == "exact" and ebitda_source_metric == "operating.operating_earnings_normalized")
        )
        else "proxy_missing_component"
    )
    repaired["provenance"] = _union_provenance(revenue_node, ebitda_node, normalized_node)
    repaired["component_breakdown"] = {
        "revenue": revenue,
        "ebitda": ebitda,
        "revenue_source_metric": "operating.revenue_ttm_provider_direct",
        "ebitda_source_metric": ebitda_source_metric,
        "formula": "ebitda / revenue",
        "period_match_type": "input_layer_ttm_fallback",
    }
    repaired["quality_flags"] = ["input_layer_ebitda_margin_repair"]
    features["operating.ebitda_margin_ttm"] = repaired
    return True


def repair_ev_ebitda(*, features: Dict[str, Any], computed_at: str) -> bool:
    target = features.get("market.ev_ebitda")
    if not target:
        return False

    enterprise_value_node = features.get("market.enterprise_value")
    ebitda_node = features.get("operating.ebitda_ltm_provider_direct")
    normalized_node = features.get("operating.operating_earnings_normalized")

    ev = _node_value(enterprise_value_node)
    ebitda = _node_value(ebitda_node)
    ebitda_source_metric = "operating.ebitda_ltm_provider_direct"
    fallback_used = "repaired_enterprise_value_plus_provider_ebitda"
    if ebitda in (None, 0):
        ebitda = _node_value(normalized_node)
        ebitda_source_metric = "operating.operating_earnings_normalized"
        fallback_used = "repaired_enterprise_value_plus_normalized_operating_earnings"

    if ev is None or ebitda in (None, 0) or ebitda <= 0:
        return False

    repaired = _base_repaired_node(target, computed_at=computed_at)
    repaired["value"] = ev / ebitda
    repaired["fallback_used"] = fallback_used
    repaired["support_mode"] = (
        "exact"
        if _node_support(enterprise_value_node) == "exact"
        and (
            (_node_support(ebitda_node) == "exact" and ebitda_source_metric == "operating.ebitda_ltm_provider_direct")
            or (_node_support(normalized_node) == "exact" and ebitda_source_metric == "operating.operating_earnings_normalized")
        )
        else "proxy_missing_component"
    )
    repaired["provenance"] = _union_provenance(enterprise_value_node, ebitda_node, normalized_node)
    repaired["component_breakdown"] = {
        "enterprise_value": ev,
        "ebitda_ttm": ebitda,
        "ebitda_source_metric": ebitda_source_metric,
        "formula": "enterprise_value / ebitda_ttm",
    }
    repaired["quality_flags"] = ["input_layer_ev_ebitda_repair"]
    features["market.ev_ebitda"] = repaired
    return True


def repair_pe_ratio(*, features: Dict[str, Any], computed_at: str) -> bool:
    target = features.get("market.pe_ratio")
    if not target:
        return False

    market_cap_node = features.get("market.market_cap_provider_direct")
    net_income_node = features.get("earnings.net_income_ttm_provider_direct")
    price_node = features.get("market.price_spot")

    market_cap = _node_value(market_cap_node)
    net_income = _node_value(net_income_node)
    price = _node_value(price_node)

    if net_income is not None and net_income > 0 and market_cap is not None:
        repaired = _base_repaired_node(target, computed_at=computed_at)
        repaired["value"] = market_cap / net_income
        repaired["fallback_used"] = "market_cap_plus_net_income_ttm"
        repaired["support_mode"] = (
            "exact"
            if _node_support(market_cap_node) == "exact" and _node_support(net_income_node) == "exact"
            else "proxy_missing_component"
        )
        repaired["provenance"] = _union_provenance(market_cap_node, net_income_node)
        repaired["component_breakdown"] = {
            "market_cap": market_cap,
            "net_income_ttm": net_income,
            "price": price,
            "formula": "market_cap_provider_direct / net_income_ttm_provider_direct",
        }
        repaired["quality_flags"] = ["input_layer_pe_ratio_repair"]
        features["market.pe_ratio"] = repaired
        return True

    if net_income is not None and net_income <= 0:
        repaired = _base_repaired_node(target, computed_at=computed_at)
        repaired["value"] = None
        repaired["missing_reason"] = "non_positive_net_income_ttm"
        repaired["fallback_used"] = "market_cap_plus_net_income_ttm"
        repaired["support_mode"] = "unsupported"
        repaired["provenance"] = _union_provenance(market_cap_node, net_income_node)
        repaired["component_breakdown"] = {
            "market_cap": market_cap,
            "net_income_ttm": net_income,
            "price": price,
            "formula": "market_cap_provider_direct / net_income_ttm_provider_direct",
        }
        repaired["quality_flags"] = ["input_layer_pe_ratio_repair", "non_positive_net_income_ttm"]
        features["market.pe_ratio"] = repaired
        return True

    return False


def build_summary(path: Path) -> Dict[str, Dict[str, int]]:
    counters: Dict[str, Counter[str]] = {metric: Counter() for metric in REPAIR_METRICS}
    for row in iter_rows(path):
        features = row.get("features") or {}
        for metric in REPAIR_METRICS:
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


def main() -> None:
    args = parse_args()
    artifact_path = Path(args.artifact_path)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    computed_at = _now_iso()

    with out_path.open("w") as out_handle:
        for row in iter_rows(artifact_path):
            features = row.get("features") or {}
            repair_enterprise_value(features=features, computed_at=computed_at)
            repair_ebitda_margin_ttm(features=features, computed_at=computed_at)
            repair_ev_ebitda(features=features, computed_at=computed_at)
            repair_pe_ratio(features=features, computed_at=computed_at)
            out_handle.write(json.dumps(row) + "\n")

    if args.summary_out:
        summary_path = Path(args.summary_out)
        summary_path.write_text(json.dumps(build_summary(out_path), indent=2))

    print(f"Repaired market-pricing metrics -> {out_path}")


if __name__ == "__main__":
    main()
