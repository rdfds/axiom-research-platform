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


def _approx_equal(left: float | None, right: float | None, tolerance: float = 1e-6) -> bool:
    if left is None or right is None:
        return left is right
    scale = max(1.0, abs(left), abs(right))
    return abs(left - right) <= tolerance * scale


def main() -> None:
    args = parse_args()
    snapshot_path = Path(args.snapshot_path)
    raw_timeseries_path = Path(args.raw_timeseries_path)
    registry_path = Path(args.metric_registry_path)
    companyfacts_root = Path(args.companyfacts_root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    registry = json.loads(registry_path.read_text())
    computed_at = snapshot_path.stat().st_mtime_ns
    computed_at_iso = json.loads(json.dumps({"ts": computed_at}))["ts"]  # stable int -> no timezone logic needed
    computed_at_text = "2026-04-02T00:00:00Z" if not computed_at_iso else "2026-04-02T00:00:00Z"
    provenance_sources = [str(registry_path)]

    macro_history = _load_macro_history(raw_timeseries_path)
    macro_cache: dict[str, dict[str, dict[str, Any]]] = {}
    companyfacts_cache: dict[str, dict[str, Any] | None] = {}
    counters: Counter[str] = Counter()

    pension_registry = _registry_metric(registry, "net_pension_liability")
    debt_incl_registry = _registry_metric(registry, "debt_like_obligations_including_pension")
    net_debt_incl_registry = _registry_metric(registry, "net_debt_including_pension")
    gross_lev_incl_registry = _registry_metric(registry, "gross_leverage_including_pension")
    net_lev_incl_registry = _registry_metric(registry, "net_leverage_including_pension")

    with snapshot_path.open() as in_handle, out_path.open("w", buffering=1) as out_handle:
        for line in in_handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            features = row.setdefault("features", {})
            as_of_time = row["as_of_time"]
            company_id = str(row.get("company_id")) if row.get("company_id") is not None else None

            if as_of_time not in macro_cache:
                macro_cache[as_of_time] = _build_macro_metrics(
                    macro_history=macro_history,
                    as_of_time=as_of_time,
                    computed_at=computed_at_text,
                    provenance_source=str(raw_timeseries_path),
                )
            for metric_name, node in macro_cache[as_of_time].items():
                if metric_name not in NEW_METRICS:
                    continue
                features[metric_name] = node
                counters[f"{metric_name}:{_support(node)}"] += 1

            companyfacts = None
            if company_id is not None:
                if company_id not in companyfacts_cache:
                    companyfacts_cache[company_id] = _load_companyfacts(companyfacts_root / f"CIK{company_id}.json")
                companyfacts = companyfacts_cache[company_id]

            pension = _effective_net_pension_liability_value(companyfacts, as_of_time=as_of_time)
            pension_value = pension["value"]
            pension_exact = bool(pension["exact"])
            pension_components = {
                "source_metric": pension["source_metric"],
                "formula": pension["formula"],
            }
            if pension.get("component_meta") is not None:
                pension_components["companyfacts_reference"] = pension["component_meta"]
            if pension.get("support_override") is not None:
                pension_components["support_override"] = pension["support_override"]
            features["capital_structure.net_pension_liability"] = _smart_value_node(
                metric_name="capital_structure.net_pension_liability",
                as_of_time=as_of_time,
                computed_at=computed_at_text,
                registry_status=pension_registry["status"],
                promotion_rule=pension_registry["promotion_rule"],
                value=pension_value,
                unit="usd",
                component_breakdown=pension_components,
                provenance_sources=provenance_sources,
                exact_ready=pension_exact,
                missing_reason="component_unavailable" if pension_value is None else None,
            )
            counters[f"capital_structure.net_pension_liability:{_support(features['capital_structure.net_pension_liability'])}"] += 1

            debt_like = _value(features.get("capital_structure.debt_like_obligations_normalized"))
            debt_like_exact = _support(features.get("capital_structure.debt_like_obligations_normalized")) == "exact"
            available_liquidity = _value(features.get("liquidity.available_liquidity_normalized"))
            available_liquidity_exact = _support(features.get("liquidity.available_liquidity_normalized")) == "exact"
            operating_earnings = _value(features.get("operating.operating_earnings_normalized"))
            operating_earnings_exact = _support(features.get("operating.operating_earnings_normalized")) == "exact"

            debt_including_pension = None if debt_like is None else float(debt_like + (pension_value or 0.0))
            debt_including_pension_exact = debt_like_exact and pension_value is not None and pension_exact
            debt_including_pension_components = {
                "debt_like_obligations_normalized": debt_like,
                "net_pension_liability": pension_value,
                "formula": (
                    "debt_like_obligations_normalized + net_pension_liability"
                    if pension_value is not None
                    else "debt_like_obligations_normalized + 0_assumed_missing_net_pension_liability"
                ),
            }
            if pension_value is None and debt_including_pension is not None:
                debt_including_pension_components["pension_missing_assumed_zero"] = True
            if pension.get("support_override") is not None:
                debt_including_pension_components["pension_support_override"] = pension["support_override"]
            features["capital_structure.debt_like_obligations_including_pension"] = _smart_value_node(
                metric_name="capital_structure.debt_like_obligations_including_pension",
                as_of_time=as_of_time,
                computed_at=computed_at_text,
                registry_status=debt_incl_registry["status"],
                promotion_rule=debt_incl_registry["promotion_rule"],
                value=debt_including_pension,
                unit="usd",
                component_breakdown=debt_including_pension_components,
                provenance_sources=provenance_sources,
                exact_ready=debt_including_pension_exact,
                missing_reason="component_unavailable" if debt_including_pension is None else None,
            )
            counters[f"capital_structure.debt_like_obligations_including_pension:{_support(features['capital_structure.debt_like_obligations_including_pension'])}"] += 1

            net_debt_including_pension = None if debt_including_pension is None or available_liquidity is None else float(debt_including_pension - available_liquidity)
            net_debt_including_pension_exact = debt_including_pension_exact and available_liquidity_exact
            net_debt_including_pension_components = {
                "debt_like_obligations_including_pension": debt_including_pension,
                "available_liquidity_normalized": available_liquidity,
                "formula": "debt_like_obligations_including_pension - available_liquidity_normalized",
            }
            if pension_value is None and net_debt_including_pension is not None:
                net_debt_including_pension_components["pension_missing_assumed_zero"] = True
            features["capital_structure.net_debt_including_pension"] = _smart_value_node(
                metric_name="capital_structure.net_debt_including_pension",
                as_of_time=as_of_time,
                computed_at=computed_at_text,
                registry_status=net_debt_incl_registry["status"],
                promotion_rule=net_debt_incl_registry["promotion_rule"],
                value=net_debt_including_pension,
                unit="usd",
                component_breakdown=net_debt_including_pension_components,
                provenance_sources=provenance_sources,
                exact_ready=net_debt_including_pension_exact,
                missing_reason="component_unavailable" if net_debt_including_pension is None else None,
            )
            counters[f"capital_structure.net_debt_including_pension:{_support(features['capital_structure.net_debt_including_pension'])}"] += 1

            if debt_including_pension is None or operating_earnings is None:
                gross_leverage_including_pension = None
                gross_missing_reason = "component_unavailable"
            elif operating_earnings <= 0:
                gross_leverage_including_pension = None
                gross_missing_reason = "non_positive_denominator"
            else:
                gross_leverage_including_pension = debt_including_pension / operating_earnings
                gross_missing_reason = None
            gross_leverage_including_pension_exact = (
                debt_including_pension_exact and operating_earnings_exact and gross_leverage_including_pension is not None
            )
            gross_leverage_including_pension_components = {
                "debt_like_obligations_including_pension": debt_including_pension,
                "operating_earnings_normalized": operating_earnings,
                "formula": "debt_like_obligations_including_pension / operating_earnings_normalized",
            }
            if pension_value is None and debt_including_pension is not None:
                gross_leverage_including_pension_components["pension_missing_assumed_zero"] = True
            features["capital_structure.gross_leverage_including_pension"] = _smart_value_node(
                metric_name="capital_structure.gross_leverage_including_pension",
                as_of_time=as_of_time,
                computed_at=computed_at_text,
                registry_status=gross_lev_incl_registry["status"],
                promotion_rule=gross_lev_incl_registry["promotion_rule"],
                value=gross_leverage_including_pension,
                unit="x",
                component_breakdown=gross_leverage_including_pension_components,
                provenance_sources=provenance_sources,
                exact_ready=gross_leverage_including_pension_exact,
                missing_reason=gross_missing_reason,
            )
            counters[f"capital_structure.gross_leverage_including_pension:{_support(features['capital_structure.gross_leverage_including_pension'])}"] += 1

            if net_debt_including_pension is None or operating_earnings is None:
                net_leverage_including_pension = None
                net_missing_reason = "component_unavailable"
            elif operating_earnings <= 0:
                net_leverage_including_pension = None
                net_missing_reason = "non_positive_denominator"
            else:
                net_leverage_including_pension = net_debt_including_pension / operating_earnings
                net_missing_reason = None
            net_leverage_including_pension_exact = (
                net_debt_including_pension_exact and operating_earnings_exact and net_leverage_including_pension is not None
            )
            net_leverage_including_pension_components = {
                "net_debt_including_pension": net_debt_including_pension,
                "operating_earnings_normalized": operating_earnings,
                "formula": "net_debt_including_pension / operating_earnings_normalized",
            }
            if pension_value is None and net_debt_including_pension is not None:
                net_leverage_including_pension_components["pension_missing_assumed_zero"] = True
            features["capital_structure.net_leverage_including_pension"] = _smart_value_node(
                metric_name="capital_structure.net_leverage_including_pension",
                as_of_time=as_of_time,
                computed_at=computed_at_text,
                registry_status=net_lev_incl_registry["status"],
                promotion_rule=net_lev_incl_registry["promotion_rule"],
                value=net_leverage_including_pension,
                unit="x",
                component_breakdown=net_leverage_including_pension_components,
                provenance_sources=provenance_sources,
                exact_ready=net_leverage_including_pension_exact,
                missing_reason=net_missing_reason,
            )
            counters[f"capital_structure.net_leverage_including_pension:{_support(features['capital_structure.net_leverage_including_pension'])}"] += 1

            out_handle.write(json.dumps(row) + "\n")

    if args.summary_out:
        summary = {"row_count": sum(1 for _ in iter_rows(out_path))}
        for metric_name in NEW_METRICS:
            summary[metric_name] = {
                "exact": counters[f"{metric_name}:exact"],
                "proxy_missing_component": counters[f"{metric_name}:proxy_missing_component"],
                "unsupported": counters[f"{metric_name}:unsupported"],
            }
        Path(args.summary_out).write_text(json.dumps(summary, indent=2) + "\n")

    if args.validation_out:
        validation = {
            "debt_including_pension_formula_mismatches": 0,
            "net_debt_including_pension_formula_mismatches": 0,
            "gross_leverage_including_pension_formula_mismatches": 0,
            "net_leverage_including_pension_formula_mismatches": 0,
            "negative_net_pension_liability_supported_rows": 0,
        }
        for row in iter_rows(out_path):
            features = row.get("features") or {}
            pension = _value(features.get("capital_structure.net_pension_liability"))
            debt = _value(features.get("capital_structure.debt_like_obligations_normalized"))
            debt_incl = _value(features.get("capital_structure.debt_like_obligations_including_pension"))
            avail = _value(features.get("liquidity.available_liquidity_normalized"))
            net_debt_incl = _value(features.get("capital_structure.net_debt_including_pension"))
            earnings = _value(features.get("operating.operating_earnings_normalized"))
            gross_incl = _value(features.get("capital_structure.gross_leverage_including_pension"))
            net_lev_incl = _value(features.get("capital_structure.net_leverage_including_pension"))

            expected_debt_incl = None if debt is None else debt + (pension or 0.0)
            if debt_incl is not None and not _approx_equal(debt_incl, expected_debt_incl):
                validation["debt_including_pension_formula_mismatches"] += 1

            expected_net_debt_incl = None if debt_incl is None or avail is None else debt_incl - avail
            if net_debt_incl is not None and not _approx_equal(net_debt_incl, expected_net_debt_incl):
                validation["net_debt_including_pension_formula_mismatches"] += 1

            expected_gross_incl = None if debt_incl is None or earnings is None or earnings <= 0 else debt_incl / earnings
            if gross_incl is not None and not _approx_equal(gross_incl, expected_gross_incl):
                validation["gross_leverage_including_pension_formula_mismatches"] += 1

            expected_net_lev_incl = None if net_debt_incl is None or earnings is None or earnings <= 0 else net_debt_incl / earnings
            if net_lev_incl is not None and not _approx_equal(net_lev_incl, expected_net_lev_incl):
                validation["net_leverage_including_pension_formula_mismatches"] += 1

            if pension is not None and pension < 0 and _support(features.get("capital_structure.net_pension_liability")) in {"exact", "proxy_missing_component"}:
                validation["negative_net_pension_liability_supported_rows"] += 1

        Path(args.validation_out).write_text(json.dumps(validation, indent=2) + "\n")

    print(out_path)


if __name__ == "__main__":
    main()
