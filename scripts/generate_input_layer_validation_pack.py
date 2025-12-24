#!/usr/bin/env python3
"""Generate a repeatable validation pack for the canonical input-layer artifact."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


APPROVED_METRICS = [
    "market.market_cap_provider_direct",
    "operating.revenue_ttm_provider_direct",
    "operating.ebitda_ltm_provider_direct",
    "earnings.net_income_ttm_provider_direct",
    "liquidity.cash_and_short_term_investments_provider_direct",
    "capital_structure.total_debt_provider_direct",
    "capital_structure.net_debt_standardized",
    "capital_structure.gross_leverage_standardized",
    "capital_structure.net_leverage_standardized",
    "operating.ebitda_margin_standardized",
    "earnings.net_margin_standardized",
    "market.price_spot",
    "market.total_return_1m_standardized",
    "market.total_return_3m_standardized",
    "market.total_return_6m_standardized",
    "market.total_return_12m_standardized",
    "macro.sofr_or_fed_funds",
    "macro.ust_2y_yield",
    "macro.ust_10y_yield",
    "macro.curve_2s10s",
    "macro.ig_oas",
    "macro.hy_oas",
    "macro.cpi_yoy",
    "macro.unemployment_rate",
    "macro.retail_sales_yoy",
    "macro.wti_crude",
    "capital_structure.debt_like_obligations_normalized",
    "liquidity.available_liquidity_normalized",
    "operating.operating_earnings_normalized",
    "capital_structure.net_debt_normalized",
    "capital_structure.gross_leverage_normalized",
    "capital_structure.net_leverage_normalized",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-path", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    return parser.parse_args()


def _value(node: dict[str, Any] | None) -> float | None:
    if not node:
        return None
    value = node.get("value")
    return None if value is None else float(value)


def _support(node: dict[str, Any] | None) -> str:
    if not node:
        return "missing_metric"
    return node.get("support_mode") or "missing_metric"


def _approx_equal(a: float | None, b: float | None, tol: float = 1e-6) -> bool:
    if a is None or b is None:
        return a is b
    scale = max(1.0, abs(a), abs(b))
    return abs(a - b) <= tol * scale


def _company_ref(row: dict[str, Any]) -> dict[str, Any]:
    feats = row.get("features") or {}
    provider_metrics = [
        "market.market_cap_provider_direct",
        "liquidity.cash_and_short_term_investments_provider_direct",
        "capital_structure.total_debt_provider_direct",
    ]
    company_name = None
    instrument = None
    for metric in provider_metrics:
        breakdown = (feats.get(metric) or {}).get("component_breakdown") or {}
        company_name = company_name or breakdown.get("provider_company_name")
        instrument = instrument or breakdown.get("reference_instrument")
    return {
        "company_id": row.get("company_id"),
        "company_name": company_name,
        "reference_instrument": instrument,
    }


def _record_issue(container: dict[str, Any], category: str, metric: str, example: dict[str, Any]) -> None:
    bucket = container.setdefault(category, {}).setdefault(metric, {"count": 0, "examples": []})
    bucket["count"] += 1
    if len(bucket["examples"]) < 5:
        bucket["examples"].append(example)


def main() -> None:
    args = parse_args()
    artifact_path = Path(args.artifact_path)
    rows = [json.loads(line) for line in artifact_path.open() if line.strip()]
    if not rows:
        raise SystemExit("artifact empty")

    coverage: dict[str, Counter[str]] = defaultdict(Counter)
    issues: dict[str, Any] = {}
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        feats = row.get("features") or {}
        ref = _company_ref(row)

        for metric in APPROVED_METRICS:
            coverage[metric][_support(feats.get(metric))] += 1

        def value(metric: str) -> float | None:
            return _value(feats.get(metric))

        def support(metric: str) -> str:
            return _support(feats.get(metric))

        revenue = value("operating.revenue_ttm_provider_direct")
        ebitda = value("operating.ebitda_ltm_provider_direct")
        net_income = value("earnings.net_income_ttm_provider_direct")
        total_debt = value("capital_structure.total_debt_provider_direct")
        cash_sti = value("liquidity.cash_and_short_term_investments_provider_direct")
        net_debt_std = value("capital_structure.net_debt_standardized")
        gross_lev_std = value("capital_structure.gross_leverage_standardized")
        net_lev_std = value("capital_structure.net_leverage_standardized")
        ebitda_margin = value("operating.ebitda_margin_standardized")
        net_margin = value("earnings.net_margin_standardized")
        debt_like = value("capital_structure.debt_like_obligations_normalized")
        avail_liq = value("liquidity.available_liquidity_normalized")
        op_earn = value("operating.operating_earnings_normalized")
        net_debt_norm = value("capital_structure.net_debt_normalized")
        gross_lev_norm = value("capital_structure.gross_leverage_normalized")
        net_lev_norm = value("capital_structure.net_leverage_normalized")
        lease = value("capital_structure.lease_liabilities_sec_exact")

        if total_debt is not None and cash_sti is not None and net_debt_std is not None and not _approx_equal(net_debt_std, total_debt - cash_sti):
            _record_issue(issues, "formula_mismatch", "capital_structure.net_debt_standardized", {**ref, "expected": total_debt - cash_sti, "actual": net_debt_std})
        if total_debt is not None and ebitda is not None and ebitda > 0 and gross_lev_std is not None and not _approx_equal(gross_lev_std, total_debt / ebitda):
            _record_issue(issues, "formula_mismatch", "capital_structure.gross_leverage_standardized", {**ref, "expected": total_debt / ebitda, "actual": gross_lev_std})
        if net_debt_std is not None and ebitda is not None and ebitda > 0 and net_lev_std is not None and not _approx_equal(net_lev_std, net_debt_std / ebitda):
            _record_issue(issues, "formula_mismatch", "capital_structure.net_leverage_standardized", {**ref, "expected": net_debt_std / ebitda, "actual": net_lev_std})
        if revenue is not None and revenue > 0 and ebitda is not None and ebitda_margin is not None and not _approx_equal(ebitda_margin, ebitda / revenue):
            _record_issue(issues, "formula_mismatch", "operating.ebitda_margin_standardized", {**ref, "expected": ebitda / revenue, "actual": ebitda_margin})
        if revenue is not None and revenue > 0 and net_income is not None and net_margin is not None and not _approx_equal(net_margin, net_income / revenue):
            _record_issue(issues, "formula_mismatch", "earnings.net_margin_standardized", {**ref, "expected": net_income / revenue, "actual": net_margin})

        if debt_like is not None and total_debt is not None and debt_like + 1e-9 < total_debt:
            _record_issue(issues, "economic_inconsistency", "capital_structure.debt_like_obligations_normalized", {**ref, "total_debt": total_debt, "debt_like": debt_like})
        if debt_like is not None and avail_liq is not None and net_debt_norm is not None and not _approx_equal(net_debt_norm, debt_like - avail_liq):
            _record_issue(issues, "formula_mismatch", "capital_structure.net_debt_normalized", {**ref, "expected": debt_like - avail_liq, "actual": net_debt_norm})
        if debt_like is not None and op_earn is not None and op_earn > 0 and gross_lev_norm is not None and not _approx_equal(gross_lev_norm, debt_like / op_earn):
            _record_issue(issues, "formula_mismatch", "capital_structure.gross_leverage_normalized", {**ref, "expected": debt_like / op_earn, "actual": gross_lev_norm})
        if net_debt_norm is not None and op_earn is not None and op_earn > 0 and net_lev_norm is not None and not _approx_equal(net_lev_norm, net_debt_norm / op_earn):
            _record_issue(issues, "formula_mismatch", "capital_structure.net_leverage_normalized", {**ref, "expected": net_debt_norm / op_earn, "actual": net_lev_norm})

        if avail_liq is not None and avail_liq < 0:
            _record_issue(issues, "economic_inconsistency", "liquidity.available_liquidity_normalized", {**ref, "actual": avail_liq})

        # Stratified sample buckets for manual review.
        if debt_like is not None and total_debt is not None:
            buckets["largest_lease_deltas"].append({**ref, "metric": debt_like - total_debt, "debt_like": debt_like, "total_debt": total_debt, "lease_liabilities": lease})
        if gross_lev_norm is not None:
            buckets["highest_normalized_gross_leverage"].append({**ref, "metric": gross_lev_norm, "gross_leverage_normalized": gross_lev_norm})
        if avail_liq is not None and avail_liq == 0.0:
            raw_before_floor = ((feats.get("liquidity.available_liquidity_normalized") or {}).get("component_breakdown") or {}).get("raw_value_before_floor")
            buckets["liquidity_zero_floor"].append({**ref, "metric": abs(raw_before_floor or 0.0), "available_liquidity": avail_liq, "raw_value_before_floor": raw_before_floor})
        if support("liquidity.available_liquidity_normalized") == "proxy_missing_component":
            buckets["proxy_liquidity"].append({**ref, "metric": avail_liq if avail_liq is not None else -1.0, "available_liquidity": avail_liq})
        if op_earn is not None and op_earn < 0:
            buckets["negative_operating_earnings"].append({**ref, "metric": abs(op_earn), "operating_earnings_normalized": op_earn})
        market_cap = value("market.market_cap_provider_direct")
        if market_cap is not None:
            buckets["largest_market_caps"].append({**ref, "metric": market_cap, "market_cap": market_cap})

    for bucket_name, records in buckets.items():
        records.sort(key=lambda x: (x["metric"] is None, x["metric"]), reverse=True)
        buckets[bucket_name] = records[:10]

    report = {
        "artifact_path": str(artifact_path),
        "row_count": len(rows),
        "coverage": {metric: dict(sorted(coverage[metric].items())) for metric in APPROVED_METRICS},
        "issues": issues,
        "stratified_samples": buckets,
    }

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.write_text(json.dumps(report, indent=2))

    lines = [
        "# Input Layer Validation Pack",
        "",
        f"- Artifact: `{artifact_path}`",
        f"- Rows: `{len(rows)}`",
        "",
        "## Issues",
    ]
    if issues:
        for category, metrics in issues.items():
            lines.append(f"- `{category}`")
            for metric, payload in metrics.items():
                lines.append(f"  - `{metric}`: {payload['count']}")
    else:
        lines.append("- none")
    lines.extend(["", "## Stratified Samples"])
    for bucket_name, records in buckets.items():
        lines.append(f"### {bucket_name}")
        if not records:
            lines.append("- none")
            continue
        for record in records[:5]:
            lines.append(
                f"- `{record.get('reference_instrument')}` {record.get('company_name')} ({record.get('company_id')}): {record}"
            )
        lines.append("")
    out_md.write_text("\n".join(lines))
    print(out_json)
    print(out_md)


if __name__ == "__main__":
    main()
