#!/usr/bin/env python3
"""Audit the six core provider-direct metrics against raw-source reconstructions.

This is intentionally a triangulation audit, not a circular "rerun the same
artifact builder and call it validated" pass.

For each metric we compare the artifact value to:
1. the live provider reference row (Refinitiv sidecar)
2. a raw-source reconstruction from SEC companyfacts and/or PIT price history

The output is a compact JSON report with support counts, source-basis counts,
gap stats, and the largest discrepancy examples.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable

import pandas as pd

import backfill_input_layer_v1_metrics as core
import backfill_market_macro_input_layer_v1 as market_macro


METRICS = (
    "market.market_cap_provider_direct",
    "operating.revenue_ttm_provider_direct",
    "operating.ebitda_ltm_provider_direct",
    "earnings.net_income_ttm_provider_direct",
    "liquidity.cash_and_short_term_investments_provider_direct",
    "capital_structure.total_debt_provider_direct",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-path", required=True)
    parser.add_argument(
        "--taxonomy-reference-path",
        default=str(Path(__file__).resolve().parents[1] / "data" / "refinitiv" / "fundamentals_all.parquet"),
    )
    parser.add_argument(
        "--entity-identifier-path",
        default=str(Path(__file__).resolve().parents[1] / "data" / "inputs_layer" / "entity_identifier.parquet"),
    )
    parser.add_argument(
        "--companyfacts-root",
        default=str(Path(__file__).resolve().parents[1] / "data" / "sec" / "companyfacts"),
    )
    parser.add_argument(
        "--raw-timeseries-path",
        default=str(Path(__file__).resolve().parents[1] / "data" / "inputs_layer" / "raw_timeseries.parquet"),
    )
    parser.add_argument("--out-json", required=True)
    return parser.parse_args()


def iter_snapshot_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _pct_gap(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    denom = max(abs(float(right)), 1.0)
    return abs(float(left) - float(right)) / denom


def _artifact_view(node: Dict[str, Any] | None) -> Dict[str, Any]:
    node = node or {}
    return {
        "value": node.get("value"),
        "support_mode": node.get("support_mode"),
        "primary_source_basis": node.get("primary_source_basis"),
        "missing_reason": node.get("missing_reason"),
        "quality_flags": node.get("quality_flags"),
    }


def _reconstructed_nodes(
    *,
    row: Dict[str, Any],
    provider_row: Dict[str, Any] | None,
    companyfacts: dict | None,
    companyfacts_path: Path | None,
    price_history: pd.DataFrame | None,
    raw_timeseries_path: Path,
) -> Dict[str, Dict[str, Any]]:
    as_of_time = row["as_of_time"]
    as_of_date = as_of_time[:10]
    computed_at = row["as_of_time"]
    nodes: Dict[str, Dict[str, Any]] = {}
    for metric_name in METRICS:
        if metric_name == "market.market_cap_provider_direct":
            provider_node = core._build_legacy_provider_metric(
                metric_name=metric_name,
                provider_row=provider_row,
                as_of_time=as_of_time,
                computed_at=computed_at,
                provenance_source="provider_reference",
                unit="usd",
            )
            market_node = None
            if companyfacts is not None and companyfacts_path is not None and price_history is not None:
                price_metrics = market_macro._build_price_metrics(
                    permno=None,
                    price_history=price_history,
                    as_of_time=as_of_time,
                    computed_at=computed_at,
                    provenance_source=str(raw_timeseries_path),
                )
                market_node = market_macro._build_market_cap_metric_from_companyfacts(
                    companyfacts=companyfacts,
                    price_node=price_metrics["market.price_spot"],
                    as_of_time=as_of_time,
                    computed_at=computed_at,
                    companyfacts_path=companyfacts_path,
                )
            nodes[metric_name] = {
                "provider": _artifact_view(provider_node),
                "reconstructed": _artifact_view(market_node),
            }
            continue

        provider_node = core._build_legacy_provider_metric(
            metric_name=metric_name,
            provider_row=provider_row,
            as_of_time=as_of_time,
            computed_at=computed_at,
            provenance_source="provider_reference",
            unit=core.DIRECT_METRIC_SPECS[metric_name]["unit"],
        )
        if companyfacts is not None:
            value, support_mode, missing_reason, component_breakdown, quality_flags = core._build_sec_core_metric(
                metric_name,
                companyfacts,
                as_of_date,
            )
            reconstructed_node = core._build_metric_from_value(
                metric_name=metric_name,
                as_of_time=as_of_time,
                computed_at=computed_at,
                provenance_source=str(companyfacts_path),
                unit=core.DIRECT_METRIC_SPECS[metric_name]["unit"],
                value=value,
                support_mode=support_mode,
                missing_reason=missing_reason,
                component_breakdown=component_breakdown,
                quality_flags=quality_flags,
                primary_source_basis="sec_companyfacts",
                provenance_artifact_type="SecCompanyFacts",
                input_layer_bucket_reason="sec_companyfacts_asof",
            )
        else:
            reconstructed_node = None
        nodes[metric_name] = {
            "provider": _artifact_view(provider_node),
            "reconstructed": _artifact_view(reconstructed_node),
        }
    return nodes


def main() -> None:
    args = parse_args()
    artifact_path = Path(args.artifact_path)
    taxonomy_reference_path = Path(args.taxonomy_reference_path)
    entity_identifier_path = Path(args.entity_identifier_path)
    companyfacts_root = Path(args.companyfacts_root)
    raw_timeseries_path = Path(args.raw_timeseries_path)
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    snapshot_rows = list(iter_snapshot_rows(artifact_path))
    snapshot_company_ids = {str(row.get("company_id")) for row in snapshot_rows if row.get("company_id")}

    provider = core._provider_reference_map(taxonomy_reference_path, entity_identifier_path)
    if snapshot_company_ids:
        provider = provider[provider["entity_id"].astype(str).isin(snapshot_company_ids)].copy()
    provider_by_entity = provider.set_index("entity_id").to_dict(orient="index")
    permnos = market_macro._permno_map(entity_identifier_path)
    if snapshot_company_ids:
        permnos = permnos[permnos["entity_id"].astype(str).isin(snapshot_company_ids)].copy()
    permno_by_entity = permnos.set_index("entity_id")["permno"].to_dict()
    price_history = market_macro._load_price_history(raw_timeseries_path, permnos["permno"].tolist())
    price_by_permno = {permno: frame.reset_index(drop=True) for permno, frame in price_history.groupby("permno")}

    summaries: Dict[str, Any] = {}
    metric_counters = {
        metric_name: {
            "artifact_support": Counter(),
            "artifact_basis": Counter(),
            "reconstructed_support": Counter(),
            "provider_support": Counter(),
            "artifact_vs_provider_gap": [],
            "artifact_vs_reconstructed_gap": [],
            "largest_provider_gaps": [],
            "largest_reconstructed_gaps": [],
        }
        for metric_name in METRICS
    }

    row_count = 0
    for row in snapshot_rows:
        row_count += 1
        entity_id = row.get("company_id")
        provider_row = provider_by_entity.get(entity_id)
        companyfacts_path = companyfacts_root / f"CIK{entity_id}.json"
        companyfacts = core._load_companyfacts(companyfacts_path)
        price_df = price_by_permno.get(permno_by_entity.get(entity_id))
        reconstructed = _reconstructed_nodes(
            row=row,
            provider_row=provider_row,
            companyfacts=companyfacts,
            companyfacts_path=companyfacts_path if companyfacts is not None else None,
            price_history=price_df,
            raw_timeseries_path=raw_timeseries_path,
        )
        features = row.get("features") or {}

        for metric_name in METRICS:
            artifact_node = _artifact_view(features.get(metric_name))
            provider_node = reconstructed[metric_name]["provider"]
            reconstructed_node = reconstructed[metric_name]["reconstructed"]
            counters = metric_counters[metric_name]

            counters["artifact_support"][artifact_node.get("support_mode") or "missing_metric"] += 1
            counters["artifact_basis"][artifact_node.get("primary_source_basis") or "missing_basis"] += 1
            counters["provider_support"][provider_node.get("support_mode") or "missing_metric"] += 1
            counters["reconstructed_support"][reconstructed_node.get("support_mode") or "missing_metric"] += 1

            gap_vs_provider = _pct_gap(artifact_node.get("value"), provider_node.get("value"))
            gap_vs_reconstructed = _pct_gap(artifact_node.get("value"), reconstructed_node.get("value"))
            if gap_vs_provider is not None:
                counters["artifact_vs_provider_gap"].append(gap_vs_provider)
                counters["largest_provider_gaps"].append(
                    {
                        "company_id": entity_id,
                        "artifact_value": artifact_node.get("value"),
                        "artifact_support_mode": artifact_node.get("support_mode"),
                        "artifact_primary_source_basis": artifact_node.get("primary_source_basis"),
                        "provider_value": provider_node.get("value"),
                        "gap_pct": gap_vs_provider,
                    }
                )
            if gap_vs_reconstructed is not None:
                counters["artifact_vs_reconstructed_gap"].append(gap_vs_reconstructed)
                counters["largest_reconstructed_gaps"].append(
                    {
                        "company_id": entity_id,
                        "artifact_value": artifact_node.get("value"),
                        "artifact_support_mode": artifact_node.get("support_mode"),
                        "artifact_primary_source_basis": artifact_node.get("primary_source_basis"),
                        "reconstructed_value": reconstructed_node.get("value"),
                        "reconstructed_support_mode": reconstructed_node.get("support_mode"),
                        "gap_pct": gap_vs_reconstructed,
                    }
                )

    for metric_name, counters in metric_counters.items():
        summaries[metric_name] = {
            "artifact_support_counts": dict(counters["artifact_support"]),
            "artifact_primary_source_basis_counts": dict(counters["artifact_basis"]),
            "provider_support_counts": dict(counters["provider_support"]),
            "reconstructed_support_counts": dict(counters["reconstructed_support"]),
            "artifact_vs_provider_gap_stats": {
                "count": len(counters["artifact_vs_provider_gap"]),
                "median_abs_pct_gap": median(counters["artifact_vs_provider_gap"]) if counters["artifact_vs_provider_gap"] else None,
                "max_abs_pct_gap": max(counters["artifact_vs_provider_gap"]) if counters["artifact_vs_provider_gap"] else None,
            },
            "artifact_vs_reconstructed_gap_stats": {
                "count": len(counters["artifact_vs_reconstructed_gap"]),
                "median_abs_pct_gap": median(counters["artifact_vs_reconstructed_gap"]) if counters["artifact_vs_reconstructed_gap"] else None,
                "max_abs_pct_gap": max(counters["artifact_vs_reconstructed_gap"]) if counters["artifact_vs_reconstructed_gap"] else None,
            },
            "largest_provider_gaps": sorted(
                counters["largest_provider_gaps"],
                key=lambda item: item["gap_pct"],
                reverse=True,
            )[:10],
            "largest_reconstructed_gaps": sorted(
                counters["largest_reconstructed_gaps"],
                key=lambda item: item["gap_pct"],
                reverse=True,
            )[:10],
        }

    report = {
        "artifact_path": str(artifact_path),
        "row_count": row_count,
        "metrics": summaries,
    }
    out_path.write_text(json.dumps(report, indent=2))
    print(out_path)


if __name__ == "__main__":
    main()
