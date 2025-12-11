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


def _pct_gap(left: float | None, right: float | None) :
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
        "missing_reason": node['missing_reason'],
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


