#!/usr/bin/env python3
"""Backfill production-safe direct provider metrics into snapshot JSONL rows.

This keeps the contract narrow on purpose: only direct provider fields with a
single stable meaning are emitted. No in-house reconstructed adjusted metrics
are added here.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd


METRIC_SPECS = {
    "operating.ebitda_ltm_provider_direct": {
        "source_column": "EBITDA",
        "unit": "usd",
        "quality_flags": None,
    },
    "liquidity.cash_and_short_term_investments_provider_direct": {
        "source_column": "Cash and Short Term Investments",
        "unit": "usd",
        "quality_flags": None,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-path", required=True, help="Input snapshot JSONL path")
    parser.add_argument("--taxonomy-reference-path", required=True, help="Provider reference parquet")
    parser.add_argument("--entity-identifier-path", required=True, help="Entity identifier parquet with ticker rows")
    parser.add_argument("--out", required=True, help="Output JSONL path")
    return parser.parse_args()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ticker_to_entity_map(entity_identifier_path: Path) -> pd.DataFrame:
    ids = pd.read_parquet(entity_identifier_path)
    ids = ids[ids["identifier_type"].astype(str).str.lower() == "ticker"].copy()
    ids["ticker"] = ids["identifier_value"].astype(str).str.upper().str.strip()
    return ids[["entity_id", "ticker"]].drop_duplicates()


def _provider_reference_map(taxonomy_reference_path: Path, entity_identifier_path: Path) -> pd.DataFrame:
    ref = pd.read_parquet(taxonomy_reference_path).copy()
    ref["ticker"] = ref["Instrument"].astype(str).str.replace(r"\..*$", "", regex=True).str.upper().str.strip()
    tickers = _ticker_to_entity_map(entity_identifier_path)
    merged = ref.merge(tickers, on="ticker", how="inner")
    merged = merged.sort_values(["entity_id", "Instrument"]).drop_duplicates("entity_id", keep="first")
    return merged


def _feature_template(
    *,
    metric_name: str,
    as_of_time: str,
    computed_at: str,
    provenance_source: str,
    support_mode: str,
    value: Any,
    unit: str,
    missing_reason: str | None,
    component_breakdown: Dict[str, Any] | None,
    quality_flags: list[str] | None,
) -> Dict[str, Any]:
    return {
        "name": metric_name,
        "value": value,
        "unit": unit,
        "computed_at": computed_at,
        "as_of_time": as_of_time,
        "window": None,
        "confidence": 1.0 if value is not None else None,
        "provenance": [
            {
                "artifact_type": "ReferenceFact",
                "artifact_id": f"provider_reference:{Path(provenance_source).name}",
                "source": provenance_source,
                "published_at": as_of_time,
                "ingested_at": computed_at,
                "hash": None,
            }
        ],
        "missing_reason": missing_reason,
        "fallback_used": None,
        "metric_policy_id": None,
        "market_owner": None,
        "primary_source_basis": "provider_direct",
        "methodology_registry_id": None,
        "methodology_metric_id": None,
        "canonical_owner_id": None,
        "canonical_owner_name": None,
        "canonical_classification": None,
        "market_layer_status": None,
        "current_alignment_status": None,
        "primary_source_document_id": None,
        "recommended_metric_name": None,
        "input_source_registry_id": None,
        "input_source_owner_id": None,
        "input_source_owner_name": None,
        "input_source_classification": "provider_direct",
        "input_source_formula_basis": None,
        "input_source_alignment_status": "aligned",
        "input_source_document_ids": None,
        "definition_requirement": None,
        "definition_requirement_reason": None,
        "methodology_execution_decision": None,
        "methodology_execution_reason": None,
        "input_layer_bucket": "reference",
        "input_layer_bucket_reason": "provider_reference_sidecar",
        "strict_market_defined": None,
        "archetype": None,
        "sector": None,
        "subsector": None,
        "override_level_applied": None,
        "support_mode": support_mode,
        "applicability_status": None,
        "component_breakdown": component_breakdown,
        "quality_flags": quality_flags,
        "view_type": None,
    }


def _build_metric_node(
    metric_name: str,
    spec: Dict[str, Any],
    provider_row: pd.Series | None,
    as_of_time: str,
    computed_at: str,
    provenance_source: str,
) -> Dict[str, Any]:
    if provider_row is None:
        return _feature_template(
            metric_name=metric_name,
            as_of_time=as_of_time,
            computed_at=computed_at,
            provenance_source=provenance_source,
            support_mode="unsupported",
            value=None,
            unit=spec["unit"],
            missing_reason="provider_row_unavailable",
            component_breakdown={"source_column": spec["source_column"]},
            quality_flags=["provider_row_unavailable"],
        )

    raw_value = provider_row.get(spec["source_column"])
    if pd.isna(raw_value):
        return _feature_template(
            metric_name=metric_name,
            as_of_time=as_of_time,
            computed_at=computed_at,
            provenance_source=provenance_source,
            support_mode="unsupported",
            value=None,
            unit=spec["unit"],
            missing_reason="provider_field_unavailable",
            component_breakdown={
                "source_column": spec["source_column"],
                "reference_instrument": provider_row.get("Instrument"),
            },
            quality_flags=["provider_field_unavailable"],
        )

    value = float(raw_value)
    return _feature_template(
        metric_name=metric_name,
        as_of_time=as_of_time,
        computed_at=computed_at,
        provenance_source=provenance_source,
        support_mode="exact",
        value=value,
        unit=spec["unit"],
        missing_reason=None,
        component_breakdown={
            "provider_field": spec["source_column"],
            "reference_instrument": provider_row.get("Instrument"),
            "provider_company_name": provider_row.get("Company Common Name"),
            "formula": "provider_direct_field",
        },
        quality_flags=spec.get("quality_flags"),
    )


def iter_snapshot_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> None:
    args = parse_args()
    snapshot_path = Path(args.snapshot_path)
    taxonomy_reference_path = Path(args.taxonomy_reference_path)
    entity_identifier_path = Path(args.entity_identifier_path)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    provider = _provider_reference_map(taxonomy_reference_path, entity_identifier_path)
    provider_by_entity = provider.set_index("entity_id").to_dict(orient="index")
    computed_at = _now_iso()
    counters: Counter[str] = Counter()

    with out_path.open("w") as out_handle:
        for row in iter_snapshot_rows(snapshot_path):
            entity_id = row.get("company_id")
            provider_row = provider_by_entity.get(entity_id)
            features = row.setdefault("features", {})
            for metric_name, spec in METRIC_SPECS.items():
                node = _build_metric_node(
                    metric_name=metric_name,
                    spec=spec,
                    provider_row=provider_row,
                    as_of_time=row.get("as_of_time"),
                    computed_at=computed_at,
                    provenance_source=str(taxonomy_reference_path),
                )
                features[metric_name] = node
                counters[f"{metric_name}:{node['support_mode']}"] += 1
            out_handle.write(json.dumps(row) + "\n")

    print(f"Wrote augmented snapshots -> {out_path}")
    print(f"provider_rows={len(provider_by_entity)}")
    for metric_name in METRIC_SPECS:
        exact = counters[f"{metric_name}:exact"]
        unsupported = counters[f"{metric_name}:unsupported"]
        print(f"{metric_name}: exact={exact} unsupported={unsupported}")


if __name__ == "__main__":
    main()
