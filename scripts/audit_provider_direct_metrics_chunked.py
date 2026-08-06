#!/usr/bin/env python3
"""Run the provider-direct external audit in chunked parallel batches.

This keeps the exact audit logic from `audit_provider_direct_metrics.py`, but
avoids one large monolithic pass over the full artifact by splitting the
artifact rows into smaller chunks, auditing each chunk independently, and then
merging the partial results back into one exact report.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import audit_provider_direct_metrics as audit_core
import backfill_input_layer_v1_metrics as core
import backfill_market_macro_input_layer_v1 as market_macro


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-path", required=True)
    parser.add_argument(
        "--taxonomy-reference-path",
        default=str(ROOT / "data" / "refinitiv" / "fundamentals_all.parquet"),
    )
    parser.add_argument(
        "--entity-identifier-path",
        default=str(ROOT / "data" / "inputs_layer" / "entity_identifier.parquet"),
    )
    parser.add_argument(
        "--companyfacts-root",
        default=str(ROOT / "data" / "sec" / "companyfacts"),
    )
    parser.add_argument(
        "--raw-timeseries-path",
        default=str(ROOT / "data" / "inputs_layer" / "raw_timeseries.parquet"),
    )
    parser.add_argument("--out-json", required=True)
    parser.add_argument(
        "--skip-market-cap",
        action="store_true",
        help="Skip market cap in this audit. Useful when the market stack has already been audited separately via CRSP.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=96,
        help="Number of rows to audit per worker chunk.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum worker processes to use.",
    )
    return parser.parse_args()


def _selected_metrics(skip_market_cap: bool) -> tuple[str, ...]:
    if not skip_market_cap:
        return tuple(audit_core.METRICS)
    return tuple(metric for metric in audit_core.METRICS if metric != "market.market_cap_provider_direct")


def _iter_snapshot_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _chunk_rows(rows: List[Dict[str, Any]], chunk_size: int) -> Iterable[List[Dict[str, Any]]]:
    for start in range(0, len(rows), chunk_size):
        yield rows[start : start + chunk_size]


def _empty_metric_counters(selected_metrics: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    return {
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
        for metric_name in selected_metrics
    }


def _top_gap_examples(examples: List[Dict[str, Any]], key: str = "gap_pct", limit: int = 10) -> List[Dict[str, Any]]:
    return sorted(examples, key=lambda item: item[key], reverse=True)[:limit]


def _reconstruct_metric_pair(
    *,
    metric_name: str,
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
        return {
            "provider": audit_core._artifact_view(provider_node),
            "reconstructed": audit_core._artifact_view(market_node),
        }

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
    return {
        "provider": audit_core._artifact_view(provider_node),
        "reconstructed": audit_core._artifact_view(reconstructed_node),
    }


def _audit_chunk(
    *,
    rows: List[Dict[str, Any]],
    selected_metrics: List[str],
    provider_by_entity: Dict[str, Dict[str, Any]],
    companyfacts_root: str,
    price_by_permno: Dict[str, Any],
    permno_by_entity: Dict[str, str],
) -> Dict[str, Any]:
    metric_counters = _empty_metric_counters(selected_metrics)
    companyfacts_root_path = Path(companyfacts_root)

    for row in rows:
        entity_id = row.get("company_id")
        provider_row = provider_by_entity.get(entity_id)
        companyfacts_path = companyfacts_root_path / f"CIK{entity_id}.json"
        companyfacts = core._load_companyfacts(companyfacts_path)
        price_df = price_by_permno.get(permno_by_entity.get(entity_id))
        features = row.get("features") or {}

        for metric_name in selected_metrics:
            artifact_node = audit_core._artifact_view(features.get(metric_name))
            reconstructed = _reconstruct_metric_pair(
                metric_name=metric_name,
                row=row,
                provider_row=provider_row,
                companyfacts=companyfacts,
                companyfacts_path=companyfacts_path if companyfacts is not None else None,
                price_history=price_df,
                raw_timeseries_path=None,
            )
            provider_node = reconstructed["provider"]
            reconstructed_node = reconstructed["reconstructed"]
            counters = metric_counters[metric_name]

            counters["artifact_support"][artifact_node.get("support_mode") or "missing_metric"] += 1
            counters["artifact_basis"][artifact_node.get("primary_source_basis") or "missing_basis"] += 1
            counters["provider_support"][provider_node.get("support_mode") or "missing_metric"] += 1
            counters["reconstructed_support"][reconstructed_node.get("support_mode") or "missing_metric"] += 1

            gap_vs_provider = audit_core._pct_gap(artifact_node.get("value"), provider_node.get("value"))
            gap_vs_reconstructed = audit_core._pct_gap(artifact_node.get("value"), reconstructed_node.get("value"))
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

    return {
        "row_count": len(rows),
        "metrics": metric_counters,
    }


def _merge_partials(partials: List[Dict[str, Any]]) -> Dict[str, Any]:
    selected_metrics = list(partials[0]["metrics"].keys()) if partials else []
    merged = _empty_metric_counters(selected_metrics)
    row_count = 0
    for partial in partials:
        row_count += partial["row_count"]
        for metric_name, counters in partial["metrics"].items():
            target = merged[metric_name]
            target["artifact_support"].update(counters["artifact_support"])
            target["artifact_basis"].update(counters["artifact_basis"])
            target["reconstructed_support"].update(counters["reconstructed_support"])
            target["provider_support"].update(counters["provider_support"])
            target["artifact_vs_provider_gap"].extend(counters["artifact_vs_provider_gap"])
            target["artifact_vs_reconstructed_gap"].extend(counters["artifact_vs_reconstructed_gap"])
            target["largest_provider_gaps"].extend(counters["largest_provider_gaps"])
            target["largest_reconstructed_gaps"].extend(counters["largest_reconstructed_gaps"])

    summaries: Dict[str, Any] = {}
    for metric_name, counters in merged.items():
        provider_gaps = counters["artifact_vs_provider_gap"]
        reconstructed_gaps = counters["artifact_vs_reconstructed_gap"]
        summaries[metric_name] = {
            "artifact_support_counts": dict(counters["artifact_support"]),
            "artifact_primary_source_basis_counts": dict(counters["artifact_basis"]),
            "provider_support_counts": dict(counters["provider_support"]),
            "reconstructed_support_counts": dict(counters["reconstructed_support"]),
            "artifact_vs_provider_gap_stats": {
                "count": len(provider_gaps),
                "median_abs_pct_gap": median(provider_gaps) if provider_gaps else None,
                "max_abs_pct_gap": max(provider_gaps) if provider_gaps else None,
            },
            "artifact_vs_reconstructed_gap_stats": {
                "count": len(reconstructed_gaps),
                "median_abs_pct_gap": median(reconstructed_gaps) if reconstructed_gaps else None,
                "max_abs_pct_gap": max(reconstructed_gaps) if reconstructed_gaps else None,
            },
            "largest_provider_gaps": _top_gap_examples(counters["largest_provider_gaps"]),
            "largest_reconstructed_gaps": _top_gap_examples(counters["largest_reconstructed_gaps"]),
        }

    return {
        "row_count": row_count,
        "metrics": summaries,
    }


def main() -> None:
    args = parse_args()
    artifact_path = Path(args.artifact_path)
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = list(_iter_snapshot_rows(artifact_path))
    if not rows:
        raise SystemExit("Artifact is empty")

    selected_metrics = list(_selected_metrics(args.skip_market_cap))
    chunks = list(_chunk_rows(rows, max(1, args.chunk_size)))
    partials: List[Dict[str, Any]] = []
    max_workers = max(1, min(args.max_workers, len(chunks)))

    provider = core._provider_reference_map(
        Path(args.taxonomy_reference_path),
        Path(args.entity_identifier_path),
    )
    snapshot_company_ids = {str(row.get("company_id")) for row in rows if row.get("company_id")}
    if snapshot_company_ids:
        provider = provider[provider["entity_id"].astype(str).isin(snapshot_company_ids)].copy()
    provider_by_entity = provider.set_index("entity_id").to_dict(orient="index")

    price_by_permno: Dict[str, Any] = {}
    permno_by_entity: Dict[str, str] = {}
    if "market.market_cap_provider_direct" in selected_metrics:
        permnos = market_macro._permno_map(Path(args.entity_identifier_path))
        if snapshot_company_ids:
            permnos = permnos[permnos["entity_id"].astype(str).isin(snapshot_company_ids)].copy()
        permno_by_entity = permnos.set_index("entity_id")["permno"].to_dict()
        if not permnos.empty:
            price_history = market_macro._load_price_history(
                Path(args.raw_timeseries_path),
                permnos["permno"].tolist(),
            )
            price_by_permno = {
                permno: frame.reset_index(drop=True)
                for permno, frame in price_history.groupby("permno")
            }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _audit_chunk,
                rows=chunk,
                selected_metrics=selected_metrics,
                provider_by_entity=provider_by_entity,
                companyfacts_root=args.companyfacts_root,
                price_by_permno=price_by_permno,
                permno_by_entity=permno_by_entity,
            )
            for chunk in chunks
        ]
        completed = 0
        for future in as_completed(futures):
            partials.append(future.result())
            completed += 1
            print(f"completed_chunks={completed}/{len(chunks)}", flush=True)

    report = _merge_partials(partials)
    report["artifact_path"] = str(artifact_path)
    report["chunk_size"] = args.chunk_size
    report["max_workers"] = max_workers
    report["selected_metrics"] = selected_metrics
    out_path.write_text(json.dumps(report, indent=2))
    print(out_path)


if __name__ == "__main__":
    main()
