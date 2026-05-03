#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path
import sys
from typing import Any, Dict, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_SNAPSHOT_CATALOG_PATH = (
    REPO_ROOT
    / "out/materialized_feedback_20260405/company_state_snapshots_audit_catalog.asof_safe_enriched_v1.jsonl.gz"
)
DEFAULT_OUT_PATH = (
    REPO_ROOT
    / "out/materialized_feedback_20260405/company_state_snapshots_input_complete_catalog.asof_safe_v1.jsonl.gz"
)
DEFAULT_COMPANYFACTS_ROOT = REPO_ROOT / "data/sec/companyfacts"
DEFAULT_ENTITY_IDENTIFIER_PATH = REPO_ROOT / "data/inputs_layer/entity_identifier.parquet"
DEFAULT_CRSP_DAILY_ROOT = REPO_ROOT / "data/wrds/crsp"

TARGET_METRICS = [
    "operating.revenue_ttm_provider_direct",
    "operating.revenue_ttm_lag_1y",
    "operating.ebitda_ltm_provider_direct",
    "cash_flow.free_cash_flow_ttm",
    "capital_structure.total_debt_provider_direct",
    "capital_structure.debt_due_next_24m",
    "liquidity.cash_and_short_term_investments_provider_direct",
    "liquidity.marketable_securities_sec_exact",
    "market.market_cap_provider_direct",
    "market.enterprise_value",
    "market.ev_ebitda",
    "market.fcf_yield",
    "market.volatility_30d",
    "market.volatility_90d",
    "market.drawdown_90d",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an as-of-safe snapshot catalog with completeness enrichment applied universe-wide."
    )
    parser.add_argument("--snapshot-catalog-path", default=str(DEFAULT_SNAPSHOT_CATALOG_PATH))
    parser.add_argument("--companyfacts-root", default=str(DEFAULT_COMPANYFACTS_ROOT))
    parser.add_argument("--entity-identifier-path", default=str(DEFAULT_ENTITY_IDENTIFIER_PATH))
    parser.add_argument("--crsp-daily-root", default=str(DEFAULT_CRSP_DAILY_ROOT))
    parser.add_argument("--crsp-market-cache-path", default="")
    parser.add_argument("--out-path", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--summary-path", default="")
    return parser.parse_args()


def _iter_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with gzip.open(path, "rt") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _support_mode(row: Dict[str, Any], metric_name: str) -> str:
    feature = (row.get("features") or {}).get(metric_name) or {}
    value = feature.get("value")
    mode = str(feature.get("support_mode") or "unsupported")
    if value is None:
        return "unsupported"
    return mode


def main() -> None:
    args = _parse_args()
    snapshot_catalog_path = Path(args.snapshot_catalog_path)
    companyfacts_root = Path(args.companyfacts_root)
    entity_identifier_path = Path(args.entity_identifier_path)
    crsp_daily_root = Path(args.crsp_daily_root)
    crsp_market_cache_path = Path(args.crsp_market_cache_path) if args.crsp_market_cache_path else None
    out_path = Path(args.out_path)
    summary_path = Path(args.summary_path) if args.summary_path else None

    from src.replay_snapshot_enrichment import enrich_snapshot_with_revenue_growth_inputs

    rows = list(_iter_rows(snapshot_catalog_path))
    total_rows = len(rows)
    pre_counts: dict[str, Counter[str]] = {metric: Counter() for metric in TARGET_METRICS}
    post_counts: dict[str, Counter[str]] = {metric: Counter() for metric in TARGET_METRICS}
    changed_rows = 0
    source_changed_counts: Counter[str] = Counter()
    metric_change_counts: Counter[str] = Counter()
    temp_out_path = out_path.with_suffix(out_path.suffix + ".tmp")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if temp_out_path.exists():
        temp_out_path.unlink()

    with gzip.open(temp_out_path, "wt") as handle:
        for index, row in enumerate(rows, start=1):
            for metric in TARGET_METRICS:
                pre_counts[metric][_support_mode(row, metric)] += 1
            enriched, changed, summary = enrich_snapshot_with_revenue_growth_inputs(
                row,
                companyfacts_root=companyfacts_root,
                entity_identifier_path=entity_identifier_path,
                crsp_market_cache_path=crsp_market_cache_path,
                crsp_daily_root=crsp_daily_root,
                company_id=str(row.get("company_id") or ""),
                as_of_time=str(row.get("as_of_time") or ""),
            )
            if changed:
                changed_rows += 1
                source_changed_counts[str(row.get("snapshot_catalog_source") or "unknown")] += 1
            for metric in (summary.get("metrics") or {}).keys():
                metric_change_counts[str(metric)] += 1
            for metric in TARGET_METRICS:
                post_counts[metric][_support_mode(enriched, metric)] += 1
            handle.write(json.dumps(enriched, sort_keys=True, default=str))
            handle.write("\n")
            if index % 50 == 0 or index == total_rows:
                print(
                    json.dumps(
                        {
                            "progress": index,
                            "total_rows": total_rows,
                            "changed_rows": changed_rows,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    temp_out_path.replace(out_path)

    metric_support_summary = {}
    for metric in TARGET_METRICS:
        metric_support_summary[metric] = {
            "before": dict(pre_counts[metric]),
            "after": dict(post_counts[metric]),
            "changed_rows": int(metric_change_counts[metric]),
        }

    summary_payload = {
        "snapshot_catalog_path": str(snapshot_catalog_path),
        "out_path": str(out_path),
        "row_count": total_rows,
        "changed_rows": changed_rows,
        "companyfacts_root": str(companyfacts_root),
        "entity_identifier_path": str(entity_identifier_path),
        "crsp_daily_root": str(crsp_daily_root),
        "crsp_market_cache_path": str(crsp_market_cache_path) if crsp_market_cache_path else None,
        "changed_rows_by_source": dict(source_changed_counts),
        "metric_support_summary": metric_support_summary,
    }
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True))
    print(json.dumps(summary_payload, sort_keys=True))


if __name__ == "__main__":
    main()
