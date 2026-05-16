#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.replay_snapshot_enrichment import enrich_snapshot_with_revenue_growth_inputs

DEFAULT_ENTITY_IDENTIFIER_PATH = ROOT / "out/manual_replay_bundle_20260405_localized/inputs/entity_identifier.parquet"
DEFAULT_CRSP_DAILY_ROOT = ROOT / "data/wrds/crsp"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enrich replay snapshots with as-of-safe matching inputs (revenue growth plus derived buyback/dividend support metrics)."
    )
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--snapshot-cache-root", required=True)
    parser.add_argument("--companyfacts-root", required=True)
    parser.add_argument("--entity-identifier-path", default=str(DEFAULT_ENTITY_IDENTIFIER_PATH))
    parser.add_argument("--crsp-daily-root", default=str(DEFAULT_CRSP_DAILY_ROOT))
    parser.add_argument("--crsp-market-cache-path", default="")
    parser.add_argument("--summary-path", default="")
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _modern_snapshot_path(snapshot_cache_root: Path, *, company_id: str, as_of_time: str) -> Path:
    as_of_date = str(as_of_time).split("T", 1)[0]
    return snapshot_cache_root / f"as_of_date={as_of_date}" / f"company_id={company_id}.json"


def _legacy_snapshot_path(snapshot_cache_root: Path, *, company_id: str, as_of_time: str) -> Path:
    ts = str(as_of_time).replace("-", "").replace(":", "")
    if ts.endswith("+0000"):
        ts = ts[:-5]
    if ts.endswith("+00:00"):
        ts = ts[:-6]
    ts = ts.replace("T", "T").replace("Z", "")
    date_part, _, time_part = str(as_of_time).partition("T")
    time_part = (time_part or "00:00:00+00:00").split("+", 1)[0].replace(":", "")
    return snapshot_cache_root / f"company_id={company_id}" / f"snapshot_as_of={date_part.replace('-', '')}T{time_part}Z.json"


def _snapshot_path(snapshot_cache_root: Path, *, company_id: str, as_of_time: str) -> Path:
    modern = _modern_snapshot_path(snapshot_cache_root, company_id=company_id, as_of_time=as_of_time)
    if modern.exists():
        return modern
    legacy = _legacy_snapshot_path(snapshot_cache_root, company_id=company_id, as_of_time=as_of_time)
    if legacy.exists():
        return legacy
    return modern


def main() -> None:
    args = _parse_args()
    manifest_path = Path(args.manifest_path)
    snapshot_cache_root = Path(args.snapshot_cache_root)
    companyfacts_root = Path(args.companyfacts_root)
    entity_identifier_path = Path(args.entity_identifier_path) if args.entity_identifier_path else None
    crsp_daily_root = Path(args.crsp_daily_root) if args.crsp_daily_root else None
    crsp_market_cache_path = Path(args.crsp_market_cache_path) if args.crsp_market_cache_path else None
    summary_path = Path(args.summary_path) if args.summary_path else None

    manifest = _load_json(manifest_path)
    cases = list(manifest.get("cases") or manifest.get("selection_rankings") or [])
    summaries: List[Dict[str, Any]] = []
    changed_count = 0
    for case in cases:
        company_id = str(case.get("company_id") or "").strip()
        as_of_time = str(case.get("as_of_time") or "").strip()
        if not company_id or not as_of_time:
            continue
        snapshot_path = _snapshot_path(snapshot_cache_root, company_id=company_id, as_of_time=as_of_time)
        if not snapshot_path.exists():
            summaries.append(
                {
                    "company_id": company_id,
                    "as_of_time": as_of_time,
                    "snapshot_path": str(snapshot_path),
                    "exists": False,
                    "changed": False,
                }
            )
            continue
        payload = _load_json(snapshot_path)
        enriched, changed, summary = enrich_snapshot_with_revenue_growth_inputs(
            payload,
            companyfacts_root=companyfacts_root,
            entity_identifier_path=entity_identifier_path,
            crsp_market_cache_path=crsp_market_cache_path,
            crsp_daily_root=crsp_daily_root,
            company_id=company_id,
            as_of_time=as_of_time,
        )
        if changed:
            snapshot_path.write_text(json.dumps(enriched, default=str))
            changed_count += 1
        summaries.append(
            {
                "company_id": company_id,
                "as_of_time": as_of_time,
                "snapshot_path": str(snapshot_path),
                "exists": True,
                "changed": changed,
                "metrics": summary.get("metrics") or {},
            }
        )

    aggregate = {
        "manifest_path": str(manifest_path),
        "snapshot_cache_root": str(snapshot_cache_root),
        "companyfacts_root": str(companyfacts_root),
        "entity_identifier_path": str(entity_identifier_path) if entity_identifier_path else None,
        "crsp_daily_root": str(crsp_daily_root) if crsp_daily_root else None,
        "crsp_market_cache_path": str(crsp_market_cache_path) if crsp_market_cache_path else None,
        "requested_cases": len(cases),
        "touched_cases": len(summaries),
        "changed_cases": changed_count,
        "case_summaries": summaries,
    }
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True))
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
