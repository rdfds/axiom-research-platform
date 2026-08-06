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

from src.company_state_builder import CompanyStateBuilder
from src.historical_recommendation_eval import _build_historical_alias_overrides, _cached_snapshot_loader


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a keyed snapshot cache for the fixed historical cases in a manifest."
    )
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--companyfacts-root", default="")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _emit(payload: Dict[str, Any]) -> None:
    print(json.dumps(payload), flush=True)


def main() -> None:
    args = _parse_args()
    manifest_path = Path(args.manifest_path)
    cache_dir = Path(args.cache_dir)
    payload = json.loads(manifest_path.read_text())
    cases: List[Dict[str, Any]] = list(payload.get("cases") or [])
    if int(args.limit) > 0:
        cases = cases[: int(args.limit)]

    root = ROOT
    companyfacts_root = Path(args.companyfacts_root) if str(args.companyfacts_root or "").strip() else (root / "data" / "sec" / "companyfacts")
    resolved_companyfacts_root = companyfacts_root if companyfacts_root.exists() else None

    builder = CompanyStateBuilder(
        raw_timeseries_path=root / "data" / "inputs_layer" / "raw_timeseries.parquet",
        macro_timeseries_path=None,
        event_store_path=root / "data" / "inputs_layer" / "event_store.parquet",
        facts_path=root / "data" / "inputs_layer" / "extracted_fact_registry_validity",
        ownership_summary_path=root / "data" / "inputs_layer" / "ownership_13f_summary.parquet",
        issuer_ratings_path=root / "data" / "inputs_layer" / "issuer_rating_history.parquet",
        entity_graph_path=root / "data" / "inputs_layer" / "entity_graph.parquet",
        entity_identifier_path=root / "data" / "inputs_layer" / "entity_identifier.parquet",
        entity_table_path=root / "data" / "inputs_layer" / "entity.parquet",
        skip_timeseries=False,
        skip_macro=False,
        skip_events=False,
        skip_peer_context=False,
        debug=bool(args.debug),
        cache_facts=True,
        cache_events=True,
        cache_timeseries=True,
        cache_ownership=True,
        cache_ratings=True,
        historical_backfill_mode=True,
        companyfacts_root=resolved_companyfacts_root,
        enable_market_relevant_smart_normalized_inputs=True,
    )

    loader = _cached_snapshot_loader(
        builder,
        cache_dir=cache_dir,
        progress_logger=_emit,
        alias_overrides=_build_historical_alias_overrides(cases),
    )

    for idx, case in enumerate(cases, start=1):
        company_id = str(case.get("company_id") or "")
        as_of_time = str(case.get("as_of_time") or "")
        anchor_action_id = str(case.get("anchor_action_id") or "")
        _emit(
            {
                "event": "case_start",
                "index": idx,
                "total": len(cases),
                "company_id": company_id,
                "as_of_time": as_of_time,
                "anchor_action_id": anchor_action_id,
            }
        )
        loader(company_id, as_of_time)
        _emit(
            {
                "event": "case_complete",
                "index": idx,
                "total": len(cases),
                "company_id": company_id,
                "as_of_time": as_of_time,
                "anchor_action_id": anchor_action_id,
            }
        )

    _emit({"event": "complete", "case_count": len(cases), "cache_dir": str(cache_dir)})


if __name__ == "__main__":
    main()
