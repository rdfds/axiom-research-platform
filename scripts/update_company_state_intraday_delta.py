import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.company_state_builder import CompanyStateBuilder
from src.company_state_delta import update_snapshot
from src.company_state_store import SnapshotStore


def _parse_company_ids(arg: Optional[str]) -> List[str]:
    if not arg:
        return []
    if "," in arg:
        return [x.strip() for x in arg.split(",") if x.strip()]
    return [arg.strip()]


def _load_company_ids_file(path: Optional[str]) -> List[str]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"--company-ids-file not found: {path}")
    ids: List[str] = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [x.strip() for x in line.replace(",", " ").split() if x.strip()]
            ids.extend(parts)
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply intraday delta updates (market/regime) to keyed CompanyState snapshots."
    )
    parser.add_argument("--root", default="data/company_state_snapshots")
    parser.add_argument("--asof", required=True, help="Intraday as-of timestamp, e.g. 2026-02-28T15:30:00Z")
    parser.add_argument("--asof-date", default=None, help="Snapshot cache date partition to update (default: date(asof))")
    parser.add_argument("--mode", choices=["market", "regime", "both"], default="both")
    parser.add_argument("--company-id", default=None, help="Single company ID or comma-separated list")
    parser.add_argument("--company-ids-file", default=None, help="Optional file of company IDs to update")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeseries-path", default="data/inputs_layer/raw_timeseries.parquet")
    parser.add_argument("--macro-path", default=None, help="Optional macro-only timeseries file")
    parser.add_argument("--facts-path", default="data/inputs_layer/extracted_fact_registry_validity")
    parser.add_argument("--events-path", default="data/inputs_layer/event_store.parquet")
    parser.add_argument("--ownership-path", default="data/inputs_layer/ownership_13f_summary.parquet")
    parser.add_argument("--issuer-ratings-path", default="data/inputs_layer/issuer_rating_history.parquet")
    parser.add_argument("--entity-table-path", default="data/inputs_layer/entity.parquet")
    parser.add_argument("--entity-graph-path", default="data/inputs_layer/entity_graph.parquet")
    parser.add_argument("--cache-events", action="store_true")
    parser.add_argument("--cache-timeseries", action="store_true")
    parser.add_argument("--cache-ownership", action="store_true")
    parser.add_argument("--cache-ratings", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    store = SnapshotStore(args.root)
    asof_date = args.asof_date or pd.to_datetime(args.asof).strftime("%Y-%m-%d")

    builder = CompanyStateBuilder(
        raw_timeseries_path=args.timeseries_path,
        macro_timeseries_path=args.macro_path,
        event_store_path=args.events_path,
        facts_path=args.facts_path,
        ownership_summary_path=args.ownership_path,
        issuer_ratings_path=args.issuer_ratings_path,
        entity_table_path=args.entity_table_path,
        entity_graph_path=args.entity_graph_path,
        cache_events=args.cache_events,
        cache_timeseries=args.cache_timeseries,
        cache_ownership=args.cache_ownership,
        cache_ratings=args.cache_ratings,
        debug=args.debug,
    )

    company_ids = _parse_company_ids(args.company_id)
    if args.company_ids_file:
        company_ids = _load_company_ids_file(args.company_ids_file) or company_ids
    if not company_ids:
        company_ids = []
        for snap in store.iter_keyed_snapshots(asof_date):
            cid = snap.get("company_id")
            if cid is not None:
                company_ids.append(str(cid))
    company_ids = list(dict.fromkeys(company_ids))
    if args.limit:
        company_ids = company_ids[: args.limit]

    updated = 0
    missing = 0
    for cid in company_ids:
        snap = store.load_keyed_snapshot(cid, asof_date)
        if snap is None:
            missing += 1
            continue
        out = update_snapshot(snap, builder, args.asof, args.mode)
        prov = out.setdefault("provenance", {})
        deltas = prov.setdefault("delta_updates", [])
        deltas.append(
            {
                "mode": args.mode,
                "as_of_time": pd.to_datetime(args.asof, utc=True).isoformat(),
            }
        )
        store.upsert_keyed_snapshot(out, asof_date)
        updated += 1
        if args.debug and updated % 100 == 0:
            print(f"[delta] updated {updated}", flush=True)

    print(f"[delta] asof_date={asof_date} updated={updated} missing={missing} mode={args.mode}")


if __name__ == "__main__":
    main()

