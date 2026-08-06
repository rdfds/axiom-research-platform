import argparse
import faulthandler
import json
import os
import sys
import time
import zlib
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Allow trace-hang before heavy imports (set TRACE_HANG=1)
if os.environ.get("TRACE_HANG") == "1":
    faulthandler.dump_traceback_later(30, repeat=True)

import pandas as pd
import duckdb

from src.company_state_builder import CompanyStateBuilder
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
            # allow comma-separated or whitespace-separated
            parts = [x.strip() for x in line.replace(",", " ").split() if x.strip()]
            ids.extend(parts)
    return ids

def _apply_shard(ids: List[str], shard: Optional[int], shard_count: Optional[int]) -> List[str]:
    if shard is None or shard_count is None:
        return ids
    if shard < 0 or shard_count <= 0 or shard >= shard_count:
        raise SystemExit("--shard must be in [0, shard_count)")
    out: List[str] = []
    for cid in ids:
        h = zlib.crc32(cid.encode("utf-8")) % shard_count
        if h == shard:
            out.append(cid)
    return out

def _count_jsonl_rows(path: Path) -> int:
    n = 0
    with path.open("r") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _default_dealscan_revolver_path(asof: str) -> Path:
    active_name = f"loanconnector_revolver_facilities_active_{asof.replace('-', '_')}.parquet"
    active_path = ROOT / "data" / "wrds" / "dealscan" / active_name
    if active_path.exists():
        return active_path
    return ROOT / "data" / "wrds" / "dealscan" / "loanconnector_revolver_facilities.parquet"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CompanyStateSnapshot JSONL.")
    parser.add_argument("--asof", required=True, help="As-of timestamp (e.g., 2026-02-28)")
    parser.add_argument("--company-id", help="Company ID or comma-separated list")
    parser.add_argument("--company-ids-file", default=None, help="Path to file with company IDs (one per line)")
    parser.add_argument("--entity-table", default="data/inputs_layer/entity.parquet")
    parser.add_argument("--entity-table-path", default=None, help="Override entity table path (e.g., /tmp/entity.parquet)")
    parser.add_argument("--entity-identifier-path", default=None, help="Override entity identifier path")
    parser.add_argument("--entity-graph-path", default=None, help="Override entity graph path")
    parser.add_argument("--out", default="data/company_state_snapshots")
    parser.add_argument("--out-format", default="jsonl", choices=["jsonl", "parquet", "both", "keyed", "all"])
    parser.add_argument("--timeseries-path", default=None)
    parser.add_argument("--macro-path", default=None, help="Optional macro-only timeseries file")
    parser.add_argument("--facts-path", default=None)
    parser.add_argument("--dealscan-revolver-path", default=None, help="Optional DealScan revolver parquet fallback")
    parser.add_argument("--events-path", default=None)
    parser.add_argument("--ownership-path", default=None, help="Optional 13F ownership summary parquet")
    parser.add_argument("--issuer-ratings-path", default=None, help="Optional issuer rating history parquet")
    parser.add_argument("--taxonomy-reference-path", default=None, help="Optional fundamentals/taxonomy reference parquet")
    parser.add_argument("--metric-policy-path", default=None, help="Optional metric policy JSON path")
    parser.add_argument("--methodology-registry-path", default=None, help="Optional methodology registry JSON path")
    parser.add_argument("--input-source-registry-path", default=None, help="Optional company-state input source registry JSON path")
    parser.add_argument("--skip-timeseries", action="store_true", help="Skip company timeseries (market features)")
    parser.add_argument("--skip-macro", action="store_true", help="Skip macro regime computation")
    parser.add_argument("--skip-events", action="store_true", help="Skip loading events (faster tests)")
    parser.add_argument("--skip-peers", action="store_true", help="Skip peer resolution/context (faster tests)")
    parser.add_argument("--debug", action="store_true", help="Print timing for each build step")
    parser.add_argument("--trace-hang", action="store_true", help="Dump stack traces every 30s")
    parser.add_argument("--facts-years", default=None, help="Comma-separated list of fact years to scan")
    parser.add_argument("--facts-window-years", type=int, default=None, help="Use last N years of facts (relative to asof)")
    parser.add_argument("--cache-facts", action="store_true", help="Cache facts once for all companies")
    parser.add_argument("--cache-events", action="store_true", help="Cache events once for all companies")
    parser.add_argument("--cache-timeseries", action="store_true", help="Cache timeseries once for all companies")
    parser.add_argument("--cache-ownership", action="store_true", help="Cache ownership summary once for all companies")
    parser.add_argument("--cache-ratings", action="store_true", help="Cache issuer ratings once for all companies")
    parser.add_argument(
        "--historical-backfill-mode",
        action="store_true",
        help="Ignore ingested_at cutoff when reconstructing older as-of snapshots from backfilled facts.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=50, help="Progress log every N companies (0 to disable)")
    parser.add_argument("--shard", type=int, default=None, help="Shard index (0-based)")
    parser.add_argument("--shard-count", type=int, default=None, help="Total shards")
    parser.add_argument(
        "--allow-smaller-overwrite",
        action="store_true",
        help="Allow overwriting an existing snapshot file with fewer rows.",
    )
    args = parser.parse_args()

    if args.trace_hang:
        faulthandler.dump_traceback_later(30, repeat=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    store = SnapshotStore(out_dir)
    dealscan_revolver_path = Path(args.dealscan_revolver_path) if args.dealscan_revolver_path else _default_dealscan_revolver_path(args.asof)
    if dealscan_revolver_path.exists():
        print(f"[snapshot] using DealScan revolver path: {dealscan_revolver_path}")

    builder = CompanyStateBuilder(
        raw_timeseries_path=args.timeseries_path or "data/inputs_layer/raw_timeseries.parquet",
        macro_timeseries_path=args.macro_path,
        event_store_path=args.events_path or "data/inputs_layer/event_store.parquet",
        facts_path=args.facts_path or "data/inputs_layer/extracted_fact_registry_validity",
        dealscan_revolver_path=dealscan_revolver_path,
        ownership_summary_path=args.ownership_path or "data/inputs_layer/ownership_13f_summary.parquet",
        issuer_ratings_path=args.issuer_ratings_path or "data/inputs_layer/issuer_rating_history.parquet",
        entity_table_path=args.entity_table_path or args.entity_table,
        entity_graph_path=args.entity_graph_path or "data/inputs_layer/entity_graph.parquet",
        entity_identifier_path=args.entity_identifier_path or "data/inputs_layer/entity_identifier.parquet",
        taxonomy_reference_path=args.taxonomy_reference_path or "data/refinitiv/fundamentals_all.parquet",
        metric_policy_path=args.metric_policy_path,
        methodology_registry_path=args.methodology_registry_path,
        input_source_registry_path=args.input_source_registry_path,
        skip_timeseries=args.skip_timeseries,
        skip_macro=args.skip_macro,
        facts_years=None,
        skip_events=args.skip_events,
        skip_peer_context=args.skip_peers,
        debug=args.debug,
        cache_facts=args.cache_facts,
        cache_events=args.cache_events,
        cache_timeseries=args.cache_timeseries,
        cache_ownership=args.cache_ownership,
        cache_ratings=args.cache_ratings,
        historical_backfill_mode=args.historical_backfill_mode,
    )
    if args.facts_years:
        try:
            years = [int(x.strip()) for x in args.facts_years.split(",") if x.strip()]
            builder.facts_years = years
        except ValueError:
            raise SystemExit("Invalid --facts-years; expected comma-separated integers")
    if args.facts_window_years:
        try:
            asof_year = pd.to_datetime(args.asof).year
            builder.facts_years = [asof_year - i for i in range(int(args.facts_window_years))]
        except Exception:
            raise SystemExit("Invalid --facts-window-years")

    company_ids = _parse_company_ids(args.company_id)
    if args.company_ids_file:
        company_ids = _load_company_ids_file(args.company_ids_file) or company_ids
    if not company_ids:
        # Default: read entity table for company IDs
        entity_path_for_ids = Path(args.entity_table_path or args.entity_table)
        if not entity_path_for_ids.exists():
            raise SystemExit("No company ids provided and entity table missing.")
        con = duckdb.connect()
        ent = con.execute(
            f"SELECT entity_id FROM read_parquet('{entity_path_for_ids.as_posix()}', union_by_name=True)"
        ).df()
        company_ids = ent["entity_id"].astype(str).tolist()
    if args.shard_count is not None or args.shard is not None:
        company_ids = _apply_shard(company_ids, args.shard, args.shard_count)

    if args.limit:
        company_ids = company_ids[: args.limit]

    expected_count = len(company_ids)
    out_jsonl = out_dir / f"company_state_snapshots_asof={args.asof}.jsonl"
    if args.out_format in ("jsonl", "both", "all") and out_jsonl.exists() and not args.allow_smaller_overwrite:
        existing_rows = _count_jsonl_rows(out_jsonl)
        if existing_rows > expected_count:
            raise SystemExit(
                "Refusing to overwrite a larger existing snapshot file with fewer rows. "
                f"existing_rows={existing_rows}, planned_rows={expected_count}. "
                "Use --allow-smaller-overwrite to override."
            )

    start_time = time.time()
    def iter_snapshots():
        for idx, cid in enumerate(company_ids):
            if args.debug:
                print(f"[snapshot] building {cid} ({idx+1}/{len(company_ids)})")
            snapshot = builder.build(cid, args.asof)
            if args.log_every and ((idx + 1) % args.log_every == 0 or (idx + 1) == len(company_ids)):
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed if elapsed > 0 else 0.0
                remaining = len(company_ids) - (idx + 1)
                eta = remaining / rate if rate > 0 else 0.0
                print(
                    f"[progress] {idx + 1}/{len(company_ids)} "
                    f"elapsed={elapsed/60:.1f}m rate={rate:.2f}/s eta={eta/60:.1f}m",
                    flush=True,
                )
            yield snapshot

    # Stream JSONL to avoid holding all snapshots in memory.
    if args.out_format == "jsonl":
        out_file = store.write_jsonl(iter_snapshots(), args.asof, expected_count=expected_count)
        print(f"Wrote snapshots -> {out_file}")
        return
    if args.out_format == "keyed":
        out_dir_keyed = store.write_keyed_json(iter_snapshots(), args.asof, expected_count=expected_count)
        print(f"Wrote keyed snapshots -> {out_dir_keyed}")
        return

    snapshots = list(iter_snapshots())
    if args.out_format in ("jsonl", "both", "all"):
        out_file = store.write_jsonl(snapshots, args.asof, expected_count=expected_count)
        print(f"Wrote snapshots -> {out_file}")
    if args.out_format in ("parquet", "both", "all"):
        out_file = store.write_parquet(snapshots, args.asof, expected_count=expected_count)
        print(f"Wrote snapshots -> {out_file}")
    if args.out_format in ("keyed", "all"):
        out_dir_keyed = store.write_keyed_json(snapshots, args.asof, expected_count=expected_count)
        print(f"Wrote keyed snapshots -> {out_dir_keyed}")


if __name__ == "__main__":
    main()
