import argparse
from pathlib import Path
from typing import List

import pandas as pd

from src.company_state_builder import CompanyStateBuilder
from src.company_state_store import SnapshotStore


def parse_dates(start: str, end: str) -> List[str]:
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    dates = pd.date_range(start_dt, end_dt, freq="D")
    return [d.strftime("%Y-%m-%d") for d in dates]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build daily CompanyState cache.")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--entity-table", default="data/inputs_layer/entity.parquet")
    parser.add_argument("--out", default="data/company_state_snapshots")
    parser.add_argument("--out-format", choices=["jsonl", "keyed", "both"], default="both")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if not Path(args.entity_table).exists():
        raise SystemExit("Entity table missing.")

    ent = pd.read_parquet(args.entity_table, columns=["entity_id"])
    company_ids = ent["entity_id"].astype(str).tolist()
    if args.limit:
        company_ids = company_ids[: args.limit]

    builder = CompanyStateBuilder()
    store = SnapshotStore(args.out)

    for d in parse_dates(args.start, args.end):
        snapshots = [builder.build(cid, d) for cid in company_ids]
        if args.out_format in ("jsonl", "both"):
            store.write_jsonl(snapshots, d, expected_count=len(snapshots))
        if args.out_format in ("keyed", "both"):
            store.write_keyed_json(snapshots, d, expected_count=len(snapshots))
        print(f"[cache] wrote {len(snapshots)} snapshots for {d} format={args.out_format}")


if __name__ == "__main__":
    main()
