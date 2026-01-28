#!/usr/bin/env python
"""
Build RawEventItem registry from warehouse corp actions (bitemporal raw).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def collect_mentions(row: pd.Series) -> List[str] | None:
    vals = []
    for col in ["entity_id", "company_id", "security_id", "ticker", "cusip"]:
        val = row.get(col)
        if pd.notna(val):
            vals.append(str(val))
    if not vals:
        return None
    # de-dup preserve order
    seen = {}
    return [seen.setdefault(v, v) for v in vals if v not in seen]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-path", default="data/warehouse/warehouse_corp_actions.parquet")
    parser.add_argument("--out-path", default="data/inputs_layer/raw_event_items.parquet")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    in_path = ROOT / args.in_path
    out_path = ROOT / args.out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not args.overwrite:
        print(f"Output exists: {out_path}")
        return

    df = pd.read_parquet(in_path)

    # timestamps
    df["event_time"] = pd.to_datetime(df.get("event_time"), errors="coerce", utc=True)
    df["available_time"] = pd.to_datetime(df.get("available_time"), errors="coerce", utc=True)
    df["ingestion_time"] = pd.to_datetime(df.get("ingestion_time"), errors="coerce", utc=True)

    published_at = df["available_time"].combine_first(df["event_time"])

    out = pd.DataFrame(
        {
            "raw_event_id": df["version_id"].astype("string"),
            "source": df["source_system"].astype("string"),
            "source_id": df["raw_payload_hash"].astype("string"),
            "published_at": published_at,
            "ingested_at": df["ingestion_time"],
            "raw_payload": df.to_dict(orient="records"),
            "status_text": df.get("source_action_subtype").combine_first(df.get("action_subtype")),
            "entity_mentions": df.apply(collect_mentions, axis=1),
            "content_hash": df["raw_payload_hash"].astype("string"),
            "effective_at": pd.to_datetime(df.get("effective_date"), errors="coerce", utc=True),
        }
    )

    out.to_parquet(out_path, index=False)
    print(f"Wrote RawEventItems -> {out_path}")


if __name__ == "__main__":
    main()
