#!/usr/bin/env python
"""
Build EntityGraph from the ID mapping table.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True)


def build_edges(df: pd.DataFrame, field: str, related_type: str, source_path: Path) -> pd.DataFrame:
    out = pd.DataFrame()
    out["entity_id"] = df["company_id"].astype("string")
    out["entity_id_type"] = "company_id"
    out["related_id"] = df[field].astype("string")
    out["related_id_type"] = related_type
    out["relationship"] = "maps_to"
    out["valid_from"] = df.get("namedt")
    out["valid_to"] = df.get("nameendt")
    out["source_id"] = "entity_id_map"
    out["source_type"] = "mapping"
    out["published_at"] = out["valid_from"].combine_first(out["valid_to"])
    out["effective_at"] = out["valid_from"]
    out["ingested_at"] = utc_now()
    out["confidence_score"] = 1.0
    out["raw_pointer"] = f"{source_path.as_posix()}#row=" + df["row_id"].astype("string") + f":{field}"
    out["edge_id"] = "edge_" + df["row_id"].astype("string") + f"_{field}"
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-path", default="data/mappings/entity_id_map.parquet")
    parser.add_argument("--out", default="data/inputs_layer/entity_graph.parquet")
    args = parser.parse_args()

    map_path = ROOT / args.map_path
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not map_path.exists():
        raise FileNotFoundError(f"Missing mapping file: {map_path}")

    df = pd.read_parquet(map_path)
    df = df.reset_index().rename(columns={"index": "row_id"})

    for c in ["namedt", "nameendt"]:
        if c in df.columns:
            df[c] = parse_dt(df[c])

    fields: Dict[str, str] = {
        "permno": "permno",
        "permco": "permco",
        "cik": "cik",
        "ticker": "ticker",
        "cusip": "cusip",
    }

    parts: List[pd.DataFrame] = []
    for field, related_type in fields.items():
        if field in df.columns:
            part = build_edges(df, field, related_type, map_path)
            parts.append(part)

    if not parts:
        raise RuntimeError("No mapping fields found to build EntityGraph.")

    out_df = pd.concat(parts, ignore_index=True)
    out_df.to_parquet(out_path, index=False)
    print(f"Saved EntityGraph -> {out_path} ({len(out_df):,} rows)")


if __name__ == "__main__":
    main()
