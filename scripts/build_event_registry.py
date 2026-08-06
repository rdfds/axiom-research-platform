#!/usr/bin/env python
"""
Build EventRegistry from the unified corporate actions master dataset.

This is a baseline generator: it normalizes core fields and preserves provenance.
Parameters/evidence_links are left null for now (can be enriched later).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True)


def coalesce(series_list: Iterable[pd.Series]) -> pd.Series:
    out = None
    for s in series_list:
        if s is None:
            continue
        out = s if out is None else out.combine_first(s)
    return out


def prefixed_str(s: pd.Series, prefix: str) -> pd.Series:
    s = s.astype("string")
    s = s.where(s.notna(), pd.NA)
    return prefix + s


def build_source_id(df: pd.DataFrame) -> pd.Series:
    source_id = pd.Series(pd.NA, index=df.index, dtype="string")

    for prefix, col in [
        ("deal_id:", "deal_id"),
        ("issue_id:", "ISSUE_ID"),
        ("issuer_id:", "ISSUER_ID"),
        ("facilityid:", "facilityid"),
        ("action_code:", "action_code"),
        ("distcd:", "distcd"),
    ]:
        if col in df.columns:
            s = prefixed_str(df[col], prefix)
            source_id = source_id.combine_first(s)

    # Fallback: hash a small stable key
    missing = source_id.isna()
    if missing.any():
        key_cols = [
            c
            for c in [
                "source",
                "source_table",
                "action_type",
                "action_subtype",
                "action_date",
                "permno",
                "gvkey",
            ]
            if c in df.columns
        ]
        key_df = df.loc[missing, key_cols].copy()
        for c in key_df.columns:
            if pd.api.types.is_datetime64_any_dtype(key_df[c]):
                key_df[c] = key_df[c].dt.strftime("%Y-%m-%d")
            key_df[c] = key_df[c].astype("string").fillna("")
        hashes = pd.util.hash_pandas_object(key_df, index=False).astype("uint64").astype("string")
        source_id.loc[missing] = "rowhash:" + hashes

    return source_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--actions-path",
        default="data/curated/corporate_actions_master.parquet",
        help="Unified corporate actions master parquet.",
    )
    parser.add_argument(
        "--out",
        default="data/inputs_layer/event_registry.parquet",
        help="Output EventRegistry parquet.",
    )
    args = parser.parse_args()

    actions_path = ROOT / args.actions_path
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not actions_path.exists():
        raise FileNotFoundError(f"Missing actions file: {actions_path}")

    pf = pq.ParquetFile(actions_path)
    cols_needed: List[str] = [
        "gvkey",
        "gvkey_x",
        "permno",
        "entity_id",
        "company_name",
        "ticker",
        "action_type",
        "action_subtype",
        "source_action_subtype",
        "action_date",
        "announce_date",
        "event_date",
        "completion_date",
        "deal_status",
        "source",
        "source_table",
        "confidence",
        "deal_id",
        "ISSUE_ID",
        "ISSUER_ID",
        "facilityid",
        "action_code",
        "distcd",
    ]
    cols = [c for c in cols_needed if c in pf.schema.names]

    df = pd.read_parquet(actions_path, columns=cols)

    # Date fields
    for c in ["action_date", "announce_date", "event_date", "completion_date"]:
        if c in df.columns:
            df[c] = parse_dt(df[c])

    # Core identity
    company_id = pd.Series(pd.NA, index=df.index, dtype="string")
    if "gvkey" in df.columns:
        company_id = company_id.combine_first(df["gvkey"].astype("string"))
    if "gvkey_x" in df.columns:
        company_id = company_id.combine_first(df["gvkey_x"].astype("string"))
    if "permno" in df.columns:
        permno_str = "permno:" + df["permno"].astype("Int64").astype("string")
        company_id = company_id.combine_first(permno_str)
    df["company_id"] = company_id

    # entity_id: use existing if present, else company_id
    if "entity_id" in df.columns:
        df["entity_id"] = df["entity_id"].astype("string").combine_first(df["company_id"])
    else:
        df["entity_id"] = df["company_id"]

    # action_subtype fallback
    if "source_action_subtype" in df.columns:
        df["action_subtype"] = df["action_subtype"].combine_first(df["source_action_subtype"])

    # announcement/effective dates
    announcement = coalesce(
        [
            df["announce_date"] if "announce_date" in df.columns else None,
            df["event_date"] if "event_date" in df.columns else None,
            df["action_date"] if "action_date" in df.columns else None,
        ]
    )
    effective = coalesce(
        [
            df["completion_date"] if "completion_date" in df.columns else None,
            df["event_date"] if "event_date" in df.columns else None,
            df["action_date"] if "action_date" in df.columns else None,
        ]
    )
    df["announcement_date"] = announcement
    df["effective_date"] = effective
    df["published_at"] = announcement
    df["effective_at"] = effective

    # status
    if "deal_status" in df.columns:
        df["status"] = df["deal_status"].astype("string")
    else:
        df["status"] = pd.NA

    # source_type
    source_type = pd.Series(pd.NA, index=df.index, dtype="string")
    if "source" in df.columns:
        source_type = source_type.combine_first(df["source"].astype("string"))
    if "source_table" in df.columns:
        source_type = source_type.combine_first(df["source_table"].astype("string"))
    df["source_type"] = source_type

    # source_id and event_id
    df["source_id"] = build_source_id(df)
    df["event_id"] = "evt_" + df["source_id"].astype("string")

    # confidence_score
    conf = pd.Series(1.0, index=df.index, dtype="float64")
    if "confidence" in df.columns:
        # map common text labels if present
        conf_map = {"high": 0.9, "medium": 0.7, "low": 0.4}
        mapped = df["confidence"].map(conf_map)
        conf = mapped.fillna(1.0).astype("float64")
    df["confidence_score"] = conf.clip(0.0, 1.0)

    # parameters/evidence_links placeholders
    df["parameters"] = None
    df["evidence_links"] = None

    # ingested_at and raw_pointer
    ingested_at = utc_now()
    df["ingested_at"] = ingested_at
    df["raw_pointer"] = f"{actions_path.as_posix()}#row=" + pd.Series(np.arange(len(df)), index=df.index).astype("string")

    # Select and write
    out_cols = [
        "event_id",
        "company_id",
        "entity_id",
        "action_type",
        "action_subtype",
        "announcement_date",
        "effective_date",
        "status",
        "parameters",
        "evidence_links",
        "source_id",
        "source_type",
        "published_at",
        "effective_at",
        "ingested_at",
        "confidence_score",
        "raw_pointer",
    ]

    out_df = df[out_cols].copy()
    out_df.to_parquet(out_path, index=False)
    print(f"Saved EventRegistry -> {out_path} ({len(out_df):,} rows)")


if __name__ == "__main__":
    main()
