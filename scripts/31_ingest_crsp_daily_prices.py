#!/usr/bin/env python
"""
Ingest CRSP Daily Stock File (dsf) into warehouse_prices_daily dataset.

This writes partitioned parquet files by year to:
  data/warehouse/warehouse_prices_daily/year=YYYY/part_*.parquet
"""

from __future__ import annotations

import argparse
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion import compute_raw_payload_hash


DATA_DIR = Path(__file__).parent.parent / "data"
WRDS_DIR = DATA_DIR / "wrds" / "crsp"
OUT_DIR = DATA_DIR / "warehouse" / "warehouse_prices_daily"


def log(msg: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}")


def parse_date(value: str) -> Optional[pd.Timestamp]:
    try:
        return pd.to_datetime(value)
    except Exception:
        return None


def adjust_close(df: pd.DataFrame) -> pd.Series:
    close = df["close"]
    if "cfacpr" in df.columns:
        factor = df["cfacpr"].replace(0, np.nan)
        return close / factor
    return close


def build_hash_series(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    # Fast stable hash using pandas, then hex string
    h = pd.util.hash_pandas_object(df[cols], index=False)
    return h.map(lambda x: f"{x:016x}")


def ingest_file(path: Path, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> int:
    df = pd.read_parquet(path)
    if df.empty:
        return 0

    df["date"] = pd.to_datetime(df["date"])
    if start is not None:
        df = df[df["date"] >= start]
    if end is not None:
        df = df[df["date"] <= end]
    if df.empty:
        return 0

    df["event_time"] = df["date"]
    df["available_time"] = df["date"] + pd.Timedelta(hours=16)
    df["entity_id"] = df["permno"].astype("Int64").astype("string")
    df["security_id"] = df["entity_id"]
    df["company_id"] = None

    # Close price (CRSP prc can be negative)
    df["close"] = df["prc"].abs()
    df["adjusted_close"] = adjust_close(df)

    # Optional OHLC (if available)
    df["open"] = df["openprc"] if "openprc" in df.columns else np.nan
    df["high"] = df["askhi"] if "askhi" in df.columns else np.nan
    df["low"] = df["bidlo"] if "bidlo" in df.columns else np.nan

    df["volume"] = df["vol"] if "vol" in df.columns else np.nan
    df["total_return_index"] = np.nan

    hash_cols = ["permno", "date", "prc", "vol"]
    hash_cols = [c for c in hash_cols if c in df.columns]
    df["raw_payload_hash"] = build_hash_series(df, hash_cols)

    df["version_id"] = build_hash_series(
        df.assign(source_system="crsp_dsf")[["permno", "date", "raw_payload_hash"]],
        ["permno", "date", "raw_payload_hash"],
    )

    df["source_system"] = "crsp_dsf"
    df["ingestion_time"] = datetime.utcnow()
    df["upstream_version_ids"] = None
    df["quality_flags"] = None

    out_cols = [
        "source_system",
        "entity_id",
        "company_id",
        "security_id",
        "event_time",
        "available_time",
        "ingestion_time",
        "version_id",
        "raw_payload_hash",
        "upstream_version_ids",
        "quality_flags",
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
        "total_return_index",
        "ret",
        "retx",
        "permno",
        "date",
    ]
    for col in out_cols:
        if col not in df.columns:
            df[col] = np.nan
    df = df[out_cols]

    df["year"] = df["event_time"].dt.year.astype("Int64")

    rows = 0
    for year, ydf in df.groupby("year"):
        if pd.isna(year):
            continue
        year_dir = OUT_DIR / f"year={int(year)}"
        year_dir.mkdir(parents=True, exist_ok=True)
        part_path = year_dir / f"part_{uuid.uuid4().hex}.parquet"
        ydf.drop(columns=["year"]).to_parquet(part_path, index=False)
        rows += len(ydf)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--pattern", default="dsf_*.parquet", help="Glob for dsf files")
    args = parser.parse_args()

    start = parse_date(args.start) if args.start else None
    end = parse_date(args.end) if args.end else None

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(WRDS_DIR.glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {args.pattern} in {WRDS_DIR}")

    total = 0
    for path in files:
        log(f"Ingesting {path.name}...")
        rows = ingest_file(path, start, end)
        total += rows
        log(f"  {path.name}: {rows:,} rows")

    log(f"Done. Total rows ingested: {total:,}")


if __name__ == "__main__":
    main()
