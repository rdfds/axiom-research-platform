#!/usr/bin/env python
"""
Build RawTimeSeriesStore by combining prices, macro, and estimates into a
single point-in-time table with provenance.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True)


def load_parquet_cols(path: Path, cols: List[str]) -> pd.DataFrame:
    pf = pq.ParquetFile(path)
    available = [c for c in cols if c in pf.schema.names]
    if not available:
        return pd.DataFrame()
    return pd.read_parquet(path, columns=available)


def build_prices(path: Path) -> pd.DataFrame:
    cols = [
        "source_system",
        "entity_id",
        "company_id",
        "security_id",
        "event_time",
        "available_time",
        "ingestion_time",
        "trade_date",
        "close",
        "adjusted_close",
        "volume",
        "ret",
        "retx",
        "cusip",
    ]
    df = load_parquet_cols(path, cols)
    if df.empty:
        return df

    df = df.reset_index().rename(columns={"index": "row_id"})
    for c in ["event_time", "available_time", "ingestion_time", "trade_date"]:
        if c in df.columns:
            df[c] = parse_dt(df[c])

    date = df["trade_date"] if "trade_date" in df.columns else df["event_time"]
    df["_date"] = date

    value_vars = [c for c in ["close", "adjusted_close", "volume", "ret", "retx"] if c in df.columns]
    m = df.melt(
        id_vars=[
            "row_id",
            "source_system",
            "entity_id",
            "company_id",
            "security_id",
            "cusip",
            "event_time",
            "available_time",
            "ingestion_time",
            "_date",
        ],
        value_vars=value_vars,
        var_name="metric",
        value_name="value",
    )

    m = m[m["value"].notna()].copy()
    unit_map: Dict[str, str] = {
        "close": "price",
        "adjusted_close": "price",
        "volume": "shares",
        "ret": "pct",
        "retx": "pct",
    }

    m["series_id"] = "price." + m["metric"].astype("string")
    m["series_type"] = "price"
    m["entity_id_type"] = "entity_id"
    m["date"] = m["_date"]
    m["unit"] = m["metric"].map(unit_map)
    m["currency"] = pd.NA
    m["frequency"] = "D"
    m["published_at"] = m["available_time"].combine_first(m["event_time"])
    m["effective_at"] = m["_date"]
    m["ingested_at"] = m["ingestion_time"]
    m["confidence_score"] = 1.0
    m["raw_pointer"] = f"{path.as_posix()}#row=" + m["row_id"].astype("string") + ":" + m["metric"].astype("string")
    m["revision_flag"] = pd.NA
    m["release_lag_days"] = pd.NA

    return m[
        [
            "series_id",
            "series_type",
            "entity_id",
            "entity_id_type",
            "date",
            "value",
            "unit",
            "currency",
            "frequency",
            "published_at",
            "effective_at",
            "ingested_at",
            "confidence_score",
            "raw_pointer",
            "revision_flag",
            "release_lag_days",
            "company_id",
            "security_id",
            "cusip",
            "source_system",
        ]
    ]


def build_macro(path: Path) -> pd.DataFrame:
    cols = [
        "source_system",
        "entity_id",
        "instrument_id",
        "instrument_type",
        "tenor",
        "event_time",
        "available_time",
        "ingestion_time",
        "value",
        "units",
    ]
    df = load_parquet_cols(path, cols)
    if df.empty:
        return df

    df = df.reset_index().rename(columns={"index": "row_id"})
    for c in ["event_time", "available_time", "ingestion_time"]:
        if c in df.columns:
            df[c] = parse_dt(df[c])

    series_id = df["instrument_id"] if "instrument_id" in df.columns else df["entity_id"]
    df["series_id"] = series_id.astype("string")
    df["series_type"] = "macro"
    df["entity_id_type"] = "macro_series"
    df["date"] = df["event_time"]
    df["unit"] = df["units"] if "units" in df.columns else pd.NA
    df["currency"] = pd.NA
    df["frequency"] = pd.NA
    df["published_at"] = df["available_time"].combine_first(df["event_time"])
    df["effective_at"] = df["event_time"]
    df["ingested_at"] = df["ingestion_time"]
    df["confidence_score"] = 1.0
    df["raw_pointer"] = f"{path.as_posix()}#row=" + df["row_id"].astype("string")
    df["revision_flag"] = pd.NA
    lag = (df["available_time"] - df["event_time"]).dt.days if "available_time" in df.columns else pd.NA
    df["release_lag_days"] = lag

    return df[
        [
            "series_id",
            "series_type",
            "entity_id",
            "entity_id_type",
            "date",
            "value",
            "unit",
            "currency",
            "frequency",
            "published_at",
            "effective_at",
            "ingested_at",
            "confidence_score",
            "raw_pointer",
            "revision_flag",
            "release_lag_days",
            "instrument_id",
            "instrument_type",
            "tenor",
            "source_system",
        ]
    ]


def build_estimates(path: Path) -> pd.DataFrame:
    cols = [
        "source_system",
        "entity_id",
        "metric",
        "period",
        "period_end",
        "event_time",
        "available_time",
        "ingestion_time",
        "consensus_value",
        "num_estimates",
        "revision_direction",
        "revision_magnitude",
    ]
    df = load_parquet_cols(path, cols)
    if df.empty:
        return df

    df = df.reset_index().rename(columns={"index": "row_id"})
    for c in ["event_time", "available_time", "ingestion_time", "period_end"]:
        if c in df.columns:
            df[c] = parse_dt(df[c])

    metric = df["metric"].astype("string").str.replace(" ", "_")
    period = df["period"].astype("string").str.replace(" ", "_")
    df["series_id"] = "estimate." + metric + "." + period + ".consensus"
    df["series_type"] = "estimate"
    df["entity_id_type"] = "entity_id"
    df["date"] = df["event_time"]
    df["value"] = df["consensus_value"]
    df["unit"] = pd.NA
    df["currency"] = pd.NA
    df["frequency"] = pd.NA
    df["published_at"] = df["available_time"].combine_first(df["event_time"])
    df["effective_at"] = df["event_time"]
    df["ingested_at"] = df["ingestion_time"]
    df["confidence_score"] = 1.0
    df["raw_pointer"] = f"{path.as_posix()}#row=" + df["row_id"].astype("string")
    df["revision_flag"] = pd.NA
    df["release_lag_days"] = pd.NA

    return df[
        [
            "series_id",
            "series_type",
            "entity_id",
            "entity_id_type",
            "date",
            "value",
            "unit",
            "currency",
            "frequency",
            "published_at",
            "effective_at",
            "ingested_at",
            "confidence_score",
            "raw_pointer",
            "revision_flag",
            "release_lag_days",
            "metric",
            "period",
            "period_end",
            "num_estimates",
            "revision_direction",
            "revision_magnitude",
            "source_system",
        ]
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices-path", default="data/warehouse/warehouse_prices.parquet")
    parser.add_argument("--macro-path", default="data/warehouse/warehouse_macro.parquet")
    parser.add_argument("--estimates-path", default="data/warehouse/warehouse_estimates.parquet")
    parser.add_argument("--out", default="data/inputs_layer/raw_timeseries.parquet")
    args = parser.parse_args()

    prices_path = ROOT / args.prices_path
    macro_path = ROOT / args.macro_path
    estimates_path = ROOT / args.estimates_path
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    parts = []
    if prices_path.exists():
        parts.append(build_prices(prices_path))
    if macro_path.exists():
        parts.append(build_macro(macro_path))
    if estimates_path.exists():
        parts.append(build_estimates(estimates_path))

    parts = [p for p in parts if p is not None and not p.empty]
    if not parts:
        raise RuntimeError("No input datasets found to build RawTimeSeriesStore.")

    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(out_path, index=False)
    print(f"Saved RawTimeSeriesStore -> {out_path} ({len(df):,} rows)")


if __name__ == "__main__":
    main()
