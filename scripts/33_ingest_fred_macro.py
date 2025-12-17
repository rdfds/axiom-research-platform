#!/usr/bin/env python
"""
Ingest macro rates / spreads / volatility from FRED into the warehouse.

Uses the public FRED CSV endpoint (no API key required):
  https://fred.stlouisfed.org/graph/fredgraph.csv?id=SERIES&cosd=YYYY-MM-DD&coed=YYYY-MM-DD

Env:
  FRED_START=2000-01-01
  FRED_END=YYYY-MM-DD (default: today UTC)
  FRED_SLEEP=0.2
  FRED_SERIES=comma,separated,list (optional override)
"""

from __future__ import annotations

import io
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import requests

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion import (
    append_canonical_records,
    compute_raw_payload_hash,
    compute_version_id,
    write_raw_records,
)


DATA_DIR = Path(__file__).parent.parent / "data"

FRED_START = os.getenv("FRED_START", "2000-01-01")
FRED_END = os.getenv("FRED_END", datetime.utcnow().strftime("%Y-%m-%d"))
FRED_SLEEP = float(os.getenv("FRED_SLEEP", "0.2"))
FRED_SERIES = os.getenv("FRED_SERIES")


SERIES_DEFAULT = [
    {"series_id": "DGS1", "instrument_type": "rate", "tenor": "1Y", "units": "pct"},
    {"series_id": "DGS2", "instrument_type": "rate", "tenor": "2Y", "units": "pct"},
    {"series_id": "DGS5", "instrument_type": "rate", "tenor": "5Y", "units": "pct"},
    {"series_id": "DGS10", "instrument_type": "rate", "tenor": "10Y", "units": "pct"},
    {"series_id": "DGS30", "instrument_type": "rate", "tenor": "30Y", "units": "pct"},
    {"series_id": "SOFR", "instrument_type": "rate", "tenor": "ON", "units": "pct"},
    {"series_id": "DFF", "instrument_type": "rate", "tenor": "ON", "units": "pct"},
    {"series_id": "AAA", "instrument_type": "rate", "tenor": "corp", "units": "pct"},
    {"series_id": "BAA", "instrument_type": "rate", "tenor": "corp", "units": "pct"},
    {"series_id": "BAMLC0A0CM", "instrument_type": "spread", "tenor": "OAS", "units": "pct"},
    {"series_id": "BAMLH0A0HYM2", "instrument_type": "spread", "tenor": "OAS", "units": "pct"},
    {"series_id": "VIXCLS", "instrument_type": "volatility", "tenor": None, "units": "index"},
    # Inflation / growth / labor
    {"series_id": "CPIAUCSL", "instrument_type": "inflation", "tenor": None, "units": "index"},
    {"series_id": "PCEPI", "instrument_type": "inflation", "tenor": None, "units": "index"},
    {"series_id": "GDPC1", "instrument_type": "gdp", "tenor": "real", "units": "bil_ch2017_usd"},
    {"series_id": "UNRATE", "instrument_type": "labor", "tenor": None, "units": "pct"},
    {"series_id": "INDPRO", "instrument_type": "production", "tenor": None, "units": "index"},
    {"series_id": "RSAFS", "instrument_type": "consumption", "tenor": None, "units": "mil_usd"},
    # FX / commodities / equity index
    {"series_id": "DTWEXBGS", "instrument_type": "fx", "tenor": "broad", "units": "index"},
    {"series_id": "DCOILWTICO", "instrument_type": "commodity", "tenor": "oil", "units": "usd_bbl"},
    # Gold proxy (FRED no longer serves LBMA spot series; use import price index)
    {"series_id": "IP7108", "instrument_type": "commodity", "tenor": "gold_import_price_index", "units": "index"},
    {"series_id": "SP500", "instrument_type": "equity_index", "tenor": None, "units": "index"},
]


def log(msg: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def normalize_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def fetch_fred_series(session: requests.Session, series_id: str, start: str, end: str) -> pd.DataFrame:
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    params = {"id": series_id, "cosd": start, "coed": end}
    resp = session.get(url, params=params, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    if df.empty:
        return df
    # FRED responses can use either DATE or observation_date
    if "DATE" in df.columns:
        date_col = "DATE"
    elif "observation_date" in df.columns:
        date_col = "observation_date"
    else:
        raise ValueError(f"Unexpected FRED response columns for {series_id}: {list(df.columns)}")

    value_col = series_id
    if value_col not in df.columns:
        # Fallback: assume second column is the value
        value_col = df.columns[1] if len(df.columns) > 1 else series_id

    df = df.rename(columns={date_col: "date", value_col: "value"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date"])
    return df


def iter_chunks(df: pd.DataFrame, chunk_size: int) -> Iterable[pd.DataFrame]:
    if chunk_size <= 0:
        yield df
        return
    for start in range(0, len(df), chunk_size):
        yield df.iloc[start : start + chunk_size]


def main() -> None:
    series_list = SERIES_DEFAULT
    if FRED_SERIES:
        requested = {s.strip().upper() for s in FRED_SERIES.split(",") if s.strip()}
        series_list = [s for s in SERIES_DEFAULT if s["series_id"].upper() in requested]
        if not series_list:
            log("No valid series selected; nothing to do.")
            return

    source_system = "fred"
    ingestion_time = datetime.utcnow()

    session = requests.Session()
    total_rows = 0

    total_series = len(series_list)
    for idx, spec in enumerate(series_list, start=1):
        series_id = spec["series_id"]
        instrument_type = spec.get("instrument_type")
        tenor = spec.get("tenor")
        units = spec.get("units")

        log(f"Pulling FRED series {series_id} ({idx}/{total_series})...")
        t0 = time.time()
        try:
            df = fetch_fred_series(session, series_id, FRED_START, FRED_END)
        except requests.RequestException as exc:
            log(f"  Failed to fetch {series_id}: {exc}")
            continue
        elapsed = time.time() - t0
        if df.empty:
            log(f"  No rows for {series_id} (elapsed {elapsed:.1f}s)")
            continue
        log(f"  Fetched {len(df):,} rows for {series_id} in {elapsed:.1f}s")

        raw_records: List[Dict[str, Any]] = []
        canonical_records: List[Dict[str, Any]] = []

        for _, row in df.iterrows():
            event_time = pd.to_datetime(row["date"])
            # FRED publication time is not explicit; use next-day availability.
            available_time = event_time + timedelta(days=1)

            value = row.get("value")
            if value is None or pd.isna(value):
                continue

            entity_id = series_id
            instrument_id = series_id

            payload = {
                "series_id": series_id,
                "date": normalize_value(event_time),
                "value": normalize_value(value),
                "instrument_type": instrument_type,
                "tenor": tenor,
                "units": units,
            }
            raw_payload_hash = compute_raw_payload_hash(payload)
            raw_version_id = compute_version_id(
                source_system=source_system,
                entity_id=entity_id,
                event_time=event_time.to_pydatetime(),
                available_time=available_time.to_pydatetime(),
                raw_payload_hash=raw_payload_hash,
            )

            raw_records.append(
                {
                    "entity_id": entity_id,
                    "company_id": None,
                    "security_id": None,
                    "event_time": event_time,
                    "available_time": available_time,
                    "payload": payload,
                }
            )

            canonical_records.append(
                {
                    "source_system": source_system,
                    "entity_id": entity_id,
                    "company_id": None,
                    "security_id": None,
                    "event_time": event_time,
                    "available_time": available_time,
                    "ingestion_time": ingestion_time,
                    "version_id": raw_version_id,
                    "raw_payload_hash": raw_payload_hash,
                    "upstream_version_ids": [raw_version_id],
                    "quality_flags": ["estimated_available_time"],
                    "instrument_id": instrument_id,
                    "instrument_type": instrument_type,
                    "tenor": tenor,
                    "value": float(value),
                    "units": units,
                }
            )

        if raw_records:
            log(f"  Writing raw records for {series_id}...")
            t1 = time.time()
            write_raw_records(source_system=source_system, records=raw_records)
            log(f"  Wrote raw records for {series_id} in {time.time() - t1:.1f}s")
        if canonical_records:
            log(f"  Appending canonical records for {series_id}...")
            t2 = time.time()
            append_canonical_records("warehouse_macro", canonical_records)
            log(f"  Appended canonical records for {series_id} in {time.time() - t2:.1f}s")
            total_rows += len(canonical_records)
            log(f"  Ingested {len(canonical_records):,} rows for {series_id}")

        time.sleep(FRED_SLEEP)

    log(f"Done. Total rows ingested: {total_rows:,}")


if __name__ == "__main__":
    main()
