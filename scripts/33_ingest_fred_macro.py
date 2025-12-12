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


