#!/usr/bin/env python
"""
Pull earnings call transcripts from FMP (B1) and ingest into canonical tables.

Requires:
  export FMP_API_KEY="..."

Writes (partitioned by year):
  data/warehouse/warehouse_documents
  data/warehouse/warehouse_doc_chunks
  data/warehouse/warehouse_text_signals

Raw payloads are stored in data/lake/raw/fmp_transcripts.

Env:
  FMP_API_KEY (required)
  FMP_BASE_URL=https://financialmodelingprep.com/stable
  FMP_SLEEP=0.2
  FMP_RETRIES=2
  FMP_START_YEAR=2000
  FMP_END_YEAR=YYYY
  FMP_LIMIT_SYMBOLS=0 (0 = all)
  FMP_MAX_TRANSCRIPTS_PER_SYMBOL=0 (0 = all)
  FMP_TARGET_SYMBOL= (optional)
  FMP_RESUME=1
  FMP_FLUSH_EVERY=200
  FMP_USE_UNIVERSE=1 (use R3000 proxy tickers)
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion import compute_raw_payload_hash, compute_version_id, write_raw_records
from src.text_processing import chunk_text, ensure_list, extract_signals, write_partitioned


DATA_DIR = Path(__file__).parent.parent / "data"
FMP_DIR = DATA_DIR / "fmp"
CRSP_DIR = DATA_DIR / "wrds" / "crsp"

FMP_API_KEY = os.getenv("FMP_API_KEY")
FMP_BASE_URL = os.getenv("FMP_BASE_URL", "https://financialmodelingprep.com/stable").rstrip("/")
FMP_SLEEP = float(os.getenv("FMP_SLEEP", "0.2"))
FMP_RETRIES = int(os.getenv("FMP_RETRIES", "2"))
FMP_TIMEOUT = float(os.getenv("FMP_TIMEOUT", "30"))
FMP_START_YEAR = int(os.getenv("FMP_START_YEAR", "2000"))
FMP_END_YEAR = int(os.getenv("FMP_END_YEAR", datetime.utcnow().year))
FMP_LIMIT_SYMBOLS = int(os.getenv("FMP_LIMIT_SYMBOLS", "0"))
FMP_MAX_TRANSCRIPTS_PER_SYMBOL = int(os.getenv("FMP_MAX_TRANSCRIPTS_PER_SYMBOL", "0"))
FMP_TARGET_SYMBOL = os.getenv("FMP_TARGET_SYMBOL")
FMP_RESUME = os.getenv("FMP_RESUME", "1") == "1"
FMP_FLUSH_EVERY = int(os.getenv("FMP_FLUSH_EVERY", "200"))
FMP_USE_UNIVERSE = os.getenv("FMP_USE_UNIVERSE", "1") == "1"
FMP_DEBUG = os.getenv("FMP_DEBUG", "0") == "1"
FMP_HEARTBEAT_SECS = int(os.getenv("FMP_HEARTBEAT_SECS", "60"))

CHUNK_TOKENS = int(os.getenv("TRANSCRIPT_CHUNK_TOKENS", "400"))
CHUNK_MIN = int(os.getenv("TRANSCRIPT_CHUNK_MIN", "300"))
CHUNK_MAX = int(os.getenv("TRANSCRIPT_CHUNK_MAX", "500"))


def log(msg: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def require_api_key() -> str:
    if not FMP_API_KEY:
        raise RuntimeError("FMP_API_KEY not set. Export your FMP API key.")
    return FMP_API_KEY


def _safe_params(params: Dict[str, object]) -> Dict[str, object]:
    safe = dict(params)
    if "apikey" in safe:
        safe["apikey"] = "***REDACTED***"
    return safe


def _request_json(url: str, params: Dict[str, object], session: requests.Session) -> Optional[List[Dict]]:
    for attempt in range(FMP_RETRIES + 1):
        try:
            if FMP_DEBUG:
                log(f"[debug] GET {url} params={_safe_params(params)}")
            resp = session.get(url, params=params, timeout=FMP_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            if FMP_SLEEP:
                time.sleep(FMP_SLEEP)
            return data
        except requests.RequestException as exc:
            if attempt < FMP_RETRIES:
                time.sleep(max(FMP_SLEEP, 0.2))
                continue
            log(f"Request failed: {url} {exc}")
            return None
    return None


def load_universe_tickers() -> List[str]:
    universe_path = DATA_DIR / "curated" / "universe_r3000_proxy.parquet"
    names_path = CRSP_DIR / "msenames_2000-01-01_to_2026-12-31.parquet"
    if not universe_path.exists() or not names_path.exists():
        return []
    universe = pd.read_parquet(universe_path)
    universe["date"] = pd.to_datetime(universe["date"])
    asof_date = universe["date"].max()
    universe = universe[universe["date"] == asof_date][["permno"]]

    names = pd.read_parquet(names_path, columns=["permno", "namedt", "nameendt", "ticker"])
    names["namedt"] = pd.to_datetime(names["namedt"])
    names["nameendt"] = pd.to_datetime(names["nameendt"])
    active = names[(names["namedt"] <= asof_date) & (names["nameendt"] >= asof_date)]
    active = active.sort_values(["permno", "nameendt"])
    latest = active.drop_duplicates(subset=["permno"], keep="last")
    merged = universe.merge(latest, on="permno", how="left")
    tickers = (
        merged["ticker"]
        .dropna()
        .astype(str)
        .str.upper()
        .str.strip()
        .tolist()
    )
    return sorted(set(tickers))


def load_symbol_list(session: requests.Session) -> List[str]:
    if FMP_TARGET_SYMBOL:
        return [FMP_TARGET_SYMBOL.upper()]

    symbols: List[str] = []
    if FMP_USE_UNIVERSE:
        symbols = load_universe_tickers()
        if symbols:
            log(f"Using universe tickers: {len(symbols):,}")

    url = f"{FMP_BASE_URL}/earnings-transcript-list"
    data = _request_json(url, params={"apikey": FMP_API_KEY}, session=session)
    if data:
        avail = [row.get("symbol") for row in data if row.get("symbol")]
        avail = [str(s).upper() for s in avail]
        if symbols:
            symbols = sorted(set(symbols).intersection(set(avail)))
        else:
            symbols = sorted(set(avail))

    if FMP_LIMIT_SYMBOLS and FMP_LIMIT_SYMBOLS > 0:
        symbols = symbols[:FMP_LIMIT_SYMBOLS]
    return symbols


def load_symbol_dates(symbol: str, session: requests.Session) -> List[Dict]:
    url = f"{FMP_BASE_URL}/earning-call-transcript-dates"
    data = _request_json(url, params={"symbol": symbol, "apikey": FMP_API_KEY}, session=session)
    if not data:
        if FMP_DEBUG:
            log(f"[debug] dates empty for {symbol}")
        return []
    if isinstance(data, dict) and data.get("Error Message"):
        if FMP_DEBUG:
            log(f"[debug] dates error for {symbol}: {data.get('Error Message')}")
        return []
    return data


