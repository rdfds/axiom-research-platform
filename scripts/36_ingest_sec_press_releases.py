#!/usr/bin/env python
"""
Ingest SEC 8-K press releases (public) into warehouse_press_releases.

This is the MVP B4 source. We use SEC submissions JSON and pull the
primary document for 8-K filings. We treat:
  event_time = reportDate if present else filingDate (estimated)
  available_time = acceptanceDateTime (SEC timestamp)

Env:
  SEC_USER_AGENT   (required)
  SEC_START=2000-01-01
  SEC_END=YYYY-MM-DD (default: today UTC)
  SEC_SLEEP=0.2
  SEC_LIMIT_CIKS=0 (0 = all)
  SEC_MAX_PER_CIK=200
  SEC_FETCH_TEXT=1 (0 = metadata only)
  SEC_UNIVERSE_DATE=YYYY-MM-DD (optional)
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
import time as _time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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
SEC_DIR = DATA_DIR / "sec"
CRSP_DIR = DATA_DIR / "wrds" / "crsp"

SEC_TICKER_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"

SEC_START = os.getenv("SEC_START", "2000-01-01")
SEC_END = os.getenv("SEC_END", datetime.utcnow().date().isoformat())
SEC_SLEEP = float(os.getenv("SEC_SLEEP", "0.2"))
SEC_TIMEOUT = float(os.getenv("SEC_TIMEOUT", "20"))
SEC_RETRIES = int(os.getenv("SEC_RETRIES", "2"))
SEC_LIMIT_CIKS = int(os.getenv("SEC_LIMIT_CIKS", "0"))
SEC_MAX_PER_CIK = int(os.getenv("SEC_MAX_PER_CIK", "200"))
SEC_FETCH_TEXT = os.getenv("SEC_FETCH_TEXT", "1") == "1"
SEC_UNIVERSE_DATE = os.getenv("SEC_UNIVERSE_DATE")
SEC_RESUME = os.getenv("SEC_RESUME", "1") == "1"
SEC_START_INDEX = int(os.getenv("SEC_START_INDEX", "0"))

WAREHOUSE_DIR = DATA_DIR / "warehouse"


def log(msg: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def require_user_agent() -> str:
    user_agent = os.getenv("SEC_USER_AGENT")
    if not user_agent:
        raise RuntimeError(
            "SEC_USER_AGENT not set. Example: export SEC_USER_AGENT='Axiom Research (you@example.com)'"
        )
    return user_agent


def ensure_dirs() -> None:
    SEC_DIR.mkdir(parents=True, exist_ok=True)
    (SEC_DIR / "submissions").mkdir(parents=True, exist_ok=True)


def fetch_json(url: str, session: requests.Session, sleep_seconds: float) -> Dict:
    last_exc: Optional[Exception] = None
    for attempt in range(SEC_RETRIES + 1):
        try:
            resp = session.get(url, timeout=SEC_TIMEOUT)
            resp.raise_for_status()
            if sleep_seconds:
                time.sleep(sleep_seconds)
            return resp.json()
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < SEC_RETRIES:
                time.sleep(max(sleep_seconds, 0.2))
                continue
            raise
    if last_exc:
        raise last_exc
    raise RuntimeError("Unknown SEC request error")


def load_sec_tickers(session: requests.Session, sleep_seconds: float) -> pd.DataFrame:
    cache_path = SEC_DIR / "company_tickers.json"
    if cache_path.exists():
        data = json.loads(cache_path.read_text())
    else:
        log("Downloading SEC ticker mapping...")
        data = fetch_json(SEC_TICKER_URL, session, sleep_seconds)
        cache_path.write_text(json.dumps(data))

    rows = []
    for _, row in data.items():
        cik = str(row["cik_str"]).zfill(10)
        rows.append(
            {
                "cik": cik,
                "ticker": str(row['ticker']).upper().strip(),
                "title": row.get("title"),
            }
        )
    return pd.DataFrame(rows).drop_duplicates()


def load_universe_tickers(universe_date: Optional[str]) -> pd.DataFrame:
    universe_path = DATA_DIR / "curated" / "universe_r3000_proxy.parquet"
    names_path = CRSP_DIR / "msenames_2000-01-01_to_2026-12-31.parquet"
    if not universe_path.exists():
        raise FileNotFoundError(f"Missing universe file: {universe_path}")
    if not names_path.exists():
        raise FileNotFoundError(f"Missing CRSP names file: {names_path}")

    universe = pd.read_parquet(universe_path)
    universe["date"] = pd.to_datetime(universe["date"])
    asof_date = pd.to_datetime(universe_date) if universe_date else universe["date"].max()
    universe = universe[universe["date"] == asof_date][["permno", "permco"]]

    names = pd.read_parquet(names_path, columns=["permno", "namedt", "nameendt", "ticker", "cusip", "comnam"])
    names["namedt"] = pd.to_datetime(names["namedt"])
    names["nameendt"] = pd.to_datetime(names["nameendt"])
    active = names[(names["namedt"] <= asof_date) & (names["nameendt"] >= asof_date)]
    active = active.sort_values(["permno", "nameendt"])
    latest = active.drop_duplicates(subset=["permno"], keep="last")
    merged = universe.merge(latest, on="permno", how="left")
    merged["ticker"] = merged["ticker"].str.upper().str.strip()
    return merged


