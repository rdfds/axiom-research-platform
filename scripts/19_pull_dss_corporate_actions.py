#!/usr/bin/env python
"""
Pull DSS Corporate Actions (US Active, 2000-present)
====================================================
Uses DataScope Select (DSS) REST API Corporate Actions extraction.

Requirements:
  - DSS token OR DSS username/password via env vars
  - Universe file from prior RIC build (defaults to data/refinitiv/universe_us_active.parquet)

Env vars:
  DSS_URL=https://selectapi.datascope.lseg.com/RestApi/v1
  DSS_TOKEN=...
  DSS_USERNAME=...
  DSS_PASSWORD=...
  DSS_APP_KEY=... (optional, if required by your DSS setup)
  DSS_REBUILD_UNIVERSE=0|1
  DSS_REQUIRE_FULL_UNIVERSE=0|1
  DSS_SCREEN_MIN_COUNT=500
  DSS_USE_ALL_FIELDS=0|1
  DSS_FIELD_NAME_SOURCE=Name|Code
  DSS_FIELDS_FILE=path/to/fields.txt|json
  DSS_CONDITION_FILE=path/to/condition.json
  DSS_BATCH_SIZE=200
  DSS_SMOKE_TEST=0|1
  DSS_SMOKE_TICKERS=50
  DSS_SMOKE_DAYS=30
  DSS_SMOKE_MAX_BATCHES=1
  DSS_START_DATE=2000-01-01
  DSS_END_DATE=2026-02-02

Outputs:
  data/refinitiv/corporate_actions_dss/ca_<year>_part_<batch>.parquet
  data/refinitiv/corporate_actions_dss/manifest.jsonl
"""

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests


DATA_DIR = Path(__file__).parent.parent / "data" / "refinitiv"
DATA_DIR.mkdir(parents=True, exist_ok=True)

UNIVERSE_PATH = Path(os.getenv("DSS_UNIVERSE_FILE", DATA_DIR / "universe_us_active.parquet"))
OUT_DIR = DATA_DIR / "corporate_actions_dss"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = OUT_DIR / "manifest.jsonl"

DSS_URL = os.getenv("DSS_URL", "https://selectapi.datascope.lseg.com/RestApi/v1").rstrip("/")
DSS_TOKEN = os.getenv("DSS_TOKEN")
DSS_USERNAME = os.getenv("DSS_USERNAME")
DSS_PASSWORD = os.getenv("DSS_PASSWORD")
DSS_APP_KEY = os.getenv("DSS_APP_KEY")

DSS_REBUILD_UNIVERSE = os.getenv("DSS_REBUILD_UNIVERSE", "0") == "1"
DSS_REQUIRE_FULL_UNIVERSE = os.getenv("DSS_REQUIRE_FULL_UNIVERSE", "0") == "1"
DSS_SCREEN_MIN_COUNT = int(os.getenv("DSS_SCREEN_MIN_COUNT", "500"))

DSS_USE_ALL_FIELDS = os.getenv("DSS_USE_ALL_FIELDS", "0") == "1"
DSS_FIELD_NAME_SOURCE = os.getenv("DSS_FIELD_NAME_SOURCE", "Name")  # Name or Code
DSS_FIELDS_FILE = os.getenv("DSS_FIELDS_FILE")
DSS_CONDITION_FILE = os.getenv("DSS_CONDITION_FILE")

START_DATE = os.getenv("DSS_START_DATE", "2000-01-01")
END_DATE = os.getenv("DSS_END_DATE", "2026-02-02")

BATCH_SIZE = int(os.getenv("DSS_BATCH_SIZE", "200"))
SLEEP_SECONDS = 0.4
POLL_SECONDS = 4
MAX_POLL = 120

SMOKE_TEST = os.getenv("DSS_SMOKE_TEST", "0") == "1"
SMOKE_TICKERS = int(os.getenv("DSS_SMOKE_TICKERS", "50"))
SMOKE_DAYS = int(os.getenv("DSS_SMOKE_DAYS", "30"))
SMOKE_MAX_BATCHES = int(os.getenv("DSS_SMOKE_MAX_BATCHES", "1"))

DEFAULT_FIELDS = [
    "Corporate Actions Type",
    "Corporate Actions Type Description",
    "Capital Change Event Type",
    "Capital Change Event Type Description",
    "Actual Adjustment Type",
    "Actual Adjustment Type Description",
    "Adjustment Factor",
    "Currency Code",
    "Exchange Code",
    "Effective Date",
    "Dividend Pay Date",
    "Dividend Rate",
    "Nominal Value",
    "Nominal Value Currency",
    "Nominal Value Date",
]


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_universe() -> List[str]:
    if DSS_REBUILD_UNIVERSE:
        rebuild_universe()
    if not UNIVERSE_PATH.exists():
        raise FileNotFoundError(
            f"Universe file not found: {UNIVERSE_PATH}. "
            "Run scripts/18_probe_and_pull_refinitiv_corp_actions.py to build it, "
            "or set DSS_UNIVERSE_FILE."
        )
    suffix = UNIVERSE_PATH.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(UNIVERSE_PATH)
        if "ric" not in df.columns and "RIC" not in df.columns:
            raise ValueError(f"Universe file missing 'ric' column: {UNIVERSE_PATH}")
        col = "ric" if "ric" in df.columns else "RIC"
        tickers = df[col].dropna().unique().tolist()
        return tickers
    if suffix in {".csv"}:
        df = pd.read_csv(UNIVERSE_PATH)
        if "ric" in df.columns or "RIC" in df.columns:
            col = "ric" if "ric" in df.columns else "RIC"
            return df[col].dropna().unique().tolist()
        # fallback to first column
        return df.iloc[:, 0].dropna().unique().tolist()
    if suffix in {".txt"}:
        with open(UNIVERSE_PATH, "r") as f:
            return [line.strip() for line in f if line.strip()]
    raise ValueError(f"Unsupported universe file type: {UNIVERSE_PATH}")


def try_screen_universe() -> Tuple[List[str], Dict[str, str]]:
    import refinitiv.data as rd

    screen_queries = [
        ("SCREEN(U(IN(Equity)), TR.ExchangeCountry='United States' AND TR.Status='Active')", "exchange_country_status"),
        ("SCREEN(U(IN(Equity)), TR.ExchangeCountry='United States')", "exchange_country"),
        ("SCREEN(U(IN(Equity)), TR.CountryOfIncorporation='United States' AND TR.Status='Active')", "incorporation_country_status"),
        ("SCREEN(U(IN(Equity)), TR.CountryOfIncorporation='United States')", "incorporation_country"),
    ]

    for screen, label in screen_queries:
        try:
            df = rd.get_data(universe=screen, fields=["TR.CommonName"])
            if df is None or len(df) == 0:
                log(f"Screen {label} returned no rows.")
                continue
            tickers = df["Instrument"].dropna().unique().tolist()
            if len(tickers) < DSS_SCREEN_MIN_COUNT:
                log(f"Screen {label} returned only {len(tickers)} tickers (too small).")
                continue
            return tickers, {"method": "screen", "label": label}
        except Exception as e:
            log(f"Screen {label} failed: {e}")

    return [], {}


def universe_from_indices() -> Tuple[List[str], Dict[str, str]]:
    import refinitiv.data as rd

    index_universes = [
        "0#.SPX",
        "0#.MID",
        "0#.SML",
        "0#.RUI",
        "0#.RUT",
        "0#.RUA",
        "0#.NDX",
    ]
    tickers: List[str] = []

    for idx in index_universes:
        try:
            df = rd.get_data(universe=idx, fields=["TR.CommonName"])
            if df is None or len(df) == 0:
                continue
            tickers.extend(df["Instrument"].dropna().tolist())
        except Exception as e:
            log(f"Index universe {idx} failed: {e}")

    tickers = sorted(list(set(tickers)))
    return tickers, {"method": "indices", "label": ",".join(index_universes)}


def rebuild_universe() -> None:
    import refinitiv.data as rd

    log("Rebuilding US active equity universe...")
    rd.open_session()
    try:
        tickers, meta = try_screen_universe()
        if not tickers:
            if DSS_REQUIRE_FULL_UNIVERSE:
                raise RuntimeError(
                    "Screen universe failed or returned too few tickers. "
                    "Set DSS_REQUIRE_FULL_UNIVERSE=0 to allow index fallback, "
                    "or provide a custom universe file."
                )
            log("Screen universe failed. Falling back to index constituents.")
            tickers, meta = universe_from_indices()

        if not tickers:
            raise RuntimeError("Failed to build any universe from Refinitiv.")

        df = pd.DataFrame({
            "ric": sorted(list(set(tickers))),
            "source_method": meta.get("method", "unknown"),
            "source_label": meta.get("label", "unknown"),
            "pulled_at": datetime.now().isoformat(),
        })
        df.to_parquet(UNIVERSE_PATH, index=False)
        log(f"Saved universe: {len(df):,} tickers -> {UNIVERSE_PATH.name}")
    finally:
        rd.close_session()


