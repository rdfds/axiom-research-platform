#!/usr/bin/env python
"""
Probe + Pull Refinitiv Corporate Actions (US Active, 2000-present)
==================================================================
This script:
1) Probes which corporate-action fields are available in your entitlement.
2) Builds a US active equity universe (attempts full universe; falls back to indices).
3) Pulls corporate actions for that universe from 2000-01-01 through 2026-02-02.

Outputs:
  data/refinitiv/universe_us_active.parquet
  data/refinitiv/corporate_actions/ca_<year>_part_<batch>.parquet
  data/refinitiv/corporate_actions/manifest.jsonl

Run:
  python -u scripts/18_probe_and_pull_refinitiv_corp_actions.py
"""

import json
import time
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from pandas.api.types import is_string_dtype
import refinitiv.data as rd


DATA_DIR = Path(__file__).parent.parent / "data" / "refinitiv"
DATA_DIR.mkdir(parents=True, exist_ok=True)

UNIVERSE_PATH = DATA_DIR / "universe_us_active.parquet"
CA_DIR = DATA_DIR / "corporate_actions"
CA_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = CA_DIR / "manifest.jsonl"

START_DATE = "2000-01-01"
END_DATE = "2026-02-02"  # Current date in this workspace

BATCH_SIZE = 75
SLEEP_SECONDS = 0.4

CA_FIELD_CANDIDATES = [
    "TR.CAActionType",
    "TR.CAEventType",
    "TR.CAType",
    "TR.CAAnnouncementDate",
    "TR.CAEffectiveDate",
    "TR.CAExDate",
    "TR.CARecordDate",
    "TR.CAPayDate",
    "TR.CAAmount",
    "TR.CAAdjustmentFactor",
    "TR.CAAdjustmentType",
    "TR.CACurrency",
    "TR.CARatio",
    "TR.CAStatus",
    "TR.CAAmount",
    "TR.CAAmountGross",
    "TR.CAAmountNet",
    "TR.CAExAmount",
    "TR.CAShareFactor",
    "TR.CAValue",
]

# If CAType is required, we will try these. Add/remove as needed.
CA_TYPE_CANDIDATES = [
    "DIV",  # Dividends (may or may not be valid)
    "DVD",
    "SSP",  # Stock splits
    "RHT",  # Rights
    "SPI",  # Spinoff
    "BON",  # Bonus issue
    "CAP",  # Capital change
    "MER",  # Merger
    "TND",  # Tender
    "LIQ",  # Liquidation
]

# Safety / quality controls
ABORT_IF_EMPTY_PROBE = True
MIN_EVENT_ROWS = 5
MIN_EVENT_RATIO = 0.01
MIN_DATE_EVENT_ROWS = 3
MIN_DATE_EVENT_RATIO = 0.01
MAX_CONSECUTIVE_EMPTY_BATCHES = 10


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def batched(items: List[str], batch_size: int) -> List[List[str]]:
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def try_screen_universe() -> Tuple[List[str], Dict[str, str]]:
    """
    Attempt to build a full US active equity universe using Refinitiv Screen.
    Returns (tickers, meta). meta is empty if no success.
    """
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
            if len(tickers) < 500:
                log(f"Screen {label} returned only {len(tickers)} tickers (too small).")
                continue
            return tickers, {"method": "screen", "label": label}
        except Exception as e:
            log(f"Screen {label} failed: {e}")

    return [], {}


def universe_from_indices() -> Tuple[List[str], Dict[str, str]]:
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


def build_universe() -> List[str]:
    if UNIVERSE_PATH.exists():
        log(f"Loading cached universe from {UNIVERSE_PATH.name}...")
        df = pd.read_parquet(UNIVERSE_PATH)
        return df["ric"].dropna().unique().tolist()

    log("Building US active equity universe...")
    tickers, meta = try_screen_universe()

    if not tickers:
        log("Full screen universe failed. Falling back to index constituents.")
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
    return df["ric"].tolist()


def probe_ca_fields(sample_tickers: List[str], start_date: str, end_date: str) -> List[str]:
    log("Probing corporate action fields...")
    working_fields = []
    for field in CA_FIELD_CANDIDATES:
        try:
            _ = rd.get_data(
                universe=sample_tickers,
                fields=[field],
                parameters={"SDate": start_date, "EDate": end_date},
            )
            working_fields.append(field)
        except Exception as e:
            log(f"Field not available: {field} ({e})")

    if not working_fields:
        raise RuntimeError("No corporate-action fields are available with current entitlements.")

    log(f"Working CA fields: {working_fields}")
    return working_fields


def probe_ca_type_requirement(sample_tickers: List[str], fields: List[str], start_date: str, end_date: str) -> bool:
    log("Probing whether CAType parameter is required...")
    try:
        _ = rd.get_data(
            universe=sample_tickers,
            fields=fields,
            parameters={"SDate": start_date, "EDate": end_date},
        )
        log("CAType not required.")
        return False
    except Exception as e:
        msg = str(e).lower()
        if "catype" in msg or "ca type" in msg:
            log("CAType appears to be required.")
            return True
        log(f"CAType probe failed for other reason: {e}")
        raise


def _event_columns(df: pd.DataFrame) -> List[str]:
    keywords = [
        "announcement date",
        "effective date",
        "ex date",
        "record date",
        "pay date",
        "adjustment factor",
        "adjustment type",
        "action type",
        "event type",
        "amount",
        "ratio",
        "currency",
        "status",
    ]
    cols = []
    for c in df.columns:
        cl = c.lower()
        if any(k in cl for k in keywords):
            cols.append(c)
    return cols


def _event_signal_columns(df: pd.DataFrame, event_cols: List[str]) -> List[str]:
    date_cols = [c for c in event_cols if "date" in c.lower()]
    value_keywords = ["amount", "ratio", "value", "currency", "factor", "share factor"]
    value_cols = [c for c in event_cols if any(k in c.lower() for k in value_keywords)]
    return date_cols + value_cols


