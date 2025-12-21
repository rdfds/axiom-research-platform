#!/usr/bin/env python
"""
Backfill GVKEYs for FMP financials ingested with company_id = symbol.

This script appends new canonical records with company_id/entity_id set to gvkey,
leaving original symbol-based records intact (append-only).

Env:
  BACKFILL_START_YEAR=1998
  BACKFILL_END_YEAR=2026
  BACKFILL_RESUME=1
  BACKFILL_FLUSH_EVERY=50000
  BACKFILL_DEBUG=0
"""

from __future__ import annotations

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion import append_canonical_records, compute_version_id


DATA_DIR = Path(__file__).parent.parent / "data"
WAREHOUSE_DIR = DATA_DIR / "warehouse"
CRSP_DIR = DATA_DIR / "wrds" / "crsp"
COMP_DIR = DATA_DIR / "wrds" / "compustat"
SEC_DIR = DATA_DIR / "sec"

BACKFILL_START_YEAR = int(os.getenv("BACKFILL_START_YEAR", "1998"))
BACKFILL_END_YEAR = int(os.getenv("BACKFILL_END_YEAR", "2026"))
BACKFILL_RESUME = os.getenv("BACKFILL_RESUME", "1") == "1"
BACKFILL_FLUSH_EVERY = int(os.getenv("BACKFILL_FLUSH_EVERY", "50000"))
BACKFILL_DEBUG = os.getenv("BACKFILL_DEBUG", "0") == "1"


def log(msg: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def load_mappings() -> tuple[pd.DataFrame, pd.DataFrame]:
    names_path = CRSP_DIR / "msenames_2000-01-01_to_2026-12-31.parquet"
    link_path = CRSP_DIR / "ccmxpf_lnkhist.parquet"
    if not names_path.exists() or not link_path.exists():
        raise FileNotFoundError("Missing CRSP mapping files for gvkey backfill.")

    names = pd.read_parquet(names_path, columns=["permno", "namedt", "nameendt", "ticker"])
    try:
        link = pd.read_parquet(link_path, columns=["permno", "gvkey", "linkdt", "linkenddt"])
    except Exception:
        link = pd.read_parquet(link_path, columns=["lpermno", "gvkey", "linkdt", "linkenddt"])
        link = link.rename(columns={"lpermno": "permno"})

    names["namedt"] = pd.to_datetime(names["namedt"], errors="coerce")
    names["nameendt"] = pd.to_datetime(names["nameendt"], errors="coerce")
    link["linkdt"] = pd.to_datetime(link["linkdt"], errors="coerce")
    link["linkenddt"] = pd.to_datetime(link["linkenddt"], errors="coerce")

    names["ticker_norm"] = names["ticker"].astype("string").str.upper().str.replace("-", ".", regex=False)
    link["permno"] = pd.to_numeric(link["permno"], errors="coerce")
    return names, link


def load_cik_gvkey() -> pd.DataFrame:
    path = COMP_DIR / "cik_gvkey.csv.gz"
    if not path.exists():
        raise FileNotFoundError("Missing Compustat CIK-GVKEY file at data/wrds/compustat/cik_gvkey.csv.gz")
    df = pd.read_csv(path, dtype=str)
    df.columns = [c.lower() for c in df.columns]
    # Normalize
    df["cik"] = df["cik"].astype(str).str.replace(r"^0+", "", regex=True)
    df["gvkey"] = df["gvkey"].astype(str).str.strip()
    if "link_start_date" in df.columns:
        df["link_start_date"] = pd.to_datetime(df["link_start_date"], errors="coerce")
    if "link_end_date" in df.columns:
        df["link_end_date"] = pd.to_datetime(df["link_end_date"], errors="coerce")
    return df


