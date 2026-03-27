#!/usr/bin/env python
"""
Pull Refinitiv Estimates with publish time + period end (A4 spec) and
ingest into warehouse_estimates.

This script probes for available date fields (publish time and period end)
and aborts if it cannot find both (to keep A4 spec-compliant).

Env:
  EST_START=2000-01-01
  EST_END=YYYY-MM-DD (default: today UTC)
  EST_PERIODS=FY1,FY2,NTM
  EST_SAMPLE=50
  EST_BATCH=75
  EST_SLEEP=0.2
  EST_PROBE_ONLY=0
  EST_TICKERS=comma,separated,override list
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import refinitiv.data as rd

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion import (
    append_canonical_records,
    compute_raw_payload_hash,
    compute_version_id,
    write_raw_records,
)


DATA_DIR = Path(__file__).parent.parent / "data"
CRSP_DIR = DATA_DIR / "wrds" / "crsp"
REF_DIR = DATA_DIR / "refinitiv"

EST_START = os.getenv("EST_START", "2000-01-01")
EST_END = os.getenv("EST_END", datetime.utcnow().strftime("%Y-%m-%d"))
EST_PERIODS = [p.strip() for p in os.getenv("EST_PERIODS", "FY1,FY2,NTM").split(",") if p.strip()]
EST_SAMPLE = int(os.getenv("EST_SAMPLE", "50"))
EST_BATCH = int(os.getenv("EST_BATCH", "75"))
EST_SLEEP = float(os.getenv("EST_SLEEP", "0.2"))
EST_PROBE_ONLY = os.getenv("EST_PROBE_ONLY", "0") == "1"
EST_TICKERS = os.getenv("EST_TICKERS")


VALUE_FIELDS = {
    "eps": ["TR.EPSMean"],
    "revenue": ["TR.RevenueMean"],
    "ebitda": ["TR.EBITDAMean"],
}

NUM_FIELDS = [
    "TR.NumberOfEstimates",
    "TR.NumOfEstimates",
]

PUBLISH_FIELDS = [
    "TR.EstimateDate",
    "TR.EstimateDateTime",
    "TR.EPSMeanDate",
    "TR.EPSMeanDateTime",
    "TR.EPSMeanLastUpdated",
    "TR.LastUpdateDate",
    "TR.LastUpdateDateTime",
]

PERIOD_FIELDS = [
    "TR.EPSMeanPeriodEndDate",
    "TR.EPSPeriodEndDate",
    "TR.PeriodEndDate",
    "TR.FiscalPeriodEndDate",
    "TR.FiscalPeriodEnd",
    "TR.FYEndDate",
]


def log(msg: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def load_universe() -> List[str]:
    if EST_TICKERS:
        return [t.strip() for t in EST_TICKERS.split(",") if t.strip()]
    path = REF_DIR / "universe_us_active.parquet"
    if not path.exists():
        raise FileNotFoundError("Missing universe_us_active.parquet")
    df = pd.read_parquet(path)
    if "ric" in df.columns:
        rics = df["ric"].dropna().astype("string").str.upper().tolist()
    else:
        rics = df.iloc[:, 0].dropna().astype("string").str.upper().tolist()
    return rics


def load_ric_map() -> pd.DataFrame:
    ric_map_path = REF_DIR / "ric_to_cusip_map.parquet"
    if not ric_map_path.exists():
        raise FileNotFoundError("Missing ric_to_cusip_map.parquet")
    ric_map = pd.read_parquet(ric_map_path)
    ric_map["ric"] = ric_map["ric"].astype("string").str.upper().str.strip()
    ric_map["cusip8"] = (
        ric_map["cusip"]
        .astype("string")
        .str.replace(r"[^0-9A-Za-z]", "", regex=True)
        .str.upper()
        .str[:8]
    )
    ric_map = ric_map[ric_map["cusip8"].notna()]
    ric_map = ric_map.drop_duplicates("ric")
    return ric_map[["ric", "cusip8", "ticker"]]


def load_names() -> pd.DataFrame:
    names_path = CRSP_DIR / "msenames_2000-01-01_to_2026-12-31.parquet"
    if not names_path.exists():
        raise FileNotFoundError("Missing msenames_2000-01-01_to_2026-12-31.parquet")
    names = pd.read_parquet(
        names_path,
        columns=["permno", "permco", "namedt", "nameendt", "ncusip", "cusip"],
    )
    names["namedt"] = pd.to_datetime(names["namedt"], errors="coerce")
    names["nameendt"] = pd.to_datetime(names["nameendt"], errors="coerce")
    names["cusip8"] = (
        names["ncusip"]
        .fillna(names["cusip"])
        .astype("string")
        .str.replace(r"[^0-9A-Za-z]", "", regex=True)
        .str.upper()
        .str[:8]
    )
    names = names[names["cusip8"].notna()]
    # Extend CRSP end date to cover estimate end date for mapping
    max_end = names["nameendt"].max()
    target_end = pd.to_datetime(EST_END, errors="coerce")
    if pd.notna(max_end) and pd.notna(target_end) and target_end > max_end:
        names.loc[names["nameendt"] == max_end, "nameendt"] = target_end
    return names[["permno", "permco", "namedt", "nameendt", "cusip8"]]


