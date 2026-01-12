#!/usr/bin/env python
"""
A4-lite: Daily estimates snapshots via Refinitiv Data Library (Workspace session),
with inferred period end using Compustat FY end.

This is NOT fully spec-compliant (no publish_time). We mark:
  - partial_coverage
  - estimated_available_time
  - estimated_period_end

Env:
  EST_PERIODS=FY1,FY2,NTM
  EST_BATCH=75
  EST_SLEEP=0.2
  EST_UNIVERSE_FILE=data/refinitiv/universe_us_active.parquet
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
import warnings
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
REF_DIR = DATA_DIR / "refinitiv"
CRSP_DIR = DATA_DIR / "wrds" / "crsp"
WAREHOUSE_DIR = DATA_DIR / "warehouse"

EST_PERIODS = [p.strip() for p in os.getenv("EST_PERIODS", "FY1,FY2,NTM").split(",") if p.strip()]
EST_BATCH = int(os.getenv("EST_BATCH", "75"))
EST_SLEEP = float(os.getenv("EST_SLEEP", "0.2"))
EST_UNIVERSE_FILE = Path(os.getenv("EST_UNIVERSE_FILE", str(REF_DIR / "universe_us_active.parquet")))
EST_DEBUG = os.getenv("EST_DEBUG", "0") == "1"


FIELD_ALIASES = {
    "eps": ["TR.EPSMean", "Earnings Per Share - Mean", "EPS Mean"],
    "revenue": ["TR.RevenueMean", "Revenue - Mean", "Revenue Mean"],
    "ebitda": ["TR.EBITDAMean", "EBITDA - Mean", "EBITDA Mean"],
}
NUM_FIELDS = [
    "TR.NumberOfEstimates",
    "TR.NumOfEstimates",
    "Number of Estimates",
    "Num of Estimates",
]
REQUEST_FIELDS = [
    "TR.EPSMean",
    "TR.RevenueMean",
    "TR.EBITDAMean",
    "TR.NumberOfEstimates",
    "TR.NumOfEstimates",
]


def log(msg: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def load_universe() -> List[str]:
    if not EST_UNIVERSE_FILE.exists():
        raise FileNotFoundError(f"Missing universe file: {EST_UNIVERSE_FILE}")
    df = pd.read_parquet(EST_UNIVERSE_FILE)
    if "ric" in df.columns:
        return df["ric"].dropna().astype("string").str.upper().tolist()
    return df.iloc[:, 0].dropna().astype("string").str.upper().tolist()


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
    ric_map = ric_map[["ric", "cusip8", "ticker"]]

    # Optional: enrich mapping with permid map if available
    permid_path = REF_DIR / "permid_map.parquet"
    if permid_path.exists():
        permid = pd.read_parquet(permid_path, columns=["ric", "cusip", "ticker"])
        permid["ric"] = permid["ric"].astype("string").str.upper().str.strip()
        permid["cusip8"] = (
            permid["cusip"]
            .astype("string")
            .str.replace(r"[^0-9A-Za-z]", "", regex=True)
            .str.upper()
            .str[:8]
        )
        permid = permid[permid["ric"].notna() & permid["cusip8"].notna()]
        permid = permid.drop_duplicates("ric")
        permid = permid[["ric", "cusip8", "ticker"]]
        ric_map = pd.concat([ric_map, permid], ignore_index=True)
        ric_map = ric_map.drop_duplicates("ric")

    return ric_map


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
    # Extend CRSP end date to cover "today" for mapping
    max_end = names["nameendt"].max()
    today = pd.Timestamp(datetime.utcnow().date())
    if pd.notna(max_end) and today > max_end:
        names.loc[names["nameendt"] == max_end, "nameendt"] = today
    return names[["permno", "permco", "namedt", "nameendt", "cusip8"]]


