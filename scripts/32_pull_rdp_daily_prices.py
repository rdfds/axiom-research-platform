#!/usr/bin/env python
"""
Pull Refinitiv (RDP) daily prices and map to permno by date.

Writes partitioned parquet to:
  data/warehouse/warehouse_prices_daily_rdp/year=YYYY/part_*.parquet

Env:
  RDP_START=2000-01-01
  RDP_END=2025-12-31
  RDP_BATCH=50
  RDP_SLEEP=0.2
  RDP_SPLIT_ON_ERROR=1
"""

import os
import time
import uuid
import concurrent.futures as futures
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import refinitiv.data as rd

DATA_DIR = Path(__file__).parent.parent / "data"
CRSP_DIR = DATA_DIR / "wrds" / "crsp"
REF_DIR = DATA_DIR / "refinitiv"
OUT_DIR = DATA_DIR / "warehouse" / "warehouse_prices_daily_rdp"

RDP_START = os.getenv("RDP_START", "2000-01-01")
RDP_END = os.getenv("RDP_END", datetime.utcnow().strftime("%Y-%m-%d"))
RDP_UNIVERSE_DATE = os.getenv("RDP_UNIVERSE_DATE", RDP_END)
BATCH_SIZE = int(os.getenv("RDP_BATCH", "50"))
SLEEP = float(os.getenv("RDP_SLEEP", "0.2"))
SPLIT_ON_ERROR = os.getenv("RDP_SPLIT_ON_ERROR", "1") == "1"
RDP_TIMEOUT = float(os.getenv("RDP_TIMEOUT", "120"))
RDP_LIMIT = int(os.getenv("RDP_LIMIT", "0"))
RDP_DEBUG = os.getenv("RDP_DEBUG", "0") == "1"
RDP_RELAX_NAME_FILTER = os.getenv("RDP_RELAX_NAME_FILTER", "0") == "1"


def log(msg: str) -> None:
    print(msg, flush=True)


def year_chunks(start: str, end: str):
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    for year in range(start_dt.year, end_dt.year + 1):
        chunk_start = max(pd.Timestamp(year=year, month=1, day=1), start_dt)
        chunk_end = min(pd.Timestamp(year=year, month=12, day=31), end_dt)
        yield chunk_start, chunk_end


def load_ric_map() :
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


