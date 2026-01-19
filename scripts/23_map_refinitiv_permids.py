#!/usr/bin/env python
"""
Map Refinitiv PermIDs to identifiers (RIC/CUSIP/ISIN/Ticker).
============================================================
Reads Refinitiv M&A deals and builds a PermID -> identifier lookup table.

Outputs:
  data/refinitiv/permid_map.parquet

Run:
  python -u scripts/23_map_refinitiv_permids.py
"""

import os
from pathlib import Path
from datetime import datetime
from typing import List
import sys
import time

import pandas as pd
import refinitiv.data as rd


DATA_DIR = Path(__file__).parent.parent / "data" / "refinitiv"
MAP_PATH = DATA_DIR / "permid_map.parquet"
PARTS_DIR = DATA_DIR / "permid_map_parts"
PARTS_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = int(os.getenv("PERMID_MAP_BATCH", "200"))
SLEEP_SECONDS = float(os.getenv("PERMID_MAP_SLEEP", "0.3"))
US_ONLY = os.getenv("PERMID_MAP_US_ONLY", "1") == "1"
SAVE_PARTS = os.getenv("PERMID_MAP_SAVE_PARTS", "1") == "1"
SKIP_EXISTING = os.getenv("PERMID_MAP_SKIP_EXISTING", "1") == "1"
RETRIES = int(os.getenv("PERMID_MAP_RETRIES", "3"))
RETRY_SLEEP = float(os.getenv("PERMID_MAP_RETRY_SLEEP", "1.5"))

FIELD_CANDIDATES = [
    "TR.CommonName",
    "TR.RIC",
    "TR.PrimaryRIC",
    "TR.TickerSymbol",
    "TR.ExchangeTicker",
    "TR.CUSIP",
    "TR.CUSIP9",
    "TR.ISIN",
]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def ensure_session() :
    try:
        _ = rd.get_data(universe="0#.SPX", fields=["TR.CommonName"])
        return True
    except Exception as e:
        log(f"Refinitiv session check failed: {e}")
        return False


