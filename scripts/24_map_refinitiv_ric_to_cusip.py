#!/usr/bin/env python
"""
Map Refinitiv RICs to identifiers via Symbology (Discovery Convert Symbols).
===========================================================================
Builds a RIC -> CUSIP/ISIN/Ticker map using Refinitiv Symbology.

Outputs:
  data/refinitiv/ric_to_cusip_map.parquet

Run:
  python -u scripts/24_map_refinitiv_ric_to_cusip.py
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd
import refinitiv.data as rd
import refinitiv.data.discovery as disc
from refinitiv.data.content import symbol_conversion as sc


DATA_DIR = Path(__file__).parent.parent / "data" / "refinitiv"
MAP_PATH = DATA_DIR / "ric_to_cusip_map.parquet"
PARTS_DIR = DATA_DIR / "ric_map_parts"
PARTS_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = int(os.getenv("RIC_MAP_BATCH", "200"))
SLEEP_SECONDS = float(os.getenv("RIC_MAP_SLEEP", "0.2"))
SAVE_PARTS = os.getenv("RIC_MAP_SAVE_PARTS", "1") == "1"
SKIP_EXISTING = os.getenv("RIC_MAP_SKIP_EXISTING", "1") == "1"
RETRIES = int(os.getenv("RIC_MAP_RETRIES", "3"))
RETRY_SLEEP = float(os.getenv("RIC_MAP_RETRY_SLEEP", "1.5"))
US_ONLY = os.getenv("RIC_MAP_US_ONLY", "1") == "1"
ASSET_STATE = os.getenv("RIC_MAP_ASSET_STATE", "ACTIVE").upper()
APPEND = os.getenv("RIC_MAP_APPEND", "1") == "1"


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def ensure_session() :
    try:
        _ = rd.get_data(universe="0#.SPX", fields=["TR.CommonName"])
        return True
    except Exception as e:
        log(f"Refinitiv session check failed: {e}")
        return False


def clean_text(series: pd.Series) -> pd.Series:
    s = series.astype("string")
    s = s.str.strip()
    s = s.where(~s.str.lower().isin(["", "nan", "none", "<na>"]))
    return s


class Progress:
    def __init__(self, total: int):
        self.total = max(total, 1)
        self.start = time.time()
        self.last_print = 0.0

    def update(self, current: int, note: str = ""):
        now = time.time()
        if now - self.last_print < 0.2 and current < self.total:
            return
        elapsed = now - self.start
        rate = current / elapsed if elapsed > 0 else 0
        remaining = (self.total - current) / rate if rate > 0 else 0
        percent = current / self.total * 100
        bar_len = 28
        filled = int(bar_len * percent / 100)
        bar = "#" * filled + "-" * (bar_len - filled)
        msg = (
            f"[{percent:5.1f}%] [{bar}] {current}/{self.total} "
            f"| {elapsed/60:5.1f}m elapsed | ETA {remaining/60:5.1f}m {note}"
        )
        sys.stdout.write("\r" + msg.ljust(120))
        sys.stdout.flush()
        self.last_print = now

    def finish(self):
        sys.stdout.write("\n")
        sys.stdout.flush()


def _pick_col(df: pd.DataFrame, *names):
    for name in names:
        if name in df.columns:
            return name
        for col in df.columns:
            if col.lower() == name.lower():
                return col
    return None


