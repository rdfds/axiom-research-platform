#!/usr/bin/env python
"""
Ingest WRDS SEC 13F Holdings Data (XML-based, complete post-2013) into warehouse.

Input: CSV or CSV.GZ from WRDS SEC 13F Holdings Data query.
Default path: data/wrds/13f/holdings.csv.gz

Env:
  WRDS_13F_INPUT=path/to/holdings.csv.gz
  WRDS_13F_CHUNK=200000
  WRDS_13F_PARTITIONED=1  (create data/warehouse/warehouse_13f_holdings/ for partitioned writes)
  WRDS_13F_LOG_EVERY=1     (log every chunk)
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion import (
    append_canonical_records,
    compute_raw_payload_hash,
    compute_version_id,
)


DATA_DIR = Path(__file__).parent.parent / "data"

INPUT_PATH = Path(os.getenv("WRDS_13F_INPUT", DATA_DIR / "wrds" / "13f" / "holdings.csv.gz"))
CHUNK = int(os.getenv("WRDS_13F_CHUNK", "200000"))
PARTITIONED = os.getenv("WRDS_13F_PARTITIONED", "1") not in ("0", "false", "False")
LOG_EVERY = int(os.getenv("WRDS_13F_LOG_EVERY", "1"))


def log(msg: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def normalize_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


