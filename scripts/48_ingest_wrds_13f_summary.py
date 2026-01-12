#!/usr/bin/env python
"""
Ingest WRDS SEC 13F Summary Data into warehouse.

Input: CSV or CSV.GZ from WRDS SEC 13F Summary Data query.
Default path: data/wrds/13f/summary.csv.gz

Env:
  WRDS_13F_SUMMARY_INPUT=path/to/summary.csv.gz
  WRDS_13F_SUMMARY_CHUNK=200000
  WRDS_13F_SUMMARY_PARTITIONED=1  (create data/warehouse/warehouse_13f_filings/)
  WRDS_13F_SUMMARY_LOG_EVERY=1
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

INPUT_PATH = Path(os.getenv("WRDS_13F_SUMMARY_INPUT", DATA_DIR / "wrds" / "13f" / "summary.csv.gz"))
CHUNK = int(os.getenv("WRDS_13F_SUMMARY_CHUNK", "200000"))
PARTITIONED = os.getenv("WRDS_13F_SUMMARY_PARTITIONED", "1") not in ("0", "false", "False")
LOG_EVERY = int(os.getenv("WRDS_13F_SUMMARY_LOG_EVERY", "1"))


PREFERRED_COLUMNS = [
    "cik",
    "coname",
    "form",
    "rdate",
    "fdate",
    "fname",
    "reportdate",
    "report_period",
    "total_value",
    "total_shares",
    "noentries",
    "othermanager",
    "amendmenttype",
    "amendmentno",
]


def log(msg: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def normalize_value(value: Any) :
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


def coerce_date(value: Any) -> pd.Timestamp | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        ts = pd.to_datetime(value, errors="coerce")
        if pd.isna(ts):
            return None
        return ts
    except Exception:
        return None


def iter_chunks(path: Path, chunksize: int) -> Iterable[pd.DataFrame]:
    reader = pd.read_csv(
        path,
        compression="infer",
        chunksize=chunksize,
        low_memory=False,
    )
    for chunk in reader:
        yield chunk


