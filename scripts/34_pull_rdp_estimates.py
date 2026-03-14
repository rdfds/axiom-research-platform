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


