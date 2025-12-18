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


