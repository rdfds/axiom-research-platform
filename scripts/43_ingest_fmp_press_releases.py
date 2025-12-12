#!/usr/bin/env python
"""
Ingest FMP Press Releases (B4) to improve text coverage.

Requires:
  export FMP_API_KEY="..."

Writes:
  data/warehouse/warehouse_press_releases (partitioned by year)
  data/warehouse/warehouse_documents (optional, default on)
  data/warehouse/warehouse_doc_chunks (optional, default on)
  data/warehouse/warehouse_text_signals (optional, default on)

Raw payloads are stored in data/lake/raw/fmp_press_releases.

Env:
  FMP_API_KEY (required)
  FMP_BASE_URL=https://financialmodelingprep.com/stable
  FMP_SLEEP=0.2
  FMP_RETRIES=2
  FMP_TIMEOUT=30
  FMP_START_DATE=2000-01-01
  FMP_END_DATE=YYYY-MM-DD (default: today UTC)
  FMP_LIMIT_SYMBOLS=0 (0 = all)
  FMP_TARGET_SYMBOL= (optional)
  FMP_USE_UNIVERSE=1 (use R3000 proxy tickers)
  FMP_SYMBOL_BATCH=25
  FMP_MAX_PAGES=5
  FMP_PAGE_LIMIT=100
  FMP_RESUME=1
  FMP_PR_REQUIRE_TEXT=1
  FMP_PR_PROCESS_DOCS=1
  FMP_PR_FLUSH_EVERY=200
"""

from __future__ import annotations

import hashlib
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion import compute_raw_payload_hash, compute_version_id, write_raw_records
from src.text_processing import chunk_text, ensure_list, extract_signals, write_partitioned


DATA_DIR = Path(__file__).parent.parent / "data"
FMP_DIR = DATA_DIR / "fmp"
CRSP_DIR = DATA_DIR / "wrds" / "crsp"
WAREHOUSE_DIR = DATA_DIR / "warehouse"

FMP_API_KEY = os.getenv("FMP_API_KEY")
FMP_BASE_URL = os.getenv("FMP_BASE_URL", "https://financialmodelingprep.com/stable").rstrip("/")
FMP_SLEEP = float(os.getenv("FMP_SLEEP", "0.2"))
FMP_RETRIES = int(os.getenv("FMP_RETRIES", "2"))
FMP_TIMEOUT = float(os.getenv("FMP_TIMEOUT", "30"))
FMP_START_DATE = os.getenv("FMP_START_DATE", "2000-01-01")
FMP_END_DATE = os.getenv("FMP_END_DATE", datetime.utcnow().date().isoformat())
FMP_LIMIT_SYMBOLS = int(os.getenv("FMP_LIMIT_SYMBOLS", "0"))
FMP_TARGET_SYMBOL = os.getenv("FMP_TARGET_SYMBOL")
FMP_USE_UNIVERSE = os.getenv("FMP_USE_UNIVERSE", "1") == "1"
FMP_SYMBOL_BATCH = int(os.getenv("FMP_SYMBOL_BATCH", "25"))
FMP_MAX_PAGES = int(os.getenv("FMP_MAX_PAGES", "5"))
FMP_PAGE_LIMIT = int(os.getenv("FMP_PAGE_LIMIT", "100"))
FMP_RESUME = os.getenv("FMP_RESUME", "1") == "1"
FMP_PR_REQUIRE_TEXT = os.getenv("FMP_PR_REQUIRE_TEXT", "1") == "1"
FMP_PR_PROCESS_DOCS = os.getenv("FMP_PR_PROCESS_DOCS", "1") == "1"
FMP_PR_FLUSH_EVERY = int(os.getenv("FMP_PR_FLUSH_EVERY", "200"))
FMP_DEBUG = os.getenv("FMP_DEBUG", "0") == "1"

PR_CHUNK_TOKENS = int(os.getenv("PR_CHUNK_TOKENS", "400"))
PR_CHUNK_MIN = int(os.getenv("PR_CHUNK_MIN", "300"))
PR_CHUNK_MAX = int(os.getenv("PR_CHUNK_MAX", "500"))


def log(msg: str) :
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


