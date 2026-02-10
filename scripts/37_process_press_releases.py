#!/usr/bin/env python
"""
Chunk press releases (B4) + extract text signals into canonical tables.

Reads:
  data/warehouse/warehouse_press_releases/year=*/part_*.parquet

Writes:
  data/warehouse/warehouse_documents/year=*/part_*.parquet
  data/warehouse/warehouse_doc_chunks/year=*/part_*.parquet
  data/warehouse/warehouse_text_signals/year=*/part_*.parquet

Env:
  PR_START_YEAR=2000
  PR_END_YEAR=YYYY
  PR_LIMIT_DOCS=0 (0 = no limit)
  PR_FLUSH_EVERY=200 (docs per flush)
  PR_CHUNK_TOKENS=400
  PR_CHUNK_MIN=300
  PR_CHUNK_MAX=500
  PR_RESUME=1
  PR_START_FILE_INDEX=0
"""

from __future__ import annotations

import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion import compute_raw_payload_hash, compute_version_id


DATA_DIR = Path(__file__).parent.parent / "data"
WAREHOUSE_DIR = DATA_DIR / "warehouse"
SEC_DIR = DATA_DIR / "sec"

PR_START_YEAR = int(os.getenv("PR_START_YEAR", "2000"))
PR_END_YEAR = int(os.getenv("PR_END_YEAR", datetime.utcnow().year))
PR_LIMIT_DOCS = int(os.getenv("PR_LIMIT_DOCS", "0"))
PR_FLUSH_EVERY = int(os.getenv("PR_FLUSH_EVERY", "200"))
PR_CHUNK_TOKENS = int(os.getenv("PR_CHUNK_TOKENS", "400"))
PR_CHUNK_MIN = int(os.getenv("PR_CHUNK_MIN", "300"))
PR_CHUNK_MAX = int(os.getenv("PR_CHUNK_MAX", "500"))
PR_RESUME = os.getenv("PR_RESUME", "1") == "1"
PR_START_FILE_INDEX = int(os.getenv("PR_START_FILE_INDEX", "0"))
PR_REQUIRE_TEXT = os.getenv("PR_REQUIRE_TEXT", "0") == "1"


def log(msg: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


