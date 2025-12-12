#!/usr/bin/env python
"""
Targeted backfill of SEC 8-K press release text for records ingested without text.

Reads warehouse_press_releases parquet parts, finds rows with missing text,
fetches the primary document from SEC, and appends new records with text.
Append-only: prior records remain; backfilled records supersede via version_id.

Env:
  SEC_USER_AGENT (required)
  SEC_BACKFILL_START=2000-01-01
  SEC_BACKFILL_END=YYYY-MM-DD (default: today UTC)
  SEC_BACKFILL_LIMIT=0 (0 = no limit)
  SEC_BACKFILL_SLEEP=0.2
  SEC_BACKFILL_TIMEOUT=20
  SEC_BACKFILL_RETRIES=2
  SEC_BACKFILL_RESUME=1
  SEC_BACKFILL_START_INDEX=0  (# of missing-text rows to skip)
  SEC_BACKFILL_BATCH=200
"""

from __future__ import annotations

import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion import (
    compute_raw_payload_hash,
    compute_version_id,
    write_raw_records,
)


DATA_DIR = Path(__file__).parent.parent / "data"
WAREHOUSE_DIR = DATA_DIR / "warehouse"
SEC_DIR = DATA_DIR / "sec"

SEC_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"

SEC_BACKFILL_START = os.getenv("SEC_BACKFILL_START", "2000-01-01")
SEC_BACKFILL_END = os.getenv("SEC_BACKFILL_END", datetime.utcnow().date().isoformat())
SEC_BACKFILL_LIMIT = int(os.getenv("SEC_BACKFILL_LIMIT", "0"))
SEC_BACKFILL_SLEEP = float(os.getenv("SEC_BACKFILL_SLEEP", "0.2"))
SEC_BACKFILL_TIMEOUT = float(os.getenv("SEC_BACKFILL_TIMEOUT", "20"))
SEC_BACKFILL_RETRIES = int(os.getenv("SEC_BACKFILL_RETRIES", "2"))
SEC_BACKFILL_MIN_TEXT = int(os.getenv("SEC_BACKFILL_MIN_TEXT", "200"))
SEC_BACKFILL_ALLOW_FULL_SUBMISSION = os.getenv("SEC_BACKFILL_ALLOW_FULL_SUBMISSION", "1") == "1"
SEC_BACKFILL_ALLOW_EXHIBIT_SEARCH = os.getenv("SEC_BACKFILL_ALLOW_EXHIBIT_SEARCH", "1") == "1"
SEC_BACKFILL_COMPACT_ONLY = os.getenv("SEC_BACKFILL_COMPACT_ONLY", "0") == "1"
SEC_BACKFILL_MAX_MB = float(os.getenv("SEC_BACKFILL_MAX_MB", "0"))
SEC_BACKFILL_SLOW_LOG_SECONDS = float(os.getenv("SEC_BACKFILL_SLOW_LOG_SECONDS", "15"))
SEC_BACKFILL_RESUME = os.getenv("SEC_BACKFILL_RESUME", "1") == "1"
SEC_BACKFILL_START_INDEX = int(os.getenv("SEC_BACKFILL_START_INDEX", "0"))
SEC_BACKFILL_BATCH = int(os.getenv("SEC_BACKFILL_BATCH", "200"))
SEC_FILE_LOG_EVERY = int(os.getenv("SEC_FILE_LOG_EVERY", "100"))
SEC_BACKFILL_START_FILE = int(os.getenv("SEC_BACKFILL_START_FILE", "0"))


def log(msg: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def require_user_agent() -> str:
    user_agent = os.getenv("SEC_USER_AGENT")
    if not user_agent:
        raise RuntimeError(
            "SEC_USER_AGENT not set. Example: export SEC_USER_AGENT='Axiom Research (you@example.com)'"
        )
    return user_agent


def ensure_list(value) -> List:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, pd.Series):
        return value.tolist()
    return [value]


def normalize_flags(value) -> List[str]:
    flat: List[str] = []
    for item in ensure_list(value):
        if item is None:
            continue
        if isinstance(item, (list, tuple, pd.Series)):
            for sub in ensure_list(item):
                if sub is None:
                    continue
                flat.append(str(sub))
            continue
        # numpy arrays or pandas arrays
        if hasattr(item, "tolist") and not isinstance(item, str):
            try:
                for sub in item.tolist():
                    if sub is None:
                        continue
                    flat.append(str(sub))
                continue
            except Exception:
                pass
        flat.append(str(item))
    return flat


def html_to_text(html: str) -> str:
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    text = re.sub(r"\\s+", " ", text)
    return text.strip()


