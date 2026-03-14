#!/usr/bin/env python
"""
Probe + Pull Refinitiv Corporate Actions (US Active, 2000-present)
==================================================================
This script:
1) Probes which corporate-action fields are available in your entitlement.
2) Builds a US active equity universe (attempts full universe; falls back to indices).
3) Pulls corporate actions for that universe from 2000-01-01 through 2026-02-02.

Outputs:
  data/refinitiv/universe_us_active.parquet
  data/refinitiv/corporate_actions/ca_<year>_part_<batch>.parquet
  data/refinitiv/corporate_actions/manifest.jsonl

Run:
  python -u scripts/18_probe_and_pull_refinitiv_corp_actions.py
"""

import json
import time
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from pandas.api.types import is_string_dtype
import refinitiv.data as rd


DATA_DIR = Path(__file__).parent.parent / "data" / "refinitiv"
DATA_DIR.mkdir(parents=True, exist_ok=True)

UNIVERSE_PATH = DATA_DIR / "universe_us_active.parquet"
CA_DIR = DATA_DIR / "corporate_actions"
CA_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = CA_DIR / "manifest.jsonl"

START_DATE = "2000-01-01"
END_DATE = "2026-02-02"  # Current date in this workspace

BATCH_SIZE = 75
SLEEP_SECONDS = 0.4

CA_FIELD_CANDIDATES = [
    "TR.CAActionType",
    "TR.CAEventType",
    "TR.CAType",
    "TR.CAAnnouncementDate",
    "TR.CAEffectiveDate",
    "TR.CAExDate",
    "TR.CARecordDate",
    "TR.CAPayDate",
    "TR.CAAmount",
    "TR.CAAdjustmentFactor",
    "TR.CAAdjustmentType",
    "TR.CACurrency",
    "TR.CARatio",
    "TR.CAStatus",
    "TR.CAAmount",
    "TR.CAAmountGross",
    "TR.CAAmountNet",
    "TR.CAExAmount",
    "TR.CAShareFactor",
    "TR.CAValue",
]

# If CAType is required, we will try these. Add/remove as needed.
CA_TYPE_CANDIDATES = [
    "DIV",  # Dividends (may or may not be valid)
    "DVD",
    "SSP",  # Stock splits
    "RHT",  # Rights
    "SPI",  # Spinoff
    "BON",  # Bonus issue
    "CAP",  # Capital change
    "MER",  # Merger
    "TND",  # Tender
    "LIQ",  # Liquidation
]

# Safety / quality controls
ABORT_IF_EMPTY_PROBE = True
MIN_EVENT_ROWS = 5
MIN_EVENT_RATIO = 0.01
MIN_DATE_EVENT_ROWS = 3
MIN_DATE_EVENT_RATIO = 0.01
MAX_CONSECUTIVE_EMPTY_BATCHES = 10


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def batched(items: List[str], batch_size: int) -> List[List[str]]:
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


