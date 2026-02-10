#!/usr/bin/env python
"""
Ingest CRSP Daily Stock File (dsf) into warehouse_prices_daily dataset.

This writes partitioned parquet files by year to:
  data/warehouse/warehouse_prices_daily/year=YYYY/part_*.parquet
"""

from __future__ import annotations

import argparse
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion import compute_raw_payload_hash


DATA_DIR = Path(__file__).parent.parent / "data"
WRDS_DIR = DATA_DIR / "wrds" / "crsp"
OUT_DIR = DATA_DIR / "warehouse" / "warehouse_prices_daily"


def log(msg: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}")


