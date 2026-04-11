#!/usr/bin/env python
"""
Scan parquet files and move any unreadable/corrupt files to a quarantine folder.

Default target:
  data/warehouse/warehouse_press_releases

Usage:
  python -u scripts/41_quarantine_corrupt_parquet.py
  python -u scripts/41_quarantine_corrupt_parquet.py --path data/warehouse/warehouse_prices_daily
  python -u scripts/41_quarantine_corrupt_parquet.py --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

import pyarrow.parquet as pq


def log(msg: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def iter_files(base: Path, pattern: str) -> List[Path]:
    return sorted(base.rglob(pattern))


def is_parquet_ok(path: Path) -> bool:
    try:
        if path.stat().st_size == 0:
            return False
        pf = pq.ParquetFile(path)
        _ = pf.metadata
        return True
    except Exception:
        return False


