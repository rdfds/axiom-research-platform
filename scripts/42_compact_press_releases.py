#!/usr/bin/env python
"""
Compact press release warehouse files to reduce per-file overhead.

Reads:
  data/warehouse/warehouse_press_releases/year=*/part_*.parquet

Writes:
  data/warehouse/warehouse_press_releases/year=*/part_compact.parquet

Options (env):
  PR_COMPACT_START_YEAR=2000
  PR_COMPACT_END_YEAR=YYYY
  PR_COMPACT_BACKUP=1      (move old part_*.parquet to backup)
  PR_COMPACT_DELETE=0      (delete old part_*.parquet instead of backup)
  PR_COMPACT_SKIP_EXISTING=1 (skip if part_compact already exists)
  PR_COMPACT_LOG_EVERY=50  (file progress per year)
"""

from __future__ import annotations

import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd


DATA_DIR = Path(__file__).parent.parent / "data"
WAREHOUSE_DIR = DATA_DIR / "warehouse"

PR_COMPACT_START_YEAR = int(os.getenv("PR_COMPACT_START_YEAR", "2000"))
PR_COMPACT_END_YEAR = int(os.getenv("PR_COMPACT_END_YEAR", datetime.utcnow().year))
PR_COMPACT_BACKUP = os.getenv("PR_COMPACT_BACKUP", "1") == "1"
PR_COMPACT_DELETE = os.getenv("PR_COMPACT_DELETE", "0") == "1"
PR_COMPACT_SKIP_EXISTING = os.getenv("PR_COMPACT_SKIP_EXISTING", "1") == "1"
PR_COMPACT_LOG_EVERY = int(os.getenv("PR_COMPACT_LOG_EVERY", "50"))


def log(msg: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def normalize_list(value: Optional[object]) -> List[str]:
    if value is None:
        return []
    try:
        import numpy as np
    except Exception:
        np = None  # type: ignore

    if np is not None and isinstance(value, np.ndarray):
        if value.size == 0:
            return []
        return [str(v) for v in value.tolist() if v is not None]
    if isinstance(value, pd.Series):
        if value.empty:
            return []
        return [str(v) for v in value.tolist() if v is not None]
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value if v is not None and not (isinstance(v, float) and pd.isna(v))]
    try:
        if pd.isna(value):
            return []
    except Exception:
        pass
    return [str(value)]


