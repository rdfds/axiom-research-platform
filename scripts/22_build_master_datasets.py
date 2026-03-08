#!/usr/bin/env python
"""
Build master datasets (best/largest per type) and a Russell 3000 proxy universe.

Outputs:
  data/curated/universe_r3000_proxy.parquet
  data/curated/corporate_actions_master.parquet
  data/curated/prices_master.parquet
  data/curated/prices_master_full.parquet
  data/curated/fundamentals_master.parquet
  data/curated/buybacks_master.parquet
  data/curated/mna_master.parquet
  data/curated/master_summary.csv
"""

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
import pandas as pd
import pyarrow.dataset as ds

try:
    import duckdb  # type: ignore
except Exception:  # pragma: no cover
    duckdb = None


DATA_DIR = Path(__file__).parent.parent / "data"
CRSP_DIR = DATA_DIR / "wrds" / "crsp"
CURATED_DIR = DATA_DIR / "curated"
CURATED_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(pattern: str):
    files = list(CRSP_DIR.glob(pattern))
    if not files:
        return None
    return ds.dataset(files, format="parquet")


