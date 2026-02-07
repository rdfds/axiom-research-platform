#!/usr/bin/env python
"""
Build issuer-level rating history in inputs layer, mapped to canonical entity_id.

Primary source:
  - data/curated/bond_ratings_fisd.parquet

Optional source (enabled when mapping files are materialized):
  - data/curated/issuer_ratings_ciq.parquet
  - data/wrds/compustat/cik_gvkey.csv.gz

Output:
  - data/inputs_layer/issuer_rating_history.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _is_materialized(path: Path) -> bool:
    if not path.exists():
        return False
    st = path.stat()
    if st.st_size <= 0:
        return False
    # Avoid triggering iCloud fetch in scripts that should be non-blocking.
    if hasattr(st, "st_blocks") and st.st_blocks == 0:
        return False
    return True


