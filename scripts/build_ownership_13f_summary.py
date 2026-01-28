#!/usr/bin/env python
"""
Build issuer-level 13F ownership summary for CompanyState.

Input:
  - data/warehouse/warehouse_13f_holdings (partitioned parquet directory)
  - data/inputs_layer/entity_identifier.parquet (CUSIP -> entity_id map)

Output:
  - data/inputs_layer/ownership_13f_summary.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional

import duckdb


ROOT = Path(__file__).resolve().parents[1]


def _is_materialized(path: Path) -> bool:
    if not path.exists():
        return False
    st = path.stat()
    if st.st_size <= 0:
        return False
    # Keep this non-blocking; do not trigger iCloud fetch here.
    if hasattr(st, "st_blocks") and st.st_blocks == 0:
        return False
    return True


def _parse_years(arg: Optional[str]) -> Optional[List[int]]:
    if not arg:
        return None
    out: List[int] = []
    for x in arg.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(int(x))
    return sorted(set(out))


