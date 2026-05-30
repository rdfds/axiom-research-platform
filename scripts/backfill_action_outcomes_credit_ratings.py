#!/usr/bin/env python
"""
Backfill credit spread change + rating migration outcome columns onto action_outcomes.parquet
without rebuilding the entire outcomes table.

Inputs (defaults):
  data/curated/action_outcomes.parquet
  data/curated/trace_btds_daily_fisduniverse.parquet
  data/curated/bond_issuances_fisd.parquet
  data/inputs_layer/issuer_rating_history.parquet

Output:
  data/curated/action_outcomes_with_credit_ratings.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]


def _parse_horizons(raw: str) -> List[int]:
    out: List[int] = []
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        out = [1, 6, 12, 24]
    return sorted(set(out))


def _quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _rating_score_case(expr: str) -> str:
    # Higher score = better rating quality.
    return f"""
    CASE {expr}
      -- S&P / Fitch style
      WHEN 'AAA' THEN 22
      WHEN 'AA+' THEN 21 WHEN 'AA' THEN 20 WHEN 'AA-' THEN 19
      WHEN 'A+' THEN 18 WHEN 'A' THEN 17 WHEN 'A-' THEN 16
      WHEN 'BBB+' THEN 15 WHEN 'BBB' THEN 14 WHEN 'BBB-' THEN 13
      WHEN 'BB+' THEN 12 WHEN 'BB' THEN 11 WHEN 'BB-' THEN 10
      WHEN 'B+' THEN 9 WHEN 'B' THEN 8 WHEN 'B-' THEN 7
      WHEN 'CCC+' THEN 6 WHEN 'CCC' THEN 5 WHEN 'CCC-' THEN 4
      WHEN 'CC' THEN 3 WHEN 'C' THEN 2 WHEN 'D' THEN 1
      -- Moody's style (normalized to upper-case)
      WHEN 'AAA' THEN 22
      WHEN 'AA1' THEN 21 WHEN 'AA2' THEN 20 WHEN 'AA3' THEN 19
      WHEN 'A1' THEN 18 WHEN 'A2' THEN 17 WHEN 'A3' THEN 16
      WHEN 'BAA1' THEN 15 WHEN 'BAA2' THEN 14 WHEN 'BAA3' THEN 13
      WHEN 'BA1' THEN 12 WHEN 'BA2' THEN 11 WHEN 'BA3' THEN 10
      WHEN 'B1' THEN 9 WHEN 'B2' THEN 8 WHEN 'B3' THEN 7
      WHEN 'CAA1' THEN 6 WHEN 'CAA2' THEN 5 WHEN 'CAA3' THEN 4
      WHEN 'CA' THEN 3
      ELSE NULL
    END
    """


