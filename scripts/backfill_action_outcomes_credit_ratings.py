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


def _parse_horizons(raw: str) :
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


