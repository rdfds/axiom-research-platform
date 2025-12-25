#!/usr/bin/env python
"""
Pull CRSP Stock (US equities) via WRDS
======================================
This script pulls the core CRSP Stock tables for US equities:
  - msenames (security master / name history)
  - msedist (distributions: dividends, splits, spin-offs, etc.)
  - msedelist (delistings)
  - msf (monthly stock file)
  - dsf (daily stock file, optional)
  - ccmxpf_lnkhist (Compustat link table)

Defaults:
  - US common stocks only (shrcd 10/11, exchcd 1/2/3)
  - 2000-01-01 through today
  - monthly + distributions + delistings + names + link table

Usage:
  python -u scripts/20_pull_crsp_stock_data.py [WRDS_USERNAME]

Examples:
  # Smoke test (last 30 days, monthly only)
  python -u scripts/20_pull_crsp_stock_data.py --start 2026-01-01 --end 2026-01-31

  # Full 2000-present monthly + distributions + delistings
  python -u scripts/20_pull_crsp_stock_data.py --start 2000-01-01

  # Re-pull distributions with broader filters (all share codes) and overwrite
  python -u scripts/20_pull_crsp_stock_data.py --start 2000-01-01 --end 2024-12-31 \\
    --no-monthly --no-delistings --no-names --no-link --all-shares --force

  # Include daily (very large)
  python -u scripts/20_pull_crsp_stock_data.py --start 2000-01-01 --daily
"""

import argparse
import json
from datetime import datetime, date
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
import wrds


DATA_DIR = Path(__file__).parent.parent / "data" / "wrds" / "crsp"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def parse_int_list(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def year_chunks(start: str, end: str, chunk_years: int) -> Iterable[Tuple[str, str]]:
    start_date = datetime.strptime(start, "%Y-%m-%d").date()
    end_date = datetime.strptime(end, "%Y-%m-%d").date()

    year = start_date.year
    while year <= end_date.year:
        chunk_start = date(year, 1, 1)
        chunk_end_year = min(year + chunk_years - 1, end_date.year)
        chunk_end = date(chunk_end_year, 12, 31)
        if year == start_date.year:
            chunk_start = start_date
        if chunk_end_year == end_date.year:
            chunk_end = end_date
        yield chunk_start.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        year += chunk_years


def build_permno_cte(common_only: bool, shrcd: List[int], exchcd: List[int]) -> str:
    where = "nameendt >= %(start_date)s AND namedt <= %(end_date)s"
    if common_only:
        where += f" AND shrcd IN ({','.join(str(x) for x in shrcd)})"
        where += f" AND exchcd IN ({','.join(str(x) for x in exchcd)})"
    return f"""
    WITH permnos AS (
        SELECT DISTINCT permno
        FROM crsp.msenames
        WHERE {where}
    )
    """


