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


