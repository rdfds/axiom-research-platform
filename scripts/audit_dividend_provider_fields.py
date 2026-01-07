#!/usr/bin/env python3
"""Inspect candidate dividend provider files for direct dividend fields.

This audit is intentionally read-only. It helps answer whether we have a truly
direct provider field for dividend per share or dividend yield, rather than an
event series we would need to aggregate ourselves.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


DEFAULT_FILES = [
    "./data/refinitiv/dividends_complete.parquet",
    "./data/refinitiv/dividends_with_amounts.parquet",
    "./data/dividend_profiles.parquet",
    "./data/dividend_actions.parquet",
    "./data/special_dividends.parquet",
    "./data/special_dividends_linked.parquet",
]

KEYWORDS = ("div", "yield", "special", "regular", "cash", "amount", "trailing", "forward")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", nargs="*", default=DEFAULT_FILES, help="Parquet files to inspect")
    parser.add_argument("--sample-rows", type=int, default=3, help="Number of sample rows to print")
    return parser.parse_args()


