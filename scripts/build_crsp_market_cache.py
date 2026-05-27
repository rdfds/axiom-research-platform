#!/usr/bin/env python3
"""Build a filtered CRSP daily market cache for the current entity universe.

This scans the large CRSP daily stock export once, keeps only the permnos that
exist in the current entity identifier file, and writes a compact parquet cache
that downstream scripts can query quickly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


def parse_args() :
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity-identifier-path", required=True)
    parser.add_argument("--crsp-daily-path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--date-from", required=True, help="Inclusive start date YYYY-MM-DD")
    parser.add_argument("--date-to", required=True, help="Inclusive end date YYYY-MM-DD")
    return parser.parse_args()


