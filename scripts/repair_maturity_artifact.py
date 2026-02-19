#!/usr/bin/env python3
"""Repair debt-due and maturity-wall metrics in a materialized company-state artifact."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd


REPAIR_METRICS = [
    "capital_structure.debt_due_0_12m",
    "capital_structure.debt_due_12_24m",
    "capital_structure.maturity_wall_ratio_24m_reported",
    "capital_structure.maturity_wall_ratio_24m_market",
    "capital_structure.maturity_wall_ratio_24m",
    "capital_structure.refi_pressure_flag_reported",
    "capital_structure.refi_pressure_flag_market",
    "capital_structure.refi_pressure_flag",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-path", required=True, help="Input company-state JSONL artifact")
    parser.add_argument("--out", required=True, help="Output repaired JSONL artifact")
    parser.add_argument("--summary-out", help="Optional summary JSON")
    parser.add_argument(
        "--private-debt-schedule-path",
        help="Optional parquet with due_0_12/due_12_24 style schedule buckets keyed by company_id",
    )
    return parser.parse_args()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


