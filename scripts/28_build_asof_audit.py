#!/usr/bin/env python
"""
As-Of Coverage Audit
====================
Generates coverage reports for warehouse tables:
- Missing required fields
- Missing months (prices)
- Missing sizes by action type (corporate actions)
"""

from __future__ import annotations

import argparse
import os
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import sys

try:
    import duckdb  # type: ignore
except Exception:  # pragma: no cover
    duckdb = None

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.asof_store import AsOfWarehouse


DATA_DIR = Path(__file__).parent.parent / "data"
WAREHOUSE_DIR = DATA_DIR / "warehouse"


REQUIRED_FIELDS = {
    "warehouse_financials": [
        "company_id",
        "event_time",
        "available_time",
        "statement_type",
        "line_item",
        "value",
        "fiscal_period_end",
        "fiscal_year",
        "fiscal_quarter",
    ],
    "warehouse_prices": ["entity_id", "event_time", "available_time", "close", "adjusted_close", "volume"],
    "warehouse_prices_daily": [
        "entity_id",
        "event_time",
        "available_time",
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
        "total_return_index",
    ],
    "warehouse_prices_daily_rdp": [
        "entity_id",
        "event_time",
        "available_time",
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
        "total_return_index",
    ],
    "warehouse_macro": [
        "entity_id",
        "event_time",
        "available_time",
        "instrument_id",
        "instrument_type",
        "tenor",
        "value",
        "units",
    ],
    "warehouse_estimates": [
        "entity_id",
        "event_time",
        "available_time",
        "metric",
        "period",
        "consensus_value",
        # "num_estimates" is optional for some sources (e.g., FMP).
        # Set AUDIT_ESTIMATES_REQUIRE_NUM=1 to enforce it.
    ],
    "warehouse_press_releases": [
        "entity_id",
        "event_time",
        "available_time",
        "document_id",
        "release_date",
        "headline",
        "text",
    ],
    "warehouse_corp_actions": ["entity_id", "event_time", "available_time", "action_type", "announcement_date", "effective_date", "size"],
    "warehouse_mna_deals": ["deal_id", "announcement_date", "deal_value", "status"],
    "warehouse_13f_holdings": [
        "company_id",
        "event_time",
        "available_time",
        "holding_cusip",
        "value_k",
        "shares",
    ],
    "warehouse_13f_filings": [
        "company_id",
        "event_time",
        "available_time",
    ],
}

if os.getenv("AUDIT_ESTIMATES_REQUIRE_NUM", "0") in ("1", "true", "True"):
    REQUIRED_FIELDS["warehouse_estimates"].append("num_estimates")


def log(msg: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}")


