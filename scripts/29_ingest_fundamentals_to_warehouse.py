#!/usr/bin/env python
"""
Ingest Compustat fundamentals into bitemporal warehouse_financials.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion import (
    append_canonical_records,
    compute_raw_payload_hash,
    compute_version_id,
    write_raw_records,
)


DATA_DIR = Path(__file__).parent.parent / "data"
FUND_PATH = DATA_DIR / "fundamentals_quarterly.parquet"


LINE_ITEMS = {
    "revtq": ("income", "Revenue"),
    "cogsq": ("income", "COGS"),
    "xsgaq": ("income", "SGA"),
    "oibdpq": ("income", "EBITDA"),
    "oiadpq": ("income", "OperatingIncome"),
    "niq": ("income", "NetIncome"),
    "xintq": ("income", "InterestExpense"),
    "atq": ("balance_sheet", "TotalAssets"),
    "actq": ("balance_sheet", "CurrentAssets"),
    "cheq": ("balance_sheet", "Cash"),
    "rectq": ("balance_sheet", "Receivables"),
    "invtq": ("balance_sheet", "Inventory"),
    "ppentq": ("balance_sheet", "PP&E"),
    "ltq": ("balance_sheet", "TotalLiabilities"),
    "lctq": ("balance_sheet", "CurrentLiabilities"),
    "dlcq": ("balance_sheet", "DebtCurrent"),
    "dlttq": ("balance_sheet", "DebtLongTerm"),
    "ceqq": ("balance_sheet", "CommonEquity"),
    "seqq": ("balance_sheet", "TotalEquity"),
    "capxy": ("cash_flow", "Capex"),
    "oancfy": ("cash_flow", "OperatingCashFlow"),
    "epspxq": ("income", "EPS"),
    "cshoq": ("balance_sheet", "SharesOut"),
}


def log(msg: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}")


def normalize_value(value) -> Optional[float]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return None


def normalize_int(value) :
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return int(value)
    except Exception:
        return None


def normalize_payload(payload: Dict) -> Dict:
    cleaned = {}
    for k, v in payload.items():
        if isinstance(v, pd.Timestamp):
            if pd.isna(v):
                cleaned[k] = None
            else:
                cleaned[k] = v.isoformat()
        elif isinstance(v, (np.integer, np.floating, np.bool_)):
            cleaned[k] = v.item()
        else:
            try:
                cleaned[k] = None if pd.isna(v) else v
            except Exception:
                cleaned[k] = v
    return cleaned


