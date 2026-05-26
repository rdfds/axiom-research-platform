#!/usr/bin/env python
"""
Build action outcome dataset for precedent matching.

Example:
  python -u scripts/51_build_action_outcomes.py \
    --action-types buyback,acquisition \
    --start-date 2000-01-01 \
    --out data/curated/action_outcomes.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import pandas as pd
import duckdb

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.action_normalization import augment_action_outcomes_df
from src.pipeline.config import load_config
from src.pipeline.features import FeatureBuilder


DATA_DIR = Path(__file__).parent.parent / "data"

FMP_INCOME_ITEMS = [
    "Revenue",
    "EBITDA",
    "NetIncome",
    "EPS",
    "EPSDiluted",
    "SharesOut",
    "SharesOutDiluted",
]
FMP_BALANCE_ITEMS = [
    "Cash",
    "ShortTermInvestments",
    "DebtCurrent",
    "DebtLongTerm",
    "TotalAssets",
]
FMP_CASH_ITEMS = [
    "OperatingCashFlow",
    "Capex",
    "FreeCashFlow",
]


def _pct_change(new: Optional[float], old: Optional[float]) -> Optional[float]:
    if new is None or old is None:
        return None
    try:
        old_val = float(old)
        new_val = float(new)
    except Exception:
        return None
    if old_val == 0 or pd.isna(old_val) or pd.isna(new_val):
        return None
    return (new_val - old_val) / abs(old_val)


def _pp_change(new: Optional[float], old: Optional[float]) -> Optional[float]:
    if new is None or old is None:
        return None
    try:
        old_val = float(old)
        new_val = float(new)
    except Exception:
        return None
    if pd.isna(old_val) or pd.isna(new_val):
        return None
    return new_val - old_val


def _pick_date(row: pd.Series, date_field: str) -> Optional[pd.Timestamp]:
    if date_field != "auto":
        val = row.get(date_field)
        return pd.to_datetime(val, errors="coerce") if val is not None else None
    for field in ("announcement_date", "event_time", "effective_date", "action_date"):
        val = row.get(field)
        if val is not None and not pd.isna(val):
            return pd.to_datetime(val, errors="coerce")
    return None


def _months_from_quarters(quarters: int) -> int:
    return int(quarters) * 3


def _date_key(value: pd.Timestamp) -> str:
    return value.normalize().strftime("%Y-%m-%d")


def first_non_null(*values: Any) -> Optional[float]:
    for value in values:
        if value is None or pd.isna(value):
            continue
        return value
    return None


def first_positive_non_null(*values: Any) -> Optional[float]:
    for value in values:
        if value is None or pd.isna(value):
            continue
        try:
            numeric = float(value)
        except Exception:
            continue
        if numeric > 0:
            return numeric
    return None


def _resolved_action_size(row: pd.Series) -> Optional[float]:
    action_type = str(row.get("action_type") or "").strip().lower()
    if action_type in {"split", "stock_split", "reverse_split"}:
        return first_positive_non_null(
            row.get("split_factor"),
            row.get("facpr"),
            row.get("ratio"),
            row.get("size"),
            row.get("amount"),
            row.get("divamt"),
            row.get("deal_value"),
            row.get("offering_amt_k"),
            row.get("principal_amt"),
            row.get("dealamount"),
        )
    return first_non_null(
        row.get("size"),
        row.get("amount"),
        row.get("ratio"),
        row.get("split_factor"),
        row.get("facpr"),
        row.get("divamt"),
        row.get("deal_value"),
        row.get("offering_amt_k"),
        row['principal_amt'],
        row.get("dealamount"),
    )


