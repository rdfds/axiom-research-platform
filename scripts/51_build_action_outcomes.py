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
        row.get("principal_amt"),
        row.get("dealamount"),
    )


def _base_available_liquidity(metrics: Dict[str, Any]) -> Optional[float]:
    value = metrics.get("available_liquidity")
    if value is not None:
        return value
    return metrics.get("cash")


def _apply_richer_base_fields(record: Dict[str, Any], base: Dict[str, Any]) -> None:
    record["base_cash"] = base.get("cash")
    record["base_total_debt"] = base.get("total_debt", base.get("debt"))
    record["base_available_liquidity"] = _base_available_liquidity(base)


def _enrich_macro_columns_from_helper(
    df: pd.DataFrame,
    *,
    macro_series: Dict[str, str],
) -> pd.DataFrame:
    if df.empty or "action_date" not in df.columns or not macro_series:
        return df

    helper = FeatureBuilder()
    dates = pd.to_datetime(df["action_date"], errors="coerce")
    unique_dates = sorted({pd.Timestamp(d).normalize() for d in dates.dropna().tolist()})
    if not unique_dates:
        return df

    macro_cache: Dict[str, Dict[str, Any]] = {}
    for as_of in unique_dates:
        macro_cache[_date_key(as_of)] = helper.compute_macro_features(as_of, macro_series)

    enriched = df.copy()
    date_keys = dates.dt.normalize().dt.strftime("%Y-%m-%d")
    candidate_columns = sorted(
        {
            col
            for payload in macro_cache.values()
            for col in payload.keys()
            if str(col).startswith("macro_")
        }
    )
    for column in candidate_columns:
        mapped = date_keys.map(lambda key: (macro_cache.get(key) or {}).get(column))
        if column not in enriched.columns:
            enriched[column] = mapped
        else:
            enriched[column] = enriched[column].where(enriched[column].notna(), mapped)
    return enriched


