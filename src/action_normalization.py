from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd


NORMALIZATION_RULES_VERSION = "v2_lossless_20260317"


def _norm_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip().lower()


def _to_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if pd.isna(out):
        return None
    return out


def _size_ratio(row: Dict[str, Any]) -> Optional[float]:
    action_size = _to_float(row.get("action_size"))
    base_market_cap = _to_float(row.get("base_market_cap"))
    if action_size is not None and base_market_cap and base_market_cap > 0:
        ratio = action_size / base_market_cap
        if ratio > 0:
            return ratio

    for key in (
        "target_size_pct_ev",
        "target_size_pct_market_cap",
        "target_size_pct_mc",
        "deal_size_pct_ev",
        "transaction_size_pct_ev",
        "target_ev_pct",
        "percent_divested",
        "percent_sold",
        "stake_pct",
    ):
        ratio = _to_float(row.get(key))
        if ratio is not None and ratio > 0:
            return ratio
    return None


