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


def _scale_bucket(row: Dict[str, Any]) -> Optional[str]:
    ratio = _size_ratio(row)
    if ratio is None:
        return None
    if ratio < 0.05:
        return "small"
    if ratio < 0.25:
        return "medium"
    return "large"


def _exact_result(
    *,
    family: str,
    subfamily: str,
    action_id: str,
    family_scale_bucket: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "normalized_action_family": family,
        "normalized_action_subfamily": subfamily,
        "normalized_action_id": action_id,
        "normalization_level": "exact",
        "normalization_confidence": 0.98,
        "family_scale_bucket": family_scale_bucket,
        "normalization_rules_version": NORMALIZATION_RULES_VERSION,
    }


def _family_result(
    *,
    family: str,
    subfamily: str,
    family_scale_bucket: Optional[str] = None,
    action_id: Optional[str] = None,
) -> Dict[str, Any]:
    level = "family_scale" if family_scale_bucket else "family"
    confidence = 0.85 if family_scale_bucket else 0.65
    return {
        "normalized_action_family": family,
        "normalized_action_subfamily": subfamily,
        "normalized_action_id": action_id,
        "normalization_level": level,
        "normalization_confidence": confidence,
        "family_scale_bucket": family_scale_bucket,
        "normalization_rules_version": NORMALIZATION_RULES_VERSION,
    }


