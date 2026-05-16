import argparse
import json
from datetime import timezone
from pathlib import Path
from typing import Dict, List, Tuple

import duckdb
import pandas as pd


RECURRING_DIVIDEND_EVENT_TYPES = {
    "dividend_regular",
    "dividend_increase",
    "dividend_cut",
    "dividend_initiate",
}
RECURRING_DIVIDEND_SUBTYPES = {
    "regular",
    "dividend_increase",
    "dividend_cut",
    "dividend_initiate",
}


def _now_iso() -> str:
    return pd.Timestamp.now(tz=timezone.utc).isoformat()


def _feature_record(
    *,
    name: str,
    value,
    unit: str,
    as_of_time: str,
    confidence,
    provenance: List[Dict[str, str]],
    missing_reason,
    fallback_used,
) -> Dict[str, object]:
    return {
        "name": name,
        "value": value,
        "unit": unit,
        "computed_at": _now_iso(),
        "as_of_time": as_of_time,
        "window": {"type": "lookback", "length_days": 730},
        "confidence": confidence,
        "provenance": provenance,
        "missing_reason": missing_reason,
        "fallback_used": fallback_used,
    }


