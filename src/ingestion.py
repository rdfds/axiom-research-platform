"""
Ingestion Utilities (MVP)
=========================
Append-only raw lake + normalized warehouse helpers with bitemporal enforcement.
"""

from __future__ import annotations

import json
import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"

QUALITY_FLAGS = [
    "missing_data",
    "delayed_data",
    "partial_coverage",
    "source_conflict",
    "outlier_detected",
    "unit_inconsistency",
    "restatement",
    "stale_data",
    "estimated_available_time",
    "estimated_event_time",
    "schema_violation",
    "balance_sheet_unbalanced",
    "cash_flow_mismatch",
    "missing_line_item",
    "halted_session",
    "missing_trade_day",
    "price_outlier",
    "curve_non_monotonic",
    "jump_outlier",
    "authorization_only",
    "execution_only",
    "open_ended",
    "missing_size",
    "withdrawn",
    "deal_failed",
    "value_missing",
]


def _canonical_json(payload: Dict[str, Any]) -> bytes:
    def _clean(value: Any) -> Any:
        if isinstance(value, datetime):
            return value.isoformat()
        try:
            import pandas as pd  # Local import to avoid hard dependency

            if isinstance(value, pd.Timestamp):
                if pd.isna(value):
                    return None
                return value.isoformat()
            if pd.isna(value):
                return None
        except Exception:
            pass
        try:
            import numpy as np  # Local import to avoid hard dependency

            if isinstance(value, (np.integer, np.floating, np.bool_)):
                return value.item()
        except Exception:
            pass
        return value

    cleaned = {k: _clean(v) for k, v in payload.items()}
    return json.dumps(cleaned, sort_keys=True, separators=(",", ":")).encode("utf-8")


def compute_raw_payload_hash(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


