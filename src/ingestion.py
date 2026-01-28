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


def compute_version_id(
    source_system: str,
    entity_id: str,
    event_time: datetime,
    available_time: datetime,
    raw_payload_hash: str,
    schema_version: str = "v1",
) -> str:
    key = f"{source_system}|{entity_id}|{event_time.isoformat()}|{available_time.isoformat()}|{raw_payload_hash}|{schema_version}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def ensure_bitemporal(event_time: datetime, available_time: datetime) -> None:
    if event_time is None or available_time is None:
        raise ValueError("event_time and available_time are required.")
    if available_time < event_time:
        raise ValueError("available_time must be >= event_time.")


def _ensure_list(value: Optional[Any]) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


