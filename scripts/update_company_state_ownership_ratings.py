#!/usr/bin/env python
"""
Fast delta updater for ownership concentration + issuer rating features.

Use this to enrich an existing CompanyState snapshot JSONL without rerunning
full feature computation.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import duckdb
import numpy as np


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sql_quote(s: str) -> str:
    return "'" + str(s).replace("'", "''") + "'"


def _asof_ts(asof: str) -> str:
    return str(np.datetime64(asof))


def _feature_record(
    name: str,
    value: Any,
    unit: str,
    as_of_time: str,
    confidence: Optional[float],
    provenance: list,
    missing_reason: Optional[str],
    window: Optional[Dict[str, Any]] = None,
    fallback_used: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "name": name,
        "value": value,
        "unit": unit,
        "computed_at": _now_iso(),
        "as_of_time": as_of_time,
        "window": window,
        "confidence": confidence,
        "provenance": provenance,
        "missing_reason": missing_reason,
        "fallback_used": fallback_used,
    }


