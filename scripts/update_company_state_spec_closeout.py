#!/usr/bin/env python
"""
Fast post-processing updater to close remaining spec gaps without re-running
full snapshot computation.

Adds/refreshes:
  - market.ev_ebitda_vs_peer_z
  - market.fcf_yield_percentile_peers
  - operating.guidance_revision_direction
  - operating.cyclicality_macro_beta_proxy
  - operating.revenue_sensitivity_proxy
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        x = float(v)
        if math.isnan(x):
            return None
        return x
    except Exception:
        return None


def _feature_value(snapshot: dict, name: str) -> Any:
    return snapshot.get("features", {}).get(name, {}).get("value")


def _feature_record(
    name: str,
    value: Any,
    unit: str,
    as_of_time: str,
    confidence: Optional[float],
    provenance: List[dict],
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


def _reference(
    artifact_type: str,
    artifact_id: str,
    source: Optional[str],
    published_at: Optional[str] = None,
    ingested_at: Optional[str] = None,
) -> dict:
    return {
        "artifact_type": artifact_type,
        "artifact_id": artifact_id,
        "source": source,
        "published_at": published_at,
        "ingested_at": ingested_at,
        "hash": None,
    }


def _load_snapshots(path: Path) -> List[dict]:
    out: List[dict] = []
    with path.open("r") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def _write_snapshots(path: Path, snapshots: List[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for s in snapshots:
            f.write(json.dumps(s) + "\n")
    tmp.replace(path)


def _choose_sector_col(df: pd.DataFrame) -> Optional[str]:
    for c in [
        "gics_sector",
        "sector",
        "industry",
        "gics_industry",
        "industry_group",
        "sic",
        "naics",
    ]:
        if c in df.columns:
            return c
    return None


