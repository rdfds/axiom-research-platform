#!/usr/bin/env python3
"""Repair credit-market metrics in a materialized company-state artifact.

This pass fills the still-empty credit regime / credit window layer using:
- exact macro IG/HY OAS anchors already present in the artifact
- macro OAS history from the raw timeseries parquet for percentiles
- company risk signals already materialized in the artifact

The repaired company-level spread metrics are intentionally tagged as heuristic
proxy values; they are not direct traded bond/CDS spreads.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd
import pyarrow.parquet as pq


REPAIR_METRICS = [
    "macro.us_ig_oas",
    "macro.us_ig_oas_percentile_history",
    "market.credit_spread_level",
    "market.credit_spread_percentile_2y",
    "market.credit_window_proxy",
]

IG_OAS_INSTRUMENT = "BAMLC0A0CM"
HY_OAS_INSTRUMENT = "BAMLH0A0HYM2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-path", required=True, help="Input company-state JSONL artifact")
    parser.add_argument("--macro-timeseries-path", help="Optional raw timeseries parquet path")
    parser.add_argument("--out", required=True, help="Output repaired JSONL artifact")
    parser.add_argument("--summary-out", help="Optional summary JSON")
    return parser.parse_args()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def iter_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _node_value(node: Dict[str, Any] | None) -> float | None:
    if not node:
        return None
    value = node.get("value")
    if value is None:
        return None
    return float(value)


def _node_support(node: Dict[str, Any] | None) -> str:
    if not node:
        return "unsupported"
    return str(node.get("support_mode") or "unsupported")


def _union_provenance(*nodes: Dict[str, Any] | None) -> list[Dict[str, Any]]:
    merged: list[Dict[str, Any]] = []
    seen = set()
    for node in nodes:
        for prov in (node or {}).get("provenance") or []:
            key = json.dumps(prov, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            merged.append(copy.deepcopy(prov))
    return merged


def _base_repaired_node(node: Dict[str, Any], *, computed_at: str) -> Dict[str, Any]:
    repaired = copy.deepcopy(node)
    repaired["computed_at"] = computed_at
    repaired["missing_reason"] = None
    repaired["quality_flags"] = repaired.get("quality_flags") or None
    return repaired


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _infer_macro_timeseries_path(artifact_path: Path) -> Path | None:
    for row in iter_rows(artifact_path):
        features = row.get("features") or {}
        for metric in ("macro.ig_oas", "macro.hy_oas", "macro.us_ig_oas"):
            node = features.get(metric) or {}
            for prov in node.get("provenance") or []:
                source = prov.get("source")
                if source and str(source).endswith(".parquet"):
                    return Path(str(source))
    return None


def _load_spread_histories(path: Path) -> Dict[str, pd.DataFrame]:
    table = pq.read_table(
        path,
        columns=["instrument_id", "event_time", "available_time", "trade_date", "value"],
    )
    df = table.to_pandas()
    df = df[df["instrument_id"].isin([IG_OAS_INSTRUMENT, HY_OAS_INSTRUMENT])].copy()
    if df.empty:
        return {}
    df["time"] = pd.to_datetime(df["available_time"], utc=True, errors="coerce")
    missing = df["time"].isna()
    if missing.any():
        df.loc[missing, "time"] = pd.to_datetime(df.loc[missing, "event_time"], utc=True, errors="coerce")
    missing = df["time"].isna()
    if missing.any():
        df.loc[missing, "time"] = pd.to_datetime(df.loc[missing, "trade_date"], utc=True, errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["instrument_id", "time", "value"]).copy()
    df = df.sort_values(["instrument_id", "time"]).drop_duplicates(["instrument_id", "time"], keep="last")
    out: Dict[str, pd.DataFrame] = {}
    for instrument_id, group in df.groupby("instrument_id", sort=False):
        out[str(instrument_id)] = group[["time", "value"]].reset_index(drop=True)
    return out


