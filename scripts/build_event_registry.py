#!/usr/bin/env python
"""
Build EventRegistry from the unified corporate actions master dataset.

This is a baseline generator: it normalizes core fields and preserves provenance.
Parameters/evidence_links are left null for now (can be enriched later).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True)


def coalesce(series_list: Iterable[pd.Series]) -> pd.Series:
    out = None
    for s in series_list:
        if s is None:
            continue
        out = s if out is None else out.combine_first(s)
    return out


def prefixed_str(s: pd.Series, prefix: str) -> pd.Series:
    s = s.astype("string")
    s = s.where(s.notna(), pd.NA)
    return prefix + s


def build_source_id(df: pd.DataFrame) -> pd.Series:
    source_id = pd.Series(pd.NA, index=df.index, dtype="string")

    for prefix, col in [
        ("deal_id:", "deal_id"),
        ("issue_id:", "ISSUE_ID"),
        ("issuer_id:", "ISSUER_ID"),
        ("facilityid:", "facilityid"),
        ("action_code:", "action_code"),
        ("distcd:", "distcd"),
    ]:
        if col in df.columns:
            s = prefixed_str(df[col], prefix)
            source_id = source_id.combine_first(s)

    # Fallback: hash a small stable key
    missing = source_id.isna()
    if missing.any():
        key_cols = [
            c
            for c in [
                "source",
                "source_table",
                "action_type",
                "action_subtype",
                "action_date",
                "permno",
                "gvkey",
            ]
            if c in df.columns
        ]
        key_df = df.loc[missing, key_cols].copy()
        for c in key_df.columns:
            if pd.api.types.is_datetime64_any_dtype(key_df[c]):
                key_df[c] = key_df[c].dt.strftime("%Y-%m-%d")
            key_df[c] = key_df[c].astype("string").fillna("")
        hashes = pd.util.hash_pandas_object(key_df, index=False).astype("uint64").astype("string")
        source_id.loc[missing] = "rowhash:" + hashes

    return source_id


