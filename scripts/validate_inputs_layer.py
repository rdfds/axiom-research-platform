#!/usr/bin/env python
"""
Validate Inputs Layer datasets against JSON schemas.

This script performs lightweight checks:
- required columns
- basic dtype compatibility
- timestamp parseability
- timestamp ordering vs ingested_at
- confidence_score bounds

Outputs a DataIntegrityLog parquet.
"""

from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow.dataset as ds


ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_schema(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset(
    path: Path,
    sample_rows: int | None = None,
    columns: List[str] | None = None,
) -> pd.DataFrame:
    if path.is_dir():
        # Partitioned parquet dataset
        if sample_rows:
            files = sorted(path.rglob("*.parquet"))
            if not files:
                return pd.DataFrame()
            parts = []
            total = 0
            for f in files:
                try:
                    df_part = pd.read_parquet(f, columns=columns)
                except Exception:
                    # skip unreadable/unstable files (e.g., stale NFS handles)
                    continue
                parts.append(df_part)
                total += len(df_part)
                if total >= sample_rows:
                    break
            df = pd.concat(parts, ignore_index=True)
            return df.head(sample_rows)
        return pd.read_parquet(path, columns=columns)
    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=columns)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix in {".json", ".ndjson"}:
        return pd.read_json(path, lines=path.suffix == ".ndjson")
    raise ValueError(f"Unsupported file type: {path}")


def is_numeric_like(s: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(s):
        return True
    # soft check for numeric-ish object columns
    coerced = pd.to_numeric(s, errors="coerce")
    return coerced.notna().mean() >= 0.95


def is_bool_like(s: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(s):
        return True
    if pd.api.types.is_numeric_dtype(s):
        vals = s.dropna().unique()
        return set(vals).issubset({0, 1})
    if pd.api.types.is_string_dtype(s) or s.dtype == object:
        # Try numeric coercion first (handles 0.0/1.0 stored as object)
        coerced = pd.to_numeric(s, errors="coerce")
        if coerced.notna().any():
            vals = set(coerced.dropna().unique())
            return vals.issubset({0, 1})
        vals = {str(v).strip().lower() for v in s.dropna().unique()}
        return vals.issubset({"true", "false", "0", "1", "yes", "no", "y", "n"})
    return False


def is_string_like(s: pd.Series) -> bool:
    return pd.api.types.is_string_dtype(s) or s.dtype == object


def is_datetime_like(s: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(s)


def validate_required_columns(
    df: pd.DataFrame,
    required: List[str],
    available_cols: List[str] | None = None,
) -> Tuple[bool, List[str]]:
    cols = available_cols if available_cols is not None else list(df.columns)
    missing = [c for c in required if c not in cols]
    return (len(missing) == 0), missing


