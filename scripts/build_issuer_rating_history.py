#!/usr/bin/env python
"""
Build issuer-level rating history in inputs layer, mapped to canonical entity_id.

Primary source:
  - data/curated/bond_ratings_fisd.parquet

Optional source (enabled when mapping files are materialized):
  - data/curated/issuer_ratings_ciq.parquet
  - data/wrds/compustat/cik_gvkey.csv.gz

Output:
  - data/inputs_layer/issuer_rating_history.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _is_materialized(path: Path) -> bool:
    if not path.exists():
        return False
    st = path.stat()
    if st.st_size <= 0:
        return False
    # Avoid triggering iCloud fetch in scripts that should be non-blocking.
    if hasattr(st, "st_blocks") and st.st_blocks == 0:
        return False
    return True


def _normalize_id(x: object) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return None
    # "81284.0" -> "81284"
    if s.endswith(".0"):
        s = s[:-2]
    return s


def _load_identifier_maps(entity_identifier_path: Path) -> tuple[Dict[str, str], Dict[str, str]]:
    ids = pd.read_parquet(entity_identifier_path)
    ids["identifier_type"] = ids["identifier_type"].astype(str).str.lower()
    ids["identifier_value"] = ids["identifier_value"].astype(str).str.strip()
    ids["entity_id"] = ids["entity_id"].astype(str).str.strip()

    permno_to_entity: Dict[str, str] = {}
    cik_to_entity: Dict[str, str] = {}
    permno = ids[ids["identifier_type"] == "permno"][["identifier_value", "entity_id"]].dropna()
    cik = ids[ids["identifier_type"] == "cik"][["identifier_value", "entity_id"]].dropna()

    for _, row in permno.iterrows():
        key = _normalize_id(row["identifier_value"])
        if key:
            permno_to_entity[key] = str(row["entity_id"])
            stripped = key.lstrip("0")
            if stripped:
                permno_to_entity[stripped] = str(row["entity_id"])

    for _, row in cik.iterrows():
        key = _normalize_id(row["identifier_value"])
        if key:
            cik_to_entity[key] = str(row["entity_id"])
            stripped = key.lstrip("0")
            if stripped:
                cik_to_entity[stripped] = str(row["entity_id"])
                cik_to_entity[stripped.zfill(10)] = str(row["entity_id"])

    return permno_to_entity, cik_to_entity


