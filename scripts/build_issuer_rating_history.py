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


def _build_from_fisd(path: Path, permno_to_entity: Dict[str, str]) -> pd.DataFrame:
    if not _is_materialized(path):
        return pd.DataFrame()
    df = pd.read_parquet(path, columns=["permno", "RATING_DATE", "RATING", "RATING_TYPE", "RATING_STATUS", "ISSUE_ID"])
    if df.empty:
        return df
    df["permno_key"] = df["permno"].map(_normalize_id)
    df["company_id"] = df["permno_key"].map(permno_to_entity)
    df["rating_date"] = pd.to_datetime(df["RATING_DATE"], utc=True, errors="coerce")
    df["rating_symbol"] = df["RATING"].astype(str).str.strip()
    df["rating_type_code"] = df["RATING_TYPE"].astype(str).str.strip()
    df["creditwatch"] = df["RATING_STATUS"]
    df = df.dropna(subset=["company_id", "rating_date", "rating_symbol"])
    if df.empty:
        return df
    out = pd.DataFrame(
        {
            "company_id": df["company_id"].astype(str),
            "rating_date": df["rating_date"],
            "rating_symbol": df["rating_symbol"],
            "current_rating_symbol": df["rating_symbol"],
            "rating_type_code": df["rating_type_code"],
            "outlook": None,
            "creditwatch": df["creditwatch"],
            "source_type": "fisd_ratings",
            "artifact_id": (
                "fisd:"
                + df["ISSUE_ID"].astype(str)
                + ":"
                + df["rating_type_code"].astype(str)
                + ":"
                + df["rating_date"].dt.strftime("%Y-%m-%d")
            ),
        }
    )
    out["effective_at"] = out["rating_date"]
    out["published_at"] = out["rating_date"]
    out["ingested_at"] = out["rating_date"]
    return out


def _build_from_ciq(
    ciq_path: Path,
    gvkey_to_cik_path: Path,
    cik_to_entity: Dict[str, str],
) -> pd.DataFrame:
    if not _is_materialized(ciq_path) or not _is_materialized(gvkey_to_cik_path):
        return pd.DataFrame()
    ciq = pd.read_parquet(
        ciq_path,
        columns=[
            "gvkey",
            "rating_date",
            "rating_symbol",
            "current_rating_symbol",
            "rating_type_code",
            "outlook",
            "creditwatch",
            "ciqcompanyid",
        ],
    )
    if ciq.empty:
        return ciq
    gv = pd.read_csv(gvkey_to_cik_path, compression="infer", dtype=str)
    gv_cols = {c.lower(): c for c in gv.columns}
    if "gvkey" not in gv_cols or "cik" not in gv_cols:
        return pd.DataFrame()
    gv = gv.rename(columns={gv_cols["gvkey"]: "gvkey", gv_cols["cik"]: "cik"})
    gv["gvkey"] = gv["gvkey"].astype(str).str.extract(r"([0-9]+)", expand=False).str.zfill(6)
    gv["cik"] = gv["cik"].astype(str).str.extract(r"([0-9]+)", expand=False)
    ciq["gvkey"] = ciq["gvkey"].astype(str).str.extract(r"([0-9]+)", expand=False).str.zfill(6)
    ciq = ciq.merge(gv[["gvkey", "cik"]], on="gvkey", how="left")
    ciq["company_id"] = ciq["cik"].map(lambda x: cik_to_entity.get(_normalize_id(x) or ""))
    ciq["rating_date"] = pd.to_datetime(ciq["rating_date"], utc=True, errors="coerce")
    ciq = ciq.dropna(subset=["company_id", "rating_date"])
    if ciq.empty:
        return ciq
    out = pd.DataFrame(
        {
            "company_id": ciq["company_id"].astype(str),
            "rating_date": ciq["rating_date"],
            "rating_symbol": ciq["rating_symbol"],
            "current_rating_symbol": ciq["current_rating_symbol"],
            "rating_type_code": ciq["rating_type_code"],
            "outlook": ciq["outlook"],
            "creditwatch": ciq["creditwatch"],
            "source_type": "ciq_ratings",
            "artifact_id": (
                "ciq:"
                + ciq["ciqcompanyid"].astype(str)
                + ":"
                + ciq["rating_type_code"].astype(str)
                + ":"
                + ciq["rating_date"].dt.strftime("%Y-%m-%d")
            ),
        }
    )
    out["effective_at"] = out["rating_date"]
    out["published_at"] = out["rating_date"]
    out["ingested_at"] = out["rating_date"]
    return out


