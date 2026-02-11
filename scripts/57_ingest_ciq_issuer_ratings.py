#!/usr/bin/env python
"""
Ingest CIQ issuer-level credit ratings into a curated parquet.

Inputs:
  data/wrds/ciq/ciq_entity_ratings.csv.gz
  data/wrds/ciq/ciq_identifiers_master.csv.gz  (for companyid -> gvkey)

Outputs:
  data/curated/issuer_ratings_ciq.parquet
  data/wrds/ciq/ciq_company_gvkey_map.parquet (cached mapping)
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CURATED_DIR = DATA_DIR / "curated"
CIQ_DIR = DATA_DIR / "wrds" / "ciq"

IN_PATH = CIQ_DIR / "ciq_entity_ratings.csv.gz"
IDENT_PATH = CIQ_DIR / "ciq_identifiers_master.csv.gz"
MAP_PATH = CIQ_DIR / "ciq_company_gvkey_map.parquet"
OUT_PATH = CURATED_DIR / "issuer_ratings_ciq.parquet"


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def _clean_gvkey(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.extract(r"([0-9]+)", expand=False)
    return cleaned.str.zfill(6)


