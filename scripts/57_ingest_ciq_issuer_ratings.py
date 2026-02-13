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


def build_company_gvkey_map() -> pd.DataFrame:
    if MAP_PATH.exists():
        log(f"Loading cached CIQ company->gvkey map: {MAP_PATH}")
        return pd.read_parquet(MAP_PATH)

    if not IDENT_PATH.exists():
        raise FileNotFoundError(f"Missing CIQ identifiers master at {IDENT_PATH}")

    chunk = int(os.getenv("CIQ_CHUNK", "2000000"))
    log_every = int(os.getenv("CIQ_LOG_EVERY", "5000000"))
    engine = "python" if os.getenv("CIQ_ENGINE") == "python" else "c"

    log(f"Building CIQ company->gvkey map from {IDENT_PATH} (chunk={chunk}, engine={engine})")
    mapping: dict[str, str] = {}
    scanned = 0
    next_log = log_every

    usecols = ["companyid", "symboltypecat", "symbolvalue"]
    for frame in pd.read_csv(
        IDENT_PATH,
        usecols=usecols,
        dtype=str,
        chunksize=chunk,
        engine=engine,
        low_memory=False if engine == "c" else None,
    ):
        scanned += len(frame)
        gv = frame[frame["symboltypecat"].str.upper() == "GVKEY"].copy()
        if not gv.empty:
            gv = gv.dropna(subset=["companyid", "symbolvalue"])
            gv["companyid"] = gv["companyid"].astype("string")
            gv["symbolvalue"] = _clean_gvkey(gv["symbolvalue"])
            gv = gv.dropna(subset=["symbolvalue"])
            for companyid, gvkey in zip(gv["companyid"], gv["symbolvalue"]):
                if companyid not in mapping:
                    mapping[companyid] = gvkey
        if scanned >= next_log:
            log(f"Scanned {scanned:,} rows | mapped companies {len(mapping):,}")
            next_log += log_every

    df = pd.DataFrame({"companyid": list(mapping.keys()), "gvkey": list(mapping.values())})
    df.to_parquet(MAP_PATH, index=False)
    log(f"Cached company->gvkey map: {MAP_PATH} ({len(df):,} rows)")
    return df


