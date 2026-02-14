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


def main() -> None:
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Missing CIQ entity ratings file at {IN_PATH}")

    CURATED_DIR.mkdir(parents=True, exist_ok=True)

    log("Loading CIQ entity ratings...")
    ratings = pd.read_csv(IN_PATH, dtype=str, low_memory=False)

    ratings["ratingdate"] = pd.to_datetime(ratings.get("ratingdate"), errors="coerce")
    ratings["creditwatchdate"] = pd.to_datetime(ratings.get("creditwatchdate"), errors="coerce")
    ratings["outlookdate"] = pd.to_datetime(ratings.get("outlookdate"), errors="coerce")

    ratings["ciqcompanyid"] = ratings.get("ciqcompanyid").astype("string")
    ratings["company_id"] = ratings.get("company_id").astype("string")

    mapping = build_company_gvkey_map()
    ratings = ratings.merge(
        mapping,
        left_on="ciqcompanyid",
        right_on="companyid",
        how="left",
    )

    before = len(ratings)
    ratings = ratings.dropna(subset=["gvkey", "ratingdate"])
    log(f"Mapped gvkey for {ratings['gvkey'].notna().sum():,} rows (kept {len(ratings):,}/{before:,})")

    keep = {
        "gvkey": "gvkey",
        "ratingdate": "rating_date",
        "ratingsymbol": "rating_symbol",
        "currentratingsymbol": "current_rating_symbol",
        "ratingtypecode": "rating_type_code",
        "ratingtypename": "rating_type_name",
        "creditwatch": "creditwatch",
        "outlook": "outlook",
        "ratingactionword": "rating_action_word",
        "cwolactionword": "cwol_action_word",
        "ratingqualifier": "rating_qualifier",
        "longtermflag": "longterm_flag",
        "shorttermflag": "shortterm_flag",
        "globalornationalscaleind": "global_or_national_scale",
        "unsol": "unsolicited",
        "entity_id": "entity_id",
        "ciqcompanyid": "ciqcompanyid",
        "entname": "entity_name",
        "countrycode": "country_code",
        "region": "region",
        "sectorcode": "sector_code",
        "industrycode": "industry_code",
    }

    out = ratings[list(keep.keys())].rename(columns=keep)
    out.to_parquet(OUT_PATH, index=False)
    log(f"Saved issuer ratings -> {OUT_PATH} ({len(out):,} rows)")


if __name__ == "__main__":
    main()
