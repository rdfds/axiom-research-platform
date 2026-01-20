#!/usr/bin/env python
"""
A4-lite: Daily estimates snapshots via Refinitiv Data Library (Workspace session),
with inferred period end using Compustat FY end.

This is NOT fully spec-compliant (no publish_time). We mark:
  - partial_coverage
  - estimated_available_time
  - estimated_period_end

Env:
  EST_PERIODS=FY1,FY2,NTM
  EST_BATCH=75
  EST_SLEEP=0.2
  EST_UNIVERSE_FILE=data/refinitiv/universe_us_active.parquet
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import refinitiv.data as rd

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion import (
    append_canonical_records,
    compute_raw_payload_hash,
    compute_version_id,
    write_raw_records,
)


DATA_DIR = Path(__file__).parent.parent / "data"
REF_DIR = DATA_DIR / "refinitiv"
CRSP_DIR = DATA_DIR / "wrds" / "crsp"
WAREHOUSE_DIR = DATA_DIR / "warehouse"

EST_PERIODS = [p.strip() for p in os.getenv("EST_PERIODS", "FY1,FY2,NTM").split(",") if p.strip()]
EST_BATCH = int(os.getenv("EST_BATCH", "75"))
EST_SLEEP = float(os.getenv("EST_SLEEP", "0.2"))
EST_UNIVERSE_FILE = Path(os.getenv("EST_UNIVERSE_FILE", str(REF_DIR / "universe_us_active.parquet")))
EST_DEBUG = os.getenv("EST_DEBUG", "0") == "1"


FIELD_ALIASES = {
    "eps": ["TR.EPSMean", "Earnings Per Share - Mean", "EPS Mean"],
    "revenue": ["TR.RevenueMean", "Revenue - Mean", "Revenue Mean"],
    "ebitda": ["TR.EBITDAMean", "EBITDA - Mean", "EBITDA Mean"],
}
NUM_FIELDS = [
    "TR.NumberOfEstimates",
    "TR.NumOfEstimates",
    "Number of Estimates",
    "Num of Estimates",
]
REQUEST_FIELDS = [
    "TR.EPSMean",
    "TR.RevenueMean",
    "TR.EBITDAMean",
    "TR.NumberOfEstimates",
    "TR.NumOfEstimates",
]


def log(msg: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def load_universe() -> List[str]:
    if not EST_UNIVERSE_FILE.exists():
        raise FileNotFoundError(f"Missing universe file: {EST_UNIVERSE_FILE}")
    df = pd.read_parquet(EST_UNIVERSE_FILE)
    if "ric" in df.columns:
        return df["ric"].dropna().astype("string").str.upper().tolist()
    return df.iloc[:, 0].dropna().astype("string").str.upper().tolist()


def load_ric_map() -> pd.DataFrame:
    ric_map_path = REF_DIR / "ric_to_cusip_map.parquet"
    if not ric_map_path.exists():
        raise FileNotFoundError("Missing ric_to_cusip_map.parquet")
    ric_map = pd.read_parquet(ric_map_path)
    ric_map["ric"] = ric_map["ric"].astype("string").str.upper().str.strip()
    ric_map["cusip8"] = (
        ric_map["cusip"]
        .astype("string")
        .str.replace(r"[^0-9A-Za-z]", "", regex=True)
        .str.upper()
        .str[:8]
    )
    ric_map = ric_map[ric_map["cusip8"].notna()]
    ric_map = ric_map.drop_duplicates("ric")
    ric_map = ric_map[["ric", "cusip8", "ticker"]]

    # Optional: enrich mapping with permid map if available
    permid_path = REF_DIR / "permid_map.parquet"
    if permid_path.exists():
        permid = pd.read_parquet(permid_path, columns=["ric", "cusip", "ticker"])
        permid["ric"] = permid["ric"].astype("string").str.upper().str.strip()
        permid["cusip8"] = (
            permid["cusip"]
            .astype("string")
            .str.replace(r"[^0-9A-Za-z]", "", regex=True)
            .str.upper()
            .str[:8]
        )
        permid = permid[permid["ric"].notna() & permid["cusip8"].notna()]
        permid = permid.drop_duplicates("ric")
        permid = permid[["ric", "cusip8", "ticker"]]
        ric_map = pd.concat([ric_map, permid], ignore_index=True)
        ric_map = ric_map.drop_duplicates("ric")

    return ric_map


def load_names() -> pd.DataFrame:
    names_path = CRSP_DIR / "msenames_2000-01-01_to_2026-12-31.parquet"
    if not names_path.exists():
        raise FileNotFoundError("Missing msenames_2000-01-01_to_2026-12-31.parquet")
    names = pd.read_parquet(
        names_path,
        columns=["permno", "permco", "namedt", "nameendt", "ncusip", "cusip"],
    )
    names["namedt"] = pd.to_datetime(names["namedt"], errors="coerce")
    names["nameendt"] = pd.to_datetime(names["nameendt"], errors="coerce")
    names["cusip8"] = (
        names["ncusip"]
        .fillna(names["cusip"])
        .astype("string")
        .str.replace(r"[^0-9A-Za-z]", "", regex=True)
        .str.upper()
        .str[:8]
    )
    names = names[names["cusip8"].notna()]
    # Extend CRSP end date to cover "today" for mapping
    max_end = names["nameendt"].max()
    today = pd.Timestamp(datetime.utcnow().date())
    if pd.notna(max_end) and today > max_end:
        names.loc[names["nameendt"] == max_end, "nameendt"] = today
    return names[["permno", "permco", "namedt", "nameendt", "cusip8"]]


def load_links() -> pd.DataFrame:
    link_path = CRSP_DIR / "ccmxpf_lnkhist.parquet"
    if not link_path.exists():
        raise FileNotFoundError("Missing ccmxpf_lnkhist.parquet")
    link = pd.read_parquet(link_path)
    link["linkdt"] = pd.to_datetime(link["linkdt"], errors="coerce")
    link["linkenddt"] = pd.to_datetime(link["linkenddt"], errors="coerce")
    # Open-ended links
    link["linkenddt"] = link["linkenddt"].fillna(pd.Timestamp("2099-12-31"))
    return link


def load_fy_end_map() :
    fin_path = WAREHOUSE_DIR / "warehouse_financials.parquet"
    if not fin_path.exists():
        raise FileNotFoundError("Missing warehouse_financials.parquet")
    cols = ["company_id", "fiscal_period_end"]
    df = pd.read_parquet(fin_path, columns=cols)
    df["fiscal_period_end"] = pd.to_datetime(df["fiscal_period_end"], errors="coerce")
    df = df.dropna(subset=["company_id", "fiscal_period_end"])
    # Most recent fiscal_period_end per company
    df = df.sort_values(["company_id", "fiscal_period_end"])
    last = df.groupby("company_id").tail(1)
    out = {}
    for _, row in last.iterrows():
        company_id = str(row["company_id"])
        date = row["fiscal_period_end"]
        out[company_id] = {"month": int(date.month), "day": int(date.day)}
    return out


def estimate_period_end(available_time: pd.Timestamp, fy_end: Optional[Dict[str, int]], period: str) -> pd.Timestamp:
    if fy_end is None:
        # Fallback: one year forward, end of month
        return (available_time + pd.DateOffset(months=12)).to_period("M").to_timestamp("M")
    month = fy_end["month"]
    day = fy_end["day"]
    year = available_time.year
    try:
        candidate = pd.Timestamp(year=year, month=month, day=day)
    except Exception:
        candidate = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    if candidate < available_time:
        candidate = candidate + pd.DateOffset(years=1)
    if period.upper() == "FY2":
        candidate = candidate + pd.DateOffset(years=1)
    if period.upper() == "NTM":
        candidate = (available_time + pd.DateOffset(months=12)).to_period("M").to_timestamp("M")
    return candidate


def map_to_company(df: pd.DataFrame, ric_map: pd.DataFrame, names: pd.DataFrame, links: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    merged = df.merge(ric_map, on="ric", how="left")
    if EST_DEBUG:
        log(f"  map: after ric_map {len(merged):,} | cusip8 missing {merged['cusip8'].isna().mean():.2%}")
    merged = merged.dropna(subset=["cusip8"])
    merged = merged.merge(names, on="cusip8", how="left")
    if EST_DEBUG:
        log(f"  map: after names {len(merged):,} | permno missing {merged['permno'].isna().mean():.2%}")
    # Prefer name-date match, but if it eliminates everything, fall back to latest name
    merged_all = merged.copy()
    name_mask = (as_of >= merged["namedt"]) & (as_of <= merged["nameendt"])
    merged = merged[name_mask]
    if merged.empty:
        if EST_DEBUG:
            log("  map: name-date filter removed all rows; falling back to latest nameendt")
        merged = merged_all
    merged = merged.sort_values(["ric", "nameendt"])
    merged = merged.drop_duplicates(subset=["ric"], keep="last")
    merged = merged.dropna(subset=["permno"])

    # Prefer gvkey if link table provides it, but do NOT drop rows if link is missing.
    merged = merged.merge(
        links,
        left_on="permno",
        right_on="lpermno",
        how="left",
    )
    if EST_DEBUG:
        log(f"  map: after links merge {len(merged):,} | gvkey missing {merged['gvkey'].isna().mean():.2%}")

    # Keep best row per ric (prefer primary links and latest linkenddt)
    merged = merged.sort_values(["ric", "linkprim", "linkenddt"])
    merged = merged.drop_duplicates(subset=["ric"], keep="last")

    merged["permno"] = merged["permno"].astype("Int64")
    merged["permco"] = merged["permco"].astype("Int64")
    merged["gvkey"] = merged["gvkey"].astype("string")
    return merged


