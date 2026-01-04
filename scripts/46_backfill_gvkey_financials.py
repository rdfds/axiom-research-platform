#!/usr/bin/env python
"""
Backfill GVKEYs for FMP financials ingested with company_id = symbol.

This script appends new canonical records with company_id/entity_id set to gvkey,
leaving original symbol-based records intact (append-only).

Env:
  BACKFILL_START_YEAR=1998
  BACKFILL_END_YEAR=2026
  BACKFILL_RESUME=1
  BACKFILL_FLUSH_EVERY=50000
  BACKFILL_DEBUG=0
"""

from __future__ import annotations

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion import append_canonical_records, compute_version_id


DATA_DIR = Path(__file__).parent.parent / "data"
WAREHOUSE_DIR = DATA_DIR / "warehouse"
CRSP_DIR = DATA_DIR / "wrds" / "crsp"
COMP_DIR = DATA_DIR / "wrds" / "compustat"
SEC_DIR = DATA_DIR / "sec"

BACKFILL_START_YEAR = int(os.getenv("BACKFILL_START_YEAR", "1998"))
BACKFILL_END_YEAR = int(os.getenv("BACKFILL_END_YEAR", "2026"))
BACKFILL_RESUME = os.getenv("BACKFILL_RESUME", "1") == "1"
BACKFILL_FLUSH_EVERY = int(os.getenv("BACKFILL_FLUSH_EVERY", "50000"))
BACKFILL_DEBUG = os.getenv("BACKFILL_DEBUG", "0") == "1"


def log(msg: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def load_mappings() -> tuple[pd.DataFrame, pd.DataFrame]:
    names_path = CRSP_DIR / "msenames_2000-01-01_to_2026-12-31.parquet"
    link_path = CRSP_DIR / "ccmxpf_lnkhist.parquet"
    if not names_path.exists() or not link_path.exists():
        raise FileNotFoundError("Missing CRSP mapping files for gvkey backfill.")

    names = pd.read_parquet(names_path, columns=["permno", "namedt", "nameendt", "ticker"])
    try:
        link = pd.read_parquet(link_path, columns=["permno", "gvkey", "linkdt", "linkenddt"])
    except Exception:
        link = pd.read_parquet(link_path, columns=["lpermno", "gvkey", "linkdt", "linkenddt"])
        link = link.rename(columns={"lpermno": "permno"})

    names["namedt"] = pd.to_datetime(names["namedt"], errors="coerce")
    names["nameendt"] = pd.to_datetime(names["nameendt"], errors="coerce")
    link["linkdt"] = pd.to_datetime(link["linkdt"], errors="coerce")
    link["linkenddt"] = pd.to_datetime(link["linkenddt"], errors="coerce")

    names["ticker_norm"] = names["ticker"].astype("string").str.upper().str.replace("-", ".", regex=False)
    link["permno"] = pd.to_numeric(link["permno"], errors="coerce")
    return names, link


def load_cik_gvkey() -> pd.DataFrame:
    path = COMP_DIR / "cik_gvkey.csv.gz"
    if not path.exists():
        raise FileNotFoundError("Missing Compustat CIK-GVKEY file at data/wrds/compustat/cik_gvkey.csv.gz")
    df = pd.read_csv(path, dtype=str)
    df.columns = [c.lower() for c in df.columns]
    # Normalize
    df["cik"] = df["cik"].astype(str).str.replace(r"^0+", "", regex=True)
    df["gvkey"] = df["gvkey"].astype(str).str.strip()
    if "link_start_date" in df.columns:
        df["link_start_date"] = pd.to_datetime(df["link_start_date"], errors="coerce")
    if "link_end_date" in df.columns:
        df["link_end_date"] = pd.to_datetime(df["link_end_date"], errors="coerce")
    return df


def load_symbol_to_cik() -> dict:
    path = SEC_DIR / "company_tickers.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    mapping = {}
    for _, row in data.items():
        ticker = str(row.get("ticker", "")).upper().strip().replace("-", ".")
        cik = str(row.get("cik_str", "")).lstrip("0")
        if ticker and cik:
            mapping[ticker] = cik
    return mapping


def load_checkpoint() -> set[str]:
    if not BACKFILL_RESUME:
        return set()
    path = DATA_DIR / "fmp" / "fmp_financials_gvkey_backfill_checkpoint.txt"
    if not path.exists():
        return set()
    return set([line.strip() for line in path.read_text().splitlines() if line.strip()])


def save_checkpoint(entries: Iterable[str]) -> None:
    path = DATA_DIR / "fmp" / "fmp_financials_gvkey_backfill_checkpoint.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for entry in entries:
            f.write(entry + "\n")


def filter_target(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df[df["source_system"] == "fmp_financials"]
    # Only symbol-based company_id (contains letters)
    mask = df["company_id"].astype("string").str.contains(r"[A-Za-z]", regex=True, na=False)
    return df[mask]


def map_gvkey(
    df: pd.DataFrame,
    names: pd.DataFrame,
    link: pd.DataFrame,
    cik_gvkey: pd.DataFrame,
    symbol_to_cik: dict,
) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["symbol_norm"] = df["company_id"].astype("string").str.upper().str.replace("-", ".", regex=False)

    # First try CIK -> GVKEY using SEC mapping + Compustat link table
    if symbol_to_cik:
        df["cik"] = df["symbol_norm"].map(symbol_to_cik)
        merged_cik = df.merge(cik_gvkey, on="cik", how="left")
        if "link_start_date" in merged_cik.columns:
            merged_cik = merged_cik[
                (merged_cik["event_time"] >= merged_cik["link_start_date"]) &
                (merged_cik["event_time"] <= merged_cik["link_end_date"])
            ]
        merged_cik = merged_cik[merged_cik["gvkey"].notna()]
    else:
        merged_cik = pd.DataFrame()

    # Fallback to CRSP ticker -> permno -> gvkey
    merged = df.merge(
        names[["permno", "namedt", "nameendt", "ticker_norm"]],
        left_on="symbol_norm",
        right_on="ticker_norm",
        how="left",
    )
    merged = merged[
        (merged["event_time"] >= merged["namedt"]) & (merged["event_time"] <= merged["nameendt"])
    ]
    merged = merged.sort_values("nameendt").groupby("version_id", as_index=False).tail(1)
    merged = merged.merge(
        link[["permno", "gvkey", "linkdt", "linkenddt"]],
        on="permno",
        how="left",
    )
    active = merged[
        (merged["event_time"] >= merged["linkdt"]) & (merged["event_time"] <= merged["linkenddt"])
    ]
    if active.empty:
        merged = merged.sort_values("linkenddt").groupby("version_id", as_index=False).tail(1)
    else:
        merged = active.sort_values("linkenddt").groupby("version_id", as_index=False).tail(1)

    if merged_cik.empty:
        return merged

    # Prefer CIK-based gvkey where available, else CRSP fallback
    merged_cik = merged_cik.rename(columns={"gvkey": "gvkey_cik"})
    merged = merged.rename(columns={"gvkey": "gvkey_crsp"})
    out = merged.merge(merged_cik[["version_id", "gvkey_cik"]], on="version_id", how="left")
    out["gvkey"] = out["gvkey_cik"].combine_first(out["gvkey_crsp"])
    return out


def build_records(mapped: pd.DataFrame, ingestion_time: datetime) -> List[Dict]:
    if mapped.empty:
        return []
    records: List[Dict] = []
    for _, row in mapped.iterrows():
        gvkey = row.get("gvkey")
        if pd.isna(gvkey):
            continue
        gvkey = str(gvkey)

        quality_flags = []
        if isinstance(row.get("quality_flags"), list):
            quality_flags = [f for f in row["quality_flags"] if f != "missing_gvkey"]

        raw_payload_hash = row["raw_payload_hash"]
        version_id = compute_version_id(
            source_system="fmp_financials",
            entity_id=f"{gvkey}:{row['statement_type']}:{row['line_item']}",
            event_time=row["event_time"].to_pydatetime(),
            available_time=row["available_time"].to_pydatetime(),
            raw_payload_hash=raw_payload_hash,
        )

        records.append(
            {
                "source_system": "fmp_financials",
                "entity_id": gvkey,
                "company_id": gvkey,
                "security_id": None,
                "event_time": row["event_time"],
                "available_time": row["available_time"],
                "ingestion_time": ingestion_time,
                "version_id": version_id,
                "raw_payload_hash": raw_payload_hash,
                "upstream_version_ids": [row["version_id"]],
                "quality_flags": quality_flags,
                "fiscal_period_end": row["fiscal_period_end"],
                "fiscal_year": row.get("fiscal_year"),
                "fiscal_quarter": row.get("fiscal_quarter"),
                "statement_type": row["statement_type"],
                "line_item": row["line_item"],
                "value": row["value"],
                "currency": row.get("currency"),
                "units": row.get("units"),
                "restatement_flag": row.get("restatement_flag", False),
            }
        )
    return records


def main() -> None:
    log("Loading mappings for gvkey backfill...")
    names, link = load_mappings()
    cik_gvkey = load_cik_gvkey()
    symbol_to_cik = load_symbol_to_cik()
    log(f"Loaded CIK-GVKEY rows: {len(cik_gvkey):,} | SEC ticker map: {len(symbol_to_cik):,}")
    checkpoint = load_checkpoint()
    ingestion_time = datetime.utcnow()

    table_dir = WAREHOUSE_DIR / "warehouse_financials"
    if not table_dir.exists():
        raise FileNotFoundError("warehouse_financials partitioned directory not found.")

    total_backfilled = 0
    buffer: List[Dict] = []

    for year in range(BACKFILL_START_YEAR, BACKFILL_END_YEAR + 1):
        year_dir = table_dir / f"year={year}"
        if not year_dir.exists():
            continue
        files = sorted(year_dir.glob("part_*.parquet"))
        for fpath in files:
            key = str(fpath)
            if key in checkpoint:
                continue
            if BACKFILL_DEBUG:
                log(f"Scanning {fpath.name}")
            try:
                df = pd.read_parquet(
                    fpath,
                    columns=[
                        "source_system",
                        "company_id",
                        "entity_id",
                        "event_time",
                        "available_time",
                        "ingestion_time",
                        "version_id",
                        "raw_payload_hash",
                        "upstream_version_ids",
                        "quality_flags",
                        "fiscal_period_end",
                        "fiscal_year",
                        "fiscal_quarter",
                        "statement_type",
                        "line_item",
                        "value",
                        "currency",
                        "units",
                        "restatement_flag",
                    ],
                )
            except Exception:
                save_checkpoint([key])
                continue

            df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
            df["available_time"] = pd.to_datetime(df["available_time"], errors="coerce")

            target = filter_target(df)
            if target.empty:
                save_checkpoint([key])
                continue

            mapped = map_gvkey(target, names, link, cik_gvkey, symbol_to_cik)
            if mapped.empty:
                save_checkpoint([key])
                continue

            records = build_records(mapped, ingestion_time)
            if records:
                buffer.extend(records)
            if len(buffer) >= BACKFILL_FLUSH_EVERY:
                append_canonical_records("warehouse_financials", buffer)
                total_backfilled += len(buffer)
                log(f"Backfilled {len(buffer):,} records (total {total_backfilled:,})")
                buffer = []

            save_checkpoint([key])

    if buffer:
        append_canonical_records("warehouse_financials", buffer)
        total_backfilled += len(buffer)
    log(f"Done. Total backfilled records: {total_backfilled:,}")


if __name__ == "__main__":
    main()
