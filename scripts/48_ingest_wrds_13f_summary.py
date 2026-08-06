#!/usr/bin/env python
"""
Ingest WRDS SEC 13F Summary Data into warehouse.

Input: CSV or CSV.GZ from WRDS SEC 13F Summary Data query.
Default path: data/wrds/13f/summary.csv.gz

Env:
  WRDS_13F_SUMMARY_INPUT=path/to/summary.csv.gz
  WRDS_13F_SUMMARY_CHUNK=200000
  WRDS_13F_SUMMARY_PARTITIONED=1  (create data/warehouse/warehouse_13f_filings/)
  WRDS_13F_SUMMARY_LOG_EVERY=1
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion import (
    append_canonical_records,
    compute_raw_payload_hash,
    compute_version_id,
)


DATA_DIR = Path(__file__).parent.parent / "data"

INPUT_PATH = Path(os.getenv("WRDS_13F_SUMMARY_INPUT", DATA_DIR / "wrds" / "13f" / "summary.csv.gz"))
CHUNK = int(os.getenv("WRDS_13F_SUMMARY_CHUNK", "200000"))
PARTITIONED = os.getenv("WRDS_13F_SUMMARY_PARTITIONED", "1") not in ("0", "false", "False")
LOG_EVERY = int(os.getenv("WRDS_13F_SUMMARY_LOG_EVERY", "1"))


PREFERRED_COLUMNS = [
    "cik",
    "coname",
    "form",
    "rdate",
    "fdate",
    "fname",
    "reportdate",
    "report_period",
    "total_value",
    "total_shares",
    "noentries",
    "othermanager",
    "amendmenttype",
    "amendmentno",
]


def log(msg: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def normalize_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def coerce_date(value: Any) -> pd.Timestamp | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        ts = pd.to_datetime(value, errors="coerce")
        if pd.isna(ts):
            return None
        return ts
    except Exception:
        return None


def iter_chunks(path: Path, chunksize: int) -> Iterable[pd.DataFrame]:
    reader = pd.read_csv(
        path,
        compression="infer",
        chunksize=chunksize,
        low_memory=False,
    )
    for chunk in reader:
        yield chunk


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing WRDS 13F summary file: {INPUT_PATH}")

    if PARTITIONED:
        (DATA_DIR / "warehouse" / "warehouse_13f_filings").mkdir(parents=True, exist_ok=True)

    source_system = "wrds_13f_summary"
    total_rows = 0
    total_chunks = 0

    log(f"Loading WRDS 13F summary from {INPUT_PATH}...")

    for chunk in iter_chunks(INPUT_PATH, CHUNK):
        total_chunks += 1
        chunk = chunk.rename(columns={c: c.strip().lower() for c in chunk.columns})

        canonical_records: List[Dict[str, Any]] = []

        available_cols = [c for c in PREFERRED_COLUMNS if c in chunk.columns]
        if not available_cols:
            available_cols = list(chunk.columns)

        for _, row in chunk.iterrows():
            cik = str(row.get("cik", "")).strip()
            rdate = coerce_date(row.get("rdate") or row.get("reportdate") or row.get("report_period"))
            fdate = coerce_date(row.get("fdate"))

            if not cik:
                continue
            if rdate is None and fdate is None:
                continue

            event_time = rdate or fdate
            available_time = fdate or rdate

            entity_id = f"{cik}|{event_time.date().isoformat()}"

            payload = {col: normalize_value(row.get(col)) for col in available_cols}
            payload["cik"] = cik
            payload["rdate"] = normalize_value(rdate)
            payload["fdate"] = normalize_value(fdate)

            raw_payload_hash = compute_raw_payload_hash(payload)
            version_id = compute_version_id(
                source_system=source_system,
                entity_id=entity_id,
                event_time=event_time.to_pydatetime(),
                available_time=available_time.to_pydatetime(),
                raw_payload_hash=raw_payload_hash,
            )

            canonical_records.append(
                {
                    "source_system": source_system,
                    "entity_id": entity_id,
                    "company_id": cik,
                    "security_id": None,
                    "event_time": event_time,
                    "available_time": available_time,
                    "ingestion_time": datetime.utcnow(),
                    "version_id": version_id,
                    "raw_payload_hash": raw_payload_hash,
                    "upstream_version_ids": [],
                    "quality_flags": [],
                    "company_id_type": "cik",
                }
            )

        if canonical_records:
            append_canonical_records("warehouse_13f_filings", canonical_records)
            total_rows += len(canonical_records)

        if LOG_EVERY > 0 and total_chunks % LOG_EVERY == 0:
            log(f"Ingested chunk {total_chunks} | total rows {total_rows:,}")

    log(f"Done. Total 13F filings ingested: {total_rows:,}")


if __name__ == "__main__":
    main()
