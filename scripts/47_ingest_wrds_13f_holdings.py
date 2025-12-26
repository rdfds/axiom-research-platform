#!/usr/bin/env python
"""
Ingest WRDS SEC 13F Holdings Data (XML-based, complete post-2013) into warehouse.

Input: CSV or CSV.GZ from WRDS SEC 13F Holdings Data query.
Default path: data/wrds/13f/holdings.csv.gz

Env:
  WRDS_13F_INPUT=path/to/holdings.csv.gz
  WRDS_13F_CHUNK=200000
  WRDS_13F_PARTITIONED=1  (create data/warehouse/warehouse_13f_holdings/ for partitioned writes)
  WRDS_13F_LOG_EVERY=1     (log every chunk)
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

INPUT_PATH = Path(os.getenv("WRDS_13F_INPUT", DATA_DIR / "wrds" / "13f" / "holdings.csv.gz"))
CHUNK = int(os.getenv("WRDS_13F_CHUNK", "200000"))
PARTITIONED = os.getenv("WRDS_13F_PARTITIONED", "1") not in ("0", "false", "False")
LOG_EVERY = int(os.getenv("WRDS_13F_LOG_EVERY", "1"))


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


def coerce_numeric(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return pd.to_numeric(value, errors="coerce")


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
        raise FileNotFoundError(f"Missing WRDS 13F holdings file: {INPUT_PATH}")

    if PARTITIONED:
        (DATA_DIR / "warehouse" / "warehouse_13f_holdings").mkdir(parents=True, exist_ok=True)

    source_system = "wrds_13f"
    total_rows = 0
    total_chunks = 0

    log(f"Loading WRDS 13F holdings from {INPUT_PATH}...")

    for chunk in iter_chunks(INPUT_PATH, CHUNK):
        total_chunks += 1
        chunk = chunk.rename(columns={c: c.strip().lower() for c in chunk.columns})

        canonical_records: List[Dict[str, Any]] = []

        for _, row in chunk.iterrows():
            cik = str(row.get("cik", "")).strip()
            cusip = str(row.get("cusip", "")).strip()
            rdate = coerce_date(row.get("rdate"))
            fdate = coerce_date(row.get("fdate"))

            if not cik or not rdate or not fdate:
                continue

            event_time = rdate
            available_time = fdate

            entity_id = f"{cik}|{cusip}|{rdate.date().isoformat()}"
            payload = {
                "cik": cik,
                "coname": normalize_value(row.get("coname")),
                "form": normalize_value(row.get("form")),
                "rdate": normalize_value(rdate),
                "fdate": normalize_value(fdate),
                "fname": normalize_value(row.get("fname")),
                "nameofissuer": normalize_value(row.get("nameofissuer")),
                "titleofclass": normalize_value(row.get("titleofclass")),
                "cusip": cusip or None,
                "value": coerce_numeric(row.get("value")),
                "sshprnamt": coerce_numeric(row.get("sshprnamt")),
                "sshprnamttype": normalize_value(row.get("sshprnamttype")),
                "putcall": normalize_value(row.get("putcall")),
                "investmentdiscretion": normalize_value(row.get("investmentdiscretion")),
                "othermanager": normalize_value(row.get("othermanager")),
                "sole": coerce_numeric(row.get("sole")),
                "shared": coerce_numeric(row.get("shared")),
                "none": coerce_numeric(row.get("none")),
            }

            raw_payload_hash = compute_raw_payload_hash(payload)
            version_id = compute_version_id(
                source_system=source_system,
                entity_id=entity_id,
                event_time=event_time.to_pydatetime(),
                available_time=available_time.to_pydatetime(),
                raw_payload_hash=raw_payload_hash,
            )

            quality_flags = []
            if not cusip:
                quality_flags.append("missing_data")

            canonical_records.append(
                {
                    "source_system": source_system,
                    "entity_id": entity_id,
                    "company_id": cik,          # filer
                    "security_id": cusip or None,  # holding CUSIP if present
                    "event_time": event_time,
                    "available_time": available_time,
                    "ingestion_time": datetime.utcnow(),
                    "version_id": version_id,
                    "raw_payload_hash": raw_payload_hash,
                    "upstream_version_ids": [],
                    "quality_flags": quality_flags,
                    "company_id_type": "cik",
                    "holding_cusip": cusip or None,
                    "report_date": event_time,
                    "filing_date": available_time,
                    "value_k": payload["value"],
                    "shares": payload["sshprnamt"],
                    "put_call": payload["putcall"],
                    "issuer_name": payload["nameofissuer"],
                    "title_of_class": payload["titleofclass"],
                    "form_type": payload["form"],
                    "filer_name": payload["coname"],
                }
            )

        if canonical_records:
            append_canonical_records("warehouse_13f_holdings", canonical_records)
            total_rows += len(canonical_records)

        if LOG_EVERY > 0 and total_chunks % LOG_EVERY == 0:
            log(f"Ingested chunk {total_chunks} | total rows {total_rows:,}")

    log(f"Done. Total 13F holdings ingested: {total_rows:,}")


if __name__ == "__main__":
    main()
