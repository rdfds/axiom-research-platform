#!/usr/bin/env python
"""
Ingest Compustat fundamentals into bitemporal warehouse_financials.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion import (
    append_canonical_records,
    compute_raw_payload_hash,
    compute_version_id,
    write_raw_records,
)


DATA_DIR = Path(__file__).parent.parent / "data"
FUND_PATH = DATA_DIR / "fundamentals_quarterly.parquet"


LINE_ITEMS = {
    "revtq": ("income", "Revenue"),
    "cogsq": ("income", "COGS"),
    "xsgaq": ("income", "SGA"),
    "oibdpq": ("income", "EBITDA"),
    "oiadpq": ("income", "OperatingIncome"),
    "niq": ("income", "NetIncome"),
    "xintq": ("income", "InterestExpense"),
    "atq": ("balance_sheet", "TotalAssets"),
    "actq": ("balance_sheet", "CurrentAssets"),
    "cheq": ("balance_sheet", "Cash"),
    "rectq": ("balance_sheet", "Receivables"),
    "invtq": ("balance_sheet", "Inventory"),
    "ppentq": ("balance_sheet", "PP&E"),
    "ltq": ("balance_sheet", "TotalLiabilities"),
    "lctq": ("balance_sheet", "CurrentLiabilities"),
    "dlcq": ("balance_sheet", "DebtCurrent"),
    "dlttq": ("balance_sheet", "DebtLongTerm"),
    "ceqq": ("balance_sheet", "CommonEquity"),
    "seqq": ("balance_sheet", "TotalEquity"),
    "capxy": ("cash_flow", "Capex"),
    "oancfy": ("cash_flow", "OperatingCashFlow"),
    "epspxq": ("income", "EPS"),
    "cshoq": ("balance_sheet", "SharesOut"),
}


def log(msg: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}")


def normalize_value(value) -> Optional[float]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return None


def normalize_int(value) -> Optional[int]:
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return int(value)
    except Exception:
        return None


def normalize_payload(payload: Dict) -> Dict:
    cleaned = {}
    for k, v in payload.items():
        if isinstance(v, pd.Timestamp):
            if pd.isna(v):
                cleaned[k] = None
            else:
                cleaned[k] = v.isoformat()
        elif isinstance(v, (np.integer, np.floating, np.bool_)):
            cleaned[k] = v.item()
        else:
            try:
                cleaned[k] = None if pd.isna(v) else v
            except Exception:
                cleaned[k] = v
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=50000)
    args = parser.parse_args()

    if not FUND_PATH.exists():
        raise FileNotFoundError(f"Missing fundamentals file: {FUND_PATH}")

    df = pd.read_parquet(FUND_PATH)
    log(f"Loaded fundamentals: {len(df):,} rows")

    df["datadate"] = pd.to_datetime(df["datadate"])
    df["rdq"] = pd.to_datetime(df["rdq"])

    ingestion_time = datetime.utcnow()
    source_system = "compustat_fundamentals"

    chunk_size = max(1, args.chunk_size)
    total = 0

    for start in range(0, len(df), chunk_size):
        chunk = df.iloc[start : start + chunk_size]
        raw_records = []
        canonical_records: List[Dict] = []

        for _, row in chunk.iterrows():
            company_id = str(row.get("gvkey"))
            if not company_id or company_id == "nan":
                continue

            event_time = row.get("datadate")
            available_time = row.get("rdq")
            if pd.isna(event_time):
                continue
            if pd.isna(available_time):
                available_time = event_time + pd.Timedelta(days=45)
                quality_flags = ["estimated_available_time"]
            else:
                quality_flags = []

            if available_time < event_time:
                available_time = event_time
                if "estimated_available_time" not in quality_flags:
                    quality_flags.append("estimated_available_time")

            raw_payload = row.to_dict()
            cleaned_payload = normalize_payload(raw_payload)
            raw_payload_hash = compute_raw_payload_hash(cleaned_payload)
            raw_version_id = compute_version_id(
                source_system=source_system,
                entity_id=company_id,
                event_time=event_time.to_pydatetime(),
                available_time=available_time.to_pydatetime(),
                raw_payload_hash=raw_payload_hash,
            )

            raw_records.append(
                {
                    "entity_id": company_id,
                    "company_id": company_id,
                    "security_id": None,
                    "event_time": event_time,
                    "available_time": available_time,
                    "payload": cleaned_payload,
                }
            )

            fiscal_year = normalize_int(row.get("fyearq"))
            fiscal_quarter = normalize_int(row.get("fqtr"))

            for col, (statement_type, line_item) in LINE_ITEMS.items():
                value = normalize_value(row.get(col))
                canonical_records.append(
                    {
                        "source_system": source_system,
                        "entity_id": company_id,
                        "company_id": company_id,
                        "security_id": None,
                        "event_time": event_time,
                        "available_time": available_time,
                        "ingestion_time": ingestion_time,
                        "version_id": compute_version_id(
                            source_system=source_system,
                            entity_id=f"{company_id}:{col}",
                            event_time=event_time.to_pydatetime(),
                            available_time=available_time.to_pydatetime(),
                            raw_payload_hash=raw_payload_hash,
                        ),
                        "raw_payload_hash": raw_payload_hash,
                        "upstream_version_ids": [raw_version_id],
                        "quality_flags": quality_flags,
                        "fiscal_period_end": event_time,
                        "fiscal_year": fiscal_year,
                        "fiscal_quarter": fiscal_quarter,
                        "statement_type": statement_type,
                        "line_item": line_item,
                        "value": value,
                        "currency": "USD",
                        "units": None,
                        "restatement_flag": False,
                    }
                )

        if raw_records:
            write_raw_records(source_system=source_system, records=raw_records)
        if canonical_records:
            append_canonical_records("warehouse_financials", canonical_records)
            total += len(canonical_records)
            log(f"Ingested fundamentals chunk: {len(canonical_records):,} (total {total:,})")

    log(f"Done. Total line items ingested: {total:,}")


if __name__ == "__main__":
    main()
