#!/usr/bin/env python
"""
Ingest curated Prices / Corporate Actions / M&A into the bitemporal warehouse.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

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
CURATED_DIR = DATA_DIR / "curated"


def log(msg: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}")


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


def row_to_payload(row: Dict[str, Any]) -> Dict[str, Any]:
    return {k: normalize_value(v) for k, v in row.items()}


def first_non_null(*values) -> Optional[Any]:
    for value in values:
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except Exception:
            pass
        return value
    return None


def iter_chunks(df: pd.DataFrame, chunk_size: int) -> Iterable[pd.DataFrame]:
    if chunk_size <= 0:
        yield df
        return
    for start in range(0, len(df), chunk_size):
        yield df.iloc[start : start + chunk_size]


def ingest_prices(path: Path, chunk_size: int) -> None:
    if not path.exists():
        log(f"Prices file not found: {path}")
        return

    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    log(f"Loaded prices: {len(df):,} rows")

    ingestion_time = datetime.utcnow()
    source_system = "crsp_rdp_prices"

    for chunk in iter_chunks(df, chunk_size):
        raw_records = []
        canonical_records = []

        for _, row in chunk.iterrows():
            trade_date = row.get("date")
            if pd.isna(trade_date):
                continue

            event_time = pd.to_datetime(trade_date)
            available_time = event_time + timedelta(hours=16)

            permno = row.get("permno")
            permco = row.get("permco")
            entity_id = str(permno) if not pd.isna(permno) else None
            company_id = str(permco) if not pd.isna(permco) else None
            security_id = entity_id

            if not entity_id:
                continue

            payload = row_to_payload(row.to_dict())
            raw_payload_hash = compute_raw_payload_hash(payload)
            raw_version_id = compute_version_id(
                source_system=source_system,
                entity_id=entity_id,
                event_time=event_time,
                available_time=available_time,
                raw_payload_hash=raw_payload_hash,
            )

            raw_records.append(
                {
                    "entity_id": entity_id,
                    "company_id": company_id,
                    "security_id": security_id,
                    "event_time": event_time,
                    "available_time": available_time,
                    "payload": payload,
                }
            )

            prc = row.get("prc")
            close_price = abs(prc) if prc is not None and not pd.isna(prc) else None
            quality_flags: List[str] = ["estimated_available_time"]
            if close_price is None:
                quality_flags.append("missing_data")

            canonical_records.append(
                {
                    "source_system": source_system,
                    "entity_id": entity_id,
                    "company_id": company_id,
                    "security_id": security_id,
                    "event_time": event_time,
                    "available_time": available_time,
                    "ingestion_time": ingestion_time,
                    "version_id": raw_version_id,
                    "raw_payload_hash": raw_payload_hash,
                    "upstream_version_ids": [raw_version_id],
                    "quality_flags": quality_flags,
                    "trade_date": event_time,
                    "open": None,
                    "high": None,
                    "low": None,
                    "close": close_price,
                    "adjusted_close": close_price,
                    "volume": normalize_value(row.get("vol")),
                    "total_return_index": None,
                    "ret": normalize_value(row.get("ret")),
                    "retx": normalize_value(row.get("retx")),
                    "cusip": normalize_value(row.get("cusip")),
                }
            )

        if raw_records:
            write_raw_records(source_system=source_system, records=raw_records)
        if canonical_records:
            append_canonical_records("warehouse_prices", canonical_records)
            log(f"Ingested prices chunk: {len(canonical_records):,}")


def ingest_corporate_actions(path: Path, chunk_size: int) -> None:
    if not path.exists():
        log(f"Corporate actions file not found: {path}")
        return

    df = pd.read_parquet(path)
    log(f"Loaded corporate actions: {len(df):,} rows")

    ingestion_time = datetime.utcnow()
    source_system = "corporate_actions_master"

    for chunk in iter_chunks(df, chunk_size):
        raw_records = []
        canonical_records = []

        for _, row in chunk.iterrows():
            action_date = first_non_null(
                row.get("action_date"),
                row.get("exdt"),
                row.get("dclrdt"),
                row.get("rcrddt"),
                row.get("paydt"),
                row.get("dlstdt"),
            )
            if action_date is None or pd.isna(action_date):
                continue

            announcement_date = first_non_null(row.get("dclrdt"), row.get("action_date"))
            event_time = pd.to_datetime(announcement_date) if announcement_date is not None else pd.to_datetime(action_date)
            available_time = pd.to_datetime(announcement_date) if announcement_date is not None else event_time

            effective_date = first_non_null(row.get("exdt"), row.get("paydt"), row.get("action_date"))

            permno = row.get("permno")
            permco = row.get("permco")
            gvkey = row.get("gvkey")
            company_id = str(gvkey) if gvkey is not None and not pd.isna(gvkey) else (str(permco) if not pd.isna(permco) else None)
            security_id = str(permno) if not pd.isna(permno) else None
            entity_id = company_id or security_id
            if not entity_id:
                continue

            if available_time < event_time:
                available_time = event_time

            payload = row_to_payload(row.to_dict())
            raw_payload_hash = compute_raw_payload_hash(payload)
            raw_version_id = compute_version_id(
                source_system=source_system,
                entity_id=entity_id,
                event_time=event_time,
                available_time=available_time,
                raw_payload_hash=raw_payload_hash,
            )

            raw_records.append(
                {
                    "entity_id": entity_id,
                    "company_id": company_id,
                    "security_id": security_id,
                    "event_time": event_time,
                    "available_time": available_time,
                    "payload": payload,
                }
            )

            size = first_non_null(
                row.get("amount"),
                row.get("divamt"),
                row.get("div_amount"),
                row.get("deal_amount"),
                row.get("buyback_amount_qtr"),
                row.get("deal_value"),
                row.get("ratio"),
            )

            quality_flags: List[str] = []
            if announcement_date is None or pd.isna(announcement_date):
                quality_flags.append("estimated_available_time")
            if size is None or pd.isna(size):
                quality_flags.append("missing_size")

            canonical_records.append(
                {
                    "source_system": source_system,
                    "entity_id": entity_id,
                    "company_id": company_id,
                    "security_id": security_id,
                    "event_time": event_time,
                    "available_time": available_time,
                    "ingestion_time": ingestion_time,
                    "version_id": raw_version_id,
                    "raw_payload_hash": raw_payload_hash,
                    "upstream_version_ids": [raw_version_id],
                    "quality_flags": quality_flags,
                    "action_type": normalize_value(row.get("action_type")),
                    "action_subtype": normalize_value(row.get("action_subtype")),
                    "announcement_date": pd.to_datetime(announcement_date) if announcement_date is not None else None,
                    "effective_date": pd.to_datetime(effective_date) if effective_date is not None else None,
                    "size": normalize_value(size),
                    "units": None,
                    "funding_source": None,
                    "source_action_type": normalize_value(row.get("source_action_type")),
                    "source_action_subtype": normalize_value(row.get("source_action_subtype")),
                    "company_name": normalize_value(row.get("company_name")),
                    "ticker": normalize_value(row.get("ticker")),
                    "cusip": normalize_value(row.get("cusip")),
                }
            )

        if raw_records:
            write_raw_records(source_system=source_system, records=raw_records)
        if canonical_records:
            append_canonical_records("warehouse_corp_actions", canonical_records)
            log(f"Ingested corp actions chunk: {len(canonical_records):,}")


def ingest_mna(path: Path, chunk_size: int) -> None:
    if not path.exists():
        log(f"M&A file not found: {path}")
        return

    df = pd.read_parquet(path)
    log(f"Loaded M&A: {len(df):,} rows")

    ingestion_time = datetime.utcnow()
    source_system = "mna_master"

    for chunk in iter_chunks(df, chunk_size):
        raw_records = []
        canonical_records = []

        for _, row in chunk.iterrows():
            announce_date = first_non_null(row.get("announce_date"), row.get("event_date"))
            if announce_date is None or pd.isna(announce_date):
                continue

            event_time = pd.to_datetime(announce_date)
            available_time = event_time

            deal_id = row.get("deal_id") or row.get("source_id")
            if deal_id is None or pd.isna(deal_id):
                deal_id = f"mna_{event_time.date()}_{row.name}"

            target_company_id = first_non_null(row.get("target_permco"), row.get("target_id"))
            acquiror_company_id = first_non_null(row.get("acquiror_permco"), row.get("acquiror_id"))

            entity_id = str(target_company_id) if target_company_id is not None and not pd.isna(target_company_id) else str(deal_id)

            payload = row_to_payload(row.to_dict())
            raw_payload_hash = compute_raw_payload_hash(payload)
            raw_version_id = compute_version_id(
                source_system=source_system,
                entity_id=str(deal_id),
                event_time=event_time,
                available_time=available_time,
                raw_payload_hash=raw_payload_hash,
            )

            raw_records.append(
                {
                    "entity_id": str(deal_id),
                    "company_id": None,
                    "security_id": None,
                    "event_time": event_time,
                    "available_time": available_time,
                    "payload": payload,
                }
            )

            deal_value = row.get("deal_value")
            quality_flags: List[str] = []
            if deal_value is None or pd.isna(deal_value):
                quality_flags.append("value_missing")

            canonical_records.append(
                {
                    "source_system": source_system,
                    "entity_id": entity_id,
                    "company_id": None,
                    "security_id": None,
                    "event_time": event_time,
                    "available_time": available_time,
                    "ingestion_time": ingestion_time,
                    "version_id": raw_version_id,
                    "raw_payload_hash": raw_payload_hash,
                    "upstream_version_ids": [raw_version_id],
                    "quality_flags": quality_flags,
                    "deal_id": normalize_value(deal_id),
                    "acquirer_company_id": normalize_value(acquiror_company_id),
                    "target_company_id": normalize_value(target_company_id),
                    "announcement_date": event_time,
                    "close_date": normalize_value(row.get("completion_date")),
                    "deal_value": normalize_value(deal_value),
                    "consideration_type": normalize_value(row.get("payment_type")),
                    "deal_type": normalize_value(row.get("deal_type")),
                    "status": normalize_value(row.get("deal_status")),
                    "target_name": normalize_value(row.get("target_name")),
                    "acquiror_name": normalize_value(row.get("acquiror_name")),
                }
            )

        if raw_records:
            write_raw_records(source_system=source_system, records=raw_records)
        if canonical_records:
            append_canonical_records("warehouse_mna_deals", canonical_records)
            log(f"Ingested M&A chunk: {len(canonical_records):,}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", action="store_true", help="Ingest prices")
    parser.add_argument("--corp-actions", action="store_true", help="Ingest corporate actions")
    parser.add_argument("--mna", action="store_true", help="Ingest M&A deals")
    parser.add_argument("--chunk-size", type=int, default=50000)
    args = parser.parse_args()

    if not (args.prices or args.corp_actions or args.mna):
        args.prices = True
        args.corp_actions = True
        args.mna = True

    if args.prices:
        ingest_prices(CURATED_DIR / "prices_master_full.parquet", args.chunk_size)
    if args.corp_actions:
        ingest_corporate_actions(CURATED_DIR / "corporate_actions_master.parquet", args.chunk_size)
    if args.mna:
        ingest_mna(CURATED_DIR / "mna_master.parquet", args.chunk_size)


if __name__ == "__main__":
    main()
