"""
Ingestion Utilities (MVP)
=========================
Append-only raw lake + normalized warehouse helpers with bitemporal enforcement.
"""

from __future__ import annotations

import json
import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"

QUALITY_FLAGS = [
    "missing_data",
    "delayed_data",
    "partial_coverage",
    "source_conflict",
    "outlier_detected",
    "unit_inconsistency",
    "restatement",
    "stale_data",
    "estimated_available_time",
    "estimated_event_time",
    "schema_violation",
    "balance_sheet_unbalanced",
    "cash_flow_mismatch",
    "missing_line_item",
    "halted_session",
    "missing_trade_day",
    "price_outlier",
    "curve_non_monotonic",
    "jump_outlier",
    "authorization_only",
    "execution_only",
    "open_ended",
    "missing_size",
    "withdrawn",
    "deal_failed",
    "value_missing",
]


def _canonical_json(payload: Dict[str, Any]) -> bytes:
    def _clean(value: Any) -> Any:
        if isinstance(value, datetime):
            return value.isoformat()
        try:
            import pandas as pd  # Local import to avoid hard dependency

            if isinstance(value, pd.Timestamp):
                if pd.isna(value):
                    return None
                return value.isoformat()
            if pd.isna(value):
                return None
        except Exception:
            pass
        try:
            import numpy as np  # Local import to avoid hard dependency

            if isinstance(value, (np.integer, np.floating, np.bool_)):
                return value.item()
        except Exception:
            pass
        return value

    cleaned = {k: _clean(v) for k, v in payload.items()}
    return json.dumps(cleaned, sort_keys=True, separators=(",", ":")).encode("utf-8")


def compute_raw_payload_hash(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def compute_version_id(
    source_system: str,
    entity_id: str,
    event_time: datetime,
    available_time: datetime,
    raw_payload_hash: str,
    schema_version: str = "v1",
) -> str:
    key = f"{source_system}|{entity_id}|{event_time.isoformat()}|{available_time.isoformat()}|{raw_payload_hash}|{schema_version}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def ensure_bitemporal(event_time: datetime, available_time: datetime) -> None:
    if event_time is None or available_time is None:
        raise ValueError("event_time and available_time are required.")
    if available_time < event_time:
        raise ValueError("available_time must be >= event_time.")


def _ensure_list(value: Optional[Any]) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def write_raw_records(
    source_system: str,
    records: Iterable[Dict[str, Any]],
    base_dir: Path = DATA_DIR / "lake",
    schema_version: str = "v1",
) -> pd.DataFrame:
    """
    Append raw payloads to the immutable lake and update raw manifest.

    Each record must provide:
      - entity_id
      - event_time
      - available_time
      - payload (raw payload dict)
      - company_id (optional)
      - security_id (optional)
      - supersedes_version_id (optional)
    """
    ingest_time = datetime.utcnow()
    ingest_date = ingest_time.strftime("%Y-%m-%d")

    raw_dir = base_dir / "raw" / source_system / f"ingest_date={ingest_date}"
    raw_dir.mkdir(parents=True, exist_ok=True)

    raw_path = raw_dir / f"payloads_{ingest_time.strftime('%H%M%S')}.jsonl"
    manifest_rows = []

    with raw_path.open("a", encoding="utf-8") as f:
        for rec in records:
            payload = rec["payload"]
            entity_id = rec["entity_id"]
            event_time = pd.to_datetime(rec["event_time"])
            available_time = pd.to_datetime(rec["available_time"])
            ensure_bitemporal(event_time, available_time)

            raw_payload_hash = compute_raw_payload_hash(payload)
            version_id = compute_version_id(
                source_system=source_system,
                entity_id=str(entity_id),
                event_time=event_time.to_pydatetime(),
                available_time=available_time.to_pydatetime(),
                raw_payload_hash=raw_payload_hash,
                schema_version=schema_version,
            )

            raw_record_id = hashlib.sha256(
                f"{source_system}|{entity_id}|{raw_payload_hash}".encode("utf-8")
            ).hexdigest()[:32]

            f.write(json.dumps({"raw_record_id": raw_record_id, "payload": payload}) + "\n")

            manifest_rows.append(
                {
                    "source_system": source_system,
                    "raw_record_id": raw_record_id,
                    "raw_payload_hash": raw_payload_hash,
                    "raw_path": str(raw_path),
                    "entity_id": entity_id,
                    "company_id": rec.get("company_id"),
                    "security_id": rec.get("security_id"),
                    "event_time": event_time,
                    "available_time": available_time,
                    "ingestion_time": ingest_time,
                    "version_id": version_id,
                    "supersedes_version_id": rec.get("supersedes_version_id"),
                }
            )

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = base_dir / "raw_manifest.parquet"
    if manifest_path.exists():
        try:
            existing = pd.read_parquet(manifest_path)
            manifest = pd.concat([existing, manifest], ignore_index=True, sort=False)
        except Exception:
            # Corrupt/empty manifest; overwrite with new rows
            pass
    manifest.to_parquet(manifest_path, index=False)
    return manifest


def append_canonical_records(
    table_name: str,
    records: Iterable[Dict[str, Any]],
    base_dir: Path = DATA_DIR / "warehouse",
    schema_version: str = "v1",
) -> pd.DataFrame:
    """
    Append normalized records to warehouse with bitemporal enforcement.
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)

    required = [
        "source_system",
        "entity_id",
        "event_time",
        "available_time",
        "ingestion_time",
        "version_id",
        "raw_payload_hash",
        "upstream_version_ids",
        "quality_flags",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    df["event_time"] = pd.to_datetime(df["event_time"])
    df["available_time"] = pd.to_datetime(df["available_time"])
    df["ingestion_time"] = pd.to_datetime(df["ingestion_time"])

    # Normalize optional numeric fields that can come in as mixed types
    if "fiscal_year" in df.columns:
        df["fiscal_year"] = pd.to_numeric(df["fiscal_year"], errors="coerce").astype("Int64")
    if "fiscal_quarter" in df.columns:
        df["fiscal_quarter"] = pd.to_numeric(df["fiscal_quarter"], errors="coerce").astype("Int64")

    for _, row in df.iterrows():
        ensure_bitemporal(row["event_time"], row["available_time"])

    df["upstream_version_ids"] = df["upstream_version_ids"].apply(_ensure_list)
    df["quality_flags"] = df["quality_flags"].apply(_ensure_list)

    dir_path = base_dir / table_name
    if dir_path.exists() and dir_path.is_dir():
        df["year"] = df["event_time"].dt.year.astype("Int64")
        rows = 0
        for year, ydf in df.groupby("year"):
            if pd.isna(year):
                continue
            year_dir = dir_path / f"year={int(year)}"
            year_dir.mkdir(parents=True, exist_ok=True)
            part_path = year_dir / f"part_{int(datetime.utcnow().timestamp())}_{os.getpid()}.parquet"
            ydf.drop(columns=["year"]).to_parquet(part_path, index=False)
            rows += len(ydf)
        return df

    path = base_dir / f"{table_name}.parquet"
    if path.exists():
        existing = pd.read_parquet(path)
        if "fiscal_year" in existing.columns:
            existing["fiscal_year"] = pd.to_numeric(existing["fiscal_year"], errors="coerce").astype("Int64")
        if "fiscal_quarter" in existing.columns:
            existing["fiscal_quarter"] = pd.to_numeric(existing["fiscal_quarter"], errors="coerce").astype("Int64")
        df = pd.concat([existing, df], ignore_index=True, sort=False)

    if "fiscal_year" in df.columns:
        df["fiscal_year"] = pd.to_numeric(df["fiscal_year"], errors="coerce").astype("Int64")
    if "fiscal_quarter" in df.columns:
        df["fiscal_quarter"] = pd.to_numeric(df["fiscal_quarter"], errors="coerce").astype("Int64")
    df.to_parquet(path, index=False)
    return df
