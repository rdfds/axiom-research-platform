#!/usr/bin/env python
"""
Targeted backfill of SEC 8-K press release text for records ingested without text.

Reads warehouse_press_releases parquet parts, finds rows with missing text,
fetches the primary document from SEC, and appends new records with text.
Append-only: prior records remain; backfilled records supersede via version_id.

Env:
  SEC_USER_AGENT (required)
  SEC_BACKFILL_START=2000-01-01
  SEC_BACKFILL_END=YYYY-MM-DD (default: today UTC)
  SEC_BACKFILL_LIMIT=0 (0 = no limit)
  SEC_BACKFILL_SLEEP=0.2
  SEC_BACKFILL_TIMEOUT=20
  SEC_BACKFILL_RETRIES=2
  SEC_BACKFILL_RESUME=1
  SEC_BACKFILL_START_INDEX=0  (# of missing-text rows to skip)
  SEC_BACKFILL_BATCH=200
"""

from __future__ import annotations

import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion import (
    compute_raw_payload_hash,
    compute_version_id,
    write_raw_records,
)


DATA_DIR = Path(__file__).parent.parent / "data"
WAREHOUSE_DIR = DATA_DIR / "warehouse"
SEC_DIR = DATA_DIR / "sec"

SEC_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"

SEC_BACKFILL_START = os.getenv("SEC_BACKFILL_START", "2000-01-01")
SEC_BACKFILL_END = os.getenv("SEC_BACKFILL_END", datetime.utcnow().date().isoformat())
SEC_BACKFILL_LIMIT = int(os.getenv("SEC_BACKFILL_LIMIT", "0"))
SEC_BACKFILL_SLEEP = float(os.getenv("SEC_BACKFILL_SLEEP", "0.2"))
SEC_BACKFILL_TIMEOUT = float(os.getenv("SEC_BACKFILL_TIMEOUT", "20"))
SEC_BACKFILL_RETRIES = int(os.getenv("SEC_BACKFILL_RETRIES", "2"))
SEC_BACKFILL_MIN_TEXT = int(os.getenv("SEC_BACKFILL_MIN_TEXT", "200"))
SEC_BACKFILL_ALLOW_FULL_SUBMISSION = os.getenv("SEC_BACKFILL_ALLOW_FULL_SUBMISSION", "1") == "1"
SEC_BACKFILL_ALLOW_EXHIBIT_SEARCH = os.getenv("SEC_BACKFILL_ALLOW_EXHIBIT_SEARCH", "1") == "1"
SEC_BACKFILL_COMPACT_ONLY = os.getenv("SEC_BACKFILL_COMPACT_ONLY", "0") == "1"
SEC_BACKFILL_MAX_MB = float(os.getenv("SEC_BACKFILL_MAX_MB", "0"))
SEC_BACKFILL_SLOW_LOG_SECONDS = float(os.getenv("SEC_BACKFILL_SLOW_LOG_SECONDS", "15"))
SEC_BACKFILL_RESUME = os.getenv("SEC_BACKFILL_RESUME", "1") == "1"
SEC_BACKFILL_START_INDEX = int(os.getenv("SEC_BACKFILL_START_INDEX", "0"))
SEC_BACKFILL_BATCH = int(os.getenv("SEC_BACKFILL_BATCH", "200"))
SEC_FILE_LOG_EVERY = int(os.getenv("SEC_FILE_LOG_EVERY", "100"))
SEC_BACKFILL_START_FILE = int(os.getenv("SEC_BACKFILL_START_FILE", "0"))


def log(msg: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def require_user_agent() -> str:
    user_agent = os.getenv("SEC_USER_AGENT")
    if not user_agent:
        raise RuntimeError(
            "SEC_USER_AGENT not set. Example: export SEC_USER_AGENT='Axiom Research (you@example.com)'"
        )
    return user_agent


def ensure_list(value) -> List:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, pd.Series):
        return value.tolist()
    return [value]


def normalize_flags(value) -> List[str]:
    flat: List[str] = []
    for item in ensure_list(value):
        if item is None:
            continue
        if isinstance(item, (list, tuple, pd.Series)):
            for sub in ensure_list(item):
                if sub is None:
                    continue
                flat.append(str(sub))
            continue
        # numpy arrays or pandas arrays
        if hasattr(item, "tolist") and not isinstance(item, str):
            try:
                for sub in item.tolist():
                    if sub is None:
                        continue
                    flat.append(str(sub))
                continue
            except Exception:
                pass
        flat.append(str(item))
    return flat


def html_to_text(html: str) -> str:
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    text = re.sub(r"\\s+", " ", text)
    return text.strip()


def _fetch_url(url: str, session: requests.Session, sleep_seconds: float) -> Optional[str]:
    last_exc: Optional[Exception] = None
    for attempt in range(SEC_BACKFILL_RETRIES + 1):
        try:
            resp = session.get(url, timeout=SEC_BACKFILL_TIMEOUT)
            resp.raise_for_status()
            if sleep_seconds:
                time.sleep(sleep_seconds)
            text = resp.text
            if "<html" in text.lower():
                return html_to_text(text)
            return text
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < SEC_BACKFILL_RETRIES:
                time.sleep(max(sleep_seconds, 0.2))
                continue
            return None
    if last_exc:
        return None
    return None


def _find_exhibit_doc(cik: str, accession_clean: str, session: requests.Session, sleep_seconds: float) -> Optional[str]:
    index_url = f"{SEC_ARCHIVES_BASE}/{int(cik)}/{accession_clean}/index.json"
    try:
        resp = session.get(index_url, timeout=SEC_BACKFILL_TIMEOUT)
        resp.raise_for_status()
        if sleep_seconds:
            time.sleep(sleep_seconds)
        data = resp.json()
    except Exception:
        return None

    items = data.get("directory", {}).get("item", [])
    if not isinstance(items, list):
        return None

    def _score(name: str) -> int:
        lname = name.lower()
        if "ex99" in lname or "ex-99" in lname or "exhibit99" in lname:
            return 3
        if "press" in lname or "release" in lname:
            return 2
        if lname.endswith((".htm", ".html", ".txt")):
            return 1
        return 0

    candidates = [item.get("name") for item in items if isinstance(item, dict) and item.get("name")]
    candidates = [c for c in candidates if _score(c) > 0]
    if not candidates:
        return None

    candidates.sort(key=_score, reverse=True)
    return candidates[0]


def fetch_filing_text(
    cik: str, accession: str, primary_doc: str, session: requests.Session, sleep_seconds: float
) -> Optional[str]:
    accession_clean = accession.replace("-", "")
    url = f"{SEC_ARCHIVES_BASE}/{int(cik)}/{accession_clean}/{primary_doc}"

    text = _fetch_url(url, session, sleep_seconds)
    if text and len(text) >= SEC_BACKFILL_MIN_TEXT:
        return text

    if SEC_BACKFILL_ALLOW_EXHIBIT_SEARCH:
        exhibit = _find_exhibit_doc(cik, accession_clean, session, sleep_seconds)
        if exhibit and exhibit != primary_doc:
            ex_url = f"{SEC_ARCHIVES_BASE}/{int(cik)}/{accession_clean}/{exhibit}"
            ex_text = _fetch_url(ex_url, session, sleep_seconds)
            if ex_text and len(ex_text) >= SEC_BACKFILL_MIN_TEXT:
                return ex_text

    if SEC_BACKFILL_ALLOW_FULL_SUBMISSION:
        full_url = f"{SEC_ARCHIVES_BASE}/{int(cik)}/{accession_clean}/{accession_clean}.txt"
        full_text = _fetch_url(full_url, session, sleep_seconds)
        if full_text and len(full_text) >= SEC_BACKFILL_MIN_TEXT:
            return full_text

    return None


def iter_press_release_files() -> List[Path]:
    base = WAREHOUSE_DIR / "warehouse_press_releases"
    if SEC_BACKFILL_COMPACT_ONLY:
        return sorted(base.rglob("part_compact.parquet"))
    return sorted(base.rglob("part_*.parquet"))


def write_partitioned(records: List[Dict[str, object]]) -> int:
    if not records:
        return 0
    df = pd.DataFrame(records)
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    df["available_time"] = pd.to_datetime(df["available_time"], errors="coerce")
    df["ingestion_time"] = pd.to_datetime(df["ingestion_time"], errors="coerce")
    for col in ["quality_flags", "upstream_version_ids"]:
        if col in df.columns:
            df[col] = df[col].apply(ensure_list)
    df["year"] = df["event_time"].dt.year.astype("Int64")

    out_dir = WAREHOUSE_DIR / "warehouse_press_releases"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = 0
    for year, ydf in df.groupby("year"):
        if pd.isna(year):
            continue
        year_dir = out_dir / f"year={int(year)}"
        year_dir.mkdir(parents=True, exist_ok=True)
        part_path = year_dir / f"part_{int(datetime.utcnow().timestamp())}_{os.getpid()}.parquet"
        ydf.drop(columns=["year"]).to_parquet(part_path, index=False)
        rows += len(ydf)
    return rows


def _read_parquet_with_progress(path: Path, columns: Optional[List[str]] = None) -> pd.DataFrame:
    if SEC_BACKFILL_MAX_MB and path.stat().st_size > SEC_BACKFILL_MAX_MB * 1024 * 1024:
        log(f"Skipping large parquet (> {SEC_BACKFILL_MAX_MB} MB): {path.name}")
        return pd.DataFrame()

    result: Dict[str, object] = {}
    done = threading.Event()

    def _run() -> None:
        try:
            if columns is None:
                result["df"] = pd.read_parquet(path)
            else:
                result["df"] = pd.read_parquet(path, columns=columns)
        except Exception as exc:
            result["error"] = exc
        finally:
            done.set()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    start = time.perf_counter()
    while not done.wait(timeout=SEC_BACKFILL_SLOW_LOG_SECONDS):
        elapsed = time.perf_counter() - start
        log(f"  still reading {path.name}... {elapsed:.0f}s")
    t.join()
    if "error" in result:
        raise result["error"]  # type: ignore[arg-type]
    return result.get("df", pd.DataFrame())  # type: ignore[return-value]


def main() -> None:
    user_agent = require_user_agent()
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})

    start_dt = pd.to_datetime(SEC_BACKFILL_START, errors="coerce")
    end_dt = pd.to_datetime(SEC_BACKFILL_END, errors="coerce")
    if pd.notna(start_dt) and start_dt.tzinfo is None:
        start_dt = start_dt.tz_localize("UTC")
    if pd.notna(end_dt) and end_dt.tzinfo is None:
        end_dt = end_dt.tz_localize("UTC")

    files = iter_press_release_files()
    if not files:
        log("No press release files found.")
        return
    if SEC_BACKFILL_START_FILE > 0:
        files = files[SEC_BACKFILL_START_FILE:]
    log(f"Found {len(files):,} press release parquet files to scan.")

    checkpoint_path = SEC_DIR / "press_release_text_backfill.txt"
    processed = set()
    if SEC_BACKFILL_RESUME and checkpoint_path.exists():
        processed = set([line.strip() for line in checkpoint_path.read_text().splitlines() if line.strip()])

    total_examined = 0
    total_backfilled = 0
    start_ts = time.perf_counter()
    last_log = start_ts

    raw_records: List[Dict[str, object]] = []
    canonical_records: List[Dict[str, object]] = []
    ingestion_time = datetime.utcnow()

    needed_cols = [
        "text",
        "accession",
        "document_id",
        "primary_document",
        "cik",
        "entity_id",
        "event_time",
        "available_time",
        "filing_date",
        "report_date",
        "acceptance_datetime",
        "headline",
        "quality_flags",
        "release_date",
        "form_type",
    ]

    for fidx, path in enumerate(files, start=1):
        if fidx == 1 or (SEC_FILE_LOG_EVERY and fidx % SEC_FILE_LOG_EVERY == 0):
            size_mb = path.stat().st_size / (1024 * 1024)
            log(f"Scanning file {fidx}/{len(files)}: {path.name} ({size_mb:.1f} MB)")
        try:
            if path.stat().st_size == 0:
                log(f"Skipping empty parquet: {path}")
                continue
            try:
                df = _read_parquet_with_progress(path, columns=needed_cols)
            except Exception:
                df = _read_parquet_with_progress(path, columns=None)
        except Exception as exc:
            log(f"Skipping unreadable parquet {path}: {exc}")
            continue
        if df.empty:
            continue

        # Missing text rows only
        if "text" not in df.columns:
            continue
        missing = df[df["text"].isna() | (df["text"] == "")]
        if missing.empty:
            continue

        for _, row in missing.iterrows():
            total_examined += 1
            if total_examined <= SEC_BACKFILL_START_INDEX:
                continue
            if SEC_BACKFILL_LIMIT and total_backfilled >= SEC_BACKFILL_LIMIT:
                break

            accession = row.get("accession") or row.get("document_id")
            primary_doc = row.get("primary_document")
            cik = row.get("cik") or row.get("entity_id")
            if not accession or not primary_doc or not cik:
                continue
            accession = str(accession)
            cik = str(cik)

            if SEC_BACKFILL_RESUME and accession in processed:
                continue

            event_time = pd.to_datetime(row.get("event_time"), errors="coerce")
            available_time = pd.to_datetime(row.get("available_time"), errors="coerce")
            if pd.isna(event_time) or pd.isna(available_time):
                continue
            if event_time.tzinfo is None:
                event_time = event_time.tz_localize("UTC")
            if available_time.tzinfo is None:
                available_time = available_time.tz_localize("UTC")

            if pd.notna(start_dt) and event_time < start_dt:
                continue
            if pd.notna(end_dt) and event_time > end_dt:
                continue

            text = fetch_filing_text(cik, accession, primary_doc, session, SEC_BACKFILL_SLEEP)
            if not text:
                continue

            quality_flags = normalize_flags(row.get("quality_flags"))
            quality_flags = [f for f in quality_flags if f not in ("missing_data", "partial_coverage")]
            if "backfilled_text" not in quality_flags:
                quality_flags.append("backfilled_text")

            payload = {
                "cik": cik,
                "accession": accession,
                "primary_document": primary_doc,
                "filing_date": row.get("filing_date"),
                "report_date": row.get("report_date"),
                "acceptance_datetime": row.get("acceptance_datetime"),
                "headline": row.get("headline"),
                "text": text,
            }

            raw_payload_hash = compute_raw_payload_hash(payload)
            version_id = compute_version_id(
                source_system="sec_8k_press_release",
                entity_id=str(cik),
                event_time=event_time.to_pydatetime(),
                available_time=available_time.to_pydatetime(),
                raw_payload_hash=raw_payload_hash,
            )

            raw_records.append(
                {
                    "entity_id": str(cik),
                    "company_id": str(cik),
                    "security_id": None,
                    "event_time": event_time,
                    "available_time": available_time,
                    "payload": payload,
                }
            )

            canonical_records.append(
                {
                    "source_system": "sec_8k_press_release",
                    "entity_id": str(cik),
                    "company_id": str(cik),
                    "security_id": None,
                    "event_time": event_time,
                    "available_time": available_time,
                    "ingestion_time": ingestion_time,
                    "version_id": version_id,
                    "raw_payload_hash": raw_payload_hash,
                    "upstream_version_ids": [version_id],
                    "quality_flags": quality_flags,
                    "document_id": accession,
                    "release_date": row.get("release_date"),
                    "headline": row.get("headline"),
                    "text": text,
                    "form_type": row.get("form_type") or "8-K",
                    "cik": cik,
                    "accession": accession,
                    "primary_document": primary_doc,
                }
            )

            processed.add(accession)
            total_backfilled += 1

            if SEC_BACKFILL_RESUME:
                with checkpoint_path.open("a") as f:
                    f.write(f"{accession}\n")

            if len(canonical_records) >= SEC_BACKFILL_BATCH:
                write_raw_records(source_system="sec_8k_press_release", records=raw_records)
                write_partitioned(canonical_records)
                raw_records.clear()
                canonical_records.clear()

            now = time.perf_counter()
            if now - last_log >= 30:
                elapsed = now - start_ts
                log(
                    f"Progress: examined {total_examined:,} | backfilled {total_backfilled:,} | elapsed {elapsed/60:.1f}m"
                )
                last_log = now

        if SEC_BACKFILL_LIMIT and total_backfilled >= SEC_BACKFILL_LIMIT:
            break

        if fidx % 100 == 0:
            elapsed = time.perf_counter() - start_ts
            log(
                f"Progress: files {fidx}/{len(files)} | backfilled {total_backfilled:,} | elapsed {elapsed/60:.1f}m"
            )

    if raw_records:
        write_raw_records(source_system="sec_8k_press_release", records=raw_records)
    if canonical_records:
        write_partitioned(canonical_records)

    log(f"Done. Backfilled {total_backfilled:,} press release records.")


if __name__ == "__main__":
    main()
