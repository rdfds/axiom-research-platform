#!/usr/bin/env python
"""
Ingest SEC 8-K press releases (public) into warehouse_press_releases.

This is the MVP B4 source. We use SEC submissions JSON and pull the
primary document for 8-K filings. We treat:
  event_time = reportDate if present else filingDate (estimated)
  available_time = acceptanceDateTime (SEC timestamp)

Env:
  SEC_USER_AGENT   (required)
  SEC_START=2000-01-01
  SEC_END=YYYY-MM-DD (default: today UTC)
  SEC_SLEEP=0.2
  SEC_LIMIT_CIKS=0 (0 = all)
  SEC_MAX_PER_CIK=200
  SEC_FETCH_TEXT=1 (0 = metadata only)
  SEC_UNIVERSE_DATE=YYYY-MM-DD (optional)
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
import time as _time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion import (
    append_canonical_records,
    compute_raw_payload_hash,
    compute_version_id,
    write_raw_records,
)


DATA_DIR = Path(__file__).parent.parent / "data"
SEC_DIR = DATA_DIR / "sec"
CRSP_DIR = DATA_DIR / "wrds" / "crsp"

SEC_TICKER_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"

SEC_START = os.getenv("SEC_START", "2000-01-01")
SEC_END = os.getenv("SEC_END", datetime.utcnow().date().isoformat())
SEC_SLEEP = float(os.getenv("SEC_SLEEP", "0.2"))
SEC_TIMEOUT = float(os.getenv("SEC_TIMEOUT", "20"))
SEC_RETRIES = int(os.getenv("SEC_RETRIES", "2"))
SEC_LIMIT_CIKS = int(os.getenv("SEC_LIMIT_CIKS", "0"))
SEC_MAX_PER_CIK = int(os.getenv("SEC_MAX_PER_CIK", "200"))
SEC_FETCH_TEXT = os.getenv("SEC_FETCH_TEXT", "1") == "1"
SEC_UNIVERSE_DATE = os.getenv("SEC_UNIVERSE_DATE")
SEC_RESUME = os.getenv("SEC_RESUME", "1") == "1"
SEC_START_INDEX = int(os.getenv("SEC_START_INDEX", "0"))

WAREHOUSE_DIR = DATA_DIR / "warehouse"


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


def ensure_dirs() -> None:
    SEC_DIR.mkdir(parents=True, exist_ok=True)
    (SEC_DIR / "submissions").mkdir(parents=True, exist_ok=True)


def fetch_json(url: str, session: requests.Session, sleep_seconds: float) -> Dict:
    last_exc: Optional[Exception] = None
    for attempt in range(SEC_RETRIES + 1):
        try:
            resp = session.get(url, timeout=SEC_TIMEOUT)
            resp.raise_for_status()
            if sleep_seconds:
                time.sleep(sleep_seconds)
            return resp.json()
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < SEC_RETRIES:
                time.sleep(max(sleep_seconds, 0.2))
                continue
            raise
    if last_exc:
        raise last_exc
    raise RuntimeError("Unknown SEC request error")


def load_sec_tickers(session: requests.Session, sleep_seconds: float) -> pd.DataFrame:
    cache_path = SEC_DIR / "company_tickers.json"
    if cache_path.exists():
        data = json.loads(cache_path.read_text())
    else:
        log("Downloading SEC ticker mapping...")
        data = fetch_json(SEC_TICKER_URL, session, sleep_seconds)
        cache_path.write_text(json.dumps(data))

    rows = []
    for _, row in data.items():
        cik = str(row["cik_str"]).zfill(10)
        rows.append(
            {
                "cik": cik,
                "ticker": str(row.get("ticker", "")).upper().strip(),
                "title": row.get("title"),
            }
        )
    return pd.DataFrame(rows).drop_duplicates()


def load_universe_tickers(universe_date: Optional[str]) -> pd.DataFrame:
    universe_path = DATA_DIR / "curated" / "universe_r3000_proxy.parquet"
    names_path = CRSP_DIR / "msenames_2000-01-01_to_2026-12-31.parquet"
    if not universe_path.exists():
        raise FileNotFoundError(f"Missing universe file: {universe_path}")
    if not names_path.exists():
        raise FileNotFoundError(f"Missing CRSP names file: {names_path}")

    universe = pd.read_parquet(universe_path)
    universe["date"] = pd.to_datetime(universe["date"])
    asof_date = pd.to_datetime(universe_date) if universe_date else universe["date"].max()
    universe = universe[universe["date"] == asof_date][["permno", "permco"]]

    names = pd.read_parquet(names_path, columns=["permno", "namedt", "nameendt", "ticker", "cusip", "comnam"])
    names["namedt"] = pd.to_datetime(names["namedt"])
    names["nameendt"] = pd.to_datetime(names["nameendt"])
    active = names[(names["namedt"] <= asof_date) & (names["nameendt"] >= asof_date)]
    active = active.sort_values(["permno", "nameendt"])
    latest = active.drop_duplicates(subset=["permno"], keep="last")
    merged = universe.merge(latest, on="permno", how="left")
    merged["ticker"] = merged["ticker"].str.upper().str.strip()
    return merged


def build_cik_universe(sec_tickers: pd.DataFrame, universe_tickers: pd.DataFrame, limit: int = 0) -> pd.DataFrame:
    merged = universe_tickers.merge(sec_tickers, on="ticker", how="left")
    merged = merged.dropna(subset=["cik"]).drop_duplicates(subset=["cik"])
    if limit and limit > 0:
        merged = merged.head(limit)
    return merged


def load_submissions(cik: str, session: requests.Session, sleep_seconds: float) -> Optional[Dict]:
    cache_path = SEC_DIR / "submissions" / f"CIK{cik}.json"
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text())
        except json.JSONDecodeError:
            # Corrupt/empty cache file; delete and refetch
            try:
                cache_path.unlink()
            except Exception:
                pass
    url = SEC_SUBMISSIONS_URL.format(cik=cik)
    try:
        payload = fetch_json(url, session, sleep_seconds)
    except requests.RequestException as exc:
        log(f"Failed CIK {cik}: {exc}")
        return None
    cache_path.write_text(json.dumps(payload))
    return payload


def html_to_text(html: str) -> str:
    # Remove script/style and tags
    html = re.sub(r"(?is)<(script|style).*?>.*?(</\\1>)", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    text = re.sub(r"\\s+", " ", text)
    return text.strip()


def fetch_filing_text(cik: str, accession: str, primary_doc: str, session: requests.Session, sleep_seconds: float) -> Optional[str]:
    acc_no = accession.replace("-", "")
    url = f"{SEC_ARCHIVES_BASE}/{int(cik)}/{acc_no}/{primary_doc}"
    resp = session.get(url, timeout=SEC_TIMEOUT)
    if resp.status_code != 200:
        return None
    if sleep_seconds:
        time.sleep(sleep_seconds)
    content_type = resp.headers.get("Content-Type", "")
    text = resp.text
    if "html" in content_type.lower() or primary_doc.lower().endswith((".htm", ".html")):
        return html_to_text(text)
    return text


def write_partitioned(records: List[Dict]) -> int:
    if not records:
        return 0
    df = pd.DataFrame(records)
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    df["available_time"] = pd.to_datetime(df["available_time"], errors="coerce")
    df["ingestion_time"] = pd.to_datetime(df["ingestion_time"], errors="coerce")
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


def parse_recent_filings(payload: Dict, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    filings = payload.get("filings", {}).get("recent", {})
    if not filings:
        return pd.DataFrame()
    df = pd.DataFrame(filings)
    if df.empty:
        return df
    df["filingDate"] = pd.to_datetime(df.get("filingDate"), errors="coerce")
    df["reportDate"] = pd.to_datetime(df.get("reportDate"), errors="coerce")
    df["acceptanceDateTime"] = pd.to_datetime(df.get("acceptanceDateTime"), errors="coerce")
    df = df[df.get("form") == "8-K"]
    if df.empty:
        return df
    # Filter by filing date
    df = df[(df["filingDate"] >= start) & (df["filingDate"] <= end)]
    return df


def main() -> None:
    ensure_dirs()
    user_agent = require_user_agent()
    start = pd.to_datetime(SEC_START)
    end = pd.to_datetime(SEC_END)

    session = requests.Session()
    session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})

    sec_tickers = load_sec_tickers(session, SEC_SLEEP)
    universe = load_universe_tickers(SEC_UNIVERSE_DATE)
    cik_universe = build_cik_universe(sec_tickers, universe, SEC_LIMIT_CIKS)
    log(f"CIKs to pull: {len(cik_universe):,}")

    source_system = "sec_8k_press_release"
    ingestion_time = datetime.utcnow()

    checkpoint_path = SEC_DIR / "press_releases_checkpoint.txt"
    processed = set()
    if SEC_RESUME and checkpoint_path.exists():
        processed = set([line.strip() for line in checkpoint_path.read_text().splitlines() if line.strip()])

    total = 0
    start_ts = _time.perf_counter()
    for idx, row in cik_universe.iterrows():
        if idx < SEC_START_INDEX:
            continue
        if (idx + 1) % 100 == 1:
            log(f"Processing CIK {idx + 1}/{len(cik_universe)}")
        cik = row["cik"]
        if SEC_RESUME and cik in processed:
            continue
        try:
            submissions = load_submissions(cik, session, SEC_SLEEP)
        except requests.RequestException as exc:
            log(f"Failed CIK {cik}: {exc}")
            continue
        if not submissions:
            if SEC_RESUME:
                with checkpoint_path.open("a") as f:
                    f.write(f"{cik}\n")
            continue
        recent = parse_recent_filings(submissions, start, end)
        if recent.empty:
            if SEC_RESUME:
                with checkpoint_path.open("a") as f:
                    f.write(f"{cik}\n")
            continue
        if SEC_MAX_PER_CIK and len(recent) > SEC_MAX_PER_CIK:
            recent = recent.head(SEC_MAX_PER_CIK)

        raw_records = []
        canonical_records = []

        for _, filing in recent.iterrows():
            accession = filing.get("accessionNumber")
            primary_doc = filing.get("primaryDocument")
            filing_date = pd.to_datetime(filing.get("filingDate"), errors="coerce")
            report_date = pd.to_datetime(filing.get("reportDate"), errors="coerce")
            acceptance = pd.to_datetime(filing.get("acceptanceDateTime"), errors="coerce")
            if pd.notna(acceptance):
                # SEC acceptanceDateTime is timezone-aware (UTC); make all timestamps UTC
                acceptance = acceptance.tz_convert("UTC") if acceptance.tzinfo else acceptance.tz_localize("UTC")
            if pd.notna(filing_date):
                filing_date = filing_date.tz_localize("UTC")
            if pd.notna(report_date):
                report_date = report_date.tz_localize("UTC")
            doc_desc = filing.get("primaryDocDescription")

            if pd.isna(filing_date):
                continue

            event_time = report_date if pd.notna(report_date) else filing_date
            available_time = acceptance if pd.notna(acceptance) else filing_date

            quality_flags = ["partial_coverage"]
            if pd.isna(report_date):
                quality_flags.append("estimated_event_time")
            if pd.isna(acceptance):
                quality_flags.append("estimated_available_time")

            if available_time < event_time:
                event_time = available_time
                if "estimated_event_time" not in quality_flags:
                    quality_flags.append("estimated_event_time")

            text = None
            if SEC_FETCH_TEXT and accession and primary_doc:
                text = fetch_filing_text(cik, accession, primary_doc, session, SEC_SLEEP)

            payload = {
                "cik": cik,
                "accession": accession,
                "primary_document": primary_doc,
                "filing_date": filing_date.isoformat() if hasattr(filing_date, "isoformat") else str(filing_date),
                "report_date": report_date.isoformat() if hasattr(report_date, "isoformat") else str(report_date),
                "acceptance_datetime": acceptance.isoformat() if hasattr(acceptance, "isoformat") else str(acceptance),
                "headline": doc_desc,
                "text": text,
            }

            raw_payload_hash = compute_raw_payload_hash(payload)
            version_id = compute_version_id(
                source_system=source_system,
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
                    "source_system": source_system,
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
                    "release_date": event_time,
                    "headline": doc_desc,
                    "text": text,
                    "form_type": "8-K",
                    "cik": cik,
                    "accession": accession,
                    "primary_document": primary_doc,
                }
            )

        if raw_records:
            write_raw_records(source_system=source_system, records=raw_records)
        if canonical_records:
            rows = write_partitioned(canonical_records)
            total += rows

        if SEC_RESUME:
            with checkpoint_path.open("a") as f:
                f.write(f"{cik}\n")

        if (idx + 1) % 100 == 0:
            elapsed = _time.perf_counter() - start_ts
            rate = (idx + 1) / elapsed if elapsed > 0 else 0.0
            remaining = (len(cik_universe) - (idx + 1))
            eta = (remaining / rate) if rate > 0 else 0.0
            log(f"Progress: {idx + 1}/{len(cik_universe)} CIKs | elapsed {elapsed/60:.1f}m | ETA {eta/60:.1f}m")

    log(f"Done. Ingested {total:,} press release records.")


if __name__ == "__main__":
    main()
