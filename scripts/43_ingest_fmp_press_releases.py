#!/usr/bin/env python
"""
Ingest FMP Press Releases (B4) to improve text coverage.

Requires:
  export FMP_API_KEY="..."

Writes:
  data/warehouse/warehouse_press_releases (partitioned by year)
  data/warehouse/warehouse_documents (optional, default on)
  data/warehouse/warehouse_doc_chunks (optional, default on)
  data/warehouse/warehouse_text_signals (optional, default on)

Raw payloads are stored in data/lake/raw/fmp_press_releases.

Env:
  FMP_API_KEY (required)
  FMP_BASE_URL=https://financialmodelingprep.com/stable
  FMP_SLEEP=0.2
  FMP_RETRIES=2
  FMP_TIMEOUT=30
  FMP_START_DATE=2000-01-01
  FMP_END_DATE=YYYY-MM-DD (default: today UTC)
  FMP_LIMIT_SYMBOLS=0 (0 = all)
  FMP_TARGET_SYMBOL= (optional)
  FMP_USE_UNIVERSE=1 (use R3000 proxy tickers)
  FMP_SYMBOL_BATCH=25
  FMP_MAX_PAGES=5
  FMP_PAGE_LIMIT=100
  FMP_RESUME=1
  FMP_PR_REQUIRE_TEXT=1
  FMP_PR_PROCESS_DOCS=1
  FMP_PR_FLUSH_EVERY=200
"""

from __future__ import annotations

import hashlib
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion import compute_raw_payload_hash, compute_version_id, write_raw_records
from src.text_processing import chunk_text, ensure_list, extract_signals, write_partitioned


DATA_DIR = Path(__file__).parent.parent / "data"
FMP_DIR = DATA_DIR / "fmp"
CRSP_DIR = DATA_DIR / "wrds" / "crsp"
WAREHOUSE_DIR = DATA_DIR / "warehouse"

FMP_API_KEY = os.getenv("FMP_API_KEY")
FMP_BASE_URL = os.getenv("FMP_BASE_URL", "https://financialmodelingprep.com/stable").rstrip("/")
FMP_SLEEP = float(os.getenv("FMP_SLEEP", "0.2"))
FMP_RETRIES = int(os.getenv("FMP_RETRIES", "2"))
FMP_TIMEOUT = float(os.getenv("FMP_TIMEOUT", "30"))
FMP_START_DATE = os.getenv("FMP_START_DATE", "2000-01-01")
FMP_END_DATE = os.getenv("FMP_END_DATE", datetime.utcnow().date().isoformat())
FMP_LIMIT_SYMBOLS = int(os.getenv("FMP_LIMIT_SYMBOLS", "0"))
FMP_TARGET_SYMBOL = os.getenv("FMP_TARGET_SYMBOL")
FMP_USE_UNIVERSE = os.getenv("FMP_USE_UNIVERSE", "1") == "1"
FMP_SYMBOL_BATCH = int(os.getenv("FMP_SYMBOL_BATCH", "25"))
FMP_MAX_PAGES = int(os.getenv("FMP_MAX_PAGES", "5"))
FMP_PAGE_LIMIT = int(os.getenv("FMP_PAGE_LIMIT", "100"))
FMP_RESUME = os.getenv("FMP_RESUME", "1") == "1"
FMP_PR_REQUIRE_TEXT = os.getenv("FMP_PR_REQUIRE_TEXT", "1") == "1"
FMP_PR_PROCESS_DOCS = os.getenv("FMP_PR_PROCESS_DOCS", "1") == "1"
FMP_PR_FLUSH_EVERY = int(os.getenv("FMP_PR_FLUSH_EVERY", "200"))
FMP_DEBUG = os.getenv("FMP_DEBUG", "0") == "1"

PR_CHUNK_TOKENS = int(os.getenv("PR_CHUNK_TOKENS", "400"))
PR_CHUNK_MIN = int(os.getenv("PR_CHUNK_MIN", "300"))
PR_CHUNK_MAX = int(os.getenv("PR_CHUNK_MAX", "500"))


def log(msg: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def require_api_key() -> str:
    if not FMP_API_KEY:
        raise RuntimeError("FMP_API_KEY not set. Export your FMP API key.")
    return FMP_API_KEY


def _safe_params(params: Dict[str, object]) -> Dict[str, object]:
    safe = dict(params)
    if "apikey" in safe:
        safe["apikey"] = "***REDACTED***"
    return safe


def _request_json(url: str, params: Dict[str, object], session: requests.Session) -> Optional[List[Dict]]:
    for attempt in range(FMP_RETRIES + 1):
        try:
            if FMP_DEBUG:
                log(f"[debug] GET {url} params={_safe_params(params)}")
            resp = session.get(url, params=params, timeout=FMP_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            if FMP_SLEEP:
                time.sleep(FMP_SLEEP)
            return data
        except requests.RequestException as exc:
            if attempt < FMP_RETRIES:
                time.sleep(max(FMP_SLEEP, 0.2))
                continue
            log(f"Request failed: {url} {exc}")
            return None
    return None


def load_universe_tickers() -> List[str]:
    universe_path = DATA_DIR / "curated" / "universe_r3000_proxy.parquet"
    names_path = CRSP_DIR / "msenames_2000-01-01_to_2026-12-31.parquet"
    if not universe_path.exists() or not names_path.exists():
        return []
    universe = pd.read_parquet(universe_path)
    universe["date"] = pd.to_datetime(universe["date"])
    asof_date = universe["date"].max()
    universe = universe[universe["date"] == asof_date][["permno"]]

    names = pd.read_parquet(names_path, columns=["permno", "namedt", "nameendt", "ticker"])
    names["namedt"] = pd.to_datetime(names["namedt"])
    names["nameendt"] = pd.to_datetime(names["nameendt"])
    active = names[(names["namedt"] <= asof_date) & (names["nameendt"] >= asof_date)]
    active = active.sort_values(["permno", "nameendt"])
    latest = active.drop_duplicates(subset=["permno"], keep="last")
    merged = universe.merge(latest, on="permno", how="left")
    tickers = (
        merged["ticker"]
        .dropna()
        .astype(str)
        .str.upper()
        .str.strip()
        .tolist()
    )
    return sorted(set(tickers))


def load_symbol_list() -> List[str]:
    if FMP_TARGET_SYMBOL:
        symbols = [FMP_TARGET_SYMBOL.upper()]
    elif FMP_USE_UNIVERSE:
        symbols = load_universe_tickers()
    else:
        symbols = []
    if FMP_LIMIT_SYMBOLS and FMP_LIMIT_SYMBOLS > 0:
        symbols = symbols[:FMP_LIMIT_SYMBOLS]
    return symbols


def load_mappings() -> Tuple[pd.DataFrame, pd.DataFrame]:
    names_path = CRSP_DIR / "msenames_2000-01-01_to_2026-12-31.parquet"
    link_path = CRSP_DIR / "ccmxpf_lnkhist.parquet"
    if not names_path.exists() or not link_path.exists():
        return pd.DataFrame(), pd.DataFrame()
    names = pd.read_parquet(names_path, columns=["permno", "namedt", "nameendt", "ticker"])
    names["namedt"] = pd.to_datetime(names["namedt"])
    names["nameendt"] = pd.to_datetime(names["nameendt"])
    names["ticker"] = names["ticker"].astype(str).str.upper().str.strip()
    try:
        link = pd.read_parquet(link_path, columns=["permno", "gvkey", "linkdt", "linkenddt"])
    except Exception:
        link = pd.read_parquet(link_path, columns=["lpermno", "gvkey", "linkdt", "linkenddt"])
        link = link.rename(columns={"lpermno": "permno"})
    link["permno"] = pd.to_numeric(link["permno"], errors="coerce")
    link["linkdt"] = pd.to_datetime(link["linkdt"], errors="coerce")
    link["linkenddt"] = pd.to_datetime(link["linkenddt"], errors="coerce")
    return names, link


def map_symbol_to_gvkey(symbol: str, event_time: pd.Timestamp, names: pd.DataFrame, link: pd.DataFrame) -> Optional[str]:
    if names.empty or link.empty or symbol is None or pd.isna(symbol):
        return None
    symbol = str(symbol).upper().strip()
    candidates = names[names["ticker"] == symbol]
    if candidates.empty:
        return None
    active = candidates[(candidates["namedt"] <= event_time) & (candidates["nameendt"] >= event_time)]
    if active.empty:
        active = candidates.sort_values("nameendt").tail(1)
    permno = active.iloc[0]["permno"]
    link_rows = link[link["permno"] == permno]
    if link_rows.empty:
        return None
    link_active = link_rows[(link_rows["linkdt"] <= event_time) & (link_rows["linkenddt"] >= event_time)]
    if link_active.empty:
        link_active = link_rows.sort_values("linkenddt").tail(1)
    gvkey = link_active.iloc[0]["gvkey"]
    return str(gvkey) if pd.notna(gvkey) else None


def iter_batches(items: List[str], batch_size: int) -> Iterable[List[str]]:
    if batch_size <= 0:
        yield items
        return
    for idx in range(0, len(items), batch_size):
        yield items[idx : idx + batch_size]


def parse_press_release(item: Dict) -> Tuple[Optional[str], Optional[pd.Timestamp], Optional[str], Optional[str], Optional[str]]:
    symbol = item.get("symbol") or item.get("ticker")
    if symbol:
        symbol = str(symbol).upper().strip()
    date_raw = item.get("date") or item.get("publishedDate") or item.get("published_date")
    headline = item.get("title") or item.get("headline")
    text = item.get("text") or item.get("content") or item.get("body") or item.get("description")
    url = item.get("url") or item.get("link")
    event_time = pd.to_datetime(date_raw, errors="coerce")
    if pd.isna(event_time):
        return symbol, None, headline, text, url
    return symbol, event_time, headline, text, url


def write_press_release_partitioned(records: List[Dict[str, object]]) -> int:
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


def main() -> None:
    require_api_key()
    FMP_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    names, link = load_mappings()
    symbols = load_symbol_list()
    if not symbols:
        log("No symbols provided (empty universe).")
        return

    log(f"Symbols to pull: {len(symbols):,}")

    checkpoint_path = FMP_DIR / "fmp_press_releases_checkpoint.txt"
    processed = set()
    if FMP_RESUME and checkpoint_path.exists():
        processed = set([line.strip() for line in checkpoint_path.read_text().splitlines() if line.strip()])

    raw_records: List[Dict[str, object]] = []
    pr_records: List[Dict[str, object]] = []
    doc_buffer: List[Dict[str, object]] = []
    chunk_buffer: List[Dict[str, object]] = []
    signal_buffer: List[Dict[str, object]] = []

    total_docs = 0
    total_chunks = 0
    total_signals = 0
    total_pr = 0
    start_ts = time.perf_counter()
    ingestion_time = datetime.utcnow()
    start_dt = pd.to_datetime(FMP_START_DATE, errors="coerce")
    end_dt = pd.to_datetime(FMP_END_DATE, errors="coerce")

    for batch_idx, batch in enumerate(iter_batches(symbols, FMP_SYMBOL_BATCH), start=1):
        batch_symbols = ",".join(batch)
        for page in range(FMP_MAX_PAGES):
            url = f"{FMP_BASE_URL}/news/press-releases"
            params = {"symbols": batch_symbols, "apikey": FMP_API_KEY}
            if FMP_PAGE_LIMIT > 0:
                params["limit"] = FMP_PAGE_LIMIT
            params["page"] = page
            data = _request_json(url, params=params, session=session)
            if not data:
                break
            if isinstance(data, dict) and data.get("Error Message"):
                break
            if isinstance(data, list) and not data:
                break

            for item in data if isinstance(data, list) else []:
                symbol, event_time, headline, text, url = parse_press_release(item)
                if event_time is None or pd.isna(event_time):
                    continue
                if pd.notna(start_dt) and event_time < start_dt:
                    continue
                if pd.notna(end_dt) and event_time > end_dt:
                    continue

                if not text and FMP_PR_REQUIRE_TEXT:
                    continue

                title_key = headline or ""
                title_hash = hashlib.sha256(title_key.encode("utf-8")).hexdigest()[:8]
                doc_key = item.get("id") or item.get("newsId")
                if doc_key:
                    document_id = f"fmp_pr:{doc_key}"
                else:
                    document_id = f"fmp_pr:{symbol}:{event_time.strftime('%Y%m%d')}:{title_hash}"

                if FMP_RESUME and document_id in processed:
                    continue

                quality_flags: List[str] = ["estimated_available_time"]
                if not symbol:
                    quality_flags.append("missing_data")
                if not text:
                    quality_flags.append("partial_coverage")

                gvkey = map_symbol_to_gvkey(symbol, event_time, names, link) if symbol else None
                if gvkey is None:
                    entity_id = symbol or "unknown"
                    company_id = None
                    quality_flags.append("estimated_company_id")
                else:
                    entity_id = gvkey
                    company_id = gvkey

                available_time = event_time
                payload = {
                    "symbol": symbol,
                    "headline": headline,
                    "text": text,
                    "date": event_time.isoformat(),
                    "url": url,
                    "raw": item,
                }

                raw_records.append(
                    {
                        "entity_id": str(entity_id),
                        "company_id": str(company_id) if company_id is not None else None,
                        "security_id": None,
                        "event_time": event_time,
                        "available_time": available_time,
                        "payload": payload,
                    }
                )

                raw_payload_hash = compute_raw_payload_hash(payload)
                version_id = compute_version_id(
                    source_system="fmp_press_releases",
                    entity_id=str(entity_id),
                    event_time=event_time.to_pydatetime(),
                    available_time=available_time.to_pydatetime(),
                    raw_payload_hash=raw_payload_hash,
                )

                pr_records.append(
                    {
                        "source_system": "fmp_press_releases",
                        "entity_id": str(entity_id),
                        "company_id": str(company_id) if company_id is not None else None,
                        "security_id": None,
                        "event_time": event_time,
                        "available_time": available_time,
                        "ingestion_time": ingestion_time,
                        "version_id": version_id,
                        "raw_payload_hash": raw_payload_hash,
                        "upstream_version_ids": [version_id],
                        "quality_flags": quality_flags,
                        "document_id": document_id,
                        "release_date": event_time,
                        "headline": headline,
                        "text": text,
                        "form_type": "PRESS_RELEASE",
                        "cik": None,
                        "accession": None,
                        "primary_document": None,
                        "source_url": url,
                    }
                )
                total_pr += 1

                if FMP_PR_PROCESS_DOCS and text:
                    doc_payload = {
                        "document_id": document_id,
                        "document_type": "press_release",
                        "title": headline,
                        "release_date": event_time.isoformat(),
                        "text": text,
                    }
                    doc_raw_hash = compute_raw_payload_hash(doc_payload)
                    doc_version_id = compute_version_id(
                        source_system="fmp_press_releases",
                        entity_id=str(entity_id),
                        event_time=event_time.to_pydatetime(),
                        available_time=available_time.to_pydatetime(),
                        raw_payload_hash=doc_raw_hash,
                    )
                    doc_buffer.append(
                        {
                            "source_system": "fmp_press_releases",
                            "entity_id": str(entity_id),
                            "company_id": str(company_id) if company_id is not None else None,
                            "security_id": None,
                            "event_time": event_time,
                            "available_time": available_time,
                            "ingestion_time": ingestion_time,
                            "version_id": doc_version_id,
                            "raw_payload_hash": doc_raw_hash,
                            "upstream_version_ids": [doc_version_id],
                            "quality_flags": quality_flags,
                            "document_id": document_id,
                            "document_type": "press_release",
                            "title": headline,
                            "publisher": "FMP",
                            "analyst": None,
                            "rating": None,
                            "price_target": None,
                            "call_date": None,
                            "publish_date": available_time,
                            "presentation_date": None,
                            "release_date": event_time,
                            "source_url": url,
                        }
                    )
                    total_docs += 1

                    chunks = chunk_text(text, PR_CHUNK_TOKENS, PR_CHUNK_MIN, PR_CHUNK_MAX)
                    chunk_versions: Dict[str, str] = {}
                    chunk_ids: List[Tuple[str, str]] = []
                    for idx, chunk in enumerate(chunks):
                        chunk_id = f"{document_id}::chunk{idx:04d}"
                        token_count = len(chunk.split())
                        chunk_payload = {
                            "chunk_id": chunk_id,
                            "document_id": document_id,
                            "chunk_index": idx,
                            "text": chunk,
                        }
                        chunk_raw_hash = compute_raw_payload_hash(chunk_payload)
                        chunk_version_id = compute_version_id(
                            source_system="fmp_press_releases",
                            entity_id=str(entity_id),
                            event_time=event_time.to_pydatetime(),
                            available_time=available_time.to_pydatetime(),
                            raw_payload_hash=chunk_raw_hash,
                        )
                        chunk_versions[chunk_id] = chunk_version_id
                        chunk_ids.append((chunk_id, chunk))
                        chunk_buffer.append(
                            {
                                "source_system": "fmp_press_releases",
                                "entity_id": str(entity_id),
                                "company_id": str(company_id) if company_id is not None else None,
                                "security_id": None,
                                "event_time": event_time,
                                "available_time": available_time,
                                "ingestion_time": ingestion_time,
                                "version_id": chunk_version_id,
                                "raw_payload_hash": chunk_raw_hash,
                                "upstream_version_ids": [doc_version_id],
                                "quality_flags": quality_flags,
                                "chunk_id": chunk_id,
                                "document_id": document_id,
                                "chunk_index": idx,
                                "slide_number": None,
                                "text": chunk,
                                "speaker": None,
                                "speaker_role": None,
                                "section_type": "press_release",
                                "token_count": token_count,
                            }
                        )
                    total_chunks += len(chunks)

                    if chunk_ids:
                        signal_defs = extract_signals(chunk_ids)
                        for sig in signal_defs:
                            supporting = sig.get("supporting_chunk_ids", [])
                            supporting_versions = [chunk_versions[cid] for cid in supporting if cid in chunk_versions]
                            sig_payload = {
                                "signal_name": sig["signal_name"],
                                "value": sig["value"],
                                "confidence": sig["confidence"],
                                "supporting_chunk_ids": supporting,
                            }
                            sig_raw_hash = compute_raw_payload_hash(sig_payload)
                            sig_version_id = compute_version_id(
                                source_system="fmp_press_releases",
                                entity_id=str(entity_id),
                                event_time=event_time.to_pydatetime(),
                                available_time=available_time.to_pydatetime(),
                                raw_payload_hash=sig_raw_hash,
                            )
                            signal_buffer.append(
                                {
                                    "source_system": "fmp_press_releases",
                                    "entity_id": str(entity_id),
                                    "company_id": str(company_id) if company_id is not None else None,
                                    "security_id": None,
                                    "event_time": event_time,
                                    "available_time": available_time,
                                    "ingestion_time": ingestion_time,
                                    "version_id": sig_version_id,
                                    "raw_payload_hash": sig_raw_hash,
                                    "upstream_version_ids": [doc_version_id] + supporting_versions,
                                    "quality_flags": quality_flags + ensure_list(sig.get("quality_flags")),
                                    "signal_name": sig["signal_name"],
                                    "value": sig["value"],
                                    "confidence": sig["confidence"],
                                    "supporting_chunk_ids": supporting,
                                }
                            )
                        total_signals += len(signal_defs)

                processed.add(document_id)
                if FMP_RESUME:
                    with checkpoint_path.open("a") as f:
                        f.write(f"{document_id}\n")

                if FMP_PR_FLUSH_EVERY and (total_pr % FMP_PR_FLUSH_EVERY == 0):
                    write_raw_records(source_system="fmp_press_releases", records=raw_records)
                    write_press_release_partitioned(pr_records)
                    raw_records.clear()
                    pr_records.clear()
                    if FMP_PR_PROCESS_DOCS:
                        write_partitioned("warehouse_documents", doc_buffer)
                        write_partitioned("warehouse_doc_chunks", chunk_buffer)
                        write_partitioned("warehouse_text_signals", signal_buffer)
                        doc_buffer.clear()
                        chunk_buffer.clear()
                        signal_buffer.clear()
                    elapsed = time.perf_counter() - start_ts
                    log(
                        f"Progress: {total_pr:,} press releases | {total_docs:,} docs | {total_chunks:,} chunks | {total_signals:,} signals | elapsed {elapsed/60:.1f}m"
                    )

        if batch_idx % 10 == 0:
            elapsed = time.perf_counter() - start_ts
            log(f"Batch {batch_idx} processed | total press releases: {total_pr:,} | elapsed {elapsed/60:.1f}m")

    if raw_records:
        write_raw_records(source_system="fmp_press_releases", records=raw_records)
    if pr_records:
        write_press_release_partitioned(pr_records)
    if FMP_PR_PROCESS_DOCS:
        write_partitioned("warehouse_documents", doc_buffer)
        write_partitioned("warehouse_doc_chunks", chunk_buffer)
        write_partitioned("warehouse_text_signals", signal_buffer)

    log(f"Done. Ingested {total_pr:,} FMP press releases.")


if __name__ == "__main__":
    main()
