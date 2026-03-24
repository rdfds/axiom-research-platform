#!/usr/bin/env python
"""
Pull earnings call transcripts from FMP (B1) and ingest into canonical tables.

Requires:
  export FMP_API_KEY="..."

Writes (partitioned by year):
  data/warehouse/warehouse_documents
  data/warehouse/warehouse_doc_chunks
  data/warehouse/warehouse_text_signals

Raw payloads are stored in data/lake/raw/fmp_transcripts.

Env:
  FMP_API_KEY (required)
  FMP_BASE_URL=https://financialmodelingprep.com/stable
  FMP_SLEEP=0.2
  FMP_RETRIES=2
  FMP_START_YEAR=2000
  FMP_END_YEAR=YYYY
  FMP_LIMIT_SYMBOLS=0 (0 = all)
  FMP_MAX_TRANSCRIPTS_PER_SYMBOL=0 (0 = all)
  FMP_TARGET_SYMBOL= (optional)
  FMP_RESUME=1
  FMP_FLUSH_EVERY=200
  FMP_USE_UNIVERSE=1 (use R3000 proxy tickers)
"""

from __future__ import annotations

import json
import os
import re
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

FMP_API_KEY = os.getenv("FMP_API_KEY")
FMP_BASE_URL = os.getenv("FMP_BASE_URL", "https://financialmodelingprep.com/stable").rstrip("/")
FMP_SLEEP = float(os.getenv("FMP_SLEEP", "0.2"))
FMP_RETRIES = int(os.getenv("FMP_RETRIES", "2"))
FMP_TIMEOUT = float(os.getenv("FMP_TIMEOUT", "30"))
FMP_START_YEAR = int(os.getenv("FMP_START_YEAR", "2000"))
FMP_END_YEAR = int(os.getenv("FMP_END_YEAR", datetime.utcnow().year))
FMP_LIMIT_SYMBOLS = int(os.getenv("FMP_LIMIT_SYMBOLS", "0"))
FMP_MAX_TRANSCRIPTS_PER_SYMBOL = int(os.getenv("FMP_MAX_TRANSCRIPTS_PER_SYMBOL", "0"))
FMP_TARGET_SYMBOL = os.getenv("FMP_TARGET_SYMBOL")
FMP_RESUME = os.getenv("FMP_RESUME", "1") == "1"
FMP_FLUSH_EVERY = int(os.getenv("FMP_FLUSH_EVERY", "200"))
FMP_USE_UNIVERSE = os.getenv("FMP_USE_UNIVERSE", "1") == "1"
FMP_DEBUG = os.getenv("FMP_DEBUG", "0") == "1"
FMP_HEARTBEAT_SECS = int(os.getenv("FMP_HEARTBEAT_SECS", "60"))

CHUNK_TOKENS = int(os.getenv("TRANSCRIPT_CHUNK_TOKENS", "400"))
CHUNK_MIN = int(os.getenv("TRANSCRIPT_CHUNK_MIN", "300"))
CHUNK_MAX = int(os.getenv("TRANSCRIPT_CHUNK_MAX", "500"))


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


def load_symbol_list(session: requests.Session) -> List[str]:
    if FMP_TARGET_SYMBOL:
        return [FMP_TARGET_SYMBOL.upper()]

    symbols: List[str] = []
    if FMP_USE_UNIVERSE:
        symbols = load_universe_tickers()
        if symbols:
            log(f"Using universe tickers: {len(symbols):,}")

    url = f"{FMP_BASE_URL}/earnings-transcript-list"
    data = _request_json(url, params={"apikey": FMP_API_KEY}, session=session)
    if data:
        avail = [row.get("symbol") for row in data if row.get("symbol")]
        avail = [str(s).upper() for s in avail]
        if symbols:
            symbols = sorted(set(symbols).intersection(set(avail)))
        else:
            symbols = sorted(set(avail))

    if FMP_LIMIT_SYMBOLS and FMP_LIMIT_SYMBOLS > 0:
        symbols = symbols[:FMP_LIMIT_SYMBOLS]
    return symbols


def load_symbol_dates(symbol: str, session: requests.Session) -> List[Dict]:
    url = f"{FMP_BASE_URL}/earning-call-transcript-dates"
    data = _request_json(url, params={"symbol": symbol, "apikey": FMP_API_KEY}, session=session)
    if not data:
        if FMP_DEBUG:
            log(f"[debug] dates empty for {symbol}")
        return []
    if isinstance(data, dict) and data.get("Error Message"):
        if FMP_DEBUG:
            log(f"[debug] dates error for {symbol}: {data.get('Error Message')}")
        return []
    return data


def load_transcript(symbol: str, year: int, quarter: int, session: requests.Session) -> Optional[Dict]:
    url = f"{FMP_BASE_URL}/earning-call-transcript"
    data = _request_json(
        url,
        params={"symbol": symbol, "year": year, "quarter": quarter, "apikey": FMP_API_KEY},
        session=session,
    )
    if not data:
        if FMP_DEBUG:
            log(f"[debug] transcript empty for {symbol} {year}Q{quarter}")
        return None
    if isinstance(data, dict) and data.get("Error Message"):
        if FMP_DEBUG:
            log(f"[debug] transcript error for {symbol} {year}Q{quarter}: {data.get('Error Message')}")
        return None
    # API returns list with single object
    if isinstance(data, list) and data:
        return data[0]
    if isinstance(data, dict):
        return data
    return None


def quarter_end_date(year: int, quarter: int) -> pd.Timestamp:
    month = {1: 3, 2: 6, 3: 9, 4: 12}.get(quarter, 12)
    day = 31 if month in (3, 12) else 30
    return pd.Timestamp(year=year, month=month, day=day)


def parse_transcript_sections(content: str) -> List[Dict[str, str]]:
    if not content:
        return []
    # Split into lines and detect speaker labels
    lines = content.splitlines()
    sections: List[Dict[str, str]] = []
    current_speaker = None
    current_role = None
    current_section = "prepared"
    buffer: List[str] = []

    def flush():
        nonlocal buffer, current_speaker, current_role, current_section
        text = " ".join([b.strip() for b in buffer if b.strip()]).strip()
        if text:
            sections.append(
                {
                    "speaker": current_speaker,
                    "speaker_role": current_role,
                    "section_type": current_section,
                    "text": text,
                }
            )
        buffer = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        lower = line.lower()
        if "question-and-answer" in lower or "question and answer" in lower or "q&a" in lower:
            current_section = "qa"
            continue
        m = re.match(r"^([A-Za-z][A-Za-z .,'&-]{1,60}):\\s*(.*)$", line)
        if m:
            flush()
            speaker = m.group(1).strip()
            rest = m.group(2).strip()
            current_speaker = speaker
            if "operator" in speaker.lower():
                current_role = "operator"
            elif "analyst" in speaker.lower():
                current_role = "analyst"
            else:
                current_role = "management"
            if rest:
                buffer.append(rest)
            continue
        buffer.append(line)
    flush()
    return sections


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


def map_symbol_to_gvkey(symbol: str, call_date: pd.Timestamp, names: pd.DataFrame, link: pd.DataFrame) -> Optional[str]:
    if names.empty or link.empty or symbol is None or pd.isna(symbol):
        return None
    symbol = str(symbol).upper().strip()
    candidates = names[names["ticker"] == symbol]
    if candidates.empty:
        return None
    active = candidates[(candidates["namedt"] <= call_date) & (candidates["nameendt"] >= call_date)]
    if active.empty:
        active = candidates.sort_values("nameendt").tail(1)
    permno = active.iloc[0]["permno"]
    link_rows = link[link["permno"] == permno]
    if link_rows.empty:
        return None
    link_active = link_rows[(link_rows["linkdt"] <= call_date) & (link_rows["linkenddt"] >= call_date)]
    if link_active.empty:
        link_active = link_rows.sort_values("linkenddt").tail(1)
    gvkey = link_active.iloc[0]["gvkey"]
    return str(gvkey) if pd.notna(gvkey) else None


def main() -> None:
    require_api_key()
    FMP_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    names, link = load_mappings()
    symbols = load_symbol_list(session)
    log(f"Symbols to pull: {len(symbols):,}")

    checkpoint_path = FMP_DIR / "fmp_transcripts_checkpoint.txt"
    processed = set()
    if FMP_RESUME and checkpoint_path.exists():
        processed = set([line.strip() for line in checkpoint_path.read_text().splitlines() if line.strip()])

    doc_buffer: List[Dict[str, object]] = []
    chunk_buffer: List[Dict[str, object]] = []
    signal_buffer: List[Dict[str, object]] = []

    total_docs = 0
    total_chunks = 0
    total_signals = 0
    start_ts = time.perf_counter()
    ingestion_time = datetime.utcnow()

    total_date_rows = 0
    total_transcripts_attempted = 0
    total_transcripts_pulled = 0
    symbols_with_dates = 0
    last_heartbeat = start_ts
    for idx, symbol in enumerate(symbols, start=1):
        dates = load_symbol_dates(symbol, session)
        if not dates:
            continue
        symbols_with_dates += 1
        if FMP_DEBUG:
            sample = dates[0] if isinstance(dates, list) and dates else dates
            log(f"[debug] dates sample for {symbol}: {sample}")
        total_date_rows += len(dates)

        if FMP_MAX_TRANSCRIPTS_PER_SYMBOL and FMP_MAX_TRANSCRIPTS_PER_SYMBOL > 0:
            dates = dates[:FMP_MAX_TRANSCRIPTS_PER_SYMBOL]

        bad_logged = 0
        pulled_for_symbol = 0
        for item in dates:
            year = item.get("year") or item.get("calendarYear") or item.get("fiscalYear")
            quarter = item.get("quarter") or item.get("fiscalQuarter") or item.get("period")
            try:
                year = int(year)
            except Exception:
                year = 0
            try:
                if isinstance(quarter, str) and quarter.lower().startswith("q"):
                    quarter = quarter.lower().replace("q", "")
                quarter = int(quarter)
            except Exception:
                quarter = 0
            if year == 0 or quarter == 0:
                date_guess = item.get("date") or item.get("callDate") or item.get("calendarDate")
                dt = pd.to_datetime(date_guess, errors="coerce")
                if pd.notna(dt):
                    year = int(dt.year)
                    quarter = int(dt.quarter)
                    if FMP_DEBUG:
                        log(f"[debug] derived year/quarter from date for {symbol}: {year}Q{quarter}")
            if FMP_DEBUG and (year == 0 or quarter == 0) and bad_logged < 3:
                log(f"[debug] bad date row for {symbol}: {item}")
                bad_logged += 1
            if year < FMP_START_YEAR or year > FMP_END_YEAR:
                continue
            if quarter < 1 or quarter > 4:
                continue

            document_id = f"fmp:{symbol}:{year}Q{quarter}"
            if FMP_RESUME and document_id in processed:
                continue

            total_transcripts_attempted += 1
            payload = load_transcript(symbol, year, quarter, session)
            if not payload:
                continue
            pulled_for_symbol += 1
            total_transcripts_pulled += 1

            content = payload.get("content") or payload.get("text")
            call_date_raw = payload.get("date") or payload.get("call_date")
            call_date = pd.to_datetime(call_date_raw, errors="coerce")
            quality_flags: List[str] = []
            if pd.isna(call_date):
                call_date = quarter_end_date(year, quarter)
                quality_flags.append("estimated_event_time")
            available_time = call_date
            quality_flags.append("estimated_available_time")

            gvkey = map_symbol_to_gvkey(symbol, call_date, names, link)
            if gvkey is None:
                entity_id = symbol
                company_id = None
                quality_flags.append("estimated_company_id")
            else:
                entity_id = gvkey
                company_id = gvkey

            sections = parse_transcript_sections(content or "")
            if not sections:
                sections = [{"speaker": None, "speaker_role": None, "section_type": "prepared", "text": content or ""}]
                quality_flags.append("partial_coverage")

            raw_payload = {
                "symbol": symbol,
                "year": year,
                "quarter": quarter,
                "date": call_date.isoformat(),
                "content": content,
                "sections": sections,
            }

            write_raw_records(
                source_system="fmp_transcripts",
                records=[
                    {
                        "entity_id": str(entity_id),
                        "company_id": str(company_id) if company_id is not None else None,
                        "security_id": None,
                        "event_time": call_date,
                        "available_time": available_time,
                        "payload": raw_payload,
                    }
                ],
            )

            doc_payload = {
                "document_id": document_id,
                "document_type": "earnings_call",
                "title": payload.get("title") or f"{symbol} {year}Q{quarter} Earnings Call",
                "publisher": payload.get("source") or "FMP",
                "call_date": call_date.isoformat(),
                "available_time": available_time.isoformat(),
            }
            doc_raw_hash = compute_raw_payload_hash(doc_payload)
            doc_version_id = compute_version_id(
                source_system="fmp_transcripts",
                entity_id=str(entity_id),
                event_time=call_date.to_pydatetime(),
                available_time=available_time.to_pydatetime(),
                raw_payload_hash=doc_raw_hash,
            )

            doc_buffer.append(
                {
                    "source_system": "fmp_transcripts",
                    "entity_id": str(entity_id),
                    "company_id": str(company_id) if company_id is not None else None,
                    "security_id": None,
                    "event_time": call_date,
                    "available_time": available_time,
                    "ingestion_time": ingestion_time,
                    "version_id": doc_version_id,
                    "raw_payload_hash": doc_raw_hash,
                    "upstream_version_ids": [doc_version_id],
                    "quality_flags": quality_flags,
                    "document_id": document_id,
                    "document_type": "earnings_call",
                    "title": payload.get("title") or f"{symbol} {year}Q{quarter} Earnings Call",
                    "publisher": payload.get("source") or "FMP",
                    "analyst": None,
                    "rating": None,
                    "price_target": None,
                    "call_date": call_date,
                    "publish_date": available_time,
                    "presentation_date": None,
                    "release_date": None,
                    "source_url": payload.get("url"),
                }
            )

            chunk_ids: List[Tuple[str, str]] = []
            chunk_versions: Dict[str, str] = {}
            for sec_idx, sec in enumerate(sections):
                text = sec.get("text") or ""
                if not text.strip():
                    continue
                chunks = chunk_text(text, CHUNK_TOKENS, CHUNK_MIN, CHUNK_MAX)
                section_type = sec.get("section_type") or "prepared"
                speaker = sec.get("speaker")
                speaker_role = sec.get("speaker_role")
                for chunk_idx, chunk in enumerate(chunks):
                    chunk_id = f"{document_id}::s{sec_idx:04d}c{chunk_idx:04d}"
                    token_count = len(chunk.split())
                    chunk_payload = {
                        "chunk_id": chunk_id,
                        "document_id": document_id,
                        "section_type": section_type,
                        "text": chunk,
                    }
                    chunk_raw_hash = compute_raw_payload_hash(chunk_payload)
                    chunk_version_id = compute_version_id(
                        source_system="fmp_transcripts",
                        entity_id=str(entity_id),
                        event_time=call_date.to_pydatetime(),
                        available_time=available_time.to_pydatetime(),
                        raw_payload_hash=chunk_raw_hash,
                    )
                    chunk_versions[chunk_id] = chunk_version_id
                    chunk_ids.append((chunk_id, chunk))
                    chunk_buffer.append(
                        {
                            "source_system": "fmp_transcripts",
                            "entity_id": str(entity_id),
                            "company_id": str(company_id) if company_id is not None else None,
                            "security_id": None,
                            "event_time": call_date,
                            "available_time": available_time,
                            "ingestion_time": ingestion_time,
                            "version_id": chunk_version_id,
                            "raw_payload_hash": chunk_raw_hash,
                            "upstream_version_ids": [doc_version_id],
                            "quality_flags": quality_flags,
                            "chunk_id": chunk_id,
                            "document_id": document_id,
                            "chunk_index": chunk_idx,
                            "slide_number": None,
                            "text": chunk,
                            "speaker": speaker,
                            "speaker_role": speaker_role,
                            "section_type": section_type,
                            "token_count": token_count,
                        }
                    )
                total_chunks += len(chunks)

            if chunk_ids:
                signal_defs = extract_signals(chunk_ids)
                for sig in signal_defs:
                    supporting = sig.get("supporting_chunk_ids", [])
                    supporting_versions = [chunk_versions[cid] for cid in supporting if cid in chunk_versions]
                    signal_payload = {
                        "signal_name": sig["signal_name"],
                        "value": sig["value"],
                        "confidence": sig["confidence"],
                        "supporting_chunk_ids": supporting,
                    }
                    sig_raw_hash = compute_raw_payload_hash(signal_payload)
                    sig_version_id = compute_version_id(
                        source_system="fmp_transcripts",
                        entity_id=str(entity_id),
                        event_time=call_date.to_pydatetime(),
                        available_time=available_time.to_pydatetime(),
                        raw_payload_hash=sig_raw_hash,
                    )
                    signal_buffer.append(
                        {
                            "source_system": "fmp_transcripts",
                            "entity_id": str(entity_id),
                            "company_id": str(company_id) if company_id is not None else None,
                            "security_id": None,
                            "event_time": call_date,
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

            total_docs += 1
            if FMP_RESUME:
                with checkpoint_path.open("a") as f:
                    f.write(f"{document_id}\n")

            if FMP_FLUSH_EVERY and (total_docs % FMP_FLUSH_EVERY == 0):
                write_partitioned("warehouse_documents", doc_buffer)
                write_partitioned("warehouse_doc_chunks", chunk_buffer)
                write_partitioned("warehouse_text_signals", signal_buffer)
                doc_buffer.clear()
                chunk_buffer.clear()
                signal_buffer.clear()
                elapsed = time.perf_counter() - start_ts
                log(
                    f"Progress: {total_docs:,} docs | {total_chunks:,} chunks | {total_signals:,} signals | elapsed {elapsed/60:.1f}m"
                )
            if FMP_HEARTBEAT_SECS > 0:
                now_ts = time.perf_counter()
                if now_ts - last_heartbeat >= FMP_HEARTBEAT_SECS:
                    elapsed = now_ts - start_ts
                    per_tx = elapsed / max(1, total_transcripts_attempted)
                    avg_tx_per_symbol = total_transcripts_attempted / max(1, symbols_with_dates)
                    remaining_symbols = len(symbols) - idx
                    est_remaining_tx = avg_tx_per_symbol * remaining_symbols
                    eta = est_remaining_tx * per_tx
                    log(
                        "Heartbeat: symbols {}/{} | transcripts {}/{} | docs {} | avg {:.1f}s/tx | ETA {:.1f}m".format(
                            idx,
                            len(symbols),
                            total_transcripts_pulled,
                            total_transcripts_attempted,
                            total_docs,
                            per_tx,
                            eta / 60,
                        )
                    )
                    last_heartbeat = now_ts

        if pulled_for_symbol > 0:
            log(f"Completed {symbol}: {pulled_for_symbol} transcripts | total docs {total_docs:,}")
        if (idx % 50) == 0:
            elapsed = time.perf_counter() - start_ts
            log(f"Symbols processed: {idx}/{len(symbols)} | elapsed {elapsed/60:.1f}m")
        if (idx % 200) == 0:
            log(f"Symbols processed: {idx}/{len(symbols)}")

    write_partitioned("warehouse_documents", doc_buffer)
    write_partitioned("warehouse_doc_chunks", chunk_buffer)
    write_partitioned("warehouse_text_signals", signal_buffer)

    elapsed = time.perf_counter() - start_ts
    log(
        f"Done. {total_docs:,} docs | {total_chunks:,} chunks | {total_signals:,} signals | elapsed {elapsed/60:.1f}m"
    )
    if FMP_DEBUG:
        log(
            f"[debug] date rows: {total_date_rows:,} | transcripts attempted: {total_transcripts_attempted:,}"
        )


if __name__ == "__main__":
    main()
