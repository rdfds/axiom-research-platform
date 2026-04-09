#!/usr/bin/env python
"""
Ingest earnings call transcripts (B1) into canonical unstructured tables.

Input formats supported:
1) JSONL documents with sections:
   {
     "document_id": "...",
     "company_id": "...",
     "call_date": "YYYY-MM-DD",
     "available_time": "YYYY-MM-DDTHH:MM:SSZ",
     "title": "...",
     "publisher": "...",
     "sections": [
        {"speaker": "...", "speaker_role": "...", "section_type": "prepared|qa", "text": "..."}
     ]
   }

2) Parquet/CSV with section-level rows:
   document_id, company_id, call_date, available_time, title, publisher,
   speaker, speaker_role, section_type, text

Reads:
  TRANSCRIPT_INPUT (file or folder; default data/transcripts)

Writes (partitioned by year):
  data/warehouse/warehouse_documents
  data/warehouse/warehouse_doc_chunks
  data/warehouse/warehouse_text_signals

Env:
  TRANSCRIPT_INPUT=path
  TRANSCRIPT_FORMAT=auto|jsonl|parquet|csv
  TRANSCRIPT_SOURCE=transcripts_public
  TRANSCRIPT_LIMIT_DOCS=0
  TRANSCRIPT_FLUSH_EVERY=200
  TRANSCRIPT_CHUNK_TOKENS=400
  TRANSCRIPT_CHUNK_MIN=300
  TRANSCRIPT_CHUNK_MAX=500
  TRANSCRIPT_RESUME=1
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion import compute_raw_payload_hash, compute_version_id, write_raw_records
from src.text_processing import chunk_text, ensure_list, extract_signals, write_partitioned


DATA_DIR = Path(__file__).parent.parent / "data"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"

TRANSCRIPT_INPUT = os.getenv("TRANSCRIPT_INPUT", str(TRANSCRIPTS_DIR))
TRANSCRIPT_FORMAT = os.getenv("TRANSCRIPT_FORMAT", "auto").lower()
TRANSCRIPT_SOURCE = os.getenv("TRANSCRIPT_SOURCE", "transcripts_public")
TRANSCRIPT_LIMIT_DOCS = int(os.getenv("TRANSCRIPT_LIMIT_DOCS", "0"))
TRANSCRIPT_FLUSH_EVERY = int(os.getenv("TRANSCRIPT_FLUSH_EVERY", "200"))
TRANSCRIPT_CHUNK_TOKENS = int(os.getenv("TRANSCRIPT_CHUNK_TOKENS", "400"))
TRANSCRIPT_CHUNK_MIN = int(os.getenv("TRANSCRIPT_CHUNK_MIN", "300"))
TRANSCRIPT_CHUNK_MAX = int(os.getenv("TRANSCRIPT_CHUNK_MAX", "500"))
TRANSCRIPT_RESUME = os.getenv("TRANSCRIPT_RESUME", "1") == "1"


def log(msg: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def _normalize_section_type(value: Optional[str]) -> str:
    if not value:
        return "prepared"
    v = str(value).strip().lower()
    if "q&a" in v or "qa" in v or "questions" in v:
        return "qa"
    if "prepared" in v:
        return "prepared"
    return v


def iter_input_files(path: Path, fmt: str) -> List[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"Missing transcripts input: {path}")
    if fmt in ("jsonl", "auto"):
        files = sorted(path.glob("*.jsonl"))
        if files:
            return files
    if fmt in ("parquet", "auto"):
        files = sorted(path.glob("*.parquet"))
        if files:
            return files
    if fmt in ("csv", "auto"):
        files = sorted(path.glob("*.csv"))
        if files:
            return files
    raise FileNotFoundError(f"No transcript files found in {path}")


def load_jsonl_docs(path: Path) -> List[Dict]:
    docs: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))
    return docs


def rows_from_docs(docs: List[Dict]) -> pd.DataFrame:
    rows = []
    for doc in docs:
        document_id = doc.get("document_id") or doc.get("doc_id")
        company_id = doc.get("company_id") or doc.get("gvkey") or doc.get("cik")
        call_date = doc.get("call_date") or doc.get("event_time")
        available_time = doc.get("available_time") or doc.get("release_time") or doc.get("publish_time")
        title = doc.get("title")
        publisher = doc.get("publisher")
        sections = doc.get("sections") or []
        if not isinstance(sections, list):
            continue
        for section in sections:
            rows.append(
                {
                    "document_id": document_id,
                    "company_id": company_id,
                    "call_date": call_date,
                    "available_time": available_time,
                    "title": title,
                    "publisher": publisher,
                    "speaker": section.get("speaker"),
                    "speaker_role": section.get("speaker_role"),
                    "section_type": section.get("section_type"),
                    "text": section.get("text"),
                }
            )
    return pd.DataFrame(rows)


def load_section_rows(path: Path, fmt: str) -> pd.DataFrame:
    if fmt == "parquet":
        return pd.read_parquet(path)
    if fmt == "csv":
        return pd.read_csv(path)
    return pd.DataFrame()


def build_documents(df: pd.DataFrame) -> Iterable[Tuple[str, pd.DataFrame]]:
    if df.empty:
        return []
    df = df.copy()
    df["document_id"] = df["document_id"].astype(str)
    return df.groupby("document_id")


def main() -> None:
    input_path = Path(TRANSCRIPT_INPUT)
    files = iter_input_files(input_path, TRANSCRIPT_FORMAT)
    log(f"Found {len(files)} transcript files.")

    checkpoint_path = TRANSCRIPTS_DIR / "transcripts_checkpoint.txt"
    processed = set()
    if TRANSCRIPT_RESUME and checkpoint_path.exists():
        processed = set([line.strip() for line in checkpoint_path.read_text().splitlines() if line.strip()])

    total_docs = 0
    total_chunks = 0
    total_signals = 0
    start_ts = time.perf_counter()

    doc_buffer: List[Dict[str, object]] = []
    chunk_buffer: List[Dict[str, object]] = []
    signal_buffer: List[Dict[str, object]] = []

    ingestion_time = datetime.utcnow()

    for file_idx, path in enumerate(files, start=1):
        log(f"Processing file {file_idx}/{len(files)}: {path.name}")
        fmt = TRANSCRIPT_FORMAT
        if fmt == "auto":
            if path.suffix.lower() == ".jsonl":
                fmt = "jsonl"
            elif path.suffix.lower() == ".parquet":
                fmt = "parquet"
            elif path.suffix.lower() == ".csv":
                fmt = "csv"
        if fmt == "jsonl":
            docs = load_jsonl_docs(path)
            df = rows_from_docs(docs)
        else:
            df = load_section_rows(path, fmt)
        if df.empty:
            continue

        for document_id, doc_df in build_documents(df):
            if TRANSCRIPT_RESUME and document_id in processed:
                continue

            row0 = doc_df.iloc[0]
            company_id = row0.get("company_id")
            call_date = pd.to_datetime(row0.get("call_date"), errors="coerce")
            available_time = pd.to_datetime(row0.get("available_time"), errors="coerce")

            if pd.isna(call_date):
                continue
            if pd.isna(available_time):
                available_time = call_date
                quality_flags = ["estimated_available_time"]
            else:
                quality_flags = []
            if available_time < call_date:
                call_date = available_time
                quality_flags.append("estimated_event_time")

            if company_id is None or (isinstance(company_id, float) and pd.isna(company_id)):
                entity_id = document_id
                company_id = None
                quality_flags.append("estimated_company_id")
            else:
                entity_id = str(company_id)

            doc_payload = {
                "document_id": document_id,
                "document_type": "earnings_call",
                "title": row0.get("title"),
                "publisher": row0.get("publisher"),
                "call_date": call_date.isoformat(),
                "available_time": available_time.isoformat(),
            }
            doc_raw_hash = compute_raw_payload_hash(doc_payload)
            doc_version_id = compute_version_id(
                source_system=TRANSCRIPT_SOURCE,
                entity_id=str(entity_id),
                event_time=call_date.to_pydatetime(),
                available_time=available_time.to_pydatetime(),
                raw_payload_hash=doc_raw_hash,
            )

            raw_payload = {
                "document_id": document_id,
                "company_id": company_id,
                "call_date": call_date.isoformat(),
                "available_time": available_time.isoformat(),
                "title": row0.get("title"),
                "publisher": row0.get("publisher"),
                "sections": doc_df[
                    ["speaker", "speaker_role", "section_type", "text"]
                ].to_dict(orient="records"),
            }

            write_raw_records(
                source_system=TRANSCRIPT_SOURCE,
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

            doc_buffer.append(
                {
                    "source_system": TRANSCRIPT_SOURCE,
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
                    "title": row0.get("title"),
                    "publisher": row0.get("publisher"),
                    "analyst": None,
                    "rating": None,
                    "price_target": None,
                    "call_date": call_date,
                    "publish_date": available_time,
                    "presentation_date": None,
                    "release_date": None,
                    "source_url": row0.get("source_url"),
                }
            )

            chunk_ids: List[Tuple[str, str]] = []
            chunk_versions: Dict[str, str] = {}
            for sec_idx, sec_row in doc_df.iterrows():
                text = sec_row.get("text")
                if isinstance(text, float) and pd.isna(text):
                    continue
                if not text:
                    continue
                section_type = _normalize_section_type(sec_row.get("section_type"))
                speaker = sec_row.get("speaker")
                speaker_role = sec_row.get("speaker_role")
                chunks = chunk_text(text, TRANSCRIPT_CHUNK_TOKENS, TRANSCRIPT_CHUNK_MIN, TRANSCRIPT_CHUNK_MAX)
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
                        source_system=TRANSCRIPT_SOURCE,
                        entity_id=str(entity_id),
                        event_time=call_date.to_pydatetime(),
                        available_time=available_time.to_pydatetime(),
                        raw_payload_hash=chunk_raw_hash,
                    )
                    chunk_versions[chunk_id] = chunk_version_id
                    chunk_ids.append((chunk_id, chunk))
                    chunk_buffer.append(
                        {
                            "source_system": TRANSCRIPT_SOURCE,
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
                        source_system=TRANSCRIPT_SOURCE,
                        entity_id=str(entity_id),
                        event_time=call_date.to_pydatetime(),
                        available_time=available_time.to_pydatetime(),
                        raw_payload_hash=sig_raw_hash,
                    )
                    signal_buffer.append(
                        {
                            "source_system": TRANSCRIPT_SOURCE,
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
            if TRANSCRIPT_RESUME:
                with checkpoint_path.open("a") as f:
                    f.write(f"{document_id}\n")

            if TRANSCRIPT_LIMIT_DOCS and total_docs >= TRANSCRIPT_LIMIT_DOCS:
                break

            if TRANSCRIPT_FLUSH_EVERY and (total_docs % TRANSCRIPT_FLUSH_EVERY == 0):
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

        if TRANSCRIPT_LIMIT_DOCS and total_docs >= TRANSCRIPT_LIMIT_DOCS:
            break

    write_partitioned("warehouse_documents", doc_buffer)
    write_partitioned("warehouse_doc_chunks", chunk_buffer)
    write_partitioned("warehouse_text_signals", signal_buffer)

    elapsed = time.perf_counter() - start_ts
    log(
        f"Done. {total_docs:,} docs | {total_chunks:,} chunks | {total_signals:,} signals | elapsed {elapsed/60:.1f}m"
    )


if __name__ == "__main__":
    main()
