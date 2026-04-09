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


