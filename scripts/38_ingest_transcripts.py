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


