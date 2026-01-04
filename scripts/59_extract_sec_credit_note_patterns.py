#!/usr/bin/env python
"""
Extract narrow SEC credit-note patterns from document text.

This is a high-precision scaffold for three note families:
  1. Revolver / credit facility availability
  2. Lease cost / liabilities / lease maturity lines
  3. Debt maturity schedules

Inputs:
  - data/inputs_layer/doc_text_map/year=YYYY/part.parquet
  - optional data/inputs_layer/raw_documents/year=YYYY/*.parquet for metadata

Outputs:
  - data/sec/note_extracts/revolver_note_extracts.parquet
  - data/sec/note_extracts/lease_note_extracts.parquet
  - data/sec/note_extracts/debt_maturity_note_extracts.parquet
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

REVOLVER_OUT_PATH = ROOT / "data" / "sec" / "note_extracts" / "revolver_note_extracts.parquet"
LEASE_OUT_PATH = ROOT / "data" / "sec" / "note_extracts" / "lease_note_extracts.parquet"
MATURITY_OUT_PATH = ROOT / "data" / "sec" / "note_extracts" / "debt_maturity_note_extracts.parquet"

MONEY_RE = re.compile(
    r"(?P<prefix>\$)?\s*(?P<number>\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)\s*(?P<unit>billion|million|thousand|bn|mm|mn|m|b|k)?",
    re.IGNORECASE,
)
YEAR_AMOUNT_RE = re.compile(
    r"^\s*(?P<label>(?:20\d{2}|thereafter))\b[^\n$]{0,40}(?P<amount>\$?\s*\d[\d,]*(?:\.\d+)?(?:\s*(?:billion|million|thousand|bn|mm|mn|m|b|k))?)",
    re.IGNORECASE,
)

REVOLVER_KEYWORDS = re.compile(
    r"revolving credit|credit facility|line of credit|senior credit facility|abl facility|asset[- ]based lending",
    re.IGNORECASE,
)
LEASE_KEYWORDS = re.compile(
    r"\bleases?\b|lease cost|lease liabilit|future lease payments|maturity analysis of lease liabilities",
    re.IGNORECASE,
)
MATURITY_KEYWORDS = re.compile(
    r"debt maturit|long-term debt maturit|principal maturit|contractual maturit|scheduled maturit|debt due",
    re.IGNORECASE,
)
STRICT_MONEY_CAPTURE = r"(?P<money>(?:\$\s*\d[\d,]*(?:\.\d+)?(?:\s*(?:billion|million|thousand|bn|mm|mn|m|b|k))?|\d[\d,]*(?:\.\d+)?\s*(?:billion|million|thousand|bn|mm|mn|m|b|k)))"


def _quoted_paths(paths: Sequence[Path]) -> str:
    return "[" + ", ".join("'" + p.as_posix().replace("'", "''") + "'" for p in paths) + "]"


