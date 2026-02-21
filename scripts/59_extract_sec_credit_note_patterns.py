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


def _parquet_columns(paths: Sequence[Path]) -> set[str]:
    if not paths:
        return set()
    con = duckdb.connect()
    try:
        first_path = paths[0].as_posix().replace("'", "''")
        df = con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{first_path}')"
        ).df()
        return set(df["column_name"].astype(str))
    except Exception:
        return set()


def _context_multiplier(text: str) -> float:
    lower = (text or "").lower()
    if "in billions" in lower or "(billions)" in lower:
        return 1_000_000_000.0
    if "in millions" in lower or "(millions)" in lower:
        return 1_000_000.0
    if "in thousands" in lower or "(thousands)" in lower:
        return 1_000.0
    return 1.0


def _money_mentions(text: str) -> List[Dict[str, object]]:
    mentions: List[Dict[str, object]] = []
    if not text:
        return mentions
    default_multiplier = _context_multiplier(text)
    for match in MONEY_RE.finditer(text):
        raw = match.group(0).strip()
        number_raw = (match.group("number") or "").replace(",", "")
        if not number_raw:
            continue
        try:
            number = float(number_raw)
        except ValueError:
            continue
        prefix = match.group("prefix")
        unit = (match.group("unit") or "").lower()
        # Skip plain years and similar false positives unless they look like money.
        if not prefix and not unit and "." not in number_raw and 1900 <= number <= 2100:
            continue
        # Skip tiny bare numbers when there is no currency/unit context. This
        # filters date fragments like "31" in "December 31, 2024" while still
        # allowing plain table values when the block says "(in millions)".
        if not prefix and not unit and "," not in number_raw and default_multiplier == 1.0 and number < 1000:
            continue
        if unit in {"billion", "bn", "b"}:
            multiplier = 1_000_000_000.0
        elif unit in {"million", "mm", "mn", "m"}:
            multiplier = 1_000_000.0
        elif unit in {"thousand", "k"}:
            multiplier = 1_000.0
        else:
            multiplier = default_multiplier
        mentions.append({"raw": raw, "value": number * multiplier, "start": match.start(), "end": match.end()})
    return mentions


def _first_money_value(text: str) -> Optional[float]:
    mentions = _money_mentions(text)
    if not mentions:
        return None
    return float(mentions[0]["value"])


