#!/usr/bin/env python3
"""Repair statement-direct debt overrides in an already-materialized artifact."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import signal
from collections import defaultdict
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from urllib.parse import urlparse

import duckdb
import requests
from bs4 import BeautifulSoup

try:
    from backfill_sec_companyfacts_components import _extract_lease_liabilities
except Exception:  # noqa: BLE001
    try:
        from scripts.backfill_sec_companyfacts_components import _extract_lease_liabilities
    except Exception:  # noqa: BLE001
        _extract_lease_liabilities = None


TARGET_MODE = "statement_direct_current_plus_noncurrent_debt"
CURRENT_DEBT_FACT = "financial.debt_current"
LONG_TERM_DEBT_FACT = "financial.debt_long_term"
TOTAL_DEBT_FACT = "financial.total_debt"
MAX_ALIGNMENT_GAP_DAYS = 45
MAX_EXACT_AGE_DAYS = 130
PARTIAL_TOTAL_DEBT_MIN_LIFT = 1.10
PARTIAL_TOTAL_DEBT_MAX_DOWNWARD_REPLACEMENT = 0.75
TOTAL_DEBT_REGRESSION_THRESHOLD = 1.50
SEC_USER_AGENT = "Codex/axiom_v1 research support"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"
SEC_EXACT_MAX_AGE_DAYS = 130

CURRENT_TOTAL_LABEL_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^short[\s\-]term borrowings$",
        r"^(?:.+\s+)?short[\s\-]term borrowings$",
        r"^short[\s\-]term debt$",
        r"^(?:.+\s+)?short[\s\-]term debt$",
        r"^current portion of long[\s\-]term debt$",
        r"^(?:.+\s+)?current portion of long[\s\-]term debt$",
        r"^current maturities of long[\s\-]term debt$",
        r"^(?:.+\s+)?current maturities of long[\s\-]term debt$",
        r"^long[\s\-]term borrowings due within one year$",
        r"^(?:.+\s+)?long[\s\-]term borrowings due within one year$",
        r"^debt due within one year$",
        r"^(?:.+\s+)?debt due within one year$",
        r"^debt payable within one year$",
        r"^(?:.+\s+)?debt payable within one year$",
        r"^current debt$",
        r"^(?:.+\s+)?current debt$",
    )
]
CURRENT_EXTRA_LABEL_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^short[\s\-]term securitization borrowings$",
        r"^(?:.+\s+)?short[\s\-]term securitization borrowings$",
        r"^current securitization borrowings$",
        r"^(?:.+\s+)?current securitization borrowings$",
        r"^securitization borrowings due within one year$",
        r"^(?:.+\s+)?securitization borrowings due within one year$",
    )
]
CURRENT_COMPONENT_LABEL_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^commercial paper$",
        r"^notes payable(?: to banks)?$",
        r"^current portion of long[\s\-]term debt$",
        r"^current maturities of long[\s\-]term debt$",
        r"^long[\s\-]term borrowings due within one year$",
        r"^short[\s\-]term securitization borrowings$",
    )
]
LONG_TOTAL_LABEL_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^long[\s\-]term borrowings$",
        r"^(?:.+\s+)?long[\s\-]term borrowings$",
        r"^long[\s\-]term debt$",
        r"^(?:.+\s+)?long[\s\-]term debt$",
        r"^long[\s\-]term debt payable after one year$",
        r"^(?:.+\s+)?long[\s\-]term debt payable after one year$",
        r"^long term debt$",
        r"^notes payable and long[\s\-]term debt$",
    )
]
TOTAL_LABEL_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^long[\s\-]term debt, including current maturities$",
        r"^debt, including current maturities$",
        r"^total debt$",
        r"^total borrowings$",
    )
]
VEHICLE_PROGRAM_DEBT_LABEL_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^debt$",
        r"^debt due to .+$",
        r"^vehicle[\s\-]backed debt$",
        r"^vehicle[\s\-]backed debt due to .+$",
    )
]
BALANCE_SHEET_CUES = (
    "balance sheet",
    "balance sheets",
    "liabilities and stockholders",
    "liabilities and shareholders",
    "liabilities and equity",
    "current liabilities",
)
FAIR_VALUE_CUES = ("fair value", "carrying amount")
VEHICLE_PROGRAM_TABLE_CUES = (
    "liabilities under vehicle programs",
    "debt under vehicle programs",
)


class _CompanyProcessingTimeoutError(RuntimeError):
    """Raised when a single company repair attempt exceeds the allowed time."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-path", required=True)
    parser.add_argument("--facts-path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--enable-sec-filing-fallback", action="store_true")
    parser.add_argument("--sec-cache-dir")
    parser.add_argument("--company-ids", nargs="*")
    parser.add_argument("--company-ids-file")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--company-processing-timeout-seconds", type=int, default=30)
    parser.add_argument("--skip-fact-registry-repair", action="store_true")
    return parser.parse_args()


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


