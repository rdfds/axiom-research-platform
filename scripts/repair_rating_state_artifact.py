#!/usr/bin/env python3
"""Repair rating-state metrics in a materialized company-state artifact."""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd


REPAIR_METRICS = ["capital_structure.rating_state"]

RATING_SCORE_MAP = {
    "AAA": 1,
    "AA+": 2,
    "AA": 3,
    "AA-": 4,
    "A+": 5,
    "A": 6,
    "A-": 7,
    "BBB+": 8,
    "BBB": 9,
    "BBB-": 10,
    "BB+": 11,
    "BB": 12,
    "BB-": 13,
    "B+": 14,
    "B": 15,
    "B-": 16,
    "CCC+": 17,
    "CCC": 18,
    "CCC-": 19,
    "CC": 20,
    "C": 21,
    "D": 22,
}

DEFAULT_RATINGS_PATHS = [
    "/tmp/issuer_rating_history.parquet",
    "./data/inputs_layer/issuer_rating_history.parquet",
    "./data/curated/issuer_ratings_ciq.parquet",
    "./data/wrds/ciq/ciq_entity_ratings.csv.gz",
]

CANONICAL_ALIASES = {
    "company_id": {
        "company_id",
        "companyid",
        "cik",
        "issuer_cik",
        "entity_cik",
        "company_cik",
    },
    "rating_symbol": {
        "rating_symbol",
        "current_rating_symbol",
        "rating",
        "current_rating",
        "ratingvalue",
        "rating_value",
        "ratingsymbol",
        "symbol",
    },
    "current_rating_symbol": {
        "current_rating_symbol",
        "rating_symbol",
        "current_rating",
        "rating",
    },
    "outlook": {
        "outlook",
        "rating_outlook",
        "current_outlook",
    },
    "creditwatch": {
        "creditwatch",
        "watchlist",
        "watch",
        "credit_watch",
        "rating_watch",
    },
    "rating_date": {
        "rating_date",
        "ratingdate",
        "effective_at",
        "effective_date",
        "effectivedate",
        "date",
        "announcedate",
        "announcement_date",
        "as_of_date",
        "published_at",
    },
    "published_at": {
        "published_at",
        "publish_date",
        "publisheddate",
        "announcement_date",
        "announcedate",
    },
    "effective_at": {
        "effective_at",
        "effective_date",
        "effectivedate",
        "rating_date",
        "ratingdate",
    },
    "artifact_id": {
        "artifact_id",
        "event_id",
        "id",
        "rating_id",
    },
    "source_type": {
        "source_type",
        "rating_agency",
        "agency",
        "provider",
        "source",
        "source_name",
        "event_subtype",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-path", required=True, help="Input company-state JSONL artifact")
    parser.add_argument("--ratings-path", help="Optional issuer ratings parquet/csv.gz path")
    parser.add_argument("--out", required=True, help="Output repaired JSONL artifact")
    parser.add_argument("--summary-out", help="Optional summary JSON")
    return parser.parse_args()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def iter_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _normalize_company_id(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    digits = "".join(ch for ch in text if ch.isdigit())
    if digits:
        return digits.zfill(10)
    return text


def _null_if_na(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, str) and not value.strip():
        return None
    return value


def _node_support(node: Dict[str, Any] | None) -> str:
    if not node:
        return "unsupported"
    return str(node.get("support_mode") or "unsupported")


def _base_repaired_node(node: Dict[str, Any], *, computed_at: str) -> Dict[str, Any]:
    repaired = copy.deepcopy(node)
    repaired["computed_at"] = computed_at
    repaired["missing_reason"] = None
    repaired["quality_flags"] = repaired.get("quality_flags") or None
    return repaired


