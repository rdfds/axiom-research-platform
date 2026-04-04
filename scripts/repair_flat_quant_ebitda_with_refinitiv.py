#!/usr/bin/env python3
"""Overlay market-grade Refinitiv market/pricing metrics onto a flat quantitative export."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict

import pandas as pd

from scripts.repair_rating_state_artifact import (
    _resolve_ratings_path,
    build_rating_index,
    load_issuer_ratings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flat-path", required=True, help="Input flat parquet export")
    parser.add_argument("--provider-reference-path", required=True, help="Refinitiv fundamentals parquet")
    parser.add_argument("--entity-identifier-path", required=True, help="Entity identifier parquet")
    parser.add_argument("--ratings-path", help="Optional issuer ratings parquet/csv.gz path")
    parser.add_argument("--out-parquet", required=True, help="Output parquet path")
    parser.add_argument("--out-csv", help="Optional output CSV path")
    parser.add_argument("--summary-out", help="Optional summary JSON path")
    return parser.parse_args()


def _ticker_identifier_map(path: Path) -> pd.DataFrame:
    identifiers = _read_parquet_with_retries(path)
    identifiers = identifiers[identifiers["identifier_type"].astype(str).str.lower() == "ticker"].copy()
    identifiers["ticker"] = identifiers["identifier_value"].astype(str).str.upper().str.strip()
    return identifiers[["entity_id", "ticker"]].drop_duplicates()


def _read_parquet_with_retries(path: Path, attempts: int = 4, sleep_seconds: float = 3.0) -> pd.DataFrame:
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            return pd.read_parquet(path).copy()
        except TimeoutError as exc:
            last_error = exc
            if attempt == attempts:
                raise
            time.sleep(sleep_seconds)
    raise last_error


def _refinitiv_map(provider_reference_path: Path, entity_identifier_path: Path) -> pd.DataFrame:
    ref = _read_parquet_with_retries(provider_reference_path)
    ref["ticker"] = ref["Instrument"].astype(str).str.replace(r"\..*$", "", regex=True).str.upper().str.strip()
    tickers = _ticker_identifier_map(entity_identifier_path)
    merged = ref.merge(tickers, on="ticker", how="inner")
    merged["entity_id"] = merged["entity_id"].astype(str)
    return merged.sort_values(["entity_id", "Instrument"]).drop_duplicates("entity_id", keep="first")


def _support_counts(series: pd.Series) -> Dict[str, int]:
    support = series.fillna("unsupported").astype(str)
    return {
        "exact": int((support == "exact").sum()),
        "proxy_missing_component": int((support == "proxy_missing_component").sum()),
        "unsupported": int((support == "unsupported").sum()),
    }


