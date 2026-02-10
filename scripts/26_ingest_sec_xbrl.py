#!/usr/bin/env python
"""
SEC XBRL (Company Facts) Ingestion
==================================
Pulls SEC companyfacts JSON, writes raw payloads to the data lake, and
normalizes financial statement facts into the bitemporal warehouse.

Requires:
  - SEC_USER_AGENT env var (e.g., "Axiom Research (you@example.com)")
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion import (
    append_canonical_records,
    compute_version_id,
    compute_raw_payload_hash,
    write_raw_records,
)
from src.financial_fact_tags import FINANCIAL_FACT_TAGS
from src.sec_companyfacts_bulk import CompanyFactsBulkSource


SEC_TICKER_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

DATA_DIR = Path(__file__).parent.parent / "data"
SEC_DIR = DATA_DIR / "sec"
MAPPINGS_DIR = DATA_DIR / "mappings"

DEFAULT_START = "2000-01-01"
DEFAULT_END = datetime.utcnow().date().isoformat()
SEC_TIMEOUT = float(os.getenv("SEC_TIMEOUT", "30"))
SEC_LOG_EVERY = int(os.getenv("SEC_LOG_EVERY", "10"))
SEC_LOG_VERBOSE = os.getenv("SEC_LOG_VERBOSE", "0") == "1"

STATEMENT_TYPE_BY_TAG = {
    # Income statement
    "Revenues": "income",
    "RevenueFromContractWithCustomerExcludingAssessedTax": "income",
    "SalesRevenueNet": "income",
    "GrossProfit": "income",
    "CostOfRevenue": "income",
    "OperatingIncomeLoss": "income",
    "IncomeLossFromContinuingOperations": "income",
    "NetIncomeLoss": "income",
    "EarningsPerShareBasic": "income",
    "EarningsPerShareDiluted": "income",
    # Balance sheet
    "Assets": "balance_sheet",
    "Liabilities": "balance_sheet",
    "StockholdersEquity": "balance_sheet",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest": "balance_sheet",
    "CashAndCashEquivalentsAtCarryingValue": "balance_sheet",
    "LongTermDebt": "balance_sheet",
    # Cash flow
    "NetCashProvidedByUsedInOperatingActivities": "cash_flow",
    "NetCashProvidedByUsedInInvestingActivities": "cash_flow",
    "NetCashProvidedByUsedInFinancingActivities": "cash_flow",
    "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsPeriodIncreaseDecreaseIncludingExchangeRateEffect": "cash_flow",
}


def log(msg: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}")


def require_user_agent() -> str:
    user_agent = os.getenv("SEC_USER_AGENT")
    if not user_agent:
        raise RuntimeError(
            "SEC_USER_AGENT not set. Example: export SEC_USER_AGENT='Axiom Research (you@example.com)'"
        )
    return user_agent


def maybe_require_user_agent(needs_network: bool) -> str:
    if needs_network:
        return require_user_agent()
    return os.getenv("SEC_USER_AGENT", "Axiom Local SEC Cache")


def ensure_dirs() -> None:
    SEC_DIR.mkdir(parents=True, exist_ok=True)
    (SEC_DIR / "companyfacts").mkdir(parents=True, exist_ok=True)
    MAPPINGS_DIR.mkdir(parents=True, exist_ok=True)


def fetch_json(url: str, session: requests.Session, sleep_seconds: float) -> Dict:
    resp = session.get(url, timeout=SEC_TIMEOUT)
    resp.raise_for_status()
    if sleep_seconds:
        time.sleep(sleep_seconds)
    return resp.json()


def load_sec_tickers(
    session: requests.Session,
    sleep_seconds: float,
    refresh: bool = False,
) -> pd.DataFrame:
    cache_path = SEC_DIR / "company_tickers.json"
    if cache_path.exists() and not refresh:
        data = json.loads(cache_path.read_text())
    else:
        log("Downloading SEC ticker mapping...")
        data = fetch_json(SEC_TICKER_URL, session, sleep_seconds)
        cache_path.write_text(json.dumps(data))

    rows = []
    for _, row in data.items():
        cik = str(row["cik_str"]).zfill(10)
        rows.append(
            {
                "cik": cik,
                "ticker": str(row.get("ticker", "")).upper().strip(),
                "title": row.get("title"),
            }
        )
    return pd.DataFrame(rows).drop_duplicates()


def load_universe_tickers(universe_date: Optional[str]) -> pd.DataFrame:
    universe_path = DATA_DIR / "curated" / "universe_r3000_proxy.parquet"
    names_path = DATA_DIR / "wrds" / "crsp" / "msenames_2000-01-01_to_2026-12-31.parquet"

    if not universe_path.exists():
        raise FileNotFoundError(f"Missing universe file: {universe_path}")
    if not names_path.exists():
        raise FileNotFoundError(f"Missing CRSP names file: {names_path}")

    universe = pd.read_parquet(universe_path)
    universe["date"] = pd.to_datetime(universe["date"])

    if universe_date:
        asof_date = pd.to_datetime(universe_date)
    else:
        asof_date = universe["date"].max()

    universe = universe[universe["date"] == asof_date].copy()
    if "permno" not in universe.columns:
        raise KeyError("Universe file must contain 'permno'")
    if "permco" not in universe.columns:
        universe["permco"] = pd.NA
    universe = universe[["permno", "permco"]]

    names = pd.read_parquet(names_path, columns=["permno", "namedt", "nameendt", "ticker", "cusip", "comnam"])
    names["namedt"] = pd.to_datetime(names["namedt"])
    names["nameendt"] = pd.to_datetime(names["nameendt"])

    active = names[(names["namedt"] <= asof_date) & (names["nameendt"] >= asof_date)]
    active = active.sort_values(["permno", "nameendt"])
    latest = active.drop_duplicates(subset=["permno"], keep="last")

    merged = universe.merge(latest, on="permno", how="left")
    merged["ticker"] = merged["ticker"].str.upper().str.strip()
    return merged


