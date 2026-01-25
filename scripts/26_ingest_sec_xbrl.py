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


def require_user_agent() :
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


