#!/usr/bin/env python
"""
Fast A1 (Financial Statements) via FMP.

Pulls income / balance sheet / cash flow statements from FMP and ingests into
warehouse_financials using the canonical schema.

Env:
  FMP_API_KEY (required)
  FMP_BASE_URL=https://financialmodelingprep.com/stable
  FMP_SLEEP=0.2
  FMP_RETRIES=2
  FMP_TIMEOUT=30
  FMP_PERIODS=quarter,annual
  FMP_STATEMENTS=income,balance,cash
  FMP_START_YEAR=2000
  FMP_END_YEAR=YYYY
  FMP_LIMIT_SYMBOLS=0 (0=all)
  FMP_TARGET_SYMBOL=
  FMP_USE_UNIVERSE=1
  FMP_RESUME=1
  FMP_FLUSH_EVERY=200
  FMP_DEBUG=0
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion import (
    append_canonical_records,
    compute_raw_payload_hash,
    compute_version_id,
    write_raw_records,
)


DATA_DIR = Path(__file__).parent.parent / "data"
FMP_DIR = DATA_DIR / "fmp"
CRSP_DIR = DATA_DIR / "wrds" / "crsp"

FMP_API_KEY = os.getenv("FMP_API_KEY")
FMP_BASE_URL = os.getenv("FMP_BASE_URL", "https://financialmodelingprep.com/stable").rstrip("/")
FMP_SLEEP = float(os.getenv("FMP_SLEEP", "0.2"))
FMP_RETRIES = int(os.getenv("FMP_RETRIES", "2"))
FMP_TIMEOUT = float(os.getenv("FMP_TIMEOUT", "30"))
FMP_PERIODS = [p.strip() for p in os.getenv("FMP_PERIODS", "quarter,annual").split(",") if p.strip()]
FMP_STATEMENTS = [s.strip() for s in os.getenv("FMP_STATEMENTS", "income,balance,cash").split(",") if s.strip()]
FMP_START_YEAR = int(os.getenv("FMP_START_YEAR", "2000"))
FMP_END_YEAR = int(os.getenv("FMP_END_YEAR", str(datetime.utcnow().year)))
FMP_LIMIT_SYMBOLS = int(os.getenv("FMP_LIMIT_SYMBOLS", "0"))
FMP_TARGET_SYMBOL = os.getenv("FMP_TARGET_SYMBOL")
FMP_USE_UNIVERSE = os.getenv("FMP_USE_UNIVERSE", "1") == "1"
FMP_RESUME = os.getenv("FMP_RESUME", "1") == "1"
FMP_FLUSH_EVERY = int(os.getenv("FMP_FLUSH_EVERY", "200"))
FMP_DEBUG = os.getenv("FMP_DEBUG", "0") == "1"


STATEMENT_ENDPOINTS = {
    "income": "income-statement",
    "balance": "balance-sheet-statement",
    "cash": "cash-flow-statement",
}

INCOME_MAP = {
    "revenue": ("income", "Revenue"),
    "costOfRevenue": ("income", "COGS"),
    "grossProfit": ("income", "GrossProfit"),
    "operatingExpenses": ("income", "OperatingExpenses"),
    "operatingIncome": ("income", "OperatingIncome"),
    "interestExpense": ("income", "InterestExpense"),
    "ebitda": ("income", "EBITDA"),
    "incomeBeforeTax": ("income", "PretaxIncome"),
    "netIncome": ("income", "NetIncome"),
    "eps": ("income", "EPS"),
    "epsdiluted": ("income", "EPSDiluted"),
    "weightedAverageShsOut": ("income", "SharesOut"),
    "weightedAverageShsOutDil": ("income", "SharesOutDiluted"),
}

BALANCE_MAP = {
    "totalAssets": ("balance_sheet", "TotalAssets"),
    "totalCurrentAssets": ("balance_sheet", "CurrentAssets"),
    "cashAndCashEquivalents": ("balance_sheet", "Cash"),
    "shortTermInvestments": ("balance_sheet", "ShortTermInvestments"),
    "netReceivables": ("balance_sheet", "Receivables"),
    "inventory": ("balance_sheet", "Inventory"),
    "propertyPlantEquipmentNet": ("balance_sheet", "PP&E"),
    "totalLiabilities": ("balance_sheet", "TotalLiabilities"),
    "totalCurrentLiabilities": ("balance_sheet", "CurrentLiabilities"),
    "shortTermDebt": ("balance_sheet", "DebtCurrent"),
    "longTermDebt": ("balance_sheet", "DebtLongTerm"),
    "totalStockholdersEquity": ("balance_sheet", "TotalEquity"),
    "commonStock": ("balance_sheet", "CommonEquity"),
    "commonStockSharesOutstanding": ("balance_sheet", "SharesOut"),
}

CASH_MAP = {
    "netCashProvidedByOperatingActivities": ("cash_flow", "OperatingCashFlow"),
    "netCashUsedForInvestingActivites": ("cash_flow", "InvestingCashFlow"),
    "netCashProvidedByFinancingActivities": ("cash_flow", "FinancingCashFlow"),
    "capitalExpenditure": ("cash_flow", "Capex"),
    "freeCashFlow": ("cash_flow", "FreeCashFlow"),
    "netIncome": ("cash_flow", "NetIncome"),
}

LINE_ITEM_MAP = {
    "income": INCOME_MAP,
    "balance": BALANCE_MAP,
    "cash": CASH_MAP,
}


def log(msg: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def require_api_key() -> str:
    if not FMP_API_KEY:
        raise RuntimeError("FMP_API_KEY not set. Export your FMP API key.")
    return FMP_API_KEY


def _safe_params(params: Dict[str, object]) -> Dict[str, object]:
    safe = dict(params)
    if "apikey" in safe:
        safe["apikey"] = "***REDACTED***"
    return safe


