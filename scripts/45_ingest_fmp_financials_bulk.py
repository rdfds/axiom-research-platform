#!/usr/bin/env python
"""
Fast A1 via FMP BULK financial statements.

Uses bulk endpoints to pull statements by (year, period) instead of per-symbol,
which is dramatically faster for full-universe backfills.

Env:
  FMP_API_KEY (required)
  FMP_BASE_URL=https://financialmodelingprep.com/stable
  FMP_SLEEP=0.2
  FMP_RETRIES=2
  FMP_TIMEOUT=30
  FMP_BULK_START_YEAR=2000
  FMP_BULK_END_YEAR=YYYY
  FMP_BULK_PERIODS=Q1,Q2,Q3,Q4,FY
  FMP_BULK_STATEMENTS=income,balance,cash
  FMP_BULK_RESUME=1
  FMP_FLUSH_EVERY=20000
  FMP_DEBUG=0
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import csv
import io

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
FMP_RETRY_SLEEP = float(os.getenv("FMP_RETRY_SLEEP", "10"))
FMP_BULK_START_YEAR = int(os.getenv("FMP_BULK_START_YEAR", "2000"))
FMP_BULK_END_YEAR = int(os.getenv("FMP_BULK_END_YEAR", str(datetime.utcnow().year)))
FMP_BULK_PERIODS = [p.strip().upper() for p in os.getenv("FMP_BULK_PERIODS", "Q1,Q2,Q3,Q4,FY").split(",") if p.strip()]
FMP_BULK_STATEMENTS = [s.strip() for s in os.getenv("FMP_BULK_STATEMENTS", "income,balance,cash").split(",") if s.strip()]
FMP_BULK_RESUME = os.getenv("FMP_BULK_RESUME", "1") == "1"
FMP_FLUSH_EVERY = int(os.getenv("FMP_FLUSH_EVERY", "20000"))
FMP_DEBUG = os.getenv("FMP_DEBUG", "0") == "1"
FMP_BULK_USE_PARTITIONED = os.getenv("FMP_BULK_USE_PARTITIONED", "1") == "1"
FMP_BULK_MIGRATE_FILE = os.getenv("FMP_BULK_MIGRATE_FILE", "1") == "1"
FMP_SKIP_GVKEY = os.getenv("FMP_SKIP_GVKEY", "0") == "1"


STATEMENT_ENDPOINTS = {
    "income": "income-statement-bulk",
    "balance": "balance-sheet-statement-bulk",
    "cash": "cash-flow-statement-bulk",
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
    "epsDiluted": ("income", "EPSDiluted"),
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


def _request_json(url: str, params: Dict[str, object], session: requests.Session) -> Optional[List[Dict]]:
    for attempt in range(FMP_RETRIES + 1):
        try:
            if FMP_DEBUG:
                log(f"[debug] GET {url} params={_safe_params(params)}")
            resp = session.get(url, params=params, timeout=FMP_TIMEOUT)
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                try:
                    wait = float(retry_after) if retry_after else max(FMP_RETRY_SLEEP, FMP_SLEEP * 5)
                except Exception:
                    wait = max(FMP_RETRY_SLEEP, FMP_SLEEP * 5)
                log(f"Rate limited (429). Sleeping {wait:.1f}s before retrying.")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            text = resp.text
            # FMP bulk endpoints often return CSV; fall back if JSON parse fails.
            try:
                data = resp.json()
            except ValueError:
                text = text.lstrip("\ufeff").strip()
                if not text:
                    return []
                reader = csv.DictReader(io.StringIO(text))
                data = list(reader)
            if FMP_SLEEP:
                time.sleep(FMP_SLEEP)
            return data
        except requests.RequestException as exc:
            if attempt < FMP_RETRIES:
                time.sleep(max(FMP_RETRY_SLEEP, FMP_SLEEP, 0.2))
                continue
            log(f"Request failed: {url} {exc}")
            return None
    return None


def load_mappings() :
    names_path = CRSP_DIR / "msenames_2000-01-01_to_2026-12-31.parquet"
    link_path = CRSP_DIR / "ccmxpf_lnkhist.parquet"
    if not names_path.exists() or not link_path.exists():
        return pd.DataFrame(), pd.DataFrame()
    names = pd.read_parquet(names_path, columns=["permno", "namedt", "nameendt", "ticker"])
    try:
        link = pd.read_parquet(link_path, columns=["permno", "gvkey", "linkdt", "linkenddt"])
    except Exception:
        link = pd.read_parquet(link_path, columns=["lpermno", "gvkey", "linkdt", "linkenddt"])
        link = link.rename(columns={"lpermno": "permno"})
    link["permno"] = pd.to_numeric(link["permno"], errors="coerce")
    link["linkdt"] = pd.to_datetime(link["linkdt"], errors="coerce")
    link["linkenddt"] = pd.to_datetime(link["linkenddt"], errors="coerce")
    return names, link


def map_symbol_to_gvkey(symbol: str, asof: pd.Timestamp, names: pd.DataFrame, link: pd.DataFrame) -> Optional[str]:
    if names.empty or link.empty or symbol is None or pd.isna(symbol):
        return None
    symbol = str(symbol).upper().strip().replace("-", ".")
    active = names[names["ticker"].astype("string").str.upper() == symbol]
    if active.empty:
        return None
    active = active.sort_values("nameendt").tail(1)
    permno = active.iloc[0]["permno"]
    link_rows = link[link["permno"] == permno]
    if link_rows.empty:
        return None
    link_active = link_rows[(link_rows["linkdt"] <= asof) & (link_rows["linkenddt"] >= asof)]
    if link_active.empty:
        link_active = link_rows.sort_values("linkenddt").tail(1)
    gvkey = link_active.iloc[0]["gvkey"]
    return str(gvkey) if pd.notna(gvkey) else None


def normalize_payload(payload: Dict) -> Dict:
    cleaned = {}
    for k, v in payload.items():
        if isinstance(v, pd.Timestamp):
            cleaned[k] = None if pd.isna(v) else v.isoformat()
        elif isinstance(v, (np.integer, np.floating, np.bool_)):
            cleaned[k] = v.item()
        else:
            try:
                cleaned[k] = None if pd.isna(v) else v
            except Exception:
                cleaned[k] = v
    return cleaned


