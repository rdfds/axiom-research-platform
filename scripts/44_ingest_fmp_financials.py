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


def _request_json(url: str, params: Dict[str, object], session: requests.Session) -> Optional[List[Dict]]:
    for attempt in range(FMP_RETRIES + 1):
        try:
            if FMP_DEBUG:
                log(f"[debug] GET {url} params={_safe_params(params)}")
            resp = session.get(url, params=params, timeout=FMP_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            if FMP_SLEEP:
                time.sleep(FMP_SLEEP)
            return data
        except requests.RequestException as exc:
            if attempt < FMP_RETRIES:
                time.sleep(max(FMP_SLEEP, 0.2))
                continue
            log(f"Request failed: {url} {exc}")
            return None
    return None


def load_universe_tickers() -> List[str]:
    universe_path = DATA_DIR / "curated" / "universe_r3000_proxy.parquet"
    names_path = CRSP_DIR / "msenames_2000-01-01_to_2026-12-31.parquet"
    if not universe_path.exists() or not names_path.exists():
        return []
    universe = pd.read_parquet(universe_path)
    universe["date"] = pd.to_datetime(universe["date"])
    asof_date = universe["date"].max()
    universe = universe[universe["date"] == asof_date][["permno"]]
    names = pd.read_parquet(names_path, columns=["permno", "namedt", "nameendt", "ticker"])
    names["namedt"] = pd.to_datetime(names["namedt"], errors="coerce")
    names["nameendt"] = pd.to_datetime(names["nameendt"], errors="coerce")
    active = names[names["nameendt"] == names["nameendt"].max()]
    active = active.sort_values(["permno", "nameendt"])
    latest = active.drop_duplicates(subset=["permno"], keep="last")
    merged = universe.merge(latest, on="permno", how="left")
    tickers = merged["ticker"].dropna().astype("string").str.upper().tolist()
    return tickers


def normalize_symbol(symbol: str) -> str:
    symbol = str(symbol).upper().strip()
    # FMP uses dashes for share classes (e.g., BRK-B)
    symbol = symbol.replace(".", "-")
    return symbol


def load_symbol_list() -> List[str]:
    symbols: List[str] = []
    if FMP_TARGET_SYMBOL:
        symbols = [normalize_symbol(FMP_TARGET_SYMBOL)]
    elif FMP_USE_UNIVERSE:
        symbols = [normalize_symbol(s) for s in load_universe_tickers()]
        if symbols:
            log(f"Using universe tickers: {len(symbols):,}")
    if not symbols:
        # fallback to financial statement symbol list
        session = requests.Session()
        data = _request_json(f"{FMP_BASE_URL}/financial-statement-symbol-list", {"apikey": FMP_API_KEY}, session)
        if data:
            symbols = [normalize_symbol(row.get("symbol")) for row in data if row.get("symbol")]
    symbols = sorted(set([s for s in symbols if s]))
    if FMP_LIMIT_SYMBOLS and FMP_LIMIT_SYMBOLS > 0:
        symbols = symbols[:FMP_LIMIT_SYMBOLS]
    return symbols


def load_mappings() -> Tuple[pd.DataFrame, pd.DataFrame]:
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


def normalize_value(value) -> Optional[float]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return None


def fetch_statement(symbol: str, period: str, statement: str, session: requests.Session) -> List[Dict]:
    endpoint = STATEMENT_ENDPOINTS[statement]
    url = f"{FMP_BASE_URL}/{endpoint}"
    params = {"symbol": symbol, "period": period, "apikey": FMP_API_KEY}
    data = _request_json(url, params=params, session=session)
    if not data:
        return []
    if isinstance(data, dict) and data.get("Error Message"):
        return []
    if isinstance(data, list):
        return data
    return []


def load_checkpoint() -> set:
    if not FMP_RESUME:
        return set()
    checkpoint_path = FMP_DIR / "fmp_financials_checkpoint.txt"
    if not checkpoint_path.exists():
        return set()
    return set([line.strip() for line in checkpoint_path.read_text().splitlines() if line.strip()])


def save_checkpoint(entries: Iterable[str]) -> None:
    FMP_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = FMP_DIR / "fmp_financials_checkpoint.txt"
    with checkpoint_path.open("a", encoding="utf-8") as f:
        for entry in entries:
            f.write(entry + "\n")


def main() -> None:
    require_api_key()
    FMP_DIR.mkdir(parents=True, exist_ok=True)

    symbols = load_symbol_list()
    if not symbols:
        raise RuntimeError("No symbols resolved for FMP financials.")
    log(f"Symbols to pull: {len(symbols):,}")

    names, link = load_mappings()
    checkpoint = load_checkpoint()

    session = requests.Session()
    total_records = 0
    raw_buffer: List[Dict] = []
    canonical_buffer: List[Dict] = []
    ingestion_time = datetime.utcnow()

    for idx, symbol in enumerate(symbols, start=1):
        for period in FMP_PERIODS:
            for statement in FMP_STATEMENTS:
                key = f"{symbol}|{period}|{statement}"
                if key in checkpoint:
                    continue

                rows = fetch_statement(symbol, period, statement, session)
                if not rows:
                    checkpoint.add(key)
                    save_checkpoint([key])
                    continue

                for row in rows:
                    date_str = row.get("date")
                    if not date_str:
                        continue
                    event_time = pd.to_datetime(date_str, errors="coerce")
                    if pd.isna(event_time):
                        continue
                    if event_time.year < FMP_START_YEAR or event_time.year > FMP_END_YEAR:
                        continue

                    company_id = map_symbol_to_gvkey(symbol, event_time, names, link)
                    if not company_id:
                        company_id = row.get("cik") or symbol
                        quality_flags = ["missing_gvkey"]
                    else:
                        quality_flags = []

                    available_time = pd.to_datetime(
                        row.get("acceptedDate") or row.get("fillingDate"), errors="coerce"
                    )
                    if pd.isna(available_time):
                        # Estimate availability window
                        lag_days = 45 if period == "quarter" else 90
                        available_time = event_time + pd.Timedelta(days=lag_days)
                        quality_flags.append("estimated_available_time")
                    if available_time < event_time:
                        available_time = event_time
                        if "estimated_available_time" not in quality_flags:
                            quality_flags.append("estimated_available_time")

                    fiscal_year = row.get("calendarYear") or event_time.year
                    fiscal_quarter = None
                    fp = row.get("period")
                    if isinstance(fp, str) and fp.upper().startswith("Q"):
                        fiscal_quarter = fp.upper().replace("Q", "")

                    raw_payload = normalize_payload(row)
                    raw_payload_hash = compute_raw_payload_hash(raw_payload)
                    raw_version_id = compute_version_id(
                        source_system="fmp_financials",
                        entity_id=str(company_id),
                        event_time=event_time.to_pydatetime(),
                        available_time=available_time.to_pydatetime(),
                        raw_payload_hash=raw_payload_hash,
                    )

                    raw_buffer.append(
                        {
                            "entity_id": str(company_id),
                            "company_id": str(company_id),
                            "security_id": None,
                            "event_time": event_time,
                            "available_time": available_time,
                            "payload": raw_payload,
                        }
                    )

                    line_map = LINE_ITEM_MAP[statement]
                    for field, (statement_type, line_item) in line_map.items():
                        value = normalize_value(row.get(field))
                        if value is None:
                            continue
                        canonical_buffer.append(
                            {
                                "source_system": "fmp_financials",
                                "entity_id": str(company_id),
                                "company_id": str(company_id),
                                "security_id": None,
                                "event_time": event_time,
                                "available_time": available_time,
                                "ingestion_time": ingestion_time,
                                "version_id": compute_version_id(
                                    source_system="fmp_financials",
                                    entity_id=f"{company_id}:{statement_type}:{line_item}",
                                    event_time=event_time.to_pydatetime(),
                                    available_time=available_time.to_pydatetime(),
                                    raw_payload_hash=raw_payload_hash,
                                ),
                                "raw_payload_hash": raw_payload_hash,
                                "upstream_version_ids": [raw_version_id],
                                "quality_flags": quality_flags,
                                "fiscal_period_end": event_time,
                                "fiscal_year": int(fiscal_year) if fiscal_year else None,
                                "fiscal_quarter": int(fiscal_quarter) if fiscal_quarter else None,
                                "statement_type": statement_type,
                                "line_item": line_item,
                                "value": value,
                                "currency": row.get("reportedCurrency"),
                                "units": None,
                                "restatement_flag": False,
                            }
                        )

                checkpoint.add(key)
                save_checkpoint([key])

                if len(canonical_buffer) >= FMP_FLUSH_EVERY:
                    if raw_buffer:
                        write_raw_records(source_system="fmp_financials", records=raw_buffer)
                        raw_buffer = []
                    append_canonical_records("warehouse_financials", canonical_buffer)
                    total_records += len(canonical_buffer)
                    log(f"Ingested {len(canonical_buffer):,} records (total {total_records:,})")
                    canonical_buffer = []

        if idx % 50 == 0:
            log(f"Symbols processed: {idx}/{len(symbols)}")

    if raw_buffer:
        write_raw_records(source_system="fmp_financials", records=raw_buffer)
    if canonical_buffer:
        append_canonical_records("warehouse_financials", canonical_buffer)
        total_records += len(canonical_buffer)
    log(f"Done. Total FMP financial records: {total_records:,}")


if __name__ == "__main__":
    main()
