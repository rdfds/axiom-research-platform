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


def load_checkpoint() -> set:
    if not FMP_BULK_RESUME:
        return set()
    checkpoint_path = FMP_DIR / "fmp_financials_bulk_checkpoint.txt"
    if not checkpoint_path.exists():
        return set()
    return set([line.strip() for line in checkpoint_path.read_text().splitlines() if line.strip()])


def save_checkpoint(entries: Iterable[str]) -> None:
    FMP_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = FMP_DIR / "fmp_financials_bulk_checkpoint.txt"
    with checkpoint_path.open("a", encoding="utf-8") as f:
        for entry in entries:
            f.write(entry + "\n")


def fetch_bulk(statement: str, year: int, period: str, session: requests.Session) -> Optional[List[Dict]]:
    endpoint = STATEMENT_ENDPOINTS[statement]
    url = f"{FMP_BASE_URL}/{endpoint}"
    params = {"year": year, "period": period, "apikey": FMP_API_KEY}
    data = _request_json(url, params=params, session=session)
    if data is None:
        return None
    if not data:
        return []
    if isinstance(data, dict) and data.get("Error Message"):
        return []
    if isinstance(data, list):
        return data
    return []


def main() -> None:
    require_api_key()
    FMP_DIR.mkdir(parents=True, exist_ok=True)

    log(
        "Starting FMP bulk financials: years "
        f"{FMP_BULK_START_YEAR}-{FMP_BULK_END_YEAR} | "
        f"periods={','.join(FMP_BULK_PERIODS)} | "
        f"statements={','.join(FMP_BULK_STATEMENTS)}"
    )
    if FMP_SKIP_GVKEY:
        log("GVKEY mapping disabled (FMP_SKIP_GVKEY=1); using symbol as company_id.")
    log("Loading CRSP mapping tables (gvkey <-> ticker)...")
    names, link = load_mappings()
    log(f"Loaded mappings: names={len(names):,} | links={len(link):,}")
    checkpoint = load_checkpoint()
    if checkpoint:
        log(f"Loaded checkpoint entries: {len(checkpoint):,}")

    if FMP_BULK_USE_PARTITIONED:
        warehouse_dir = DATA_DIR / "warehouse"
        table_dir = warehouse_dir / "warehouse_financials"
        file_path = warehouse_dir / "warehouse_financials.parquet"
        if not table_dir.exists():
            table_dir.mkdir(parents=True, exist_ok=True)
        if file_path.exists() and FMP_BULK_MIGRATE_FILE:
            dest = table_dir / f"part_legacy_{int(datetime.utcnow().timestamp())}.parquet"
            file_path.replace(dest)
            log("Moved existing warehouse_financials.parquet into partitioned directory.")

    session = requests.Session()
    ingestion_time = datetime.utcnow()

    raw_buffer: List[Dict] = []
    canonical_buffer: List[Dict] = []
    total_records = 0

    for year in range(FMP_BULK_START_YEAR, FMP_BULK_END_YEAR + 1):
        for period in FMP_BULK_PERIODS:
            for statement in FMP_BULK_STATEMENTS:
                key = f"{statement}|{year}|{period}"
                if key in checkpoint:
                    continue

                log(f"Fetching {statement} {year} {period}...")
                rows = fetch_bulk(statement, year, period, session)
                if rows is None:
                    log(f"Skipping checkpoint for {statement} {year} {period} due to request failure.")
                    continue
                if not rows:
                    checkpoint.add(key)
                    save_checkpoint([key])
                    continue

                line_map = LINE_ITEM_MAP[statement]

                for row in rows:
                    symbol = row.get("symbol")
                    if not symbol:
                        continue
                    symbol_norm = str(symbol).upper().strip().replace("-", ".")

                    date_str = row.get("date")
                    if not date_str:
                        continue
                    event_time = pd.to_datetime(date_str, errors="coerce")
                    if pd.isna(event_time):
                        continue

                    if FMP_SKIP_GVKEY:
                        company_id = symbol_norm
                        quality_flags = ["missing_gvkey"]
                    else:
                        company_id = map_symbol_to_gvkey(symbol_norm, event_time, names, link)
                        if not company_id:
                            company_id = row.get("cik") or symbol_norm
                            quality_flags = ["missing_gvkey"]
                        else:
                            quality_flags = []

                    available_time = pd.to_datetime(
                        row.get("acceptedDate") or row.get("filingDate"), errors="coerce"
                    )
                    if pd.isna(available_time):
                        lag_days = 45 if period.upper().startswith("Q") else 90
                        available_time = event_time + pd.Timedelta(days=lag_days)
                        quality_flags.append("estimated_available_time")
                    if available_time < event_time:
                        available_time = event_time
                        if "estimated_available_time" not in quality_flags:
                            quality_flags.append("estimated_available_time")

                    fiscal_year = row.get("calendarYear") or event_time.year
                    fiscal_quarter = None
                    fp = row.get("period") or period
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

                    if len(canonical_buffer) >= FMP_FLUSH_EVERY:
                        if raw_buffer:
                            write_raw_records(source_system="fmp_financials", records=raw_buffer)
                            raw_buffer = []
                        append_canonical_records("warehouse_financials", canonical_buffer)
                        total_records += len(canonical_buffer)
                        log(f"Ingested {len(canonical_buffer):,} records (total {total_records:,})")
                        canonical_buffer = []

                checkpoint.add(key)
                save_checkpoint([key])
                log(f"Completed {statement} {year} {period}: {len(rows):,} rows")

    if raw_buffer:
        write_raw_records(source_system="fmp_financials", records=raw_buffer)
    if canonical_buffer:
        append_canonical_records("warehouse_financials", canonical_buffer)
        total_records += len(canonical_buffer)
    log(f"Done. Total FMP financial records: {total_records:,}")


if __name__ == "__main__":
    main()
