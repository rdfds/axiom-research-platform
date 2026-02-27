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


def build_cik_universe(
    sec_tickers: pd.DataFrame,
    universe_tickers: pd.DataFrame,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    merged = universe_tickers.merge(sec_tickers, on="ticker", how="left")
    merged = merged.dropna(subset=["cik"])
    merged = merged.drop_duplicates(subset=["cik"])
    if limit:
        merged = merged.head(limit)
    return merged


def load_companyfacts(
    cik: str,
    session: requests.Session,
    sleep_seconds: float,
    refresh: bool = False,
) -> Optional[Dict]:
    cache_path = SEC_DIR / "companyfacts" / f"CIK{cik}.json"
    if cache_path.exists() and not refresh:
        if SEC_LOG_VERBOSE:
            log(f"Using cached companyfacts for CIK {cik}")
        return json.loads(cache_path.read_text())

    url = SEC_COMPANYFACTS_URL.format(cik=cik)
    try:
        if SEC_LOG_VERBOSE:
            log(f"Downloading companyfacts for CIK {cik}")
        payload = fetch_json(url, session, sleep_seconds)
    except requests.RequestException as exc:
        log(f"Failed CIK {cik}: {exc}")
        return None

    cache_path.write_text(json.dumps(payload))
    return payload


def parse_value(val) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except Exception:
        return None


def normalize_currency(unit: str) -> Optional[str]:
    if not unit:
        return None
    if "/" in unit:
        unit = unit.split("/")[0]
    unit = unit.strip()
    if len(unit) == 3 and unit.isalpha():
        return unit.upper()
    return None


def statement_type_for_tag(tag: str) -> str:
    return STATEMENT_TYPE_BY_TAG.get(tag, "unknown")


def parse_companyfacts(
    payload: Dict,
    company_id: str,
    permno: Optional[int],
    permco: Optional[int],
    ticker: Optional[str],
    cusip: Optional[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    include_8k: bool,
    min_available_time: Optional[pd.Timestamp] = None,
    allowed_tags: Optional[set] = None,
) -> Tuple[List[Dict], Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    facts = payload.get("facts", {})
    records: List[Dict] = []
    min_event: Optional[pd.Timestamp] = None
    max_filed: Optional[pd.Timestamp] = None
    start_date_str = start_date.date().isoformat()
    end_date_str = end_date.date().isoformat()
    min_available_date_str = (
        min_available_time.date().isoformat() if min_available_time is not None else None
    )

    for taxonomy, taxonomy_data in facts.items():
        for tag, tag_data in taxonomy_data.items():
            if allowed_tags is not None and tag not in allowed_tags:
                continue
            units = tag_data.get("units", {})
            for unit_name, values in units.items():
                for entry in values:
                    raw_end = entry.get("end")
                    raw_filed = entry.get("filed")

                    # Fast path for incremental/local bulk runs: skip obviously stale
                    # entries before paying the pandas timestamp parsing cost.
                    if min_available_date_str is not None:
                        if raw_filed:
                            filed_prefix = str(raw_filed)[:10]
                            if len(filed_prefix) == 10 and filed_prefix <= min_available_date_str:
                                continue
                        elif raw_end:
                            end_prefix = str(raw_end)[:10]
                            if len(end_prefix) == 10 and end_prefix <= min_available_date_str:
                                continue

                    if raw_end:
                        end_prefix = str(raw_end)[:10]
                        if len(end_prefix) == 10 and (end_prefix < start_date_str or end_prefix > end_date_str):
                            continue

                    end = pd.to_datetime(entry['end'], errors="coerce")
                    filed = pd.to_datetime(entry.get("filed"), errors="coerce")
                    form = entry.get("form")

                    if pd.isna(end):
                        continue
                    if end < start_date or end > end_date:
                        continue

                    if form:
                        form = str(form).upper()
                        if not (form.startswith("10-K") or form.startswith("10-Q") or (include_8k and form.startswith("8-K"))):
                            continue

                    value = parse_value(entry.get("val"))
                    if value is None:
                        continue

                    fiscal_year = entry.get("fy")
                    fiscal_period = entry.get("fp")
                    fiscal_quarter = None
                    if isinstance(fiscal_period, str) and fiscal_period.startswith("Q"):
                        fiscal_quarter = fiscal_period.replace("Q", "")

                    available_time = filed if not pd.isna(filed) else end
                    quality_flags: List[str] = []
                    if pd.isna(filed) or available_time < end:
                        available_time = end
                        quality_flags.append("estimated_available_time")

                    if min_available_time is not None and not pd.isna(available_time):
                        if available_time <= min_available_time:
                            continue

                    restatement_flag = bool(form and form.endswith("/A"))
                    if restatement_flag:
                        quality_flags.append("restatement")

                    min_event = end if min_event is None else min(min_event, end)
                    max_filed = available_time if max_filed is None else max(max_filed, available_time)

                    records.append(
                        {
                            "company_id": company_id,
                            "entity_id": company_id,
                            "permno": permno,
                            "permco": permco,
                            "ticker": ticker,
                            "cusip": cusip,
                            "taxonomy": taxonomy,
                            "line_item": tag,
                            "statement_type": statement_type_for_tag(tag),
                            "value": value,
                            "currency": normalize_currency(unit_name),
                            "units": unit_name,
                            "fiscal_period_end": end,
                            "fiscal_year": fiscal_year,
                            "fiscal_quarter": fiscal_quarter,
                            "form_type": form,
                            "frame": entry.get("frame"),
                            "accession": entry.get("accn"),
                            "event_time": end,
                            "available_time": available_time,
                            "quality_flags": quality_flags,
                            "restatement_flag": restatement_flag,
                        }
                    )

    return records, min_event, max_filed


def build_entity_id_map(mapping: pd.DataFrame) -> None:
    path = MAPPINGS_DIR / "entity_id_map.parquet"
    mapping = mapping.copy()
    if "company_id" not in mapping.columns:
        mapping["company_id"] = mapping["cik"]
    if path.exists():
        existing = pd.read_parquet(path)
        mapping = pd.concat([existing, mapping], ignore_index=True, sort=False)
        mapping = mapping.drop_duplicates(subset=["company_id"], keep="last")
    mapping.to_parquet(path, index=False)


def load_incremental_cutoffs() -> Dict[str, pd.Timestamp]:
    path = DATA_DIR / "warehouse" / "warehouse_financials.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(path, columns=["company_id", "available_time"])
    df["available_time"] = pd.to_datetime(df["available_time"])
    cutoffs = df.groupby("company_id")["available_time"].max().to_dict()
    return cutoffs


