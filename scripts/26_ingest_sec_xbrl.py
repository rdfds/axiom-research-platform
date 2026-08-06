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

                    end = pd.to_datetime(entry.get("end"), errors="coerce")
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--universe-date", default=None)
    parser.add_argument("--ciks", default=None, help="Comma-separated CIKs (override universe)")
    parser.add_argument("--tickers", default=None, help="Comma-separated tickers (override universe)")
    parser.add_argument("--include-8k", action="store_true")
    parser.add_argument("--refresh", action="store_true", help="Redownload SEC payloads")
    parser.add_argument("--incremental", action="store_true", help="Only ingest filings newer than warehouse max available_time")
    parser.add_argument("--sleep", type=float, default=float(os.getenv("SEC_SLEEP_SECONDS", "0.2")))
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--fast", action="store_true", help="Only ingest a small tag set for faster runs")
    parser.add_argument("--chunk-index", type=int, default=0, help="0-based index of this worker")
    parser.add_argument("--chunk-count", type=int, default=1, help="Total number of workers")
    parser.add_argument(
        "--companyfacts-dir",
        default=None,
        help="Read local companyfacts JSONs from this directory before attempting SEC downloads",
    )
    parser.add_argument(
        "--companyfacts-zip",
        default=None,
        help="Read local SEC companyfacts.zip bulk archive instead of per-company network fetches when possible",
    )
    parser.add_argument(
        "--hydrate-cache",
        action="store_true",
        help="When reading from --companyfacts-zip, write requested payloads into data/sec/companyfacts",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Do not fall back to live SEC requests when a local companyfacts dir/zip is provided",
    )
    parser.add_argument(
        "--skip-raw-lake",
        action="store_true",
        help="Skip writing duplicate raw payloads into data/lake during local bulk/cache ingestion",
    )
    args = parser.parse_args()

    ensure_dirs()
    local_companyfacts_dir = Path(args.companyfacts_dir) if args.companyfacts_dir else (SEC_DIR / "companyfacts")
    default_companyfacts_zip = SEC_DIR / "companyfacts.zip"
    local_companyfacts_zip = (
        Path(args.companyfacts_zip)
        if args.companyfacts_zip
        else (default_companyfacts_zip if default_companyfacts_zip.exists() else None)
    )
    needs_sec_ticker_download = (not args.ciks) and (args.refresh or not (SEC_DIR / "company_tickers.json").exists())
    needs_companyfacts_network = (not args.local_only) and (
        args.refresh or local_companyfacts_zip is None
    )
    user_agent = maybe_require_user_agent(needs_sec_ticker_download or needs_companyfacts_network)
    skip_raw_lake = args.skip_raw_lake or (args.local_only and local_companyfacts_zip is not None)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Accept": "application/json",
        }
    )

    start_date = pd.to_datetime(args.start)
    end_date = pd.to_datetime(args.end)

    if args.ciks:
        cik_list = [c.strip().zfill(10) for c in args.ciks.split(",") if c.strip()]
        mapping = pd.DataFrame({"cik": cik_list})
    else:
        sec_tickers = load_sec_tickers(session, args.sleep, refresh=args.refresh)
        if args.tickers:
            tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
            mapping = sec_tickers[sec_tickers["ticker"].isin(tickers)]
        else:
            universe = load_universe_tickers(args.universe_date)
            mapping = build_cik_universe(sec_tickers, universe, limit=args.limit)

    if mapping.empty:
        raise RuntimeError("No CIKs resolved from universe/tickers.")

    build_entity_id_map(mapping)
    mapping.to_parquet(MAPPINGS_DIR / "sec_ticker_cik.parquet", index=False)

    log(f"Resolved {len(mapping):,} CIKs")
    cutoffs = load_incremental_cutoffs() if args.incremental else {}

    rows = mapping.to_dict(orient="records")
    if args.chunk_count < 1:
        raise ValueError("--chunk-count must be >= 1")
    if args.chunk_index < 0 or args.chunk_index >= args.chunk_count:
        raise ValueError("--chunk-index must be in [0, chunk-count)")
    if args.chunk_count > 1:
        rows = [row for idx, row in enumerate(rows) if idx % args.chunk_count == args.chunk_index]
        log(f"Chunking enabled: worker {args.chunk_index+1}/{args.chunk_count} handling {len(rows):,} CIKs")
    ingestion_time = datetime.utcnow()
    total_records = 0
    batch_size = max(1, args.batch_size)
    total_ciks = len(rows)
    processed_ciks = 0
    start_ts = time.perf_counter()
    allowed_tags: Optional[set] = None
    fast_enabled = args.fast or (os.getenv("SEC_FAST", "0") == "1")
    if os.getenv("SEC_TAGS"):
        allowed_tags = {t.strip() for t in os.getenv("SEC_TAGS", "").split(",") if t.strip()}
        log(f"Tag filter enabled via SEC_TAGS ({len(allowed_tags)} tags)")
    elif fast_enabled:
        allowed_tags = set(FINANCIAL_FACT_TAGS)
        log(f"Fast mode enabled: limiting to {len(allowed_tags)} tags")

    with CompanyFactsBulkSource(
        companyfacts_dir=local_companyfacts_dir,
        companyfacts_zip=local_companyfacts_zip,
        hydrate_cache=args.hydrate_cache,
        prefer_zip=local_companyfacts_zip is not None,
    ) as bulk_source:
        if local_companyfacts_zip is not None:
            log(f"Using local SEC bulk archive: {local_companyfacts_zip}")
        elif local_companyfacts_dir.exists():
            log(f"Using local SEC companyfacts cache: {local_companyfacts_dir}")
        if skip_raw_lake:
            log("Skipping raw-lake payload writes for local bulk/cache ingestion")

        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            canonical_batch: List[Dict] = []

            for row in batch:
                cik = str(row["cik"]).zfill(10)
                processed_ciks += 1

                if processed_ciks == 1 or (SEC_LOG_EVERY and processed_ciks % SEC_LOG_EVERY == 0):
                    elapsed = time.perf_counter() - start_ts
                    rate = processed_ciks / elapsed if elapsed > 0 else 0.0
                    remaining = total_ciks - processed_ciks
                    eta = (remaining / rate) if rate > 0 else 0.0
                    log(
                        f"Progress: {processed_ciks}/{total_ciks} CIKs | elapsed {elapsed/60:.1f}m | ETA {eta/60:.1f}m"
                    )

                cik_start = time.perf_counter()
                payload = None
                raw_payload_hash = None
                payload_origin = None
                if not args.refresh:
                    payload, raw_payload_hash, payload_origin = bulk_source.load_with_metadata(cik)
                if payload is None:
                    if args.local_only:
                        if SEC_LOG_VERBOSE:
                            log(f"Skipping CIK {cik}: not found in local bulk/cache source")
                        continue
                    payload = load_companyfacts(cik, session, args.sleep, refresh=args.refresh)
                    payload_origin = "network"
                if SEC_LOG_VERBOSE:
                    log(f"Fetched CIK {cik} in {(time.perf_counter() - cik_start):.1f}s via {payload_origin or 'unknown'}")
                if not payload:
                    continue

                min_available = cutoffs.get(cik)
                records, min_event, max_filed = parse_companyfacts(
                    payload=payload,
                    company_id=cik,
                    permno=row.get("permno"),
                    permco=row.get("permco"),
                    ticker=row.get("ticker"),
                    cusip=row.get("cusip"),
                    start_date=start_date,
                    end_date=end_date,
                    include_8k=args.include_8k,
                    min_available_time=min_available,
                    allowed_tags=allowed_tags,
                )
                if SEC_LOG_VERBOSE:
                    log(f"Parsed CIK {cik}: {len(records):,} records")

                if not records:
                    continue

                raw_event = min_event or ingestion_time
                raw_available = max_filed or raw_event
                if raw_payload_hash is None:
                    raw_payload_hash = compute_raw_payload_hash(payload)
                raw_version_id = compute_version_id(
                    source_system="sec_edgar_xbrl",
                    entity_id=cik,
                    event_time=raw_event.to_pydatetime() if hasattr(raw_event, "to_pydatetime") else raw_event,
                    available_time=raw_available.to_pydatetime() if hasattr(raw_available, "to_pydatetime") else raw_available,
                    raw_payload_hash=raw_payload_hash,
                )

                upstream_version_ids: List[str]
                if skip_raw_lake:
                    upstream_version_ids = []
                else:
                    write_raw_records(
                        source_system="sec_edgar_xbrl",
                        records=[
                            {
                                "entity_id": cik,
                                "company_id": cik,
                                "event_time": raw_event,
                                "available_time": raw_available,
                                "payload": payload,
                            }
                        ],
                    )
                    upstream_version_ids = [raw_version_id]

                for rec in records:
                    rec.update(
                        {
                            "source_system": "sec_edgar_xbrl",
                            "entity_id": cik,
                            "company_id": cik,
                            "security_id": None,
                            "ingestion_time": ingestion_time,
                            "raw_payload_hash": raw_payload_hash,
                            "version_id": compute_version_id(
                                source_system="sec_edgar_xbrl",
                                entity_id=cik,
                                event_time=rec["event_time"].to_pydatetime()
                                if hasattr(rec["event_time"], "to_pydatetime")
                                else rec["event_time"],
                                available_time=rec["available_time"].to_pydatetime()
                                if hasattr(rec["available_time"], "to_pydatetime")
                            else rec["available_time"],
                                raw_payload_hash=raw_payload_hash,
                            ),
                            "upstream_version_ids": upstream_version_ids,
                        }
                    )
                    canonical_batch.append(rec)

            if canonical_batch:
                append_canonical_records("warehouse_financials", canonical_batch)
                total_records += len(canonical_batch)
                log(f"Ingested {len(canonical_batch):,} records (total {total_records:,})")

    log(f"Done. Total financial statement records: {total_records:,}")


if __name__ == "__main__":
    main()
