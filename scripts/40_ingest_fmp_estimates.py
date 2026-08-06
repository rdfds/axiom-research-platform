#!/usr/bin/env python
"""
A4-lite via FMP analyst estimates (EPS/Revenue/EBITDA).

Notes:
  - FMP does not provide a reliable publish timestamp in this endpoint.
  - We mark quality flags: partial_coverage, estimated_available_time,
    and estimated_period_end when inferred.

Writes:
  data/lake/raw/fmp_estimates
  data/warehouse/warehouse_estimates.parquet

Env:
  FMP_API_KEY (required)
  FMP_BASE_URL=https://financialmodelingprep.com/stable
  FMP_SLEEP=0.2
  FMP_RETRIES=2
  FMP_TIMEOUT=30
  FMP_PERIOD=annual
  FMP_START_YEAR=2000
  FMP_END_YEAR=YYYY
  FMP_LIMIT_SYMBOLS=0 (0=all)
  FMP_TARGET_SYMBOL=
  FMP_USE_UNIVERSE=1 (use R3000 proxy tickers)
  FMP_RESUME=1
  FMP_FLUSH_EVERY=200
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
WAREHOUSE_DIR = DATA_DIR / "warehouse"

FMP_API_KEY = os.getenv("FMP_API_KEY")
FMP_BASE_URL = os.getenv("FMP_BASE_URL", "https://financialmodelingprep.com/stable").rstrip("/")
FMP_SLEEP = float(os.getenv("FMP_SLEEP", "0.2"))
FMP_RETRIES = int(os.getenv("FMP_RETRIES", "2"))
FMP_TIMEOUT = float(os.getenv("FMP_TIMEOUT", "30"))
FMP_PERIOD = os.getenv("FMP_PERIOD", "annual")
FMP_START_YEAR = int(os.getenv("FMP_START_YEAR", "2000"))
FMP_END_YEAR = int(os.getenv("FMP_END_YEAR", str(datetime.utcnow().year)))
FMP_LIMIT_SYMBOLS = int(os.getenv("FMP_LIMIT_SYMBOLS", "0"))
FMP_TARGET_SYMBOL = os.getenv("FMP_TARGET_SYMBOL")
FMP_USE_UNIVERSE = os.getenv("FMP_USE_UNIVERSE", "1") == "1"
FMP_RESUME = os.getenv("FMP_RESUME", "1") == "1"
FMP_FLUSH_EVERY = int(os.getenv("FMP_FLUSH_EVERY", "200"))
FMP_DEBUG = os.getenv("FMP_DEBUG", "0") == "1"


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
    symbol = str(symbol).upper()
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


def load_fy_end_map() -> Dict[str, Dict[str, int]]:
    fin_path = WAREHOUSE_DIR / "warehouse_financials.parquet"
    if not fin_path.exists():
        return {}
    cols = ["company_id", "fiscal_period_end"]
    df = pd.read_parquet(fin_path, columns=cols)
    df["fiscal_period_end"] = pd.to_datetime(df["fiscal_period_end"], errors="coerce")
    df = df.dropna(subset=["company_id", "fiscal_period_end"])
    df = df.sort_values(["company_id", "fiscal_period_end"])
    last = df.groupby("company_id").tail(1)
    out = {}
    for _, row in last.iterrows():
        company_id = str(row["company_id"])
        date = row["fiscal_period_end"]
        out[company_id] = {"month": int(date.month), "day": int(date.day)}
    return out


def period_end_for_year(year: int, fy_end: Optional[Dict[str, int]]) -> pd.Timestamp:
    if fy_end:
        month = fy_end["month"]
        day = fy_end["day"]
        try:
            return pd.Timestamp(year=year, month=month, day=day)
        except Exception:
            return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    return pd.Timestamp(year=year, month=12, day=31)


def pick_value(row: Dict[str, object], metric: str) -> Optional[float]:
    key_candidates = []
    if metric == "eps":
        key_candidates = [
            "estimatedEpsAvg",
            "estimatedEPSAvg",
            "estimatedEpsMean",
            "epsMean",
            "epsAvg",
        ]
    elif metric == "revenue":
        key_candidates = [
            "estimatedRevenueAvg",
            "estimatedRevenueMean",
            "revenueAvg",
            "revenueMean",
        ]
    elif metric == "ebitda":
        key_candidates = [
            "estimatedEbitdaAvg",
            "estimatedEBITDAAvg",
            "estimatedEbitdaMean",
            "ebitdaAvg",
            "ebitdaMean",
        ]

    for k in key_candidates:
        if k in row:
            return row.get(k)

    # Fallback: search by substrings
    for k, v in row.items():
        lk = str(k).lower()
        if metric in lk and ("avg" in lk or "mean" in lk or "consensus" in lk):
            return v
    return None


def pick_num_estimates(row: Dict[str, object], metric: str) -> Optional[float]:
    key_candidates = []
    if metric == "eps":
        key_candidates = [
            "numberAnalystsEstimatedEps",
            "numberAnalystEstimatedEps",
            "numberAnalystsEstimatedEPS",
            "numberAnalystEstimatedEPS",
        ]
    elif metric == "revenue":
        key_candidates = [
            "numberAnalystEstimatedRevenue",
            "numberAnalystsEstimatedRevenue",
        ]
    elif metric == "ebitda":
        key_candidates = [
            "numberAnalystEstimatedEbitda",
            "numberAnalystsEstimatedEbitda",
            "numberAnalystEstimatedEBITDA",
        ]

    for k in key_candidates:
        if k in row:
            return row.get(k)

    for k, v in row.items():
        lk = str(k).lower()
        if "analyst" in lk and ("num" in lk or "number" in lk):
            if metric in lk:
                return v
    return None


def derive_period_end(row: Dict[str, object]) -> Tuple[Optional[pd.Timestamp], bool, Optional[int]]:
    date_val = row.get("date")
    if date_val:
        dt = pd.to_datetime(date_val, errors="coerce")
        if pd.notna(dt):
            return dt, False, int(dt.year)
    year_val = row.get("fiscalYear") or row.get("calendarYear") or row.get("year")
    if year_val:
        try:
            year = int(year_val)
            return pd.Timestamp(year=year, month=12, day=31), True, year
        except Exception:
            return None, True, None
    return None, True, None


def main() -> None:
    api_key = require_api_key()
    FMP_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    symbols: List[str] = []
    if FMP_TARGET_SYMBOL:
        symbols = [FMP_TARGET_SYMBOL.strip().upper()]
    elif FMP_USE_UNIVERSE:
        symbols = load_universe_tickers()
    else:
        raise RuntimeError("No universe available; set FMP_TARGET_SYMBOL or FMP_USE_UNIVERSE=1")

    symbols = [s for s in symbols if s]
    if FMP_LIMIT_SYMBOLS and len(symbols) > FMP_LIMIT_SYMBOLS:
        symbols = symbols[:FMP_LIMIT_SYMBOLS]

    log(f"Symbols to pull: {len(symbols):,}")

    names, link = load_mappings()
    fy_end_map = load_fy_end_map()

    checkpoint_path = FMP_DIR / "fmp_estimates_checkpoint.txt"
    processed = set()
    if FMP_RESUME and checkpoint_path.exists():
        processed = set([line.strip() for line in checkpoint_path.read_text().splitlines() if line.strip()])

    ingestion_time = datetime.utcnow()
    available_time = pd.Timestamp(ingestion_time)
    source_system = "fmp_estimates"

    existing_path = WAREHOUSE_DIR / "warehouse_estimates.parquet"
    existing_latest = None
    if existing_path.exists():
        existing = pd.read_parquet(
            existing_path, columns=["entity_id", "metric", "period", "available_time", "consensus_value"]
        )
        existing["available_time"] = pd.to_datetime(existing["available_time"], errors="coerce")
        existing = existing.sort_values(["entity_id", "metric", "period", "available_time"])
        existing_latest = existing.groupby(["entity_id", "metric", "period"]).tail(1)
        existing_latest = existing_latest.rename(columns={"consensus_value": "prev_value"})

    all_records: List[Dict] = []
    all_raw: List[Dict] = []

    for idx, symbol in enumerate(symbols, start=1):
        if FMP_RESUME and symbol in processed:
            continue

        url = f"{FMP_BASE_URL}/analyst-estimates"
        params = {"symbol": symbol, "period": FMP_PERIOD, "apikey": api_key}
        rows = _request_json(url, params, session) or []

        if not rows:
            processed.add(symbol)
            if FMP_RESUME:
                with checkpoint_path.open("a") as f:
                    f.write(f"{symbol}\n")
            if idx % 50 == 0:
                log(f"Progress: {idx}/{len(symbols)} symbols")
            continue

        # Derive period_end for each row
        parsed_rows = []
        for row in rows:
            period_end, estimated_pe, year = derive_period_end(row)
            if period_end is None:
                continue
            if period_end.year < FMP_START_YEAR or period_end.year > FMP_END_YEAR:
                continue
            parsed_rows.append((row, period_end, estimated_pe, year))

        if not parsed_rows:
            processed.add(symbol)
            if FMP_RESUME:
                with checkpoint_path.open("a") as f:
                    f.write(f"{symbol}\n")
            continue

        # Label FY1/FY2 based on future periods
        period_ends = sorted({p[1] for p in parsed_rows})
        future = [dt for dt in period_ends if dt.date() >= available_time.date()]
        if not future:
            future = period_ends[-2:] if len(period_ends) >= 2 else period_ends
        label_map = {dt: f"FY{idx+1}" for idx, dt in enumerate(future[:2])}

        for row, period_end, estimated_pe, year in parsed_rows:
            if period_end not in label_map:
                continue
            period_label = label_map[period_end]

            gvkey = map_symbol_to_gvkey(symbol, period_end, names, link)
            company_id = gvkey if gvkey is not None else None
            entity_id = gvkey if gvkey is not None else symbol

            if gvkey and estimated_pe and year is not None:
                fy_end = fy_end_map.get(str(gvkey))
                period_end = period_end_for_year(year, fy_end)

            quality_flags = ["partial_coverage", "estimated_available_time"]
            if estimated_pe:
                quality_flags.append("estimated_period_end")
            if gvkey is None:
                quality_flags.append("estimated_company_id")

            for metric in ("eps", "revenue", "ebitda"):
                value = pick_value(row, metric)
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    continue
                num_est = pick_num_estimates(row, metric)
                try:
                    value_f = float(value)
                except Exception:
                    continue

                event_time = period_end
                if event_time > available_time:
                    event_time = available_time
                    quality_flags = list(quality_flags) + ["estimated_event_time"]

                payload = {
                    "symbol": symbol,
                    "metric": metric,
                    "period": period_label,
                    "consensus_value": value_f,
                    "num_estimates": num_est,
                    "period_end": period_end.isoformat(),
                    "capture_time": available_time.isoformat(),
                }
                raw_payload_hash = compute_raw_payload_hash(payload)
                version_id = compute_version_id(
                    source_system=source_system,
                    entity_id=str(entity_id),
                    event_time=event_time.to_pydatetime(),
                    available_time=available_time.to_pydatetime(),
                    raw_payload_hash=raw_payload_hash,
                )

                all_raw.append(
                    {
                        "entity_id": str(entity_id),
                        "company_id": str(company_id) if company_id is not None else None,
                        "security_id": None,
                        "event_time": event_time,
                        "available_time": available_time,
                        "payload": payload,
                    }
                )

                all_records.append(
                    {
                        "source_system": source_system,
                        "entity_id": str(entity_id),
                        "company_id": str(company_id) if company_id is not None else None,
                        "security_id": None,
                        "event_time": event_time,
                        "available_time": available_time,
                        "ingestion_time": ingestion_time,
                        "version_id": version_id,
                        "raw_payload_hash": raw_payload_hash,
                        "upstream_version_ids": [version_id],
                        "quality_flags": quality_flags,
                        "metric": metric,
                        "period": period_label,
                        "consensus_value": value_f,
                        "num_estimates": num_est,
                        "revision_direction": None,
                        "revision_magnitude": None,
                        "period_end": period_end,
                    }
                )

                # Optional NTM derived from FY1
                if period_label == "FY1":
                    ntm_period_end = (available_time + pd.DateOffset(months=12)).to_period("M").to_timestamp("M")
                    ntm_event_time = ntm_period_end
                    ntm_flags = list(quality_flags) + ["derived_ntm_from_fy1"]
                    if ntm_event_time > available_time:
                        ntm_event_time = available_time
                        ntm_flags.append("estimated_event_time")
                    ntm_payload = dict(payload)
                    ntm_payload["period"] = "NTM"
                    ntm_payload["period_end"] = ntm_period_end.isoformat()
                    ntm_raw_hash = compute_raw_payload_hash(ntm_payload)
                    ntm_version = compute_version_id(
                        source_system=source_system,
                        entity_id=str(entity_id),
                        event_time=ntm_event_time.to_pydatetime(),
                        available_time=available_time.to_pydatetime(),
                        raw_payload_hash=ntm_raw_hash,
                    )
                    all_raw.append(
                        {
                            "entity_id": str(entity_id),
                            "company_id": str(company_id) if company_id is not None else None,
                            "security_id": None,
                            "event_time": ntm_event_time,
                            "available_time": available_time,
                            "payload": ntm_payload,
                        }
                    )
                    all_records.append(
                        {
                            "source_system": source_system,
                            "entity_id": str(entity_id),
                            "company_id": str(company_id) if company_id is not None else None,
                            "security_id": None,
                            "event_time": ntm_event_time,
                            "available_time": available_time,
                            "ingestion_time": ingestion_time,
                            "version_id": ntm_version,
                            "raw_payload_hash": ntm_raw_hash,
                            "upstream_version_ids": [ntm_version],
                            "quality_flags": ntm_flags,
                            "metric": metric,
                            "period": "NTM",
                            "consensus_value": value_f,
                            "num_estimates": num_est,
                            "revision_direction": None,
                            "revision_magnitude": None,
                            "period_end": ntm_period_end,
                        }
                    )

        processed.add(symbol)
        if FMP_RESUME:
            with checkpoint_path.open("a") as f:
                f.write(f"{symbol}\n")

        if idx % 50 == 0:
            log(f"Progress: {idx}/{len(symbols)} symbols")

        if len(all_records) >= FMP_FLUSH_EVERY:
            records_df = pd.DataFrame(all_records)
            if existing_latest is not None and not existing_latest.empty:
                records_df = records_df.merge(
                    existing_latest[["entity_id", "metric", "period", "prev_value"]],
                    on=["entity_id", "metric", "period"],
                    how="left",
                )
                diff = records_df["consensus_value"] - records_df["prev_value"]
                records_df["revision_magnitude"] = diff.where(records_df["prev_value"].notna())
                records_df["revision_direction"] = np.where(
                    records_df["prev_value"].notna(),
                    np.where(diff > 0, "up", np.where(diff < 0, "down", "flat")),
                    None,
                )
                records_df = records_df.drop(columns=["prev_value"])

            write_raw_records(source_system=source_system, records=all_raw)
            append_canonical_records("warehouse_estimates", records_df.to_dict("records"))
            all_records.clear()
            all_raw.clear()

    if all_records:
        records_df = pd.DataFrame(all_records)
        if existing_latest is not None and not existing_latest.empty:
            records_df = records_df.merge(
                existing_latest[["entity_id", "metric", "period", "prev_value"]],
                on=["entity_id", "metric", "period"],
                how="left",
            )
            diff = records_df["consensus_value"] - records_df["prev_value"]
            records_df["revision_magnitude"] = diff.where(records_df["prev_value"].notna())
            records_df["revision_direction"] = np.where(
                records_df["prev_value"].notna(),
                np.where(diff > 0, "up", np.where(diff < 0, "down", "flat")),
                None,
            )
            records_df = records_df.drop(columns=["prev_value"])

        write_raw_records(source_system=source_system, records=all_raw)
        append_canonical_records("warehouse_estimates", records_df.to_dict("records"))

    log("Done. Ingested FMP estimates (A4-lite).")


if __name__ == "__main__":
    main()
