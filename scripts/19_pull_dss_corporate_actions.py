#!/usr/bin/env python
"""
Pull DSS Corporate Actions (US Active, 2000-present)
====================================================
Uses DataScope Select (DSS) REST API Corporate Actions extraction.

Requirements:
  - DSS token OR DSS username/password via env vars
  - Universe file from prior RIC build (defaults to data/refinitiv/universe_us_active.parquet)

Env vars:
  DSS_URL=https://selectapi.datascope.lseg.com/RestApi/v1
  DSS_TOKEN=...
  DSS_USERNAME=...
  DSS_PASSWORD=...
  DSS_APP_KEY=... (optional, if required by your DSS setup)
  DSS_REBUILD_UNIVERSE=0|1
  DSS_REQUIRE_FULL_UNIVERSE=0|1
  DSS_SCREEN_MIN_COUNT=500
  DSS_USE_ALL_FIELDS=0|1
  DSS_FIELD_NAME_SOURCE=Name|Code
  DSS_FIELDS_FILE=path/to/fields.txt|json
  DSS_CONDITION_FILE=path/to/condition.json
  DSS_BATCH_SIZE=200
  DSS_SMOKE_TEST=0|1
  DSS_SMOKE_TICKERS=50
  DSS_SMOKE_DAYS=30
  DSS_SMOKE_MAX_BATCHES=1
  DSS_START_DATE=2000-01-01
  DSS_END_DATE=2026-02-02

Outputs:
  data/refinitiv/corporate_actions_dss/ca_<year>_part_<batch>.parquet
  data/refinitiv/corporate_actions_dss/manifest.jsonl
"""

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests


DATA_DIR = Path(__file__).parent.parent / "data" / "refinitiv"
DATA_DIR.mkdir(parents=True, exist_ok=True)

UNIVERSE_PATH = Path(os.getenv("DSS_UNIVERSE_FILE", DATA_DIR / "universe_us_active.parquet"))
OUT_DIR = DATA_DIR / "corporate_actions_dss"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = OUT_DIR / "manifest.jsonl"

DSS_URL = os.getenv("DSS_URL", "https://selectapi.datascope.lseg.com/RestApi/v1").rstrip("/")
DSS_TOKEN = os.getenv("DSS_TOKEN")
DSS_USERNAME = os.getenv("DSS_USERNAME")
DSS_PASSWORD = os.getenv("DSS_PASSWORD")
DSS_APP_KEY = os.getenv("DSS_APP_KEY")

DSS_REBUILD_UNIVERSE = os.getenv("DSS_REBUILD_UNIVERSE", "0") == "1"
DSS_REQUIRE_FULL_UNIVERSE = os.getenv("DSS_REQUIRE_FULL_UNIVERSE", "0") == "1"
DSS_SCREEN_MIN_COUNT = int(os.getenv("DSS_SCREEN_MIN_COUNT", "500"))

DSS_USE_ALL_FIELDS = os.getenv("DSS_USE_ALL_FIELDS", "0") == "1"
DSS_FIELD_NAME_SOURCE = os.getenv("DSS_FIELD_NAME_SOURCE", "Name")  # Name or Code
DSS_FIELDS_FILE = os.getenv("DSS_FIELDS_FILE")
DSS_CONDITION_FILE = os.getenv("DSS_CONDITION_FILE")

START_DATE = os.getenv("DSS_START_DATE", "2000-01-01")
END_DATE = os.getenv("DSS_END_DATE", "2026-02-02")

BATCH_SIZE = int(os.getenv("DSS_BATCH_SIZE", "200"))
SLEEP_SECONDS = 0.4
POLL_SECONDS = 4
MAX_POLL = 120

SMOKE_TEST = os.getenv("DSS_SMOKE_TEST", "0") == "1"
SMOKE_TICKERS = int(os.getenv("DSS_SMOKE_TICKERS", "50"))
SMOKE_DAYS = int(os.getenv("DSS_SMOKE_DAYS", "30"))
SMOKE_MAX_BATCHES = int(os.getenv("DSS_SMOKE_MAX_BATCHES", "1"))

DEFAULT_FIELDS = [
    "Corporate Actions Type",
    "Corporate Actions Type Description",
    "Capital Change Event Type",
    "Capital Change Event Type Description",
    "Actual Adjustment Type",
    "Actual Adjustment Type Description",
    "Adjustment Factor",
    "Currency Code",
    "Exchange Code",
    "Effective Date",
    "Dividend Pay Date",
    "Dividend Rate",
    "Nominal Value",
    "Nominal Value Currency",
    "Nominal Value Date",
]


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_universe() -> List[str]:
    if DSS_REBUILD_UNIVERSE:
        rebuild_universe()
    if not UNIVERSE_PATH.exists():
        raise FileNotFoundError(
            f"Universe file not found: {UNIVERSE_PATH}. "
            "Run scripts/18_probe_and_pull_refinitiv_corp_actions.py to build it, "
            "or set DSS_UNIVERSE_FILE."
        )
    suffix = UNIVERSE_PATH.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(UNIVERSE_PATH)
        if "ric" not in df.columns and "RIC" not in df.columns:
            raise ValueError(f"Universe file missing 'ric' column: {UNIVERSE_PATH}")
        col = "ric" if "ric" in df.columns else "RIC"
        tickers = df[col].dropna().unique().tolist()
        return tickers
    if suffix in {".csv"}:
        df = pd.read_csv(UNIVERSE_PATH)
        if "ric" in df.columns or "RIC" in df.columns:
            col = "ric" if "ric" in df.columns else "RIC"
            return df[col].dropna().unique().tolist()
        # fallback to first column
        return df.iloc[:, 0].dropna().unique().tolist()
    if suffix in {".txt"}:
        with open(UNIVERSE_PATH, "r") as f:
            return [line.strip() for line in f if line.strip()]
    raise ValueError(f"Unsupported universe file type: {UNIVERSE_PATH}")


def try_screen_universe() -> Tuple[List[str], Dict[str, str]]:
    import refinitiv.data as rd

    screen_queries = [
        ("SCREEN(U(IN(Equity)), TR.ExchangeCountry='United States' AND TR.Status='Active')", "exchange_country_status"),
        ("SCREEN(U(IN(Equity)), TR.ExchangeCountry='United States')", "exchange_country"),
        ("SCREEN(U(IN(Equity)), TR.CountryOfIncorporation='United States' AND TR.Status='Active')", "incorporation_country_status"),
        ("SCREEN(U(IN(Equity)), TR.CountryOfIncorporation='United States')", "incorporation_country"),
    ]

    for screen, label in screen_queries:
        try:
            df = rd.get_data(universe=screen, fields=["TR.CommonName"])
            if df is None or len(df) == 0:
                log(f"Screen {label} returned no rows.")
                continue
            tickers = df["Instrument"].dropna().unique().tolist()
            if len(tickers) < DSS_SCREEN_MIN_COUNT:
                log(f"Screen {label} returned only {len(tickers)} tickers (too small).")
                continue
            return tickers, {"method": "screen", "label": label}
        except Exception as e:
            log(f"Screen {label} failed: {e}")

    return [], {}


def universe_from_indices() -> Tuple[List[str], Dict[str, str]]:
    import refinitiv.data as rd

    index_universes = [
        "0#.SPX",
        "0#.MID",
        "0#.SML",
        "0#.RUI",
        "0#.RUT",
        "0#.RUA",
        "0#.NDX",
    ]
    tickers: List[str] = []

    for idx in index_universes:
        try:
            df = rd.get_data(universe=idx, fields=["TR.CommonName"])
            if df is None or len(df) == 0:
                continue
            tickers.extend(df["Instrument"].dropna().tolist())
        except Exception as e:
            log(f"Index universe {idx} failed: {e}")

    tickers = sorted(list(set(tickers)))
    return tickers, {"method": "indices", "label": ",".join(index_universes)}


def rebuild_universe() -> None:
    import refinitiv.data as rd

    log("Rebuilding US active equity universe...")
    rd.open_session()
    try:
        tickers, meta = try_screen_universe()
        if not tickers:
            if DSS_REQUIRE_FULL_UNIVERSE:
                raise RuntimeError(
                    "Screen universe failed or returned too few tickers. "
                    "Set DSS_REQUIRE_FULL_UNIVERSE=0 to allow index fallback, "
                    "or provide a custom universe file."
                )
            log("Screen universe failed. Falling back to index constituents.")
            tickers, meta = universe_from_indices()

        if not tickers:
            raise RuntimeError("Failed to build any universe from Refinitiv.")

        df = pd.DataFrame({
            "ric": sorted(list(set(tickers))),
            "source_method": meta.get("method", "unknown"),
            "source_label": meta.get("label", "unknown"),
            "pulled_at": datetime.now().isoformat(),
        })
        df.to_parquet(UNIVERSE_PATH, index=False)
        log(f"Saved universe: {len(df):,} tickers -> {UNIVERSE_PATH.name}")
    finally:
        rd.close_session()


def batched(items: List[str], batch_size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def request_token(session: requests.Session) -> str:
    if DSS_TOKEN:
        return DSS_TOKEN
    if not DSS_USERNAME or not DSS_PASSWORD:
        raise RuntimeError("Set DSS_TOKEN or DSS_USERNAME/DSS_PASSWORD env vars.")

    url = f"{DSS_URL}/Authentication/RequestToken"
    payloads = [
        {"Username": DSS_USERNAME, "Password": DSS_PASSWORD},
        {"Credentials": {"Username": DSS_USERNAME, "Password": DSS_PASSWORD}},
    ]
    if DSS_APP_KEY:
        payloads.append({"Username": DSS_USERNAME, "Password": DSS_PASSWORD, "AppKey": DSS_APP_KEY})
        payloads.append({"Credentials": {"Username": DSS_USERNAME, "Password": DSS_PASSWORD, "AppKey": DSS_APP_KEY}})

    last_error = None
    for payload in payloads:
        resp = session.post(url, json=payload, timeout=60)
        if resp.status_code == 400:
            last_error = resp.text
            continue
        resp.raise_for_status()
        data = resp.json()
        token = data.get("value") or data.get("Token") or data.get("token")
        if not token:
            raise RuntimeError(f"Token not found in response: {data}")
        return token

    raise RuntimeError(
        "DSS token request failed with HTTP 400 for all payload formats. "
        f"Response: {last_error}"
    )


def headers(token: str, prefer_async: bool = True) -> Dict[str, str]:
    hdrs = {"Authorization": f"Token {token}"}
    if prefer_async:
        hdrs["Prefer"] = "respond-async"
    return hdrs


def get_valid_fields(session: requests.Session, token: str) -> List[str]:
    url = (
        f"{DSS_URL}/Extractions/GetValidContentFieldTypes"
        "(ReportTemplateType=DataScope.Select.Api.Extractions.ReportTemplates.ReportTemplateTypes'CorporateActions')"
    )
    resp = session.get(url, headers=headers(token, prefer_async=False), timeout=60)
    resp.raise_for_status()
    data = resp.json()
    values = data.get("value", [])
    if DSS_FIELD_NAME_SOURCE.lower() == "code":
        return [v["Code"] for v in values if "Code" in v]
    return [v["Name"] for v in values if "Name" in v]


def load_fields(session: requests.Session, token: str) -> List[str]:
    if DSS_FIELDS_FILE:
        p = Path(DSS_FIELDS_FILE)
        if not p.exists():
            raise FileNotFoundError(f"Field list not found: {p}")
        if p.suffix.lower() in {".json", ".jsonl"}:
            with open(p, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                fields = data.get("fields", [])
            else:
                fields = data
            return list(fields)
        with open(p, "r") as f:
            return [line.strip() for line in f if line.strip()]

    if DSS_USE_ALL_FIELDS:
        log("Fetching full corporate-actions field list from DSS...")
        fields = get_valid_fields(session, token)
        log(f"Loaded {len(fields)} fields.")
        return fields

    return DEFAULT_FIELDS


def load_condition() -> Dict:
    if DSS_CONDITION_FILE:
        p = Path(DSS_CONDITION_FILE)
        if not p.exists():
            raise FileNotFoundError(f"Condition file not found: {p}")
        with open(p, "r") as f:
            cond = json.load(f)
        return cond

    return {
        "ReportDateRangeType": "Range",
        "RangeStartDate": "{start_date}",
        "RangeEndDate": "{end_date}",
        "ExcludeDeletedEvents": True,
        "IncludeCapitalChangeEvents": True,
        "IncludeDividendEvents": True,
        "IncludeEarningsEvents": True,
        "IncludeMergersAndAcquisitionsEvents": True,
        "IncludeNominalValueEvents": True,
        "IncludePublicEquityOfferingsEvents": True,
        "IncludeSharesOutstandingEvents": True,
        "IncludeVotingRightsEvents": True,
        "CorporateActionsCapitalChangeType": "CapitalChangeExDate",
        "CorporateActionsDividendsType": "DividendPayDate",
        "CorporateActionsEarningsType": "PeriodEndDate",
        "ShareAmountTypes": [],
    }


def apply_condition_dates(condition: Dict, start_date: str, end_date: str) -> Dict:
    raw = json.dumps(condition)
    raw = raw.replace("{start_date}", start_date).replace("{end_date}", end_date)
    return json.loads(raw)


def extract_with_notes(
    session: requests.Session,
    token: str,
    fields: List[str],
    identifiers: List[str],
    condition: Dict,
) -> Tuple[Dict, str]:
    url = f"{DSS_URL}/Extractions/ExtractWithNotes"
    payload = {
        "ExtractionRequest": {
            "@odata.type": "#DataScope.Select.Api.Extractions.ExtractionRequests.CorporateActionsStandardExtractionRequest",
            "ContentFieldNames": fields,
            "IdentifierList": {
                "@odata.type": "#DataScope.Select.Api.Extractions.ExtractionRequests.InstrumentIdentifierList",
                "InstrumentIdentifiers": [
                    {"Identifier": ric, "IdentifierType": "Ric"} for ric in identifiers
                ],
            },
            "Condition": condition,
        }
    }

    resp = session.post(url, headers=headers(token), json=payload, timeout=120)
    if resp.status_code == 401 and DSS_TOKEN is None and DSS_USERNAME and DSS_PASSWORD:
        token = request_token(session)
        resp = session.post(url, headers=headers(token), json=payload, timeout=120)

    if resp.status_code == 202:
        location = resp.headers.get("Location")
        if not location:
            raise RuntimeError("Async response missing Location header.")
        data, token = poll_location(session, token, location)
        return data, token

    resp.raise_for_status()
    return resp.json(), token


def poll_location(session: requests.Session, token: str, location: str) -> Tuple[Dict, str]:
    for _ in range(MAX_POLL):
        time.sleep(POLL_SECONDS)
        resp = session.get(location, headers=headers(token, prefer_async=False), timeout=120)
        if resp.status_code == 202:
            continue
        if resp.status_code == 401 and DSS_TOKEN is None and DSS_USERNAME and DSS_PASSWORD:
            token = request_token(session)
            continue
        resp.raise_for_status()
        data = resp.json()
        if "Contents" in data or "Notes" in data:
            return data, token
        if data.get("Status") in {"Completed", "CompletedWithWarnings"}:
            return data, token
    raise TimeoutError("Timed out waiting for DSS extraction.")


def save_manifest(entry: Dict) -> None:
    with open(MANIFEST_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def write_results(
    data: Dict,
    out_path: Path,
    year: int,
    batch_index: int,
    start_date: str,
    end_date: str,
) -> int:
    contents = data.get("Contents") or []
    if not contents:
        return 0
    df = pd.DataFrame(contents)
    df["pull_start"] = start_date
    df["pull_end"] = end_date
    df["batch_index"] = batch_index
    df["year"] = year
    df.to_parquet(out_path, index=False)
    save_manifest({
        "file": out_path.name,
        "rows": len(df),
        "year": year,
        "batch_index": batch_index,
        "timestamp": datetime.now().isoformat(),
    })
    return len(df)


def main() -> None:
    log("Loading universe...")
    tickers = load_universe()
    log(f"Universe: {len(tickers):,} RICs")

    session = requests.Session()
    token = request_token(session)

    fields = load_fields(session, token)
    log(f"Using {len(fields)} fields.")

    condition_template = load_condition()

    start_date = START_DATE
    end_date = END_DATE

    if SMOKE_TEST:
        today = datetime.utcnow().date()
        start_date = (today - timedelta(days=SMOKE_DAYS)).strftime("%Y-%m-%d")
        end_date = today.strftime("%Y-%m-%d")
        tickers = tickers[:SMOKE_TICKERS]
        log(f"Smoke test enabled: {len(tickers)} tickers, {start_date} -> {end_date}")

    start_year = int(start_date[:4])
    end_year = int(end_date[:4])

    batches = list(batched(tickers, BATCH_SIZE))
    if SMOKE_TEST and len(batches) > SMOKE_MAX_BATCHES:
        batches = batches[:SMOKE_MAX_BATCHES]
        log(f"Smoke test: limiting to {len(batches)} batch(es)")
    log(f"Pulling in {len(batches)} batches per year...")

    for year in range(start_year, end_year + 1):
        year_start = f"{year}-01-01"
        year_end = f"{year}-12-31"
        if year == end_year:
            year_end = end_date
        if year == start_year:
            year_start = start_date

        log(f"Year {year}: {year_start} -> {year_end}")

        for b_idx, batch in enumerate(batches):
            out_path = OUT_DIR / f"ca_{year}_part_{b_idx:04d}.parquet"
            if out_path.exists():
                continue
            condition = apply_condition_dates(condition_template, year_start, year_end)
            try:
                data, token = extract_with_notes(session, token, fields, batch, condition)
                rows = write_results(data, out_path, year, b_idx, year_start, year_end)
                log(f"Saved {rows:,} rows -> {out_path.name}")
            except Exception as e:
                log(f"Batch {b_idx} failed: {e}")
            time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    main()
