"""
Pull FMP Equity Offerings (Form D) and build a curated action file.

This uses the FMP fundraising endpoint (Form D / exempt offerings) by CIK.
Output: data/curated/equity_offerings_fmp.parquet

Env vars:
  FMP_API_KEY (required)
  FMP_BASE_URL=https://financialmodelingprep.com/stable
  FMP_SLEEP=0.2
  FMP_RETRIES=2
  FMP_TIMEOUT=30
  FMP_START_DATE=2000-01-01
  FMP_END_DATE=YYYY-MM-DD (default: today UTC)
  FMP_TARGET_CIK= (optional single cik)
  FMP_USE_CIK_MAP=1 (use WRDS cik->gvkey map)
  FMP_LIMIT_CIKS=0 (0=all)
  FMP_RESUME=1
  FMP_FLUSH_EVERY=200
  FMP_LOG_EVERY=200
  FMP_DEBUG=0
  FMP_CIK_MAP_PATH=data/wrds/compustat/cik_gvkey.csv.gz
  FMP_EQUITY_OUT_PATH=data/curated/equity_offerings_fmp.parquet
"""

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests


DATA_DIR = Path(__file__).parent.parent / "data"
FMP_DIR = DATA_DIR / "fmp"
CURATED_DIR = DATA_DIR / "curated"

FMP_API_KEY = os.getenv("FMP_API_KEY")
FMP_BASE_URL = os.getenv("FMP_BASE_URL", "https://financialmodelingprep.com/stable").rstrip("/")
FMP_SLEEP = float(os.getenv("FMP_SLEEP", "0.2"))
FMP_RETRIES = int(os.getenv("FMP_RETRIES", "2"))
FMP_TIMEOUT = float(os.getenv("FMP_TIMEOUT", "30"))
FMP_START_DATE = os.getenv("FMP_START_DATE", "2000-01-01")
FMP_END_DATE = os.getenv("FMP_END_DATE", datetime.utcnow().date().isoformat())
FMP_TARGET_CIK = os.getenv("FMP_TARGET_CIK")
FMP_USE_CIK_MAP = os.getenv("FMP_USE_CIK_MAP", "1") == "1"
FMP_LIMIT_CIKS = int(os.getenv("FMP_LIMIT_CIKS", "0"))
FMP_RESUME = os.getenv("FMP_RESUME", "1") == "1"
FMP_FLUSH_EVERY = int(os.getenv("FMP_FLUSH_EVERY", "200"))
FMP_LOG_EVERY = int(os.getenv("FMP_LOG_EVERY", "200"))
FMP_DEBUG = os.getenv("FMP_DEBUG", "0") == "1"

CIK_MAP_PATH = Path(
    os.getenv("FMP_CIK_MAP_PATH", DATA_DIR / "wrds" / "compustat" / "cik_gvkey.csv.gz")
)
OUT_PATH = Path(
    os.getenv("FMP_EQUITY_OUT_PATH", CURATED_DIR / "equity_offerings_fmp.parquet")
)
CHECKPOINT_PATH = FMP_DIR / "fmp_equity_offerings_checkpoint.txt"


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def normalize_cik(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    if not digits:
        return None
    return digits.zfill(10)


def load_cik_map(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype={"cik": "string", "gvkey": "string", "source": "string"})
    df = df[df["cik"].notna() & df["gvkey"].notna()].copy()
    df["cik"] = df["cik"].apply(normalize_cik)

    # Prefer Compustat Company, then Security, then CRSP/Compustat, then Capital IQ
    priority = {
        "Compustat Company": 0,
        "Compustat Security": 1,
        "CRSP/Compustat Merged": 2,
        "Capital IQ": 3,
    }
    df["priority"] = df["source"].map(priority).fillna(99)
    df = df.sort_values(["cik", "priority"])
    df = df.drop_duplicates("cik", keep="first")
    return dict(zip(df["cik"], df["gvkey"]))


def load_cik_list(cik_map: Dict[str, str]) -> List[str]:
    if FMP_TARGET_CIK:
        return [normalize_cik(FMP_TARGET_CIK)]
    if not FMP_USE_CIK_MAP:
        raise RuntimeError("No CIK list available. Set FMP_TARGET_CIK or FMP_USE_CIK_MAP=1.")
    ciks = [cik for cik in cik_map.keys() if cik]
    if FMP_LIMIT_CIKS and len(ciks) > FMP_LIMIT_CIKS:
        ciks = ciks[:FMP_LIMIT_CIKS]
    return ciks


def load_checkpoint() -> Set[str]:
    if not FMP_RESUME or not CHECKPOINT_PATH.exists():
        return set()
    return {line.strip() for line in CHECKPOINT_PATH.read_text().splitlines() if line.strip()}


def save_checkpoint(cik: str) -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CHECKPOINT_PATH.open("a") as handle:
        handle.write(f"{cik}\n")


def request_json(url: str, params: dict, session: requests.Session) -> list:
    for attempt in range(FMP_RETRIES + 1):
        try:
            if FMP_DEBUG:
                log(f"GET {url} params={params}")
            resp = session.get(url, params=params, timeout=FMP_TIMEOUT)
            if resp.status_code == 429:
                time.sleep(max(FMP_SLEEP * 5, 2.0))
                continue
            resp.raise_for_status()
            data = resp.json()
            if FMP_SLEEP:
                time.sleep(FMP_SLEEP)
            return data if isinstance(data, list) else []
        except Exception as exc:
            if attempt >= FMP_RETRIES:
                log(f"Request failed: {exc}")
                return []
            time.sleep(max(FMP_SLEEP, 0.5))
    return []


def build_record(item: dict, cik: str, gvkey: Optional[str]) -> Optional[dict]:
    def get_date(key: str) -> Optional[pd.Timestamp]:
        val = item.get(key)
        if not val:
            return None
        return pd.to_datetime(val, errors="coerce")

    def get_float(key: str) -> Optional[float]:
        val = item.get(key)
        if val is None or val == "":
            return None
        try:
            return float(val)
        except Exception:
            return None

    date_of_first_sale = get_date("dateOfFirstSale")
    event_date = date_of_first_sale or get_date("date") or get_date("filingDate")
    if event_date is None or pd.isna(event_date):
        return None
    equity_flag = item.get("securitiesOfferedAreOfEquityType")
    if equity_flag is False:
        return None

    return {
        "cik": cik,
        "gvkey": gvkey,
        "company_name": item.get("companyName") or item.get("entityName"),
        "action_date": event_date,
        "filing_date": get_date("filingDate"),
        "accepted_date": get_date("acceptedDate"),
        "form_type": item.get("formType"),
        "form_signification": item.get("formSignification"),
        "issuer_state": item.get("issuerStateOrCountry") or item.get("issuerStateOrCountryDescription"),
        "issuer_country": item.get("issuerStateOrCountryDescription"),
        "equity_flag": equity_flag,
        "is_amendment": item.get("isAmendment"),
        "offering_amount": get_float("totalOfferingAmount"),
        "amount_sold": get_float("totalAmountSold"),
        "amount_remaining": get_float("totalAmountRemaining"),
        "source": "fmp_form_d",
        "action_type": "form_d",
        "action_subtype": "form_d",
    }


def main() -> None:
    if not FMP_API_KEY:
        raise RuntimeError("FMP_API_KEY not set. Export your FMP API key.")

    cik_map = load_cik_map(CIK_MAP_PATH)
    ciks = load_cik_list(cik_map)
    if not ciks:
        raise RuntimeError("No CIKs available to query.")

    start_dt = pd.to_datetime(FMP_START_DATE, errors="coerce")
    end_dt = pd.to_datetime(FMP_END_DATE, errors="coerce")

    processed = load_checkpoint()
    records: List[dict] = []
    writer: Optional[pq.ParquetWriter] = None

    session = requests.Session()
    FMP_DIR.mkdir(parents=True, exist_ok=True)
    CURATED_DIR.mkdir(parents=True, exist_ok=True)

    log(f"Pulling FMP Form D offerings for {len(ciks):,} CIKs...")

    for idx, cik in enumerate(ciks, start=1):
        if FMP_RESUME and cik in processed:
            continue

        url = f"{FMP_BASE_URL}/fundraising"
        params = {"cik": cik, "apikey": FMP_API_KEY}
        payload = request_json(url, params=params, session=session)

        for item in payload:
            rec = build_record(item, cik, cik_map.get(cik))
            if rec is None:
                continue
            if start_dt is not None and rec["action_date"] < start_dt:
                continue
            if end_dt is not None and rec["action_date"] > end_dt:
                continue
            records.append(rec)

        if FMP_RESUME:
            processed.add(cik)
            save_checkpoint(cik)

        if FMP_LOG_EVERY and idx % FMP_LOG_EVERY == 0:
            log(f"Processed {idx:,}/{len(ciks):,} CIKs | records {len(records):,}")

        if FMP_FLUSH_EVERY and len(records) >= FMP_FLUSH_EVERY:
            df = pd.DataFrame.from_records(records)
            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(OUT_PATH, table.schema, compression="zstd")
            writer.write_table(table)
            records = []

    if records:
        df = pd.DataFrame.from_records(records)
        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(OUT_PATH, table.schema, compression="zstd")
        writer.write_table(table)

    if writer is not None:
        writer.close()

    log(f"Done. Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
