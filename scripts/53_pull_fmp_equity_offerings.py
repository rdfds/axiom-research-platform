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


