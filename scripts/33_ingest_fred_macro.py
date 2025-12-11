#!/usr/bin/env python
"""
Ingest macro rates / spreads / volatility from FRED into the warehouse.

Uses the public FRED CSV endpoint (no API key required):
  https://fred.stlouisfed.org/graph/fredgraph.csv?id=SERIES&cosd=YYYY-MM-DD&coed=YYYY-MM-DD

Env:
  FRED_START=2000-01-01
  FRED_END=YYYY-MM-DD (default: today UTC)
  FRED_SLEEP=0.2
  FRED_SERIES=comma,separated,list (optional override)
"""

from __future__ import annotations

import io
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

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

FRED_START = os.getenv("FRED_START", "2000-01-01")
FRED_END = os.getenv("FRED_END", datetime.utcnow().strftime("%Y-%m-%d"))
FRED_SLEEP = float(os.getenv("FRED_SLEEP", "0.2"))
FRED_SERIES = os.getenv("FRED_SERIES")


SERIES_DEFAULT = [
    {"series_id": "DGS1", "instrument_type": "rate", "tenor": "1Y", "units": "pct"},
    {"series_id": "DGS2", "instrument_type": "rate", "tenor": "2Y", "units": "pct"},
    {"series_id": "DGS5", "instrument_type": "rate", "tenor": "5Y", "units": "pct"},
    {"series_id": "DGS10", "instrument_type": "rate", "tenor": "10Y", "units": "pct"},
    {"series_id": "DGS30", "instrument_type": "rate", "tenor": "30Y", "units": "pct"},
    {"series_id": "SOFR", "instrument_type": "rate", "tenor": "ON", "units": "pct"},
    {"series_id": "DFF", "instrument_type": "rate", "tenor": "ON", "units": "pct"},
    {"series_id": "AAA", "instrument_type": "rate", "tenor": "corp", "units": "pct"},
    {"series_id": "BAA", "instrument_type": "rate", "tenor": "corp", "units": "pct"},
    {"series_id": "BAMLC0A0CM", "instrument_type": "spread", "tenor": "OAS", "units": "pct"},
    {"series_id": "BAMLH0A0HYM2", "instrument_type": "spread", "tenor": "OAS", "units": "pct"},
    {"series_id": "VIXCLS", "instrument_type": "volatility", "tenor": None, "units": "index"},
    # Inflation / growth / labor
    {"series_id": "CPIAUCSL", "instrument_type": "inflation", "tenor": None, "units": "index"},
    {"series_id": "PCEPI", "instrument_type": "inflation", "tenor": None, "units": "index"},
    {"series_id": "GDPC1", "instrument_type": "gdp", "tenor": "real", "units": "bil_ch2017_usd"},
    {"series_id": "UNRATE", "instrument_type": "labor", "tenor": None, "units": "pct"},
    {"series_id": "INDPRO", "instrument_type": "production", "tenor": None, "units": "index"},
    {"series_id": "RSAFS", "instrument_type": "consumption", "tenor": None, "units": "mil_usd"},
    # FX / commodities / equity index
    {"series_id": "DTWEXBGS", "instrument_type": "fx", "tenor": "broad", "units": "index"},
    {"series_id": "DCOILWTICO", "instrument_type": "commodity", "tenor": "oil", "units": "usd_bbl"},
    # Gold proxy (FRED no longer serves LBMA spot series; use import price index)
    {"series_id": "IP7108", "instrument_type": "commodity", "tenor": "gold_import_price_index", "units": "index"},
    {"series_id": "SP500", "instrument_type": "equity_index", "tenor": None, "units": "index"},
]


def log(msg: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


