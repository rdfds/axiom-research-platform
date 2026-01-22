"""
Build ECM proxy actions from Compustat fundamentals (share count changes).

Output: data/curated/equity_offerings_proxy.parquet

Env vars:
  FUNDAMENTALS_PATH=data/fundamentals_quarterly.parquet
  ECM_PROXY_OUT_PATH=data/curated/equity_offerings_proxy.parquet
  ECM_MIN_SHARE_CHANGE=0.03
  ECM_MAX_SHARE_CHANGE=0.50
  ECM_REQUIRE_PRICE=1
  ECM_USE_RDQ=1
  ECM_MIN_MKT_CAP=0
"""

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


DATA_DIR = Path(__file__).parent.parent / "data"
FUND_PATH = Path(os.getenv("FUNDAMENTALS_PATH", DATA_DIR / "fundamentals_quarterly.parquet"))
OUT_PATH = Path(os.getenv("ECM_PROXY_OUT_PATH", DATA_DIR / "curated" / "equity_offerings_proxy.parquet"))

MIN_SHARE_CHANGE = float(os.getenv("ECM_MIN_SHARE_CHANGE", "0.03"))
MAX_SHARE_CHANGE = float(os.getenv("ECM_MAX_SHARE_CHANGE", "0.50"))
REQUIRE_PRICE = os.getenv("ECM_REQUIRE_PRICE", "1") == "1"
USE_RDQ = os.getenv("ECM_USE_RDQ", "1") == "1"
MIN_MKT_CAP = float(os.getenv("ECM_MIN_MKT_CAP", "0"))


def log(msg: str) :
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


