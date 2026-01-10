#!/usr/bin/env python
"""
Pull Refinitiv (RDP) monthly prices for 2025 and map to CRSP gvkey/permno.

Outputs:
  data/refinitiv/prices_monthly_rdp_2025.parquet   (raw, RIC-keyed)
  data/prices_monthly_rdp_2025.parquet             (mapped, gvkey/permno keyed)

Optional:
  MERGE_INTO_PRICES_MONTHLY=1 will append into data/prices_monthly.parquet
  (dedupe on gvkey + date).
"""

import os
import time
from pathlib import Path
import pandas as pd
import refinitiv.data as rd


DATA_DIR = Path(__file__).parent.parent / "data"
CRSP_DIR = DATA_DIR / "wrds" / "crsp"
REF_DIR = DATA_DIR / "refinitiv"
REF_DIR.mkdir(parents=True, exist_ok=True)

RDP_START = os.getenv("RDP_START", "2024-12-01")
RDP_END = os.getenv("RDP_END", "2025-12-31")
UNIVERSE_DATE = os.getenv("RDP_UNIVERSE_DATE", "2024-12-31")
BATCH_SIZE = int(os.getenv("RDP_BATCH", "50"))
SLEEP = float(os.getenv("RDP_SLEEP", "0.2"))
SPLIT_ON_ERROR = os.getenv("RDP_SPLIT_ON_ERROR", "1") == "1"
SKIP_PULL = os.getenv("RDP_SKIP_PULL", "0") == "1"

MERGE = os.getenv("MERGE_INTO_PRICES_MONTHLY", "0") == "1"


def log(msg: str) -> None:
    print(msg, flush=True)


