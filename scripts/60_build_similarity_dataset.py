#!/usr/bin/env python
"""
Build similarity feature table for action matching.

Inputs:
  data/curated/action_outcomes.parquet
  data/curated/fundamentals_master.parquet

Outputs:
  data/curated/similarity_features.parquet

This table includes:
  - baseline features at t0
  - delta features (t0 -> t1)
  - macro context at t0
  - outcomes (3m/6m/12m)
  - sector (sic2) + time bucket (year)
  - robust z-scores (sector+year for firm features, year for macro)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CURATED_DIR = DATA_DIR / "curated"

ACTION_OUTCOMES_PATH = Path(os.getenv("SIM_ACTION_OUTCOMES_PATH", CURATED_DIR / "action_outcomes.parquet"))
FUNDAMENTALS_PATH = Path(os.getenv("SIM_FUNDAMENTALS_PATH", CURATED_DIR / "fundamentals_master.parquet"))
PRICES_PATH = Path(os.getenv("SIM_PRICES_PATH", CURATED_DIR / "prices_master.parquet"))
OUT_PATH = Path(os.getenv("SIM_OUT_PATH", CURATED_DIR / "similarity_features.parquet"))
MNA_PATH = Path(os.getenv("SIM_MNA_PATH", CURATED_DIR / "mna_master.parquet"))

MIN_GROUP = int(os.getenv("SIM_MIN_GROUP", "30"))
Z_CAP = float(os.getenv("SIM_Z_CAP", "6.0"))  # cap z-scores to avoid extreme outliers
BUCKET_Q = int(os.getenv("SIM_BUCKET_Q", "3"))
DEAL_BUCKET_Q_STRICT = int(os.getenv("SIM_DEAL_BUCKET_Q_STRICT", "5"))


def log(msg: str) -> None:
    print(msg, flush=True)


