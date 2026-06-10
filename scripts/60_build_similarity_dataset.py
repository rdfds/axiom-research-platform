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


def robust_zscore(
    df: pd.DataFrame,
    col: str,
    group_cols: Iterable[str],
    min_group: int = 30,
    cap: Optional[float] = 6.0,
) -> pd.Series:
    """Compute robust z-score using median and MAD within groups, fallback to global."""
    series = pd.to_numeric(df[col], errors="coerce")
    group = df.groupby(list(group_cols))[col]

    med = group.transform("median")
    mad = group.transform(lambda x: (x - x.median()).abs().median())
    size = group.transform("size")

    global_med = series.median()
    global_mad = (series - global_med).abs().median()
    global_scale = 1.4826 * (global_mad if pd.notna(global_mad) and global_mad != 0 else 1.0)

    scale = 1.4826 * mad.replace(0, np.nan)
    z = (series - med) / scale

    # fallback for small or degenerate groups
    fallback = (series - global_med) / global_scale
    z = np.where((size < min_group) | scale.isna(), fallback, z)

    if cap is not None:
        z = np.clip(z, -cap, cap)
    return pd.Series(z, index=df.index, name=f"z_{col}")


def winsorize_by_group(
    df: pd.DataFrame,
    col: str,
    group_col: str = "action_type",
    p: float = 0.01,
    min_group: int = 200,
) :
    """Winsorize column by action_type (fallback to global if group too small)."""
    series = pd.to_numeric(df[col], errors="coerce")
    global_lo = series.quantile(p)
    global_hi = series.quantile(1 - p)

    def clip_group(x: pd.Series) -> pd.Series:
        if x.notna().sum() < min_group:
            return x.clip(global_lo, global_hi)
        lo = x.quantile(p)
        hi = x.quantile(1 - p)
        return x.clip(lo, hi)

    return series.groupby(df[group_col]).transform(clip_group)


