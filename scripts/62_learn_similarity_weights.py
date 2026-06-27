#!/usr/bin/env python
"""
Learn action-specific feature weights for similarity matching.

Uses ridge regression on z-scored delta features to predict outcome.
Outputs a per-action-type weight table.

Inputs:
  data/curated/similarity_features.parquet

Output:
  data/curated/similarity_weights.parquet
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CURATED_DIR = ROOT / "data" / "curated"

FEATURES_PATH = Path(os.getenv("SIM_FEATURES_PATH", CURATED_DIR / "similarity_features.parquet"))
OUT_PATH = Path(os.getenv("SIM_WEIGHTS_PATH", CURATED_DIR / "similarity_weights.parquet"))
TARGET_MAP_PATH = Path(os.getenv("SIM_TARGET_MAP_PATH", CURATED_DIR / "similarity_best_targets.parquet"))

TARGET = os.getenv("SIM_TARGET")
MIN_N = int(os.getenv("SIM_MIN_N", "200"))
LAMBDA = float(os.getenv("SIM_RIDGE_LAMBDA", "1.0"))


DELTA_FEATURES = [
    "z_revenue_delta",
    "z_margin_delta",
    "z_leverage_delta",
    "z_eps_delta",
    "z_roic_delta",
    "z_fcf_margin_delta",
]


def log(msg: str) -> None:
    print(msg, flush=True)


def ridge_weights(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Solve ridge regression weights: (X'X + lam I)^-1 X'y."""
    n_features = X.shape[1]
    xtx = X.T @ X
    ridge = xtx + lam * np.eye(n_features)
    xty = X.T @ y
    coef = np.linalg.solve(ridge, xty)
    return coef


