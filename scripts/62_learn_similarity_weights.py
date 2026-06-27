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


def main() -> None:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing similarity features at {FEATURES_PATH}")

    df = pd.read_parquet(FEATURES_PATH)
    target_map = None
    if TARGET is None and TARGET_MAP_PATH.exists():
        target_map = pd.read_parquet(TARGET_MAP_PATH)
        if "action_type" not in target_map.columns or "target" not in target_map.columns:
            target_map = None

    # Single-target mode
    if TARGET is not None:
        if TARGET not in df.columns:
            raise RuntimeError(f"Target column not found: {TARGET}")
        targets_by_action = {k: TARGET for k in df["action_type"].unique()}
    else:
        # Per-action target map (fallback to default if missing)
        default_target = "outcome_pe_12m_w"
        targets_by_action = {k: default_target for k in df["action_type"].unique()}
        if target_map is not None:
            for _, row in target_map.iterrows():
                targets_by_action[str(row["action_type"])] = str(row["target"])

    rows = []
    for action_type, group in df.groupby("action_type"):
        target = targets_by_action.get(action_type, "outcome_pe_12m_w")
        if target not in df.columns:
            continue
        grp = group.copy()
        grp = grp[grp[target].notna()]
        if len(grp) < MIN_N:
            continue

        X = grp[DELTA_FEATURES].copy()
        # z-scores are mean ~0, so fill NaN with 0 (neutral)
        X = X.fillna(0.0).to_numpy(dtype=float)
        y = grp[target].astype(float).to_numpy()
        # center y
        y = y - np.nanmean(y)

        coef = ridge_weights(X, y, LAMBDA)
        abs_coef = np.abs(coef)
        if abs_coef.sum() == 0:
            weights = np.ones_like(abs_coef) / len(abs_coef)
        else:
            weights = abs_coef / abs_coef.sum()


        for feat, c, w in zip(DELTA_FEATURES, coef, weights):
            rows.append(
                {
                    "action_type": action_type,
                    "feature": feat,
                    "coef": float(c),
                    "weight": float(w),
                    "target": target,
                    "n_obs": len(grp),
                    "lambda": LAMBDA,
                    "method": "ridge_abs_coef",
                }
            )

    # Global fallback weights per target
    all_targets = set(targets_by_action.values())
    for target in sorted(all_targets):
        if target not in df.columns:
            continue
        grp = df[df[target].notna()].copy()
        if len(grp) < MIN_N:
            continue
        X = grp[DELTA_FEATURES].fillna(0.0).to_numpy(dtype=float)
        y = grp[target].astype(float).to_numpy()
        y = y - np.nanmean(y)
        coef = ridge_weights(X, y, LAMBDA)
        abs_coef = np.abs(coef)
        weights = abs_coef / abs_coef.sum() if abs_coef.sum() else np.ones_like(abs_coef) / len(abs_coef)
        for feat, c, w in zip(DELTA_FEATURES, coef, weights):
            rows.append(
                {
                    "action_type": "ALL",
                    "feature": feat,
                    "coef": float(c),
                    "weight": float(w),
                    "target": target,
                    "n_obs": len(grp),
                    "lambda": LAMBDA,
                    "method": "ridge_abs_coef",
                }
            )

    out = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    log(f"Saved similarity weights -> {OUT_PATH} ({len(out):,} rows)")


if __name__ == "__main__":
    main()
