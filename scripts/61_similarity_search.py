#!/usr/bin/env python
"""
Similarity search over historical actions.

Given a query (company_id + action_date + action_type), returns top-K matches
based on profile + change + macro distances using robust z-scores.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CURATED_DIR = ROOT / "data" / "curated"

FEATURES_PATH = CURATED_DIR / "similarity_features.parquet"
WEIGHTS_PATH = CURATED_DIR / "similarity_weights.parquet"
TARGET_MAP_PATH = CURATED_DIR / "similarity_best_targets.parquet"
HYPERPARAMS_PATH = CURATED_DIR / "similarity_hyperparams.parquet"

BASE_FEATURES = [
    "z_base_market_cap_log",
    "z_base_leverage",
    "z_base_margin",
    "z_base_revenue_ttm_log",
    "z_base_roic",
    "z_base_fcf_margin",
    "z_base_pe",
    "z_base_ev_ebitda",
    "z_gross_margin_vol_12q",
    "z_gross_margin_vol_20q",
    "z_roic_vol_12q",
    "z_roic_vol_20q",
    "z_fcf_margin_vol_12q",
    "z_fcf_margin_vol_20q",
    "z_revenue_cagr_1y",
    "z_revenue_cagr_3y",
    "z_fcf_cagr_1y",
    "z_fcf_cagr_3y",
    "z_eps_cagr_1y",
    "z_eps_cagr_3y",
    "z_ev_ebitda_pct_5y",
    "z_interest_coverage",
    "z_net_debt_ebitda",
    "z_mom_6m",
    "z_mom_12m",
    "z_vol_6m",
    "z_vol_12m",
    "z_max_drawdown_12m",
    "z_deal_size_to_mcap",
    "z_mna_target_public",
    "z_mna_cross_border",
    "z_mna_deal_type_stake",
    "z_mna_deal_type_lbo",
    "z_mna_deal_type_tender",
    "z_mna_deal_type_merger",
    "z_mna_pct_cash",
    "z_mna_pct_stock",
    "z_mna_payment_cash",
    "z_mna_payment_stock",
    "z_mna_payment_mixed",
    "z_mna_premium_1d",
    "z_mna_premium_1w",
    "z_mna_premium_4w",
    "z_mna_deal_completed",
    "z_mna_same_sic2",
    "z_mna_same_sic4",
    "z_mna_same_country",
]

DELTA_FEATURES = [
    "z_revenue_delta",
    "z_margin_delta",
    "z_leverage_delta",
    "z_eps_delta",
    "z_roic_delta",
    "z_fcf_margin_delta",
]

MACRO_FEATURES = [
    "z_macro_rate_10y",
    "z_macro_rate_2y",
    "z_macro_sofr",
    "z_macro_ig_oas",
    "z_macro_hy_oas",
    "z_macro_vix",
]


def load_weights() -> pd.DataFrame | None:
    if not WEIGHTS_PATH.exists():
        return None
    return pd.read_parquet(WEIGHTS_PATH)


def classify_regime(row: pd.Series, thresh: float = 0.5) -> tuple[str, str, str]:
    vix = row.get("z_macro_vix")
    ig = row.get("z_macro_ig_oas")
    hy = row.get("z_macro_hy_oas")
    r10 = row.get("z_macro_rate_10y")

    risk = "risk_off" if pd.notna(vix) and vix >= thresh else "risk_on"
    credit = "credit_tight" if pd.notna(ig) and pd.notna(hy) and max(ig, hy) >= thresh else "credit_loose"
    rate = "rate_high" if pd.notna(r10) and r10 >= thresh else "rate_low"
    return risk, credit, rate


def quantile_bins(series: pd.Series, q: int) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    try:
        edges = s.quantile(np.linspace(0, 1, q + 1)).values
    except Exception:
        return pd.Series(index=s.index, data=np.nan)
    edges = np.unique(edges)
    if len(edges) < 2:
        return pd.Series(index=s.index, data=np.nan)
    # pad edges to include min/max
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    return pd.cut(s, bins=edges, labels=False, include_lowest=True)


def weight_lookup(weights: pd.DataFrame, action_type: str, features: List[str], target: str | None = None) -> np.ndarray:
    if weights is None or weights.empty:
        return np.ones(len(features), dtype=float)
    subset = weights[(weights["action_type"] == action_type) & (weights["feature"].isin(features))]
    if target is not None and "target" in weights.columns:
        subset = subset[subset["target"] == target]
    if subset.empty:
        subset = weights[(weights["action_type"] == "ALL") & (weights["feature"].isin(features))]
        if target is not None and "target" in weights.columns:
            subset = subset[subset["target"] == target]
    if subset.empty:
        return np.ones(len(features), dtype=float)
    mapping = dict(zip(subset["feature"], subset["weight"]))
    return np.array([mapping.get(f, 1.0) for f in features], dtype=float)


def load_target_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    if "action_type" not in df.columns or "target" not in df.columns:
        return {}
    return dict(zip(df["action_type"].astype(str), df["target"].astype(str)))


def load_hyperparams(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    if "action_type" not in df.columns:
        return {}
    out: dict[str, dict[str, object]] = {}
    for _, row in df.iterrows():
        out[str(row["action_type"])] = row.to_dict()
    return out


def weighted_distance(row: pd.Series, query: pd.Series, features: List[str], weights: np.ndarray) -> float:
    diffs = row[features] - query[features]
    mask = diffs.notna() & query[features].notna()
    if mask.sum() == 0:
        return np.nan
    w = weights[mask.to_numpy()]
    d = np.sqrt(np.sum(w * (diffs[mask].to_numpy() ** 2)) / np.sum(w))
    return float(d)


