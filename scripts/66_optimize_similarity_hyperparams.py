#!/usr/bin/env python
"""
Grid search similarity hyperparameters per action_type.

Writes:
  data/curated/similarity_hyperparams.parquet
  data/curated/similarity_hyperparams_scores.parquet (optional)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import time

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CURATED_DIR = ROOT / "data" / "curated"

FEATURES_PATH = CURATED_DIR / "similarity_features.parquet"
TARGET_MAP_PATH = CURATED_DIR / "similarity_best_targets.parquet"
OUT_PATH = CURATED_DIR / "similarity_hyperparams.parquet"

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


def parse_list(raw: str | None, cast):
    if not raw:
        return None
    return [cast(x.strip()) for x in raw.split(",") if x.strip() != ""]


def quantile_bins(series: pd.Series, q: int) :
    s = pd.to_numeric(series, errors="coerce")
    try:
        edges = s.quantile(np.linspace(0, 1, q + 1)).values
    except Exception:
        return pd.Series(index=s.index, data=np.nan)
    edges = np.unique(edges)
    if len(edges) < 2:
        return pd.Series(index=s.index, data=np.nan)
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    return pd.cut(s, bins=edges, labels=False, include_lowest=True)


def build_mask(
    dfa: pd.DataFrame,
    query: pd.Series,
    use_sector: bool,
    use_hard: bool,
    q_regime: Tuple[str, str, str] | None,
) -> np.ndarray:
    mask = np.ones(len(dfa), dtype=bool)
    if use_sector and "sic2" in dfa.columns:
        sec_mask = (dfa["sic2"] == query.get("sic2")).fillna(False).to_numpy(dtype=bool)
        mask &= sec_mask
    if use_hard and q_regime is not None:
        mask &= (dfa["_risk"] == q_regime[0]).to_numpy(dtype=bool)
        mask &= (dfa["_credit"] == q_regime[1]).to_numpy(dtype=bool)
        mask &= (dfa["_rate"] == q_regime[2]).to_numpy(dtype=bool)
    mask[query.name] = False
    return mask


def classify_regime(row: pd.Series, thresh: float = 0.5) -> tuple[str, str, str]:
    vix = row['z_macro_vix']
    ig = row.get("z_macro_ig_oas")
    hy = row.get("z_macro_hy_oas")
    r10 = row.get("z_macro_rate_10y")

    risk = "risk_off" if pd.notna(vix) and vix >= thresh else "risk_on"
    credit = "credit_tight" if pd.notna(ig) and pd.notna(hy) and max(ig, hy) >= thresh else "credit_loose"
    rate = "rate_high" if pd.notna(r10) and r10 >= thresh else "rate_low"
    return risk, credit, rate


