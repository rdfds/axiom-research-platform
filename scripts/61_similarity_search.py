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


def feature_contribs(row: pd.Series, query: pd.Series, features: List[str], weights: np.ndarray) -> List[tuple[str, float]]:
    diffs = row[features] - query[features]
    mask = diffs.notna() & query[features].notna()
    if mask.sum() == 0:
        return []
    w = weights[mask.to_numpy()]
    contrib = w * (diffs[mask].to_numpy() ** 2)
    denom = np.sum(w) if np.sum(w) > 0 else 1.0
    contrib = contrib / denom
    feats = [f for f, m in zip(features, mask) if m]
    return sorted(zip(feats, contrib.tolist()), key=lambda x: x[1], reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--company-id", required=True, help="Company ID (gvkey) for query row.")
    parser.add_argument("--action-type", required=True, help="Action type to match (e.g., acquisition, buyback).")
    parser.add_argument("--action-date", help="Action date (YYYY-MM-DD). Uses closest row if multiple.")
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--same-sector", action="store_true", default=True, help="Filter to same sic2.")
    parser.add_argument("--no-same-sector", dest="same_sector", action="store_false")
    parser.add_argument("--profile-weight", type=float, default=0.3)
    parser.add_argument("--change-weight", type=float, default=0.6)
    parser.add_argument("--macro-weight", type=float, default=0.1)
    parser.add_argument("--missing-penalty", type=float, default=0.1)
    parser.add_argument("--regime-filter", action="store_true", default=True)
    parser.add_argument("--no-regime-filter", dest="regime_filter", action="store_false")
    parser.add_argument("--regime-threshold", type=float, default=0.5)
    parser.add_argument("--regime-mode", choices=["hard", "soft", "off"], default=None)
    parser.add_argument("--regime-bins", type=int, default=4)
    parser.add_argument("--regime-weight", type=float, default=0.2)
    parser.add_argument("--profile-buckets", action="store_true", default=True)
    parser.add_argument("--no-profile-buckets", dest="profile_buckets", action="store_false")
    parser.add_argument("--max-year-gap", type=int, default=None, help="Max |action_year - query_year| for candidates.")
    parser.add_argument("--min-candidates", type=int, default=200, help="Minimum candidate pool before relaxing filters.")
    parser.add_argument("--sector-penalty", type=float, default=0.2, help="Penalty added when sector differs (if sector filter is relaxed).")
    parser.add_argument("--deal-size-penalty", type=float, default=0.0, help="Penalty weight for deal_size_to_mcap difference.")
    parser.add_argument("--deal-size-factor", type=float, default=None, help="Require deal_size_to_mcap within factor range of query (e.g., 3.0 = within 3x).")
    parser.add_argument("--mna-public-penalty", type=float, default=0.0, help="Penalty for M&A target public/private mismatch (acquisitions only).")
    parser.add_argument("--mna-crossborder-penalty", type=float, default=0.0, help="Penalty for M&A cross-border mismatch (acquisitions only).")
    parser.add_argument("--mna-dealtype-penalty", type=float, default=0.0, help="Penalty for M&A deal type mismatch (acquisitions only).")
    parser.add_argument("--mna-payment-penalty", type=float, default=0.0, help="Penalty for M&A payment type mismatch (cash/stock/mixed).")
    parser.add_argument("--mna-completion-penalty", type=float, default=0.0, help="Penalty for M&A completion status mismatch.")
    parser.add_argument("--scenario-macro", help="Override macro regime using comma-separated z-scores: vix,ig_oas,hy_oas,rate_10y")
    parser.add_argument("--predict-outcomes", action="store_true", default=True)
    parser.add_argument("--no-predict-outcomes", dest="predict_outcomes", action="store_false")
    parser.add_argument("--target", help="Optional target name for weight selection.")
    parser.add_argument("--target-map", default=str(TARGET_MAP_PATH), help="Path to per-action best target map.")
    parser.add_argument("--weighting", choices=["kernel", "inverse", "uniform"], default="kernel")
    parser.add_argument("--kernel-sigma", type=float, default=None)
    parser.add_argument("--shrinkage", type=float, default=5.0, help="Shrink predictions toward action-type mean (0 to disable).")
    parser.add_argument("--use-hyperparams", action="store_true", default=True)
    parser.add_argument("--no-hyperparams", dest="use_hyperparams", action="store_false")
    parser.add_argument("--out", help="Optional CSV output path.")
    args = parser.parse_args()

    if args.regime_mode is None:
        regime_mode = "soft" if args.regime_filter else "off"
    else:
        regime_mode = args.regime_mode

    df = pd.read_parquet(FEATURES_PATH)
    df = df[df["action_type"] == args.action_type].copy().reset_index(drop=True)
    if df.empty:
        raise RuntimeError("No rows for action_type.")

    # select query row
    q = df[df["company_id"] == args.company_id].copy()
    if q.empty:
        raise RuntimeError("No rows for company_id in similarity_features.")
    if args.action_date:
        q["action_date"] = pd.to_datetime(q["action_date"], errors="coerce")
        target = pd.to_datetime(args.action_date)
        q["date_diff"] = (q["action_date"] - target).abs()
        q = q.sort_values("date_diff").head(1)
    else:
        q = q.sort_values("action_date").tail(1)
    query = q.iloc[0]

    # Scenario macro override (z-scores)
    if args.scenario_macro:
        parts = [p.strip() for p in args.scenario_macro.split(",")]
        if len(parts) != 4:
            raise RuntimeError("scenario_macro must have 4 comma-separated values: vix,ig_oas,hy_oas,rate_10y")
        query["z_macro_vix"] = float(parts[0])
        query["z_macro_ig_oas"] = float(parts[1])
        query["z_macro_hy_oas"] = float(parts[2])
        query["z_macro_rate_10y"] = float(parts[3])

    # Precompute regime labels/bins for potential soft/hard filtering
    if regime_mode == "hard":
        q_regime = classify_regime(query, args.regime_threshold)
        df["_risk"], df["_credit"], df["_rate"] = zip(*df.apply(lambda r: classify_regime(r, args.regime_threshold), axis=1))
    if regime_mode in ("soft", "hard"):
        for feat in MACRO_FEATURES:
            df[f"_bin_{feat}"] = quantile_bins(df[feat], args.regime_bins)
        q_bins = {feat: df.loc[q.index[0], f"_bin_{feat}"] for feat in MACRO_FEATURES}

    weights = load_weights()
    target_map = load_target_map(Path(args.target_map)) if args.target is None else {}
    hyper_map = load_hyperparams(HYPERPARAMS_PATH) if args.use_hyperparams else {}
    if hyper_map.get(args.action_type):
        hp = hyper_map[args.action_type]
        args.top_k = int(hp.get("k", args.top_k))
        args.weighting = str(hp.get("weighting", args.weighting))
        args.regime_mode = str(hp.get("regime_mode", args.regime_mode)) if hp.get("regime_mode") else args.regime_mode
        args.sector_penalty = float(hp.get("sector_penalty", args.sector_penalty))
        args.min_candidates = int(hp.get("min_candidates", args.min_candidates))
        args.shrinkage = float(hp.get("shrinkage", args.shrinkage))
        if hp.get("max_year_gap") is not None and not pd.isna(hp.get("max_year_gap")):
            args.max_year_gap = int(hp.get("max_year_gap"))
        if hp.get("deal_size_penalty") is not None and not pd.isna(hp.get("deal_size_penalty")):
            args.deal_size_penalty = float(hp.get("deal_size_penalty"))
        if hp.get("deal_size_factor") is not None and not pd.isna(hp.get("deal_size_factor")):
            args.deal_size_factor = float(hp.get("deal_size_factor"))
        if hp.get("mna_public_penalty") is not None and not pd.isna(hp.get("mna_public_penalty")):
            args.mna_public_penalty = float(hp.get("mna_public_penalty"))
        if hp.get("mna_crossborder_penalty") is not None and not pd.isna(hp.get("mna_crossborder_penalty")):
            args.mna_crossborder_penalty = float(hp.get("mna_crossborder_penalty"))
        if hp.get("mna_dealtype_penalty") is not None and not pd.isna(hp.get("mna_dealtype_penalty")):
            args.mna_dealtype_penalty = float(hp.get("mna_dealtype_penalty"))
        if hp.get("mna_payment_penalty") is not None and not pd.isna(hp.get("mna_payment_penalty")):
            args.mna_payment_penalty = float(hp.get("mna_payment_penalty"))
        if hp.get("mna_completion_penalty") is not None and not pd.isna(hp.get("mna_completion_penalty")):
            args.mna_completion_penalty = float(hp.get("mna_completion_penalty"))
        print(f"[similarity_search] using hyperparams: k={args.top_k}, weighting={args.weighting}, regime={args.regime_mode}, shrinkage={args.shrinkage}", flush=True)
    target_for_weights = args.target or target_map.get(args.action_type)
    if target_for_weights:
        print(f"Using weight target: {target_for_weights}", flush=True)
    w_delta = weight_lookup(weights, args.action_type, DELTA_FEATURES, target_for_weights)
    w_base = np.ones(len(BASE_FEATURES), dtype=float)
    w_macro = np.ones(len(MACRO_FEATURES), dtype=float)

    # Build candidate mask with fallback ladder
    q_idx = q.index[0]
    use_sector = args.same_sector and pd.notna(query.get("sic2"))
    use_buckets = args.profile_buckets
    use_deal_size_range = args.deal_size_factor is not None
    use_hard = regime_mode == "hard"
    effective_regime_mode = regime_mode

    def build_mask(use_sector_flag: bool, use_buckets_flag: bool, use_hard_flag: bool, use_deal_size_flag: bool) -> np.ndarray:
        mask = np.ones(len(df), dtype=bool)
        if args.max_year_gap is not None and "action_year" in df.columns and pd.notna(query.get("action_year")):
            year_diff = (df["action_year"] - query.get("action_year")).abs()
            mask &= (year_diff <= args.max_year_gap).fillna(False).to_numpy(dtype=bool)
        if use_deal_size_flag and "deal_size_to_mcap" in df.columns:
            q_ds = query.get("deal_size_to_mcap")
            if pd.notna(q_ds):
                q_abs = abs(float(q_ds))
                if q_abs > 0 and args.deal_size_factor:
                    cand = df["deal_size_to_mcap"].astype(float).abs()
                    lower = q_abs / float(args.deal_size_factor)
                    upper = q_abs * float(args.deal_size_factor)
                    mask &= (cand >= lower) & (cand <= upper)
        if use_sector_flag and "sic2" in df.columns:
            sec_mask = (df["sic2"] == query.get("sic2")).fillna(False).to_numpy(dtype=bool)
            mask &= sec_mask
        if use_buckets_flag:
            bucket_cols = ["size_bucket", "leverage_bucket", "margin_bucket"]
            if args.action_type in {"acquisition", "divestiture"}:
                if "deal_size_bucket_strict" in df.columns:
                    bucket_cols.append("deal_size_bucket_strict")
                elif "deal_size_bucket" in df.columns:
                    bucket_cols.append("deal_size_bucket")
            for bucket in bucket_cols:
                if bucket in df.columns:
                    qval = query.get(bucket)
                    if pd.isna(qval):
                        continue
                    comp = pd.to_numeric(df[bucket], errors="coerce")
                    bucket_mask = (comp == float(qval)).fillna(False).to_numpy(dtype=bool)
                    mask &= bucket_mask
        if use_hard_flag:
            mask &= (df["_risk"] == q_regime[0]).to_numpy(dtype=bool)
            mask &= (df["_credit"] == q_regime[1]).to_numpy(dtype=bool)
            mask &= (df["_rate"] == q_regime[2]).to_numpy(dtype=bool)
        mask[q_idx] = False
        return mask

    mask = build_mask(use_sector, use_buckets, use_hard, use_deal_size_range)
    cand_count = int(mask.sum())
    if args.min_candidates and cand_count < args.min_candidates:
        if use_deal_size_range:
            use_deal_size_range = False
            mask = build_mask(use_sector, use_buckets, use_hard, use_deal_size_range)
            cand_count = int(mask.sum())
        if use_buckets:
            use_buckets = False
            mask = build_mask(use_sector, use_buckets, use_hard, use_deal_size_range)
            cand_count = int(mask.sum())
        if cand_count < args.min_candidates and use_sector:
            use_sector = False
            mask = build_mask(use_sector, use_buckets, use_hard, use_deal_size_range)
            cand_count = int(mask.sum())
        if cand_count < args.min_candidates and use_hard:
            use_hard = False
            effective_regime_mode = "soft" if regime_mode != "off" else "off"
            mask = build_mask(use_sector, use_buckets, use_hard, use_deal_size_range)
            cand_count = int(mask.sum())
    if cand_count < args.min_candidates:
        print(f"[similarity_search] candidates={cand_count} (<{args.min_candidates}) after fallbacks", flush=True)

    sector_penalty_active = args.same_sector and pd.notna(query.get("sic2")) and not use_sector
    df_cand = df[mask]

    results = []
    for _, row in df_cand.iterrows():
        d_base = weighted_distance(row, query, BASE_FEATURES, w_base)
        d_change = weighted_distance(row, query, DELTA_FEATURES, w_delta)
        d_macro = weighted_distance(row, query, MACRO_FEATURES, w_macro)

        # Fallback if distance is missing
        d_base = 9.9 if np.isnan(d_base) else d_base
        d_change = 9.9 if np.isnan(d_change) else d_change
        d_macro = 9.9 if np.isnan(d_macro) else d_macro

        # missingness penalty
        total_features = len(BASE_FEATURES) + len(DELTA_FEATURES) + len(MACRO_FEATURES)
        missing = sum(pd.isna(row[f]) or pd.isna(query[f]) for f in BASE_FEATURES + DELTA_FEATURES + MACRO_FEATURES)
        missing_frac = missing / total_features

        total = (
            args.profile_weight * d_base
            + args.change_weight * d_change
            + args.macro_weight * d_macro
            + args.missing_penalty * missing_frac
        )
        if effective_regime_mode == "soft":
            diffs = []
            for feat in MACRO_FEATURES:
                rb = row.get(f"_bin_{feat}")
                qb = q_bins.get(feat)
                if pd.notna(rb) and pd.notna(qb) and args.regime_bins > 1:
                    diffs.append(abs(rb - qb) / (args.regime_bins - 1))
            regime_dist = float(np.mean(diffs)) if diffs else 0.0
            total += args.regime_weight * regime_dist
        if sector_penalty_active:
            r_sic = row.get("sic2")
            q_sic = query.get("sic2")
            if pd.notna(r_sic) and pd.notna(q_sic) and r_sic != q_sic:
                total += args.sector_penalty
        if args.deal_size_penalty > 0:
            r_ds = row.get("z_deal_size_to_mcap")
            q_ds = query.get("z_deal_size_to_mcap")
            if pd.notna(r_ds) and pd.notna(q_ds):
                total += args.deal_size_penalty * abs(float(r_ds) - float(q_ds))
        if args.action_type == "acquisition":
            if args.mna_public_penalty > 0:
                r_pub = row.get("mna_target_public")
                q_pub = query.get("mna_target_public")
                if pd.notna(r_pub) and pd.notna(q_pub) and float(r_pub) != float(q_pub):
                    total += args.mna_public_penalty
            if args.mna_crossborder_penalty > 0:
                r_cb = row.get("mna_cross_border")
                q_cb = query.get("mna_cross_border")
                if pd.notna(r_cb) and pd.notna(q_cb) and float(r_cb) != float(q_cb):
                    total += args.mna_crossborder_penalty
            if args.mna_dealtype_penalty > 0:
                deal_cols = [
                    "mna_deal_type_stake",
                    "mna_deal_type_lbo",
                    "mna_deal_type_tender",
                    "mna_deal_type_merger",
                ]
                q_vec = pd.to_numeric(query[deal_cols], errors="coerce").to_numpy(dtype=float)
                if np.isfinite(q_vec).any():
                    r_vec = pd.to_numeric(row[deal_cols], errors="coerce").to_numpy(dtype=float)
                    if np.isfinite(r_vec).any():
                        match = (r_vec > 0.5) & (q_vec > 0.5)
                        if not match.any():
                            total += args.mna_dealtype_penalty
            if args.mna_payment_penalty > 0:
                pay_cols = [
                    "mna_payment_cash",
                    "mna_payment_stock",
                    "mna_payment_mixed",
                ]
                q_vec = pd.to_numeric(query[pay_cols], errors="coerce").to_numpy(dtype=float)
                if np.isfinite(q_vec).any():
                    r_vec = pd.to_numeric(row[pay_cols], errors="coerce").to_numpy(dtype=float)
                    if np.isfinite(r_vec).any():
                        diff = np.nansum(np.abs(r_vec - q_vec))
                        total += args.mna_payment_penalty * diff
            if args.mna_completion_penalty > 0:
                r_comp = row.get("mna_deal_completed")
                q_comp = query.get("mna_deal_completed")
                if pd.notna(r_comp) and pd.notna(q_comp):
                    total += args.mna_completion_penalty * abs(float(r_comp) - float(q_comp))

        delta_contribs = feature_contribs(row, query, DELTA_FEATURES, w_delta)
        base_contribs = feature_contribs(row, query, BASE_FEATURES, w_base)
        macro_contribs = feature_contribs(row, query, MACRO_FEATURES, w_macro)
        top_delta = ", ".join([f"{f}:{v:.3f}" for f, v in delta_contribs[:3]]) if delta_contribs else ""
        top_base = ", ".join([f"{f}:{v:.3f}" for f, v in base_contribs[:3]]) if base_contribs else ""
        top_macro = ", ".join([f"{f}:{v:.3f}" for f, v in macro_contribs[:3]]) if macro_contribs else ""

        results.append(
            {
                "company_id": row.get("company_id"),
                "action_type": row.get("action_type"),
                "action_date": row.get("action_date"),
                "action_subtype": row.get("action_subtype"),
                "distance": total,
                "d_profile": d_base,
                "d_change": d_change,
                "d_macro": d_macro,
                "missing_frac": round(missing_frac, 3),
                "top_delta_contrib": top_delta,
                "top_profile_contrib": top_base,
                "top_macro_contrib": top_macro,
                "outcome_pe_3m_w": row.get("outcome_pe_3m_w"),
                "outcome_pe_6m_w": row.get("outcome_pe_6m_w"),
                "outcome_pe_12m_w": row.get("outcome_pe_12m_w"),
                "outcome_ev_ebitda_3m_w": row.get("outcome_ev_ebitda_3m_w"),
                "outcome_ev_ebitda_6m_w": row.get("outcome_ev_ebitda_6m_w"),
                "outcome_ev_ebitda_12m_w": row.get("outcome_ev_ebitda_12m_w"),
                "outcome_pe_12m": row.get("outcome_pe_12m"),
                "outcome_ev_ebitda_12m": row.get("outcome_ev_ebitda_12m"),
                "source_dataset": row.get("source_dataset"),
                "source_id": row.get("source_id"),
            }
        )

    out = pd.DataFrame(results).sort_values("distance").head(args.top_k)
    # include regime labels for context
    if not out.empty:
        out["_risk"], out["_credit"], out["_rate"] = zip(*out.apply(lambda r: classify_regime(r, args.regime_threshold), axis=1))

        if args.predict_outcomes:
            dist = out["distance"].to_numpy(dtype=float)
            if args.weighting == "uniform":
                w = np.ones_like(dist, dtype=float)
            elif args.weighting == "inverse":
                w = 1.0 / (dist + 1e-6)
            else:
                sigma = args.kernel_sigma
                if sigma is None:
                    sigma = float(np.median(dist[~np.isnan(dist)])) if np.any(np.isfinite(dist)) else 1.0
                sigma = max(sigma, 1e-6)
                w = np.exp(-dist / sigma)
                if not np.any(np.isfinite(w)) or np.nansum(w) <= 0:
                    w = 1.0 / (dist + 1e-6)

            outcome_cols = [
                c for c in out.columns
                if c.startswith("outcome_") and c.endswith("_w") and c in df.columns
            ]
            preds = {}
            action_means = df[outcome_cols].mean(skipna=True).to_dict() if args.shrinkage > 0 else {}
            for col in outcome_cols:
                vals = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float)
                mask = np.isfinite(vals)
                if mask.sum() == 0:
                    continue
                w_sub = w[mask]
                if not np.any(np.isfinite(w_sub)) or np.nansum(w_sub) <= 0:
                    w_sub = np.ones_like(vals[mask], dtype=float)
                pred = float(np.average(vals[mask], weights=w_sub))
                if args.shrinkage > 0 and col in action_means and np.isfinite(action_means[col]):
                    sum_w = float(np.sum(w_sub))
                    pred = float((np.sum(w_sub * vals[mask]) + args.shrinkage * action_means[col]) / (sum_w + args.shrinkage))
                preds[col] = pred

            if preds:
                print("\nPredicted outcomes (dist-weighted, top-k):")
                for k, v in preds.items():
                    print(f"{k}={v:.4f}")
    print(out.to_string(index=False))
    if args.out:
        out.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
