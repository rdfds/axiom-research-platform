#!/usr/bin/env python
"""
Validate similarity matching by comparing KNN outcome predictions vs baselines.

This is a lightweight backtest on a sample:
  - For each sampled row, find top-K neighbors by distance
  - Predict outcome as mean of neighbors' outcomes
  - Compare MAE vs baseline (global mean) and random-K baseline
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CURATED_DIR = ROOT / "data" / "curated"

FEATURES_PATH = CURATED_DIR / "similarity_features.parquet"
WEIGHTS_PATH = CURATED_DIR / "similarity_weights.parquet"

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
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    return pd.cut(s, bins=edges, labels=False, include_lowest=True)


def load_weights() -> pd.DataFrame | None:
    if not WEIGHTS_PATH.exists():
        return None
    return pd.read_parquet(WEIGHTS_PATH)


def weight_lookup(weights: Optional[pd.DataFrame], action_type: str, features: List[str]) -> np.ndarray:
    if weights is None or weights.empty:
        return np.ones(len(features), dtype=float)
    subset = weights[(weights["action_type"] == action_type) & (weights["feature"].isin(features))]
    if subset.empty:
        subset = weights[(weights["action_type"] == "ALL") & (weights["feature"].isin(features))]
    if subset.empty:
        return np.ones(len(features), dtype=float)
    mapping = dict(zip(subset["feature"], subset["weight"]))
    return np.array([mapping.get(f, 1.0) for f in features], dtype=float)


def weighted_distance(row: pd.Series, query: pd.Series, features: List[str], weights: np.ndarray) -> float:
    diffs = row[features] - query[features]
    mask = diffs.notna() & query[features].notna()
    if mask.sum() == 0:
        return np.nan
    w = weights[mask.to_numpy()]
    d = np.sqrt(np.sum(w * (diffs[mask].to_numpy() ** 2)) / np.sum(w))
    return float(d)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action-type", default=None, help="Optional action_type filter")
    parser.add_argument("--target", default="outcome_pe_12m_w")
    parser.add_argument("--auto-target", action="store_true", default=False)
    parser.add_argument("--k", type=int, default=25)
    parser.add_argument("--sample", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--same-sector", action="store_true", default=True)
    parser.add_argument("--no-same-sector", dest="same_sector", action="store_false")
    parser.add_argument("--regime-filter", action="store_true", default=True)
    parser.add_argument("--no-regime-filter", dest="regime_filter", action="store_false")
    parser.add_argument("--regime-threshold", type=float, default=0.5)
    parser.add_argument("--regime-mode", choices=["hard", "soft", "off"], default=None)
    parser.add_argument("--regime-bins", type=int, default=4)
    parser.add_argument("--regime-weight", type=float, default=0.2)
    parser.add_argument("--profile-buckets", action="store_true", default=True)
    parser.add_argument("--no-profile-buckets", dest="profile_buckets", action="store_false")
    parser.add_argument("--sector-penalty", type=float, default=0.0, help="Penalty added when sector differs (if sector filter is relaxed).")
    parser.add_argument("--deal-size-penalty", type=float, default=0.0, help="Penalty weight for deal_size_to_mcap difference.")
    parser.add_argument("--deal-size-factor", type=float, default=None, help="Require deal_size_to_mcap within factor range of query (e.g., 3.0 = within 3x).")
    parser.add_argument("--mna-public-penalty", type=float, default=0.0, help="Penalty for M&A target public/private mismatch (acquisitions only).")
    parser.add_argument("--mna-crossborder-penalty", type=float, default=0.0, help="Penalty for M&A cross-border mismatch (acquisitions only).")
    parser.add_argument("--mna-dealtype-penalty", type=float, default=0.0, help="Penalty for M&A deal type mismatch (acquisitions only).")
    parser.add_argument("--mna-payment-penalty", type=float, default=0.0, help="Penalty for M&A payment type mismatch (cash/stock/mixed).")
    parser.add_argument("--mna-completion-penalty", type=float, default=0.0, help="Penalty for M&A completion status mismatch.")
    parser.add_argument("--min-candidates", type=int, default=200, help="Minimum candidate pool before relaxing filters.")
    parser.add_argument("--max-year-gap", type=int, default=None, help="Max |action_year - query_year| for candidates.")
    parser.add_argument("--distance-weighted", action="store_true", default=True)
    parser.add_argument("--no-distance-weighted", dest="distance_weighted", action="store_false")
    parser.add_argument("--kernel-weighted", action="store_true", default=False)
    parser.add_argument("--kernel-sigma", type=float, default=None)
    parser.add_argument("--shrinkage", type=float, default=0.0, help="Shrink predictions toward action-type mean (0 to disable).")
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()

    if args.regime_mode is None:
        regime_mode = "soft" if args.regime_filter else "off"
    else:
        regime_mode = args.regime_mode

    df = pd.read_parquet(FEATURES_PATH)
    if args.action_type:
        df = df[df["action_type"] == args.action_type].copy()
    if df.empty:
        raise RuntimeError("No rows available for validation.")

    # determine targets
    if args.auto_target:
        outcome_cols = [c for c in df.columns if c.startswith("outcome_") and c.endswith("_w")]
        if not outcome_cols:
            raise RuntimeError("No winsorized outcome targets found.")
    else:
        outcome_cols = [args.target]

    weights_full = load_weights()
    results = []
    best = None
    best_beats = None

    for target in outcome_cols:
        dft = df[df[target].notna()].copy().reset_index(drop=True)
        if dft.empty:
            continue
        print(f"Evaluating target: {target}", flush=True)

        weights = weights_full
        if weights is not None and "target" in weights.columns:
            wsub = weights[weights["target"] == target]
            if not wsub.empty:
                weights = wsub

        w_delta = weight_lookup(weights, args.action_type or "ALL", DELTA_FEATURES)
        w_base = np.ones(len(BASE_FEATURES), dtype=float)
        w_macro = np.ones(len(MACRO_FEATURES), dtype=float)

        # Precompute regime labels once
        if regime_mode == "hard":
            dft["_risk"], dft["_credit"], dft["_rate"] = zip(
                *dft.apply(lambda r: classify_regime(r, args.regime_threshold), axis=1)
            )
        if regime_mode in ("soft", "hard"):
            for feat in MACRO_FEATURES:
                dft[f"_bin_{feat}"] = quantile_bins(dft[feat], args.regime_bins)
            bin_cols = [f"_bin_{f}" for f in MACRO_FEATURES]
            B_macro = dft[bin_cols].to_numpy(dtype=float)

        # Precompute feature matrices
        X_base = dft[BASE_FEATURES].to_numpy(dtype=float)
        X_delta = dft[DELTA_FEATURES].to_numpy(dtype=float)
        X_macro = dft[MACRO_FEATURES].to_numpy(dtype=float)

        rng = np.random.default_rng(args.seed)
        sample_df = dft.sample(min(args.sample, len(dft)), random_state=args.seed)

        preds = []
        actuals = []
        baseline_mean = dft[target].mean()
        random_errors = []

        for i, (_, query) in enumerate(sample_df.iterrows(), start=1):
            use_sector = args.same_sector and pd.notna(query.get("sic2"))
            use_buckets = args.profile_buckets
            use_deal_size = args.deal_size_factor is not None
            use_hard = regime_mode == "hard"
            effective_regime_mode = regime_mode

            def build_mask(use_sector_flag: bool, use_buckets_flag: bool, use_hard_flag: bool, use_deal_size_flag: bool) -> np.ndarray:
                mask = np.ones(len(dft), dtype=bool)
                if args.max_year_gap is not None and "action_year" in dft.columns and pd.notna(query.get("action_year")):
                    year_diff = (dft["action_year"] - query.get("action_year")).abs()
                    mask &= (year_diff <= args.max_year_gap).fillna(False).to_numpy(dtype=bool)
                if use_deal_size_flag and "deal_size_to_mcap" in dft.columns:
                    q_ds = query.get("deal_size_to_mcap")
                    if pd.notna(q_ds):
                        q_abs = abs(float(q_ds))
                        if q_abs > 0 and args.deal_size_factor:
                            cand = dft["deal_size_to_mcap"].astype(float).abs()
                            lower = q_abs / float(args.deal_size_factor)
                            upper = q_abs * float(args.deal_size_factor)
                            mask &= (cand >= lower) & (cand <= upper)
                if use_sector_flag and pd.notna(query.get("sic2")):
                    mask &= (dft["sic2"] == query.get("sic2")).to_numpy()
                if use_buckets_flag:
                    bucket_cols = ["size_bucket", "leverage_bucket", "margin_bucket"]
                    if args.action_type in {"acquisition", "divestiture"}:
                        if "deal_size_bucket_strict" in dft.columns:
                            bucket_cols.append("deal_size_bucket_strict")
                        elif "deal_size_bucket" in dft.columns:
                            bucket_cols.append("deal_size_bucket")
                    else:
                        if "deal_size_bucket" in dft.columns:
                            bucket_cols.append("deal_size_bucket")
                    for bucket in bucket_cols:
                        if bucket in dft.columns:
                            qval = query.get(bucket)
                            if pd.isna(qval):
                                continue
                            comp = pd.to_numeric(dft[bucket], errors="coerce")
                            bucket_mask = (comp == float(qval))
                            bucket_mask = bucket_mask.fillna(False).to_numpy(dtype=bool)
                            mask &= bucket_mask
                if use_hard_flag:
                    q_regime = classify_regime(query, args.regime_threshold)
                    mask &= (dft["_risk"] == q_regime[0]).to_numpy()
                    mask &= (dft["_credit"] == q_regime[1]).to_numpy()
                    mask &= (dft["_rate"] == q_regime[2]).to_numpy()
                return mask

            mask = build_mask(use_sector, use_buckets, use_hard, use_deal_size)
            cand_count = int(mask.sum())
            if args.min_candidates and cand_count < args.min_candidates:
                if use_deal_size:
                    use_deal_size = False
                    mask = build_mask(use_sector, use_buckets, use_hard, use_deal_size)
                    cand_count = int(mask.sum())
                if cand_count < args.min_candidates and use_buckets:
                    use_buckets = False
                    mask = build_mask(use_sector, use_buckets, use_hard, use_deal_size)
                    cand_count = int(mask.sum())
                if cand_count < args.min_candidates and use_sector:
                    use_sector = False
                    mask = build_mask(use_sector, use_buckets, use_hard, use_deal_size)
                    cand_count = int(mask.sum())
                if cand_count < args.min_candidates and use_hard:
                    use_hard = False
                    effective_regime_mode = "soft" if regime_mode != "off" else "off"
                    mask = build_mask(use_sector, use_buckets, use_hard, use_deal_size)
                    cand_count = int(mask.sum())

            mask[query.name] = False
            cand_idx = np.where(mask)[0]
            if cand_idx.size == 0:
                continue

            q_base = np.array(query[BASE_FEATURES], dtype=float)
            q_delta = np.array(query[DELTA_FEATURES], dtype=float)
            q_macro = np.array(query[MACRO_FEATURES], dtype=float)

            # base distance
            diff = X_base[cand_idx] - q_base
            mask_base = ~np.isnan(diff)
            diff[~mask_base] = 0.0
            num = (diff ** 2) @ w_base
            den = (mask_base * w_base).sum(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                d_base = np.sqrt(np.where(den > 0, num / den, np.nan))

            # delta distance
            diff = X_delta[cand_idx] - q_delta
            mask_delta = ~np.isnan(diff)
            diff[~mask_delta] = 0.0
            num = (diff ** 2) @ w_delta
            den = (mask_delta * w_delta).sum(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                d_change = np.sqrt(np.where(den > 0, num / den, np.nan))

            # macro distance
            diff = X_macro[cand_idx] - q_macro
            mask_macro = ~np.isnan(diff)
            diff[~mask_macro] = 0.0
            num = (diff ** 2) @ w_macro
            den = (mask_macro * w_macro).sum(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                d_macro = np.sqrt(np.where(den > 0, num / den, np.nan))

            total_features = len(BASE_FEATURES) + len(DELTA_FEATURES) + len(MACRO_FEATURES)
            missing = total_features - (mask_base.sum(axis=1) + mask_delta.sum(axis=1) + mask_macro.sum(axis=1))
            missing_frac = missing / total_features

            d_base = np.where(np.isnan(d_base), 9.9, d_base)
            d_change = np.where(np.isnan(d_change), 9.9, d_change)
            d_macro = np.where(np.isnan(d_macro), 9.9, d_macro)

            total = 0.3 * d_base + 0.6 * d_change + 0.1 * d_macro + 0.1 * missing_frac
            if effective_regime_mode == "soft":
                q_bins = B_macro[query.name]
                diff = np.abs(B_macro[cand_idx] - q_bins)
                mask_bins = ~np.isnan(diff)
                denom = mask_bins.sum(axis=1)
                if args.regime_bins > 1:
                    scale = (args.regime_bins - 1)
                else:
                    scale = 1.0
                regime_dist = np.where(denom > 0, diff.sum(axis=1) / denom / scale, 0.0)
                total += args.regime_weight * regime_dist
            sector_penalty_active = args.same_sector and pd.notna(query.get("sic2")) and not use_sector
            if sector_penalty_active and args.sector_penalty > 0 and "sic2" in dft.columns:
                q_sic = query.get("sic2")
                if pd.notna(q_sic):
                    cand_sic = dft.loc[cand_idx, "sic2"].to_numpy()
                    mismatch = (cand_sic != q_sic) & pd.notna(cand_sic)
                    total += args.sector_penalty * mismatch.astype(float)

            if args.deal_size_penalty > 0 and "z_deal_size_to_mcap" in dft.columns:
                q_ds = query.get("z_deal_size_to_mcap")
                if pd.notna(q_ds):
                    cand_ds = dft.loc[cand_idx, "z_deal_size_to_mcap"].to_numpy(dtype=float)
                    ds_diff = np.abs(cand_ds - float(q_ds))
                    ds_diff = np.where(np.isfinite(ds_diff), ds_diff, 0.0)
                    total += args.deal_size_penalty * ds_diff
            if args.action_type == "acquisition":
                if args.mna_public_penalty > 0 and "mna_target_public" in dft.columns:
                    q_pub = query.get("mna_target_public")
                    if pd.notna(q_pub):
                        cand = pd.to_numeric(dft.loc[cand_idx, "mna_target_public"], errors="coerce").to_numpy(dtype=float)
                        mismatch = np.isfinite(cand) & (cand != float(q_pub))
                        total += args.mna_public_penalty * mismatch.astype(float)
                if args.mna_crossborder_penalty > 0 and "mna_cross_border" in dft.columns:
                    q_cb = query.get("mna_cross_border")
                    if pd.notna(q_cb):
                        cand = pd.to_numeric(dft.loc[cand_idx, "mna_cross_border"], errors="coerce").to_numpy(dtype=float)
                        mismatch = np.isfinite(cand) & (cand != float(q_cb))
                        total += args.mna_crossborder_penalty * mismatch.astype(float)
                if args.mna_dealtype_penalty > 0 and "mna_deal_type_stake" in dft.columns:
                    deal_cols = [
                        "mna_deal_type_stake",
                        "mna_deal_type_lbo",
                        "mna_deal_type_tender",
                        "mna_deal_type_merger",
                    ]
                    q_vec = pd.to_numeric(query[deal_cols], errors="coerce").to_numpy(dtype=float)
                    if np.isfinite(q_vec).any():
                        cand = dft.loc[cand_idx, deal_cols].to_numpy(dtype=float)
                        cand_active = cand > 0.5
                        q_active = q_vec > 0.5
                        cand_has = np.isfinite(cand).any(axis=1)
                        match = (cand_active & q_active).any(axis=1)
                        mismatch = cand_has & (~match)
                        total += args.mna_dealtype_penalty * mismatch.astype(float)
                if args.mna_payment_penalty > 0 and "mna_payment_cash" in dft.columns:
                    pay_cols = [
                        "mna_payment_cash",
                        "mna_payment_stock",
                        "mna_payment_mixed",
                    ]
                    q_vec = pd.to_numeric(query[pay_cols], errors="coerce").to_numpy(dtype=float)
                    if np.isfinite(q_vec).any():
                        cand = dft.loc[cand_idx, pay_cols].to_numpy(dtype=float)
                        diff = np.nansum(np.abs(cand - q_vec), axis=1)
                        diff = np.where(np.isfinite(diff), diff, 0.0)
                        total += args.mna_payment_penalty * diff
                if args.mna_completion_penalty > 0 and "mna_deal_completed" in dft.columns:
                    q_comp = query.get("mna_deal_completed")
                    if pd.notna(q_comp):
                        cand = pd.to_numeric(dft.loc[cand_idx, "mna_deal_completed"], errors="coerce").to_numpy(dtype=float)
                        diff = np.abs(cand - float(q_comp))
                        diff = np.where(np.isfinite(diff), diff, 0.0)
                        total += args.mna_completion_penalty * diff

            order = np.argsort(total)[: args.k]
            topk_idx = cand_idx[order]
            topk = dft.iloc[topk_idx]
            if topk.empty:
                continue

            d = total[order]
            vals = pd.to_numeric(topk[target], errors="coerce").to_numpy(dtype=float)
            maskv = np.isfinite(vals)
            if maskv.sum() == 0:
                continue
            d = d[maskv]
            vals = vals[maskv]
            if args.kernel_weighted:
                sigma = args.kernel_sigma
                if sigma is None:
                    sigma = float(np.median(d[~np.isnan(d)])) if np.any(np.isfinite(d)) else 1.0
                sigma = max(sigma, 1e-6)
                w = np.exp(-d / sigma)
            elif args.distance_weighted:
                w = 1.0 / (d + 1e-6)
            else:
                w = np.ones_like(vals, dtype=float)

            pred = float(np.average(vals, weights=w))
            if args.shrinkage > 0:
                target_mean = float(dft[target].mean())
                sum_w = float(np.sum(w))
                pred = float((np.sum(w * vals) + args.shrinkage * target_mean) / (sum_w + args.shrinkage))
            if not np.isfinite(pred):
                continue
            preds.append(float(pred))
            actuals.append(float(query[target]))

            rand_idx = rng.choice(cand_idx, size=min(args.k, len(cand_idx)), replace=False)
            rand = dft.iloc[rand_idx]
            rand_vals = pd.to_numeric(rand[target], errors="coerce").to_numpy(dtype=float)
            rand_mask = np.isfinite(rand_vals)
            if rand_mask.sum() > 0:
                random_errors.append(abs(np.mean(rand_vals[rand_mask]) - float(query[target])))

            if args.log_every and i % args.log_every == 0:
                print(f"[validate] processed {i}/{len(sample_df)}", flush=True)

        preds = np.array(preds)
        actuals = np.array(actuals)
        if len(preds) == 0:
            print("No valid validation rows after filtering.")
            continue

        mae = np.mean(np.abs(preds - actuals))
        mae_baseline = np.mean(np.abs(baseline_mean - actuals))
        mae_random = np.mean(random_errors) if random_errors else np.nan
        if len(preds) < 2 or np.nanstd(preds) < 1e-9 or np.nanstd(actuals) < 1e-9:
            corr = 0.0
        else:
            corr = np.corrcoef(preds, actuals)[0, 1]
        hit = np.mean(np.sign(preds) == np.sign(actuals))

        print(f"n={len(preds)} | MAE={mae:.4f} | MAE_baseline={mae_baseline:.4f} | MAE_random={mae_random:.4f}")
        print(f"corr(pred, actual)={corr:.3f} | sign_hit_rate={hit:.3f}")

        results.append(
            {
                "target": target,
                "n": len(preds),
                "mae": mae,
                "mae_baseline": mae_baseline,
                "mae_random": mae_random,
                "corr": corr,
                "sign_hit": hit,
            }
        )
        if best is None or corr > best["corr"]:
            best = results[-1]
        if mae <= mae_baseline:
            if best_beats is None or corr > best_beats["corr"]:
                best_beats = results[-1]

    if args.auto_target and results:
        if best_beats is not None:
            print("\nBest target (corr) that beats baseline MAE:", best_beats)
        else:
            print("\nNo target beat baseline MAE; best corr fallback:", best)


if __name__ == "__main__":
    main()
