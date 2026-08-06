#!/usr/bin/env python
"""
Select best outcome target per action_type based on KNN validation.

Writes: data/curated/similarity_best_targets.parquet
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
OUT_PATH = CURATED_DIR / "similarity_best_targets.parquet"

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action-type", default=None, help="Optional action_type filter")
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
    parser.add_argument("--profile-buckets", action="store_true", default=False)
    parser.add_argument("--no-profile-buckets", dest="profile_buckets", action="store_false")
    parser.add_argument("--kernel-weighted", action="store_true", default=True)
    parser.add_argument("--kernel-sigma", type=float, default=None)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--min-candidates", type=int, default=200, help="Minimum candidate pool before relaxing filters.")
    parser.add_argument("--sector-penalty", type=float, default=0.2, help="Penalty added when sector differs (if sector filter is relaxed).")
    parser.add_argument("--shrinkage", type=float, default=5.0, help="Shrink predictions toward action-type mean (0 to disable).")
    parser.add_argument("--debug-action", default=None, help="Print debug stats for a single action_type and exit.")
    parser.add_argument("--out", default=str(OUT_PATH))
    parser.add_argument("--full-out", default=None, help="Optional path to save full target scores.")
    parser.add_argument("--min-n", type=int, default=200, help="Minimum valid observations per target.")
    args = parser.parse_args()

    if args.regime_mode is None:
        regime_mode = "soft" if args.regime_filter else "off"
    else:
        regime_mode = args.regime_mode

    df = pd.read_parquet(FEATURES_PATH)
    if args.action_type:
        df = df[df["action_type"] == args.action_type].copy()
    if df.empty:
        raise RuntimeError("No rows available for selection.")

    outcome_cols = [c for c in df.columns if c.startswith("outcome_") and c.endswith("_w")]
    if not outcome_cols:
        raise RuntimeError("No winsorized outcome targets found.")

    results: List[Dict[str, object]] = []
    best_rows: List[Dict[str, object]] = []

    for action_type, dfa in df.groupby("action_type"):
        dfa = dfa.copy().reset_index(drop=True)
        if dfa.empty:
            continue

        # Precompute regime labels or bins
        if regime_mode == "hard":
            dfa["_risk"], dfa["_credit"], dfa["_rate"] = zip(
                *dfa.apply(lambda r: classify_regime(r, args.regime_threshold), axis=1)
            )
        if regime_mode in ("soft", "hard"):
            for feat in MACRO_FEATURES:
                dfa[f"_bin_{feat}"] = quantile_bins(dfa[feat], args.regime_bins)
            bin_cols = [f"_bin_{f}" for f in MACRO_FEATURES]
            B_macro = dfa[bin_cols].to_numpy(dtype=float)

        X_base = dfa[BASE_FEATURES].to_numpy(dtype=float)
        X_delta = dfa[DELTA_FEATURES].to_numpy(dtype=float)
        X_macro = dfa[MACRO_FEATURES].to_numpy(dtype=float)
        sic2_arr = dfa["sic2"].to_numpy() if "sic2" in dfa.columns else None

        rng = np.random.default_rng(args.seed)
        sample_df = dfa.sample(min(args.sample, len(dfa)), random_state=args.seed)

        # Storage per target
        preds = {t: [] for t in outcome_cols}
        actuals = {t: [] for t in outcome_cols}
        random_errors = {t: [] for t in outcome_cols}
        mean_outcomes = dfa[outcome_cols].mean(skipna=True).to_dict() if args.shrinkage > 0 else {}

        for i, (_, query) in enumerate(sample_df.iterrows(), start=1):
            use_sector = args.same_sector and pd.notna(query.get("sic2"))
            use_buckets = args.profile_buckets
            use_hard = regime_mode == "hard"
            effective_regime_mode = regime_mode
            q_regime = classify_regime(query, args.regime_threshold) if use_hard else None

            def build_mask(use_sector_flag: bool, use_buckets_flag: bool, use_hard_flag: bool) -> np.ndarray:
                mask = np.ones(len(dfa), dtype=bool)
                if use_sector_flag and "sic2" in dfa.columns:
                    sec_mask = (dfa["sic2"] == query.get("sic2")).fillna(False).to_numpy(dtype=bool)
                    mask &= sec_mask
                if use_buckets_flag:
                    bucket_cols = ["size_bucket", "leverage_bucket", "margin_bucket"]
                    if action_type in {"acquisition", "divestiture"}:
                        if "deal_size_bucket_strict" in dfa.columns:
                            bucket_cols.append("deal_size_bucket_strict")
                        elif "deal_size_bucket" in dfa.columns:
                            bucket_cols.append("deal_size_bucket")
                    else:
                        if "deal_size_bucket" in dfa.columns:
                            bucket_cols.append("deal_size_bucket")
                    for bucket in bucket_cols:
                        if bucket in dfa.columns:
                            qval = query.get(bucket)
                            if pd.isna(qval):
                                continue
                            comp = pd.to_numeric(dfa[bucket], errors="coerce")
                            bucket_mask = (comp == float(qval)).fillna(False).to_numpy(dtype=bool)
                            mask &= bucket_mask
                if use_hard_flag and q_regime is not None:
                    mask &= (dfa["_risk"] == q_regime[0]).to_numpy(dtype=bool)
                    mask &= (dfa["_credit"] == q_regime[1]).to_numpy(dtype=bool)
                    mask &= (dfa["_rate"] == q_regime[2]).to_numpy(dtype=bool)
                mask[query.name] = False
                return mask

            mask = build_mask(use_sector, use_buckets, use_hard)
            cand_count = int(mask.sum())
            if args.min_candidates and cand_count < args.min_candidates:
                if use_buckets:
                    use_buckets = False
                    mask = build_mask(use_sector, use_buckets, use_hard)
                    cand_count = int(mask.sum())
                if cand_count < args.min_candidates and use_sector:
                    use_sector = False
                    mask = build_mask(use_sector, use_buckets, use_hard)
                    cand_count = int(mask.sum())
                if cand_count < args.min_candidates and use_hard:
                    use_hard = False
                    effective_regime_mode = "soft" if regime_mode != "off" else "off"
                    mask = build_mask(use_sector, use_buckets, use_hard)
                    cand_count = int(mask.sum())

            cand_idx = np.where(mask)[0]
            if cand_idx.size == 0:
                continue
            sector_penalty_active = args.same_sector and pd.notna(query.get("sic2")) and not use_sector

            q_base = np.array(query[BASE_FEATURES], dtype=float)
            q_delta = np.array(query[DELTA_FEATURES], dtype=float)
            q_macro = np.array(query[MACRO_FEATURES], dtype=float)

            # base distance
            diff = X_base[cand_idx] - q_base
            mask_base = ~np.isnan(diff)
            diff[~mask_base] = 0.0
            num = (diff ** 2).sum(axis=1)
            den = mask_base.sum(axis=1)
            d_base = np.sqrt(np.divide(num, den, out=np.full_like(num, np.nan, dtype=float), where=den > 0))

            # delta distance
            diff = X_delta[cand_idx] - q_delta
            mask_delta = ~np.isnan(diff)
            diff[~mask_delta] = 0.0
            num = (diff ** 2).sum(axis=1)
            den = mask_delta.sum(axis=1)
            d_change = np.sqrt(np.divide(num, den, out=np.full_like(num, np.nan, dtype=float), where=den > 0))

            # macro distance
            diff = X_macro[cand_idx] - q_macro
            mask_macro = ~np.isnan(diff)
            diff[~mask_macro] = 0.0
            num = (diff ** 2).sum(axis=1)
            den = mask_macro.sum(axis=1)
            d_macro = np.sqrt(np.divide(num, den, out=np.full_like(num, np.nan, dtype=float), where=den > 0))

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
                scale = (args.regime_bins - 1) if args.regime_bins > 1 else 1.0
                regime_dist = np.where(denom > 0, diff.sum(axis=1) / denom / scale, 0.0)
                total += args.regime_weight * regime_dist
            if sector_penalty_active and sic2_arr is not None:
                q_sic = query.get("sic2")
                if pd.notna(q_sic):
                    cand_sic = sic2_arr[cand_idx]
                    mismatch = (cand_sic != q_sic) & pd.notna(cand_sic)
                    total += args.sector_penalty * mismatch.astype(float)

            order = np.argsort(total)[: args.k]
            topk_idx = cand_idx[order]
            topk = dfa.iloc[topk_idx]
            if topk.empty:
                continue

            d = total[order]
            if args.kernel_weighted:
                sigma = args.kernel_sigma
                if sigma is None:
                    sigma = float(np.median(d[~np.isnan(d)])) if np.any(np.isfinite(d)) else 1.0
                sigma = max(sigma, 1e-6)
                w = np.exp(-d / sigma)
                if not np.any(np.isfinite(w)) or np.nansum(w) <= 0:
                    w = 1.0 / (d + 1e-6)
            else:
                w = 1.0 / (d + 1e-6)

            for t in outcome_cols:
                if pd.isna(query.get(t)):
                    continue
                vals = pd.to_numeric(topk[t], errors="coerce").to_numpy(dtype=float)
                maskv = np.isfinite(vals)
                if maskv.sum() == 0:
                    continue
                w_sub = w[maskv]
                if not np.any(np.isfinite(w_sub)) or np.nansum(w_sub) <= 0:
                    w_sub = np.ones_like(vals[maskv], dtype=float)
                pred = float(np.average(vals[maskv], weights=w_sub))
                if args.shrinkage > 0 and t in mean_outcomes and np.isfinite(mean_outcomes[t]):
                    sum_w = float(np.sum(w_sub))
                    pred = float((np.sum(w_sub * vals[maskv]) + args.shrinkage * mean_outcomes[t]) / (sum_w + args.shrinkage))
                preds[t].append(pred)
                actuals[t].append(float(query[t]))

                rand_idx = rng.choice(cand_idx, size=min(args.k, len(cand_idx)), replace=False)
                rand = dfa.iloc[rand_idx]
                rand_vals = pd.to_numeric(rand[t], errors="coerce").to_numpy(dtype=float)
                rand_mask = np.isfinite(rand_vals)
                if rand_mask.sum() > 0:
                    random_errors[t].append(
                        abs(np.mean(rand_vals[rand_mask]) - float(query[t]))
                    )

            if args.log_every and i % args.log_every == 0:
                print(f"[{action_type}] processed {i}/{len(sample_df)}", flush=True)

        if args.debug_action and action_type == args.debug_action:
            print(f"[debug] action_type={action_type}", flush=True)
            print({t: len(preds[t]) for t in outcome_cols}, flush=True)
            print({t: int(sample_df[t].notna().sum()) for t in outcome_cols}, flush=True)
            return

        # Score targets
        best = None
        best_beats = None
        for t in outcome_cols:
            if len(preds[t]) == 0:
                continue
            p = np.array(preds[t])
            a = np.array(actuals[t])
            if len(p) < args.min_n:
                continue
            mae = np.mean(np.abs(p - a))
            baseline = np.mean(np.abs(np.mean(a) - a))
            mae_rand = np.mean(random_errors[t]) if random_errors[t] else np.nan
            if len(p) < 2 or np.nanstd(p) < 1e-9 or np.nanstd(a) < 1e-9:
                corr = 0.0
            else:
                corr = np.corrcoef(p, a)[0, 1]
            hit = np.mean(np.sign(p) == np.sign(a))
            if not np.isfinite(mae) or not np.isfinite(corr):
                continue
            rec = {
                "action_type": action_type,
                "target": t,
                "n": len(p),
                "mae": mae,
                "mae_baseline": baseline,
                "mae_random": mae_rand,
                "corr": corr,
                "sign_hit": hit,
            }
            results.append(rec)
            if best is None or corr > best["corr"]:
                best = rec
            if mae <= baseline:
                if best_beats is None or corr > best_beats["corr"]:
                    best_beats = rec

        chosen = None
        if best_beats is not None:
            chosen = best_beats
            print(f"[{action_type}] best target (beats baseline): {best_beats}", flush=True)
        elif best is not None:
            chosen = best
            print(f"[{action_type}] best target (corr fallback): {best}", flush=True)
        else:
            # fallback if no valid target
            fallback = {
                "action_type": action_type,
                "target": "outcome_ev_ebitda_12m_w",
                "n": 0,
                "mae": np.nan,
                "mae_baseline": np.nan,
                "mae_random": np.nan,
                "corr": np.nan,
                "sign_hit": np.nan,
            }
            chosen = fallback
            print(f"[{action_type}] no valid targets; using fallback {fallback['target']}", flush=True)

        best_rows.append(
            {k: chosen[k] for k in ["action_type", "target"]}
        )

    out = pd.DataFrame(results)
    if args.full_out:
        out.to_parquet(Path(args.full_out), index=False)
        print(f"Saved full target scores -> {args.full_out} ({len(out):,} rows)", flush=True)

    best_df = pd.DataFrame(best_rows).drop_duplicates(subset=["action_type"], keep="last")
    best_df.to_parquet(Path(args.out), index=False)
    print(f"Saved best targets -> {args.out} ({len(best_df):,} rows)", flush=True)


if __name__ == "__main__":
    main()
