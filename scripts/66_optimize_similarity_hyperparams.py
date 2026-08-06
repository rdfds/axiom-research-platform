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
    vix = row.get("z_macro_vix")
    ig = row.get("z_macro_ig_oas")
    hy = row.get("z_macro_hy_oas")
    r10 = row.get("z_macro_rate_10y")

    risk = "risk_off" if pd.notna(vix) and vix >= thresh else "risk_on"
    credit = "credit_tight" if pd.notna(ig) and pd.notna(hy) and max(ig, hy) >= thresh else "credit_loose"
    rate = "rate_high" if pd.notna(r10) and r10 >= thresh else "rate_low"
    return risk, credit, rate


def score_config(
    dfa: pd.DataFrame,
    target: str,
    sample: int,
    seed: int,
    k: int,
    regime_mode: str,
    regime_bins: int,
    regime_weight: float,
    use_sector: bool,
    sector_penalty: float,
    min_candidates: int,
    weighting: str,
    shrinkage: float,
    min_n: int,
) -> Dict[str, float] | None:
    if target not in dfa.columns:
        return None
    dft = dfa[dfa[target].notna()].copy().reset_index(drop=True)
    if dft.empty:
        return None

    # Precompute regime labels/bins
    if regime_mode == "hard":
        dft["_risk"], dft["_credit"], dft["_rate"] = zip(
            *dft.apply(lambda r: classify_regime(r, 0.5), axis=1)
        )
    if regime_mode in ("soft", "hard"):
        for feat in MACRO_FEATURES:
            dft[f"_bin_{feat}"] = quantile_bins(dft[feat], regime_bins)
        bin_cols = [f"_bin_{f}" for f in MACRO_FEATURES]
        B_macro = dft[bin_cols].to_numpy(dtype=float)
    else:
        B_macro = None

    X_base = dft[BASE_FEATURES].to_numpy(dtype=float)
    X_delta = dft[DELTA_FEATURES].to_numpy(dtype=float)
    X_macro = dft[MACRO_FEATURES].to_numpy(dtype=float)
    sic2_arr = dft["sic2"].to_numpy() if "sic2" in dft.columns else None

    rng = np.random.default_rng(seed)
    sample_df = dft.sample(min(sample, len(dft)), random_state=seed)
    target_mean = float(dft[target].mean()) if shrinkage > 0 else np.nan

    preds: List[float] = []
    actuals: List[float] = []

    for _, query in sample_df.iterrows():
        qval = query.get(target)
        if not np.isfinite(qval):
            continue
        use_hard = regime_mode == "hard"
        q_regime = classify_regime(query, 0.5) if use_hard else None

        mask = build_mask(dft, query, use_sector, use_hard, q_regime)
        cand_count = int(mask.sum())
        if cand_count < min_candidates:
            if use_sector:
                use_sector = False
                mask = build_mask(dft, query, use_sector, use_hard, q_regime)
                cand_count = int(mask.sum())
            if cand_count < min_candidates and use_hard:
                use_hard = False
                mask = build_mask(dft, query, use_sector, use_hard, None)
                cand_count = int(mask.sum())

        cand_idx = np.where(mask)[0]
        if cand_idx.size == 0:
            continue

        q_base = np.array(query[BASE_FEATURES], dtype=float)
        q_delta = np.array(query[DELTA_FEATURES], dtype=float)
        q_macro = np.array(query[MACRO_FEATURES], dtype=float)

        diff = X_base[cand_idx] - q_base
        mask_base = ~np.isnan(diff)
        diff[~mask_base] = 0.0
        num = (diff ** 2).sum(axis=1)
        den = mask_base.sum(axis=1)
        d_base = np.sqrt(np.divide(num, den, out=np.full_like(num, np.nan, dtype=float), where=den > 0))

        diff = X_delta[cand_idx] - q_delta
        mask_delta = ~np.isnan(diff)
        diff[~mask_delta] = 0.0
        num = (diff ** 2).sum(axis=1)
        den = mask_delta.sum(axis=1)
        d_change = np.sqrt(np.divide(num, den, out=np.full_like(num, np.nan, dtype=float), where=den > 0))

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

        if regime_mode == "soft" and B_macro is not None:
            q_bins = B_macro[query.name]
            diffb = np.abs(B_macro[cand_idx] - q_bins)
            mask_bins = ~np.isnan(diffb)
            denom = mask_bins.sum(axis=1)
            scale = (regime_bins - 1) if regime_bins > 1 else 1.0
            regime_dist = np.where(denom > 0, diffb.sum(axis=1) / denom / scale, 0.0)
            total += regime_weight * regime_dist

        if not use_sector and sector_penalty > 0 and sic2_arr is not None:
            q_sic = query.get("sic2")
            if pd.notna(q_sic):
                cand_sic = sic2_arr[cand_idx]
                mismatch = (cand_sic != q_sic) & pd.notna(cand_sic)
                total += sector_penalty * mismatch.astype(float)

        order = np.argsort(total)[:k]
        topk = dft.iloc[cand_idx[order]]
        d = total[order]

        if weighting == "uniform":
            w = np.ones_like(d, dtype=float)
        elif weighting == "inverse":
            w = 1.0 / (d + 1e-6)
        else:
            sigma = float(np.median(d[~np.isnan(d)])) if np.any(np.isfinite(d)) else 1.0
            sigma = max(sigma, 1e-6)
            w = np.exp(-d / sigma)
            if not np.any(np.isfinite(w)) or np.nansum(w) <= 0:
                w = 1.0 / (d + 1e-6)

        vals = pd.to_numeric(topk[target], errors="coerce").to_numpy(dtype=float)
        maskv = np.isfinite(vals)
        if maskv.sum() == 0:
            continue

        w_sub = w[maskv]
        if not np.any(np.isfinite(w_sub)) or np.nansum(w_sub) <= 0:
            w_sub = np.ones_like(vals[maskv], dtype=float)
        pred = float(np.average(vals[maskv], weights=w_sub))
        if shrinkage > 0 and np.isfinite(target_mean):
            sum_w = float(np.sum(w_sub))
            pred = float((np.sum(w_sub * vals[maskv]) + shrinkage * target_mean) / (sum_w + shrinkage))

        if not np.isfinite(pred):
            continue
        preds.append(pred)
        actuals.append(float(qval))

    if len(preds) < min_n:
        return None

    p = np.array(preds)
    a = np.array(actuals)
    mae = float(np.mean(np.abs(p - a)))
    baseline = float(np.mean(np.abs(np.mean(a) - a)))
    if len(p) < 2 or np.nanstd(p) < 1e-9 or np.nanstd(a) < 1e-9:
        corr = 0.0
    else:
        corr = float(np.corrcoef(p, a)[0, 1])
    sign_hit = float(np.mean(np.sign(p) == np.sign(a)))
    score = 0.5 * corr + 0.3 * sign_hit + 0.2 * (1 - (mae / baseline if baseline > 0 else 1.0))

    return {
        "n": len(p),
        "mae": mae,
        "mae_baseline": baseline,
        "corr": corr,
        "sign_hit": sign_hit,
        "score": score,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-n", type=int, default=200)
    parser.add_argument("--actions", default=None, help="Comma-separated action types to optimize.")
    parser.add_argument("--preset", choices=["quick", "thorough"], default="quick")
    parser.add_argument("--k-list", default=None)
    parser.add_argument("--shrinkage-list", default=None)
    parser.add_argument("--weighting-list", default=None)
    parser.add_argument("--regime-modes", default=None)
    parser.add_argument("--sector-penalties", default=None)
    parser.add_argument("--min-candidates-list", default=None)
    parser.add_argument("--log-every", type=int, default=20, help="Progress log frequency per action_type.")
    parser.add_argument("--full-out", default=str(CURATED_DIR / "similarity_hyperparams_scores.parquet"))
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()

    df = pd.read_parquet(FEATURES_PATH)
    if args.actions:
        actions = [a.strip() for a in args.actions.split(",") if a.strip()]
        df = df[df["action_type"].isin(actions)].copy()
    action_types = sorted(df["action_type"].unique().tolist())

    target_map = {}
    if TARGET_MAP_PATH.exists():
        tm = pd.read_parquet(TARGET_MAP_PATH)
        if "action_type" in tm.columns and "target" in tm.columns:
            target_map = dict(zip(tm["action_type"].astype(str), tm["target"].astype(str)))

    if args.preset == "quick":
        k_list = parse_list(args.k_list, int) or [20, 40]
        shrink_list = parse_list(args.shrinkage_list, float) or [0.0, 5.0, 10.0]
        weighting_list = parse_list(args.weighting_list, str) or ["kernel", "inverse"]
        regime_modes = parse_list(args.regime_modes, str) or ["soft", "off"]
        sector_penalties = parse_list(args.sector_penalties, float) or [0.0, 0.2]
        min_candidates_list = parse_list(args.min_candidates_list, int) or [50, 200]
    else:
        k_list = parse_list(args.k_list, int) or [15, 25, 40, 60]
        shrink_list = parse_list(args.shrinkage_list, float) or [0.0, 5.0, 10.0, 20.0]
        weighting_list = parse_list(args.weighting_list, str) or ["kernel", "inverse", "uniform"]
        regime_modes = parse_list(args.regime_modes, str) or ["soft", "off"]
        sector_penalties = parse_list(args.sector_penalties, float) or [0.0, 0.1, 0.2, 0.3]
        min_candidates_list = parse_list(args.min_candidates_list, int) or [50, 200, 500]

    grid = [
        (k, shrink, weighting, regime_mode, sector_penalty, min_candidates)
        for k in k_list
        for shrink in shrink_list
        for weighting in weighting_list
        for regime_mode in regime_modes
        for sector_penalty in sector_penalties
        for min_candidates in min_candidates_list
    ]
    print(f"[optimize] actions={len(action_types)} grid={len(grid)}", flush=True)

    results: List[Dict[str, object]] = []
    best_rows: List[Dict[str, object]] = []

    for action_type in action_types:
        dfa = df[df["action_type"] == action_type].copy().reset_index(drop=True)
        if dfa.empty:
            continue
        target = target_map.get(action_type, "outcome_ev_ebitda_12m_w")

        print(f"[{action_type}] optimizing {len(grid)} configs (target={target})", flush=True)
        t0 = time.time()
        best = None
        for idx, (k, shrink, weighting, regime_mode, sector_penalty, min_candidates) in enumerate(grid, start=1):
            rec = score_config(
                dfa,
                target,
                sample=args.sample,
                seed=args.seed,
                k=k,
                regime_mode=regime_mode,
                regime_bins=4,
                regime_weight=0.2,
                use_sector=True,
                sector_penalty=sector_penalty,
                min_candidates=min_candidates,
                weighting=weighting,
                shrinkage=shrink,
                min_n=args.min_n,
            )
            if rec is None:
                continue
            row = {
                "action_type": action_type,
                "target": target,
                "k": k,
                "shrinkage": shrink,
                "weighting": weighting,
                "regime_mode": regime_mode,
                "sector_penalty": sector_penalty,
                "min_candidates": min_candidates,
                **rec,
            }
            results.append(row)
            if best is None or row["score"] > best["score"]:
                best = row
            if args.log_every and idx % args.log_every == 0:
                elapsed = time.time() - t0
                rate = idx / elapsed if elapsed > 0 else 0.0
                eta = (len(grid) - idx) / rate if rate > 0 else 0.0
                print(f"[{action_type}] {idx}/{len(grid)} elapsed={elapsed:.1f}s eta={eta:.1f}s", flush=True)

        if best is None:
            best = {
                "action_type": action_type,
                "target": target,
                "k": 25,
                "shrinkage": 5.0,
                "weighting": "kernel",
                "regime_mode": "soft",
                "sector_penalty": 0.2,
                "min_candidates": 200,
                "score": np.nan,
                "corr": np.nan,
                "sign_hit": np.nan,
                "mae": np.nan,
                "mae_baseline": np.nan,
                "n": 0,
            }
        best_rows.append(best)

    out = pd.DataFrame(results)
    if args.full_out:
        out.to_parquet(Path(args.full_out), index=False)

    best_df = pd.DataFrame(best_rows)
    out_path = Path(args.out)
    if out_path.exists():
        try:
            existing = pd.read_parquet(out_path)
        except Exception:
            existing = None
        if existing is not None and "action_type" in existing.columns:
            # Bring forward any extra columns (e.g., custom penalties) from existing hyperparams
            extra_cols = [c for c in existing.columns if c not in best_df.columns]
            if extra_cols:
                existing_extra = existing[["action_type"] + extra_cols].drop_duplicates("action_type")
                best_df = best_df.merge(existing_extra, on="action_type", how="left")
            # If optimizing a subset of actions, keep rows for other actions
            if args.actions:
                missing_actions = existing[~existing["action_type"].isin(best_df["action_type"])]
                if not missing_actions.empty:
                    best_df = pd.concat([best_df, missing_actions], ignore_index=True, sort=False)

    best_df.to_parquet(out_path, index=False)
    print(f"Saved hyperparams -> {args.out} ({len(best_df):,} rows)", flush=True)


if __name__ == "__main__":
    main()
