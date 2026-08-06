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
) -> pd.Series:
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


def bucket_by_group(
    df: pd.DataFrame,
    col: str,
    group_cols: Iterable[str],
    q: int = 3,
    min_group: int = 30,
) -> pd.Series:
    """Assign quantile buckets within groups; fallback to global edges."""
    series = pd.to_numeric(df[col], errors="coerce").where(lambda s: np.isfinite(s), np.nan)
    # Global edges
    try:
        global_edges = series.dropna().quantile(np.linspace(0, 1, q + 1)).values
    except Exception:
        global_edges = np.array([series.min(), series.max()])
    global_edges = global_edges[np.isfinite(global_edges)]
    global_edges = np.unique(global_edges)

    def assign_bucket(x: pd.Series) -> pd.Series:
        x = pd.to_numeric(x, errors="coerce").where(lambda s: np.isfinite(s), np.nan)
        if x.notna().sum() < min_group:
            edges = global_edges
        else:
            edges = x.dropna().quantile(np.linspace(0, 1, q + 1)).values
            edges = edges[np.isfinite(edges)]
            edges = np.unique(edges)
            if len(edges) < q + 1 or not np.all(np.diff(edges) > 0):
                edges = global_edges
        if len(edges) < 2 or not np.all(np.diff(edges) > 0):
            return pd.Series(index=x.index, data=np.nan)
        return pd.cut(x, bins=edges, labels=False, include_lowest=True)

    # Use list of group columns (not a DataFrame) for grouping
    group_keys = [df[c] for c in group_cols]
    return series.groupby(group_keys).transform(assign_bucket).astype("Int64")


def main() -> None:
    if not ACTION_OUTCOMES_PATH.exists():
        raise FileNotFoundError(f"Missing action_outcomes: {ACTION_OUTCOMES_PATH}")
    if not FUNDAMENTALS_PATH.exists():
        raise FileNotFoundError(f"Missing fundamentals_master: {FUNDAMENTALS_PATH}")
    if not PRICES_PATH.exists():
        raise FileNotFoundError(f"Missing prices_master: {PRICES_PATH}")

    log("Loading action outcomes + sector (sic) + permno + price features as-of action_date...")
    con = duckdb.connect()
    query = f"""
    WITH ao AS (
        SELECT
            row_number() OVER () AS action_id,
            *
        FROM read_parquet('{ACTION_OUTCOMES_PATH.as_posix()}')
    ),
    fund AS (
        SELECT gvkey, datadate, sic, permno
        FROM read_parquet('{FUNDAMENTALS_PATH.as_posix()}')
    ),
    joined AS (
        SELECT
            ao.*,
            fund.sic,
            fund.permno,
            row_number() OVER (
                PARTITION BY ao.action_id
                ORDER BY fund.datadate DESC
            ) AS rn
        FROM ao
        LEFT JOIN fund
          ON fund.gvkey = ao.company_id
         AND fund.datadate <= ao.action_date
    ),
    base AS (
        SELECT * FROM joined WHERE rn = 1 OR rn IS NULL
    ),
    perms AS (
        SELECT DISTINCT permno FROM base WHERE permno IS NOT NULL
    ),
    prices AS (
        SELECT
            permno,
            CAST(date AS TIMESTAMP) AS date,
            ret,
            ABS(prc) AS price
        FROM read_parquet('{PRICES_PATH.as_posix()}')
        WHERE permno IN (SELECT permno FROM perms)
    ),
    prices_roll AS (
        SELECT
            permno,
            date,
            ret,
            price,
            CASE WHEN ret IS NOT NULL AND ret > -1 THEN LN(1 + ret) END AS logret,
            SUM(CASE WHEN ret IS NOT NULL AND ret > -1 THEN LN(1 + ret) END)
              OVER w126 AS sumlog_6m,
            SUM(CASE WHEN ret IS NOT NULL AND ret > -1 THEN LN(1 + ret) END)
              OVER w252 AS sumlog_12m,
            STDDEV_SAMP(ret) OVER w126 AS vol_6m,
            STDDEV_SAMP(ret) OVER w252 AS vol_12m,
            MAX(price) OVER w252 AS roll_max_12m
        FROM prices
        WINDOW
            w126 AS (PARTITION BY permno ORDER BY date ROWS BETWEEN 125 PRECEDING AND CURRENT ROW),
            w252 AS (PARTITION BY permno ORDER BY date ROWS BETWEEN 251 PRECEDING AND CURRENT ROW)
    ),
    prices_feat AS (
        SELECT
            permno,
            date,
            EXP(sumlog_6m) - 1 AS mom_6m,
            EXP(sumlog_12m) - 1 AS mom_12m,
            vol_6m,
            vol_12m,
            price / roll_max_12m - 1 AS drawdown,
            MIN(price / roll_max_12m - 1) OVER w252 AS max_drawdown_12m
        FROM prices_roll
        WINDOW w252 AS (PARTITION BY permno ORDER BY date ROWS BETWEEN 251 PRECEDING AND CURRENT ROW)
    ),
    price_asof AS (
        SELECT
            b.action_id,
            pf.mom_6m,
            pf.mom_12m,
            pf.vol_6m,
            pf.vol_12m,
            pf.max_drawdown_12m
        FROM base b
        LEFT JOIN prices_feat pf
          ON pf.permno = b.permno
         AND pf.date <= b.action_date
        QUALIFY row_number() OVER (PARTITION BY b.action_id ORDER BY pf.date DESC) = 1
    )
    SELECT
        base.*,
        price_asof.mom_6m,
        price_asof.mom_12m,
        price_asof.vol_6m,
        price_asof.vol_12m,
        price_asof.max_drawdown_12m
    FROM base
    LEFT JOIN price_asof USING (action_id)
    """
    df = con.execute(query).df()
    con.close()
    log(f"Loaded base rows: {len(df):,}")

    if df.empty:
        raise RuntimeError("No rows produced for similarity dataset.")

    df["action_date"] = pd.to_datetime(df["action_date"], errors="coerce")
    df["action_year"] = df["action_date"].dt.year
    df["sic"] = pd.to_numeric(df["sic"], errors="coerce")
    df["sic2"] = (df["sic"] // 100).astype("Int64")

    # Transform heavy-tailed base features
    df["base_market_cap_log"] = np.log10(pd.to_numeric(df["base_market_cap"], errors="coerce").abs() + 1.0)
    df["base_revenue_ttm_log"] = np.log10(pd.to_numeric(df["base_revenue_ttm"], errors="coerce").abs() + 1.0)
    df["deal_size_to_mcap"] = pd.to_numeric(df["action_size"], errors="coerce") / pd.to_numeric(
        df["base_market_cap"], errors="coerce"
    )

    # Acquisition-specific M&A attributes (where available)
    df["mna_target_public"] = np.nan
    df["mna_cross_border"] = np.nan
    df["mna_deal_type_stake"] = np.nan
    df["mna_deal_type_lbo"] = np.nan
    df["mna_deal_type_tender"] = np.nan
    df["mna_deal_type_merger"] = np.nan
    df["mna_payment_cash"] = np.nan
    df["mna_payment_stock"] = np.nan
    df["mna_payment_mixed"] = np.nan
    df["mna_deal_completed"] = np.nan
    df["mna_same_sic2"] = np.nan
    df["mna_same_sic4"] = np.nan
    df["mna_same_country"] = np.nan

    if MNA_PATH.exists():
        acq_mask = (
            (df["action_type"] == "acquisition")
            & (df["source_dataset"] == "refinitiv_mna")
            & df["source_id"].notna()
        )
        if acq_mask.any():
            log("Merging M&A deal attributes for acquisitions...")
            acq_ids = df.loc[acq_mask, "source_id"].astype(str).unique()
            mna_cols = [
                "source_id",
                "deal_type",
                "deal_value",
                "payment_type",
                "pct_cash",
                "pct_stock",
                "premium_1day",
                "premium_1week",
                "premium_4week",
                "deal_status",
                "target_sic",
                "acquiror_sic",
                "target_country",
                "acquiror_country",
                "target_us_flag",
                "acquiror_us_flag",
                "target_permno_valid",
                "target_in_universe",
            ]
            mna = pd.read_parquet(MNA_PATH, columns=mna_cols)
            mna["source_id"] = mna["source_id"].astype(str)
            mna = mna[mna["source_id"].isin(set(acq_ids))]
            mna = mna.rename(
                columns={
                    "deal_type": "mna_deal_type",
                    "deal_value": "mna_deal_value",
                    "payment_type": "mna_payment_type",
                    "pct_cash": "mna_pct_cash",
                    "pct_stock": "mna_pct_stock",
                    "premium_1day": "mna_premium_1d",
                    "premium_1week": "mna_premium_1w",
                    "premium_4week": "mna_premium_4w",
                    "deal_status": "mna_deal_status",
                    "target_sic": "mna_target_sic",
                    "acquiror_sic": "mna_acquiror_sic",
                    "target_country": "mna_target_country",
                    "acquiror_country": "mna_acquiror_country",
                    "target_us_flag": "mna_target_us_flag",
                    "acquiror_us_flag": "mna_acquiror_us_flag",
                    "target_permno_valid": "mna_target_permno_valid",
                    "target_in_universe": "mna_target_in_universe",
                }
            )
            df = df.merge(mna, on="source_id", how="left")

            acq_mask = (
                (df["action_type"] == "acquisition")
                & (df["source_dataset"] == "refinitiv_mna")
                & df["source_id"].notna()
            )

            # Ensure optional M&A columns exist even if missing in the source
            for col in [
                "mna_pct_cash",
                "mna_pct_stock",
                "mna_premium_1d",
                "mna_premium_1w",
                "mna_premium_4w",
                "mna_payment_type",
                "mna_deal_status",
                "mna_target_sic",
                "mna_acquiror_sic",
                "mna_target_country",
                "mna_acquiror_country",
            ]:
                if col not in df.columns:
                    df[col] = np.nan
            t_perm = df.loc[acq_mask, "mna_target_permno_valid"]
            t_univ = df.loc[acq_mask, "mna_target_in_universe"]
            public = (t_perm.fillna(False).astype(bool) | t_univ.fillna(False).astype(bool)).astype(float)
            df.loc[acq_mask, "mna_target_public"] = np.where(
                t_perm.notna() | t_univ.notna(), public, np.nan
            )

            t_us = df.loc[acq_mask, "mna_target_us_flag"]
            a_us = df.loc[acq_mask, "mna_acquiror_us_flag"]
            df.loc[acq_mask, "mna_cross_border"] = np.where(
                t_us.notna() & a_us.notna(), (t_us.astype(bool) != a_us.astype(bool)).astype(float), np.nan
            )

            dt = df.loc[acq_mask, "mna_deal_type"].astype(str).str.lower()
            has_dt = df.loc[acq_mask, "mna_deal_type"].notna()
            df.loc[acq_mask, "mna_deal_type_stake"] = np.where(
                has_dt, dt.str.contains("stake").astype(float), np.nan
            )
            df.loc[acq_mask, "mna_deal_type_lbo"] = np.where(
                has_dt, dt.str.contains("lbo").astype(float), np.nan
            )
            df.loc[acq_mask, "mna_deal_type_tender"] = np.where(
                has_dt, dt.str.contains("tender").astype(float), np.nan
            )
            df.loc[acq_mask, "mna_deal_type_merger"] = np.where(
                has_dt, dt.str.contains("merger").astype(float), np.nan
            )

            # Payment mix and premiums
            df.loc[acq_mask, "mna_pct_cash"] = pd.to_numeric(
                df.loc[acq_mask, "mna_pct_cash"], errors="coerce"
            )
            df.loc[acq_mask, "mna_pct_stock"] = pd.to_numeric(
                df.loc[acq_mask, "mna_pct_stock"], errors="coerce"
            )
            df.loc[acq_mask, "mna_premium_1d"] = pd.to_numeric(
                df.loc[acq_mask, "mna_premium_1d"], errors="coerce"
            )
            df.loc[acq_mask, "mna_premium_1w"] = pd.to_numeric(
                df.loc[acq_mask, "mna_premium_1w"], errors="coerce"
            )
            df.loc[acq_mask, "mna_premium_4w"] = pd.to_numeric(
                df.loc[acq_mask, "mna_premium_4w"], errors="coerce"
            )

            pt = df.loc[acq_mask, "mna_payment_type"].astype(str).str.lower()
            has_pt = df.loc[acq_mask, "mna_payment_type"].notna()
            pay_cash = pt.str.contains("cash")
            pay_stock = pt.str.contains("stock")
            df.loc[acq_mask, "mna_payment_cash"] = np.where(
                has_pt, pay_cash.astype(float), np.nan
            )
            df.loc[acq_mask, "mna_payment_stock"] = np.where(
                has_pt, pay_stock.astype(float), np.nan
            )
            df.loc[acq_mask, "mna_payment_mixed"] = np.where(
                has_pt, (pay_cash & pay_stock).astype(float), np.nan
            )

            ds = df.loc[acq_mask, "mna_deal_status"].astype(str).str.lower()
            has_ds = df.loc[acq_mask, "mna_deal_status"].notna()
            df.loc[acq_mask, "mna_deal_completed"] = np.where(
                has_ds, ds.str.contains("complete|closed").astype(float), np.nan
            )

            t_sic = pd.to_numeric(df.loc[acq_mask, "mna_target_sic"], errors="coerce")
            a_sic = pd.to_numeric(df.loc[acq_mask, "mna_acquiror_sic"], errors="coerce")
            t_sic2 = (t_sic // 100).astype("Int64")
            a_sic2 = (a_sic // 100).astype("Int64")
            df.loc[acq_mask, "mna_same_sic2"] = np.where(
                t_sic2.notna() & a_sic2.notna(), (t_sic2 == a_sic2).astype(float), np.nan
            )
            df.loc[acq_mask, "mna_same_sic4"] = np.where(
                t_sic.notna() & a_sic.notna(), (t_sic == a_sic).astype(float), np.nan
            )

            t_cty = df.loc[acq_mask, "mna_target_country"]
            a_cty = df.loc[acq_mask, "mna_acquiror_country"]
            df.loc[acq_mask, "mna_same_country"] = np.where(
                t_cty.notna() & a_cty.notna(), (t_cty == a_cty).astype(float), np.nan
            )

            # If action_size is missing, backfill deal_size_to_mcap from M&A deal value
            if "mna_deal_value" in df.columns:
                dv = pd.to_numeric(df.loc[acq_mask, "mna_deal_value"], errors="coerce")
                bm = pd.to_numeric(df.loc[acq_mask, "base_market_cap"], errors="coerce")
                backfill = (dv / bm).astype("float64")
                existing = pd.to_numeric(df.loc[acq_mask, "deal_size_to_mcap"], errors="coerce").astype("float64")
                df.loc[acq_mask, "deal_size_to_mcap"] = existing.fillna(backfill)

    # Build additional fundamentals-based features from quarterly history
    log("Computing fundamentals-derived stability/growth/valuation features...")
    con = duckdb.connect()
    fund_query = f"""
    WITH actions_ids AS (
        SELECT DISTINCT company_id AS gvkey
        FROM read_parquet('{ACTION_OUTCOMES_PATH.as_posix()}')
    ),
    fund AS (
        SELECT f.gvkey, f.datadate, f.revtq, f.cogsq, f.oibdpq, f.niq, f.epspxq,
               f.xintq, f.oancfy, f.capxy, f.atq, f.cheq,
               COALESCE(f.dlttq,0) + COALESCE(f.dlcq,0) AS debt,
               f.cshoq, f.prccq, f.mkvaltq
        FROM read_parquet('{FUNDAMENTALS_PATH.as_posix()}') f
        JOIN actions_ids a ON a.gvkey = f.gvkey
    ),
    snap AS (
        SELECT
            gvkey,
            CAST(datadate AS TIMESTAMP) AS datadate,
            SUM(revtq) OVER w AS revenue_ttm,
            SUM(oibdpq) OVER w AS ebitda_ttm,
            SUM(niq) OVER w AS net_income_ttm,
            SUM(epspxq) OVER w AS eps_ttm,
            SUM(xintq) OVER w AS xint_ttm,
            SUM(oancfy) OVER w AS oancf_ttm,
            SUM(capxy) OVER w AS capx_ttm,
            SUM(revtq - cogsq) OVER w AS gross_profit_ttm,
            atq AS total_assets,
            cheq AS cash,
            debt,
            cshoq AS shares_out,
            prccq AS price,
            mkvaltq AS market_cap_raw
        FROM fund
        WINDOW w AS (
            PARTITION BY gvkey
            ORDER BY datadate
            ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
        )
    )
    SELECT * FROM snap
    """
    fund_df = con.execute(fund_query).df()
    con.close()
    log(f"Loaded fundamentals history rows: {len(fund_df):,}")

    if not fund_df.empty:
        fund_df["datadate"] = pd.to_datetime(fund_df["datadate"], errors="coerce").astype("datetime64[ns]")
        fund_df = fund_df.sort_values(["gvkey", "datadate"])
        g = fund_df.groupby("gvkey", group_keys=False)

        fund_df["gross_margin_ttm"] = fund_df["gross_profit_ttm"] / fund_df["revenue_ttm"]
        fund_df["roic_ttm"] = fund_df["net_income_ttm"] / fund_df["total_assets"]
        fund_df["fcf_ttm"] = fund_df["oancf_ttm"] - fund_df["capx_ttm"]
        fund_df["fcf_margin_ttm"] = fund_df["fcf_ttm"] / fund_df["revenue_ttm"]
        fund_df["interest_coverage"] = fund_df["ebitda_ttm"] / fund_df["xint_ttm"]
        fund_df["net_debt"] = fund_df["debt"] - fund_df["cash"].fillna(0)
        fund_df["net_debt_ebitda"] = fund_df["net_debt"] / fund_df["ebitda_ttm"]
        fund_df["ev_ebitda_ttm"] = (
            (fund_df["market_cap_raw"].fillna(fund_df["price"] * fund_df["shares_out"]) + fund_df["net_debt"])
            / fund_df["ebitda_ttm"]
        )

        # Rolling volatility (12q, 20q)
        for col, out12, out20 in [
            ("gross_margin_ttm", "gross_margin_vol_12q", "gross_margin_vol_20q"),
            ("roic_ttm", "roic_vol_12q", "roic_vol_20q"),
            ("fcf_margin_ttm", "fcf_margin_vol_12q", "fcf_margin_vol_20q"),
        ]:
            fund_df[out12] = g[col].rolling(12, min_periods=6).std().reset_index(level=0, drop=True)
            fund_df[out20] = g[col].rolling(20, min_periods=8).std().reset_index(level=0, drop=True)

        # CAGR helpers
        def _cagr(series: pd.Series, periods: int, years: float) -> pd.Series:
            prev = series.shift(periods)
            ratio = series / prev
            out = ratio ** (1.0 / years) - 1.0
            out[(series <= 0) | (prev <= 0)] = np.nan
            return out

        fund_df["revenue_cagr_1y"] = g["revenue_ttm"].transform(lambda s: _cagr(s, 4, 1.0))
        fund_df["revenue_cagr_3y"] = g["revenue_ttm"].transform(lambda s: _cagr(s, 12, 3.0))
        fund_df["fcf_cagr_1y"] = g["fcf_ttm"].transform(lambda s: _cagr(s, 4, 1.0))
        fund_df["fcf_cagr_3y"] = g["fcf_ttm"].transform(lambda s: _cagr(s, 12, 3.0))
        fund_df["eps_cagr_1y"] = g["eps_ttm"].transform(lambda s: _cagr(s, 4, 1.0))
        fund_df["eps_cagr_3y"] = g["eps_ttm"].transform(lambda s: _cagr(s, 12, 3.0))

        # Valuation percentile vs 5y history (20 quarters)
        def _rolling_pct(s: pd.Series, window: int = 20) -> pd.Series:
            return s.rolling(window, min_periods=8).apply(lambda x: float(np.mean(x <= x.iloc[-1])), raw=False)

        fund_df["ev_ebitda_pct_5y"] = g["ev_ebitda_ttm"].transform(lambda s: _rolling_pct(s, 20))

        # Merge as-of to action dates
        fund_df = fund_df.rename(columns={"gvkey": "company_id"})
        fund_df["company_id"] = fund_df["company_id"].astype("string")
        fund_df = fund_df[fund_df["datadate"].notna() & fund_df["company_id"].notna()].copy()
        # merge_asof requires the "on" key to be globally sorted
        fund_df = fund_df.sort_values(["datadate", "company_id"], kind="mergesort")

        df["action_date"] = pd.to_datetime(df["action_date"], errors="coerce").astype("datetime64[ns]")
        df["company_id"] = df["company_id"].astype("string")
        df = df[df["action_date"].notna() & df["company_id"].notna()].copy()
        # merge_asof requires the "on" key to be globally sorted
        df = df.sort_values(["action_date", "company_id"], kind="mergesort")
        df = pd.merge_asof(
            df,
            fund_df[
                [
                    "company_id",
                    "datadate",
                    "gross_margin_vol_12q",
                    "gross_margin_vol_20q",
                    "roic_vol_12q",
                    "roic_vol_20q",
                    "fcf_margin_vol_12q",
                    "fcf_margin_vol_20q",
                    "revenue_cagr_1y",
                    "revenue_cagr_3y",
                    "fcf_cagr_1y",
                    "fcf_cagr_3y",
                    "eps_cagr_1y",
                    "eps_cagr_3y",
                    "ev_ebitda_pct_5y",
                    "interest_coverage",
                    "net_debt_ebitda",
                ]
            ],
            left_on="action_date",
            right_on="datadate",
            by="company_id",
            direction="backward",
        )
        log("Merged fundamentals-derived features.")

    base_features = [
        "base_market_cap_log",
        "base_leverage",
        "base_margin",
        "base_revenue_ttm_log",
        "base_roic",
        "base_fcf_margin",
        "base_pe",
        "base_ev_ebitda",
        "gross_margin_vol_12q",
        "gross_margin_vol_20q",
        "roic_vol_12q",
        "roic_vol_20q",
        "fcf_margin_vol_12q",
        "fcf_margin_vol_20q",
        "revenue_cagr_1y",
        "revenue_cagr_3y",
        "fcf_cagr_1y",
        "fcf_cagr_3y",
        "eps_cagr_1y",
        "eps_cagr_3y",
        "ev_ebitda_pct_5y",
        "interest_coverage",
        "net_debt_ebitda",
        "mom_6m",
        "mom_12m",
        "vol_6m",
        "vol_12m",
        "max_drawdown_12m",
        "deal_size_to_mcap",
        "mna_target_public",
        "mna_cross_border",
        "mna_deal_type_stake",
        "mna_deal_type_lbo",
        "mna_deal_type_tender",
        "mna_deal_type_merger",
        "mna_pct_cash",
        "mna_pct_stock",
        "mna_payment_cash",
        "mna_payment_stock",
        "mna_payment_mixed",
        "mna_premium_1d",
        "mna_premium_1w",
        "mna_premium_4w",
        "mna_deal_completed",
        "mna_same_sic2",
        "mna_same_sic4",
        "mna_same_country",
    ]
    delta_features = [
        "revenue_delta",
        "margin_delta",
        "leverage_delta",
        "eps_delta",
        "roic_delta",
        "fcf_margin_delta",
    ]
    macro_features = [
        "macro_rate_10y",
        "macro_rate_2y",
        "macro_sofr",
        "macro_ig_oas",
        "macro_hy_oas",
        "macro_vix",
    ]

    # Compute z-scores
    group_firm = ["sic2", "action_year"]
    group_macro = ["action_year"]

    for col in base_features + delta_features:
        if col in df.columns:
            df[f"z_{col}"] = robust_zscore(df, col, group_firm, min_group=MIN_GROUP, cap=Z_CAP)

    for col in macro_features:
        if col in df.columns:
            df[f"z_{col}"] = robust_zscore(df, col, group_macro, min_group=MIN_GROUP, cap=Z_CAP)

    # Profile buckets (for tighter matching)
    df["size_bucket"] = bucket_by_group(df, "base_market_cap_log", group_firm, q=BUCKET_Q, min_group=MIN_GROUP)
    df["leverage_bucket"] = bucket_by_group(df, "base_leverage", group_firm, q=BUCKET_Q, min_group=MIN_GROUP)
    df["margin_bucket"] = bucket_by_group(df, "base_margin", group_firm, q=BUCKET_Q, min_group=MIN_GROUP)
    df["deal_size_bucket"] = bucket_by_group(df, "deal_size_to_mcap", group_firm, q=BUCKET_Q, min_group=MIN_GROUP)
    df["deal_size_bucket_strict"] = bucket_by_group(
        df, "deal_size_to_mcap", group_firm, q=DEAL_BUCKET_Q_STRICT, min_group=MIN_GROUP
    )

    # Keep a clean column order
    keep = [
        "company_id",
        "action_type",
        "action_subtype",
        "action_date",
        "action_size",
        "source_dataset",
        "source_id",
        "mapping_source",
        "sic",
        "sic2",
        "action_year",
        "size_bucket",
        "leverage_bucket",
        "margin_bucket",
        "deal_size_bucket",
        "deal_size_bucket_strict",
    ]
    keep += base_features + delta_features + macro_features
    outcome_cols = [c for c in df.columns if c.startswith("outcome_")]

    # Winsorize outcomes to avoid exploding % changes from tiny denominators
    for col in outcome_cols:
        df[f"{col}_w"] = winsorize_by_group(df, col, group_col="action_type", p=0.01, min_group=200)

    keep += outcome_cols
    keep += [f"{c}_w" for c in outcome_cols]
    keep += [c for c in df.columns if c.startswith("z_")]
    keep = [c for c in keep if c in df.columns]

    out = df[keep].copy()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    log(f"Saved similarity features -> {OUT_PATH} ({len(out):,} rows)")


if __name__ == "__main__":
    main()
