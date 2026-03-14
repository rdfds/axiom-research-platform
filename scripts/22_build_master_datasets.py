#!/usr/bin/env python
"""
Build master datasets (best/largest per type) and a Russell 3000 proxy universe.

Outputs:
  data/curated/universe_r3000_proxy.parquet
  data/curated/corporate_actions_master.parquet
  data/curated/prices_master.parquet
  data/curated/prices_master_full.parquet
  data/curated/fundamentals_master.parquet
  data/curated/buybacks_master.parquet
  data/curated/mna_master.parquet
  data/curated/master_summary.csv
"""

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
import pandas as pd
import pyarrow.dataset as ds

try:
    import duckdb  # type: ignore
except Exception:  # pragma: no cover
    duckdb = None


DATA_DIR = Path(__file__).parent.parent / "data"
CRSP_DIR = DATA_DIR / "wrds" / "crsp"
CURATED_DIR = DATA_DIR / "curated"
CURATED_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(pattern: str):
    files = list(CRSP_DIR.glob(pattern))
    if not files:
        return None
    return ds.dataset(files, format="parquet")


def build_r3000_proxy(max_rank: Optional[int] = 3000):
    dataset = load_dataset("msf_*.parquet")
    if dataset is None:
        raise FileNotFoundError("CRSP msf files not found. Run scripts/20_pull_crsp_stock_data.py first.")

    if max_rank and max_rank > 0:
        table = dataset.to_table(columns=["permno", "permco", "date", "prc", "shrout"])
        df = table.to_pandas()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["prc"] = pd.to_numeric(df["prc"], errors="coerce")
        df["shrout"] = pd.to_numeric(df["shrout"], errors="coerce")
        df["mktcap"] = df["prc"].abs() * df["shrout"]

        df = df.dropna(subset=["date", "permno"])
        df = df.dropna(subset=["mktcap"])
        df = df.sort_values(["date", "mktcap"], ascending=[True, False])
        df["rank"] = df.groupby("date")["mktcap"].rank(method="first", ascending=False)
        universe = df[df["rank"] <= max_rank].copy()
    else:
        # Use msenames to include all CRSP-listed permno-months (not just msf coverage).
        names_dataset = load_dataset("msenames_*.parquet")
        if names_dataset is None:
            raise FileNotFoundError("CRSP msenames files not found. Run scripts/20_pull_crsp_stock_data.py first.")
        if duckdb is None:
            raise RuntimeError("duckdb is required to build full universe from msenames.")
        pattern = str(CRSP_DIR / "msenames_*.parquet")
        con = duckdb.connect(database=":memory:")
        query = f"""
        WITH names AS (
            SELECT
                permno,
                CAST(namedt AS DATE) AS namedt,
                CAST(nameendt AS DATE) AS nameendt
            FROM read_parquet('{pattern}')
            WHERE namedt IS NOT NULL AND nameendt IS NOT NULL
        ),
        expanded AS (
            SELECT
                permno,
                (date_trunc('month', gs.value) + INTERVAL '1 month' - INTERVAL '1 day')::DATE AS date
            FROM names
            CROSS JOIN generate_series(
                date_trunc('month', namedt),
                date_trunc('month', nameendt),
                INTERVAL '1 month'
            ) AS gs(value)
        )
        SELECT DISTINCT permno, date
        FROM expanded
        """
        universe = con.execute(query).df()

    # Ensure uniqueness per permno-month to avoid merge inflation downstream.
    universe = universe.drop_duplicates(subset=["date", "permno"])

    out_path = CURATED_DIR / "universe_r3000_proxy.parquet"
    universe.to_parquet(out_path, index=False)
    return universe


def filter_by_universe(df, universe, date_col: str, id_col: str):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["month_end"] = df[date_col].dt.to_period("M").dt.to_timestamp("M")
    u = universe[["date", "permno"]].copy()
    u["date"] = pd.to_datetime(u["date"], errors="coerce")
    u = u.rename(columns={"date": "month_end", "permno": id_col})
    merged = df.merge(u, on=["month_end", id_col], how="inner")
    return merged.drop(columns=["month_end"])


def filter_by_universe_carryforward(df, universe, date_col: str, id_col: str):
    """
    Filter by universe, carrying forward the last available universe month_end.
    This is useful when data extends beyond the CRSP-based universe window.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["month_end"] = df[date_col].dt.to_period("M").dt.to_timestamp("M")

    u = universe[["date", "permno"]].copy()
    u["date"] = pd.to_datetime(u["date"], errors="coerce")
    u = u.rename(columns={"date": "month_end", "permno": id_col})
    max_month_end = u["month_end"].max()
    df.loc[df["month_end"] > max_month_end, "month_end"] = max_month_end

    merged = df.merge(u, on=["month_end", id_col], how="inner")
    return merged.drop(columns=["month_end"])


def build_corporate_actions_master(universe):
    frames = []

    link_path = CRSP_DIR / "ccmxpf_lnkhist.parquet"
    link = pd.read_parquet(link_path) if link_path.exists() else None

    def standardize_crsp_actions(frame):
        if frame is None or frame.empty:
            return frame
        df = frame.copy()
        df["action_date"] = pd.to_datetime(df["action_date"], errors="coerce")
        df["source"] = df.get("source", "wrds_crsp")
        df["source_action_type"] = df.get("action_type")
        df["source_action_subtype"] = df.get("action_subtype")

        facpr = pd.to_numeric(df.get("facpr"), errors="coerce")
        df["action_type"] = df["source_action_type"]
        df["action_subtype"] = df["source_action_subtype"]

        dividend_mask = df["source_action_type"] == "dividend"
        special_mask = df["source_action_subtype"].isin(["special", "irregular", "liquidating"])
        df.loc[dividend_mask & special_mask, "action_type"] = "dividend_special"
        df.loc[dividend_mask & ~special_mask, "action_type"] = "dividend_regular"

        split_mask = df["source_action_type"] == "split"
        reverse_mask = split_mask & ((df["source_action_subtype"] == "reverse") | (facpr < 1))
        df.loc[reverse_mask, "action_type"] = "reverse_split"
        df.loc[split_mask & ~reverse_mask, "action_type"] = "stock_split"

        df.loc[df["source_action_type"] == "spinoff", "action_type"] = "spinoff"
        df.loc[df["source_action_type"] == "return_of_capital", "action_type"] = "return_of_capital"
        df.loc[df["source_action_type"] == "rights_offering", "action_type"] = "rights_offering"
        df.loc[df["source_action_type"] == "stock_distribution", "action_type"] = "stock_distribution"
        df.loc[df["source_action_type"] == "distribution_other", "action_type"] = "distribution_other"
        df.loc[df["source_action_type"] == "delisting", "action_type"] = "delisting"

        df["amount"] = pd.to_numeric(df.get("divamt"), errors="coerce").fillna(
            pd.to_numeric(df.get("dlamt"), errors="coerce")
        )
        df["ratio"] = facpr

        df = attach_names_by_permno(df, date_col="action_date")
        return df

    def standardize_action_frame(frame, action_type, source, date_col="action_date"):
        if frame is None or frame.empty:
            return frame
        df = frame.copy()
        df["action_date"] = pd.to_datetime(df[date_col], errors="coerce")
        df["source"] = df.get("source", source)
        df["source_action_type"] = df.get("action_type")
        df["source_action_subtype"] = df.get("action_type")
        df["action_type"] = action_type
        df["action_subtype"] = df["source_action_type"]
        df = attach_names_by_permno(df, date_col="action_date")
        return df

    # 1) CRSP corporate actions (base)
    crsp_path = DATA_DIR / "curated" / "corporate_actions_crsp.parquet"
    if crsp_path.exists():
        crsp = pd.read_parquet(crsp_path)
        # Drop delisting codes that we re-classify elsewhere (acquisitions/bankruptcies)
        if (DATA_DIR / "acquisitions_clean.parquet").exists() or (DATA_DIR / "bankruptcies_clean.parquet").exists():
            dlst_mask = crsp["action_code_type"].eq("dlstcd")
            codes = pd.to_numeric(crsp["action_code"], errors="coerce")
            crsp = crsp[~(dlst_mask & codes.between(200, 499))]
        frames.append(standardize_crsp_actions(crsp))

    # 2) Buybacks (Compustat)
    buybacks_path = DATA_DIR / "buybacks_clean.parquet"
    if buybacks_path.exists():
        buybacks = pd.read_parquet(buybacks_path)
        buybacks = attach_permno_by_gvkey(buybacks, link, date_col="action_date")
        buybacks["amount"] = pd.to_numeric(buybacks.get("buyback_amount_qtr"), errors="coerce")
        buybacks["ratio"] = pd.NA
        buybacks = standardize_action_frame(buybacks, "buyback", "compustat_prstkcy")
        buybacks["action_subtype"] = pd.NA
        frames.append(buybacks)

    # 2b) ECM proxy (Compustat share count changes)
    ecm_proxy_path = CURATED_DIR / "equity_offerings_proxy.parquet"
    if ecm_proxy_path.exists():
        ecm_proxy = pd.read_parquet(ecm_proxy_path)
        ecm_proxy = attach_permno_by_gvkey(ecm_proxy, link, date_col="action_date")
        ecm_proxy["amount"] = pd.to_numeric(ecm_proxy.get("amount"), errors="coerce")
        ecm_proxy["ratio"] = pd.NA
        ecm_proxy = standardize_action_frame(ecm_proxy, "equity_offering_public_proxy", "compustat_proxy")
        frames.append(ecm_proxy)

    # 2c) FMP Form D equity offerings (private/exempt)
    fmp_eq_path = CURATED_DIR / "equity_offerings_fmp.parquet"
    if fmp_eq_path.exists():
        try:
            fmp_eq = pd.read_parquet(fmp_eq_path)
        except Exception as exc:
            log(f"Skipping FMP equity offerings (unreadable): {exc}")
            fmp_eq = None
        if fmp_eq is not None and not fmp_eq.empty:
            fmp_eq = attach_permno_by_gvkey(fmp_eq, link, date_col="action_date")
            fmp_eq["amount"] = pd.to_numeric(fmp_eq.get("offering_amount"), errors="coerce")
            fmp_eq["ratio"] = pd.NA
            fmp_eq = standardize_action_frame(fmp_eq, "equity_offering_private", "fmp_form_d")
            frames.append(fmp_eq)

    # 3) Dividend change actions (CRSP-derived)
    div_actions_path = DATA_DIR / "dividend_actions.parquet"
    if div_actions_path.exists():
        div_actions = pd.read_parquet(div_actions_path)
        div_actions = attach_permno_by_gvkey(div_actions, link, date_col="action_date")
        div_actions["amount"] = pd.to_numeric(div_actions.get("div_amount"), errors="coerce")
        div_actions["ratio"] = pd.NA
        div_actions["source_action_type"] = div_actions.get("action_type")
        div_actions["source_action_subtype"] = div_actions.get("action_type")
        div_actions["action_subtype"] = pd.NA
        div_actions["source"] = "crsp_dividend_actions"
        div_actions = attach_names_by_permno(div_actions, date_col="action_date")
        frames.append(div_actions)

    # 4) Acquisition delistings (CRSP clean)
    acq_path = DATA_DIR / "acquisitions_clean.parquet"
    if acq_path.exists():
        acq = pd.read_parquet(acq_path)
        acq["amount"] = pd.to_numeric(acq.get("deal_amount"), errors="coerce")
        acq["ratio"] = pd.NA
        acq = standardize_action_frame(acq, "acquisition", "crsp_delist")
        frames.append(acq)

    # 4b) Refinitiv M&A (acquiror actions)
    mna_path = CURATED_DIR / "mna_master.parquet"
    if mna_path.exists():
        mna_countries = [c.strip().lower() for c in os.getenv("MNA_COUNTRIES", "United States").split(",") if c.strip()]
        base_mna = pd.read_parquet(
            mna_path,
            columns=[
                "deal_id",
                "announce_date",
                "event_date",
                "completion_date",
                "deal_status",
                "deal_type",
                "deal_value",
                "acquiror_permno",
                "acquiror_gvkey",
                "acquiror_name",
                "acquiror_ticker",
                "acquiror_country",
                "target_permno",
                "target_gvkey",
                "target_name",
                "target_ticker",
                "target_country",
            ],
        )
        if not base_mna.empty:
            base_mna = base_mna[base_mna["deal_status"] == "Completed"].copy()
            base_mna["announce_date"] = pd.to_datetime(base_mna["announce_date"], errors="coerce")
            base_mna["event_date"] = pd.to_datetime(base_mna["event_date"], errors="coerce")
            base_mna["completion_date"] = pd.to_datetime(base_mna["completion_date"], errors="coerce")
            base_mna["action_date"] = (
                base_mna["announce_date"].fillna(base_mna["event_date"]).fillna(base_mna["completion_date"])
            )
            base_mna = base_mna[base_mna["action_date"].notna()].copy()

            # Acquiror-side acquisitions
            mna = base_mna.copy()
            if mna_countries and "acquiror_country" in mna.columns:
                mna["_country_norm"] = (
                    mna["acquiror_country"].astype("string").str.strip().str.lower()
                )
                mna["_country_norm"] = mna["_country_norm"].replace({"u.s.": "united states", "usa": "united states"})
                mna = mna[mna["_country_norm"].isin(mna_countries)]
            mna["permno"] = pd.to_numeric(mna.get("acquiror_permno"), errors="coerce")
            mna["gvkey"] = mna.get("acquiror_gvkey").astype("string")

            if link is not None and not link.empty:
                mna_known = mna[mna["permno"].notna()].copy()
                mna_missing = mna[mna["permno"].isna() & mna["gvkey"].notna()].copy()
                if not mna_missing.empty:
                    for col in ("permno", "permco"):
                        if col in mna_missing.columns:
                            mna_missing = mna_missing.drop(columns=[col])
                    mna_missing = attach_permno_by_gvkey(mna_missing, link, date_col="action_date")
                mna = pd.concat([mna_known, mna_missing], ignore_index=True, sort=False)

            mna["amount"] = pd.to_numeric(mna.get("deal_value"), errors="coerce")
            mna["ratio"] = pd.NA
            mna["source"] = "refinitiv_mna"
            mna["source_action_type"] = "acquisition"
            mna["source_action_subtype"] = mna.get("deal_type")
            mna["action_type"] = "acquisition"
            mna["action_subtype"] = mna.get("deal_type")
            mna = attach_names_by_permno(mna, date_col="action_date")
            frames.append(mna)

            # Target-side divestitures (target sold)
            mna_t = base_mna.copy()
            if mna_countries and "target_country" in mna_t.columns:
                mna_t["_country_norm"] = (
                    mna_t["target_country"].astype("string").str.strip().str.lower()
                )
                mna_t["_country_norm"] = mna_t["_country_norm"].replace({"u.s.": "united states", "usa": "united states"})
                mna_t = mna_t[mna_t["_country_norm"].isin(mna_countries)]
            mna_t["permno"] = pd.to_numeric(mna_t.get("target_permno"), errors="coerce")
            mna_t["gvkey"] = mna_t.get("target_gvkey").astype("string")

            if link is not None and not link.empty:
                mna_known = mna_t[mna_t["permno"].notna()].copy()
                mna_missing = mna_t[mna_t["permno"].isna() & mna_t["gvkey"].notna()].copy()
                if not mna_missing.empty:
                    for col in ("permno", "permco"):
                        if col in mna_missing.columns:
                            mna_missing = mna_missing.drop(columns=[col])
                    mna_missing = attach_permno_by_gvkey(mna_missing, link, date_col="action_date")
                mna_t = pd.concat([mna_known, mna_missing], ignore_index=True, sort=False)

            mna_t["amount"] = pd.to_numeric(mna_t.get("deal_value"), errors="coerce")
            mna_t["ratio"] = pd.NA
            mna_t["source"] = "refinitiv_mna_target"
            mna_t["source_action_type"] = "divestiture"
            mna_t["source_action_subtype"] = mna_t.get("deal_type")
            mna_t["action_type"] = "divestiture"
            mna_t["action_subtype"] = mna_t.get("deal_type")
            mna_t = attach_names_by_permno(mna_t, date_col="action_date")
            frames.append(mna_t)

    # 5) Bankruptcies (CRSP clean)
    bank_path = DATA_DIR / "bankruptcies_clean.parquet"
    if bank_path.exists():
        bank = pd.read_parquet(bank_path)
        bank["amount"] = pd.NA
        bank["ratio"] = pd.NA
        bank = standardize_action_frame(bank, "bankruptcy", "crsp_delist")
        frames.append(bank)

    # 6) Going private (CRSP clean)
    gp_path = DATA_DIR / "going_private.parquet"
    if gp_path.exists():
        gp = pd.read_parquet(gp_path)
        gp["amount"] = pd.NA
        gp["ratio"] = pd.NA
        gp = standardize_action_frame(gp, "going_private", "crsp_delist")
        gp["action_subtype"] = pd.NA
        frames.append(gp)

    # 7) Ticker changes
    tc_path = DATA_DIR / "ticker_changes_linked.parquet"
    if tc_path.exists():
        tc = pd.read_parquet(tc_path)
    else:
        tc_path = DATA_DIR / "ticker_changes.parquet"
        tc = pd.read_parquet(tc_path) if tc_path.exists() else None
    if tc is not None:
        tc["amount"] = pd.NA
        tc["ratio"] = pd.NA
        tc = standardize_action_frame(tc, "ticker_change", "crsp_names")
        tc["action_subtype"] = pd.NA
        frames.append(tc)

    # 8) Bond issuances (FISD)
    bond_path = CURATED_DIR / "bond_issuances_fisd.parquet"
    if bond_path.exists():
        bonds = pd.read_parquet(bond_path)
        bonds["amount"] = pd.to_numeric(bonds.get("amount"), errors="coerce")
        bonds["ratio"] = pd.NA
        bonds = standardize_action_frame(bonds, "bond_issuance", "fisd", date_col="offering_date")
        frames.append(bonds)

    # 9) Bond redemptions (FISD)
    red_path = CURATED_DIR / "bond_redemptions_fisd.parquet"
    if red_path.exists():
        red = pd.read_parquet(red_path)
        red["amount"] = pd.to_numeric(red.get("amount"), errors="coerce")
        red["ratio"] = pd.NA
        red = standardize_action_frame(red, "bond_redemption", "fisd", date_col="action_date")
        frames.append(red)

    # 10) Loan actions (DealScan)
    loan_path = CURATED_DIR / "loan_actions_dealscan.parquet"
    if loan_path.exists():
        loans = pd.read_parquet(loan_path)
        loans = attach_permno_by_gvkey(loans, link, date_col="action_date")
        loans["amount"] = pd.to_numeric(loans.get("amount"), errors="coerce")
        loans["ratio"] = pd.NA
        loans["action_date"] = pd.to_datetime(loans["action_date"], errors="coerce")
        loans["source"] = "dealscan"
        loans["source_action_type"] = loans.get("action_type")
        loans["source_action_subtype"] = loans.get("action_subtype")
        loans["action_type"] = loans.get("action_type")
        loans["action_subtype"] = loans.get("action_subtype")
        loans = attach_names_by_permno(loans, date_col="action_date")
        frames.append(loans)

    # 11) Issuer credit ratings (CIQ)
    ratings_path = CURATED_DIR / "issuer_ratings_ciq.parquet"
    if ratings_path.exists():
        ratings = pd.read_parquet(ratings_path)
        if "rating_date" in ratings.columns:
            ratings = attach_permno_by_gvkey(ratings, link, date_col="rating_date")
            ratings["amount"] = pd.NA
            ratings["ratio"] = pd.NA
            ratings["action_date"] = pd.to_datetime(ratings["rating_date"], errors="coerce")
            ratings["source"] = "ciq_ratings"
            ratings["source_action_type"] = "issuer_rating"
            ratings["source_action_subtype"] = ratings.get("rating_type_code")
            ratings["action_type"] = "issuer_rating"
            ratings["action_subtype"] = ratings.get("rating_type_code")
            ratings = attach_names_by_permno(ratings, date_col="action_date")
            frames.append(ratings)

    if not frames:
        return None

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["action_date"] = pd.to_datetime(combined["action_date"], errors="coerce")

    # Normalize permno for joins
    combined["permno"] = pd.to_numeric(combined.get("permno"), errors="coerce")

    # Fill gvkey for permno-based actions (if missing)
    combined = attach_gvkey_by_permno(combined, link, date_col="action_date")

    # Require permno for universe filtering
    combined = combined.dropna(subset=["permno", "action_date"])

    filtered = filter_by_universe(combined, universe, "action_date", "permno")
    if "sic" in filtered.columns:
        filtered["sic"] = filtered["sic"].astype("string")
    out_path = CURATED_DIR / "corporate_actions_master.parquet"
    filtered.to_parquet(out_path, index=False)
    return filtered


def build_prices_master(universe):
    dataset = load_dataset("msf_*.parquet")
    if dataset is None:
        return None
    table = dataset.to_table()
    df = table.to_pandas()
    filtered = filter_by_universe(df, universe, "date", "permno")
    out_path = CURATED_DIR / "prices_master.parquet"
    filtered.to_parquet(out_path, index=False)
    return filtered


def build_prices_master_full(universe):
    dataset = load_dataset("msf_*.parquet")
    if dataset is None:
        return None
    table = dataset.to_table()
    crsp = table.to_pandas()

    frames = [crsp]

    rdp_path = DATA_DIR / "prices_monthly_rdp_2025.parquet"
    if rdp_path.exists():
        rdp = pd.read_parquet(rdp_path)
        # Align to CRSP column set
        for col in crsp.columns:
            if col not in rdp.columns:
                rdp[col] = pd.NA
        # Preserve extra columns from RDP
        rdp = rdp.reindex(columns=crsp.columns, fill_value=pd.NA)
        frames.append(rdp)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    filtered = filter_by_universe_carryforward(combined, universe, "date", "permno")
    out_path = CURATED_DIR / "prices_master_full.parquet"
    filtered.to_parquet(out_path, index=False)
    return filtered


