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


def attach_permno_by_gvkey(df, link, date_col="datadate"):
    if df is None or df.empty or link is None or link.empty:
        return df
    link = link.copy()
    link["linkdt"] = pd.to_datetime(link["linkdt"], errors="coerce")
    link["linkenddt"] = pd.to_datetime(link["linkenddt"], errors="coerce").fillna(pd.Timestamp("2099-12-31"))

    out = df.merge(link, on="gvkey", how="left")
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out[(out[date_col] >= out["linkdt"]) & (out[date_col] <= out["linkenddt"])]
    out = out.rename(columns={"lpermno": "permno", "lpermco": "permco"})
    return out


def attach_gvkey_by_permno(df, link, date_col="action_date"):
    if df is None or df.empty or link is None or link.empty:
        return df
    link = link.copy()
    link["linkdt"] = pd.to_datetime(link["linkdt"], errors="coerce")
    link["linkenddt"] = pd.to_datetime(link["linkenddt"], errors="coerce").fillna(pd.Timestamp("2099-12-31"))

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    tmp = out[["permno", date_col]].reset_index()
    merge = tmp.merge(link, left_on="permno", right_on="lpermno", how="left")
    merge = merge[(merge[date_col] >= merge["linkdt"]) & (merge[date_col] <= merge["linkenddt"])]
    merge = merge.sort_values(["index", "linkdt"], ascending=[True, False])
    merge = merge.drop_duplicates("index", keep="first")

    out = out.merge(
        merge[["index", "gvkey", "lpermco"]],
        left_index=True,
        right_on="index",
        how="left",
    )
    if "gvkey" in out.columns and "gvkey_y" in out.columns:
        out["gvkey"] = out["gvkey"].fillna(out["gvkey_y"])
    elif "gvkey_y" in out.columns:
        out["gvkey"] = out["gvkey_y"]
    if "lpermco" in out.columns:
        if "permco" in out.columns:
            out["permco"] = out["permco"].fillna(out["lpermco"])
        else:
            out["permco"] = out["lpermco"]
    out = out.drop(columns=[c for c in ["index", "gvkey_y", "lpermco"] if c in out.columns])
    return out


def attach_names_by_permno(df, date_col="action_date"):
    names_path = CRSP_DIR / "msenames_2000-01-01_to_2024-12-31.parquet"
    if df is None or df.empty or not names_path.exists():
        return df

    names = pd.read_parquet(
        names_path,
        columns=["permno", "namedt", "nameendt", "comnam", "ticker", "siccd", "cusip", "ncusip"],
    )
    names["namedt"] = pd.to_datetime(names["namedt"], errors="coerce")
    names["nameendt"] = pd.to_datetime(names["nameendt"], errors="coerce")
    names["cusip8"] = (
        names["ncusip"].fillna(names["cusip"]).astype("string").str.replace(r"[^0-9A-Za-z]", "", regex=True).str.upper()
    )
    names["cusip8"] = names["cusip8"].where(
        ~names["cusip8"].str.lower().isin(["", "nan", "none", "<na>"])
    ).str[:8]

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    tmp = out[["permno", date_col]].reset_index()
    merge = tmp.merge(names, on="permno", how="left")
    merge = merge[(merge[date_col] >= merge["namedt"]) & (merge[date_col] <= merge["nameendt"])]
    merge = merge.sort_values(["index", "namedt"], ascending=[True, False]).drop_duplicates("index", keep="first")

    merge = merge.rename(
        columns={
            "comnam": "company_name_map",
            "ticker": "ticker_map",
            "siccd": "sic_map",
            "cusip8": "cusip_map",
        }
    )
    out = out.merge(
        merge[["index", "company_name_map", "ticker_map", "sic_map", "cusip_map"]],
        left_index=True,
        right_on="index",
        how="left",
    )
    if "company_name" in out.columns:
        out["company_name"] = out["company_name"].fillna(out["company_name_map"])
    else:
        out["company_name"] = out["company_name_map"]
    if "ticker" in out.columns:
        out["ticker"] = out["ticker"].fillna(out["ticker_map"])
    else:
        out["ticker"] = out["ticker_map"]
    if "sic" in out.columns:
        out["sic"] = out["sic"].fillna(out["sic_map"])
    else:
        out["sic"] = out["sic_map"]
    if "cusip" in out.columns:
        out["cusip"] = out["cusip"].fillna(out["cusip_map"])
    else:
        out["cusip"] = out["cusip_map"]

    out = out.drop(columns=[c for c in ["index", "company_name_map", "ticker_map", "sic_map", "cusip_map"] if c in out.columns])
    return out


def build_fundamentals_master(universe):
    path = DATA_DIR / "fundamentals_quarterly.parquet"
    link_path = CRSP_DIR / "ccmxpf_lnkhist.parquet"
    if not path.exists() or not link_path.exists():
        return None
    df = pd.read_parquet(path)
    link = pd.read_parquet(link_path)
    if "lpermno" not in link.columns:
        link = link.rename(columns={"permno": "lpermno"}) if "permno" in link.columns else link
    df = attach_permno_by_gvkey(df, link)
    filtered = filter_by_universe(df, universe, "datadate", "permno")
    out_path = CURATED_DIR / "fundamentals_master.parquet"
    filtered.to_parquet(out_path, index=False)
    return filtered


def build_buybacks_master(universe):
    path = DATA_DIR / "buybacks_clean.parquet"
    link_path = CRSP_DIR / "ccmxpf_lnkhist.parquet"
    if not path.exists() or not link_path.exists():
        return None
    df = pd.read_parquet(path)
    link = pd.read_parquet(link_path)
    if "lpermno" not in link.columns:
        link = link.rename(columns={"permno": "lpermno"}) if "permno" in link.columns else link
    df = attach_permno_by_gvkey(df, link)
    filtered = filter_by_universe(df, universe, "action_date", "permno")
    out_path = CURATED_DIR / "buybacks_master.parquet"
    filtered.to_parquet(out_path, index=False)
    return filtered


def build_mna_master(universe):
    def first_col(frame, *names):
        for name in names:
            if name in frame.columns:
                return frame[name]
        return None

    def mark_universe_flags(frame, date_col, permno_col, flag_col):
        tmp = frame[[date_col, permno_col]].copy()
        tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
        tmp["month_end"] = tmp[date_col].dt.to_period("M").dt.to_timestamp("M")
        tmp[permno_col] = pd.to_numeric(tmp[permno_col], errors="coerce")

        u = universe[["date", "permno"]].copy()
        u["date"] = pd.to_datetime(u["date"], errors="coerce")
        u = u.rename(columns={"date": "month_end", "permno": permno_col})
        u[permno_col] = pd.to_numeric(u[permno_col], errors="coerce")
        u = u.dropna(subset=["month_end", permno_col]).drop_duplicates()

        u_index = pd.MultiIndex.from_frame(u[["month_end", permno_col]])
        tmp_index = pd.MultiIndex.from_frame(tmp[["month_end", permno_col]])
        frame[flag_col] = tmp_index.isin(u_index)
        return frame

    def load_ciq_identifiers():
        ciq_dir = DATA_DIR / "wrds" / "ciq"
        if not ciq_dir.exists():
            return None

        cached = ciq_dir / "ciq_identifiers_map.parquet"
        if cached.exists():
            cached_df = pd.read_parquet(cached)
            if cached_df is not None and not cached_df.empty:
                return cached_df

        candidates = []
        candidates.extend(sorted(ciq_dir.glob("*ident*.*")))
        candidates.extend(sorted(ciq_dir.glob("*Identifier*.*")))
        candidates = [c for c in candidates if c.suffix in [".parquet", ".csv", ".gz"]]
        if not candidates:
            candidates = [c for c in ciq_dir.iterdir() if c.suffix in [".parquet", ".csv", ".gz"]]
        if not candidates:
            return None

        path = candidates[0]

        def _clean_text(series: pd.Series) -> pd.Series:
            s = series.astype("string")
            s = s.str.strip()
            s = s.where(~s.str.lower().isin(["", "nan", "none", "<na>"]))
            return s

        ciq_chunk = int(os.getenv("CIQ_CHUNK", "2000000"))
        ciq_log_every = int(os.getenv("CIQ_LOG_EVERY", "5000000"))

        if path.suffix != ".parquet":
            header = pd.read_csv(path, nrows=0, compression="infer")
            header_cols = list(header.columns)
            wanted = {
                "companyid", "symboltypecat", "symbolvalue",
                "gvkey", "gvkey_id",
                "cusip", "cusip8", "cusip_8", "cusip9", "cusip_9",
                "isin", "isin_code",
                "ticker", "tic", "symbol", "ticker_symbol",
            }
            usecols = [c for c in header_cols if c.lower() in wanted]

            if {"companyid", "symboltypecat", "symbolvalue"}.issubset({c.lower() for c in usecols}):
                log(f"CIQ identifiers: scanning {path.name} for GVKEY map...")
                total = 0
                gv_map = {}
                next_log = ciq_log_every
                debug = os.getenv("CIQ_DEBUG") == "1"
                ciq_engine = "python" if os.getenv("CIQ_ENGINE") == "python" else "c"
                read_kwargs = dict(
                    usecols=usecols,
                    chunksize=ciq_chunk,
                    compression="infer",
                    dtype=str,
                    engine=ciq_engine,
                )
                if ciq_engine == "c":
                    read_kwargs["low_memory"] = False
                for chunk in pd.read_csv(path, **read_kwargs):
                    total += len(chunk)
                    chunk.columns = [c.lower() for c in chunk.columns]
                    chunk["companyid"] = chunk["companyid"].astype("string").str.strip()
                    chunk["symboltypecat"] = chunk["symboltypecat"].astype("string").str.strip()
                    chunk["symbolvalue"] = chunk["symbolvalue"].astype("string").str.strip()
                    gv = chunk[chunk["symboltypecat"].str.contains("GVKEY", case=False, na=False)].copy()
                    if debug and total == len(chunk):
                        sample_counts = chunk["symboltypecat"].value_counts().head(5).to_dict()
                        log(f"CIQ debug: first chunk types {sample_counts}")
                        log(f"CIQ debug: first chunk gv rows {len(gv):,}")
                    if not gv.empty:
                        gv["gvkey"] = gv["symbolvalue"].str.extract(r"(\d+)", expand=False)
                        gv["gvkey"] = gv["gvkey"].where(~gv["gvkey"].str.lower().isin(["", "nan", "none", "<na>"]))
                        gv["gvkey"] = gv["gvkey"].str.zfill(6)
                        gv = gv.dropna(subset=["companyid", "gvkey"]).drop_duplicates("companyid")
                        gv_map.update(dict(zip(gv["companyid"], gv["gvkey"])))
                    if total >= next_log:
                        log(f"CIQ identifiers: scanned {total:,} rows | gvkey companies {len(gv_map):,}")
                        next_log += ciq_log_every

                if not gv_map:
                    log("CIQ identifiers: no GVKEY rows found; skipping.")
                    return None

                log(f"CIQ identifiers: scanning {path.name} for CUSIP/ISIN/TICKER...")
                total = 0
                next_log = ciq_log_every
                cusip_rows = []
                isin_rows = []
                ticker_rows = []
                for chunk in pd.read_csv(path, **read_kwargs):
                    total += len(chunk)
                    chunk.columns = [c.lower() for c in chunk.columns]
                    chunk["companyid"] = chunk["companyid"].astype("string").str.strip()
                    chunk["symboltypecat"] = chunk["symboltypecat"].astype("string").str.strip()
                    chunk["symbolvalue"] = chunk["symbolvalue"].astype("string").str.strip()
                    chunk = chunk[chunk["companyid"].isin(gv_map)]
                    if chunk.empty:
                        if total >= next_log:
                            log(f"CIQ identifiers: scanned {total:,} rows | mappings {len(cusip_rows)+len(isin_rows)+len(ticker_rows):,}")
                            next_log += ciq_log_every
                        continue
                    chunk["gvkey"] = chunk["companyid"].map(gv_map)

                    cus = chunk[chunk["symboltypecat"].str.contains("CUSIP", case=False, na=False)].copy()
                    if not cus.empty:
                        cus["cusip8"] = cus["symbolvalue"].str.replace(r"[^0-9A-Za-z]", "", regex=True).str.upper().str.slice(0, 8)
                        cus = cus.dropna(subset=["gvkey", "cusip8"])
                        cusip_rows.append(cus[["gvkey", "cusip8"]])

                    isin = chunk[chunk["symboltypecat"].str.contains("ISIN", case=False, na=False)].copy()
                    if not isin.empty:
                        isin = isin.dropna(subset=["gvkey", "symbolvalue"])
                        isin_rows.append(isin[["gvkey", "symbolvalue"]].rename(columns={"symbolvalue": "isin"}))

                    tic = chunk[chunk["symboltypecat"].str.contains("TICKER", case=False, na=False)].copy()
                    if not tic.empty:
                        tic = tic.dropna(subset=["gvkey", "symbolvalue"])
                        ticker_rows.append(tic[["gvkey", "symbolvalue"]].rename(columns={"symbolvalue": "ticker"}))

                    if total >= next_log:
                        log(f"CIQ identifiers: scanned {total:,} rows | mappings {len(cusip_rows)+len(isin_rows)+len(ticker_rows):,}")
                        next_log += ciq_log_every

                out_parts = []
                if cusip_rows:
                    out_parts.append(pd.concat(cusip_rows, ignore_index=True))
                if isin_rows:
                    out_parts.append(pd.concat(isin_rows, ignore_index=True))
                if ticker_rows:
                    out_parts.append(pd.concat(ticker_rows, ignore_index=True))
                if not out_parts:
                    log("CIQ identifiers: no identifier mappings found.")
                    return None
                out = pd.concat(out_parts, ignore_index=True, sort=False)
                out = out.dropna(subset=["gvkey"]).drop_duplicates()
                out.to_parquet(cached, index=False)
                log(f"CIQ identifiers: cached {len(out):,} rows -> {cached}")
                return out

            # Fall back to loading full file if structure doesn't match expected master table
            df = pd.read_csv(path, usecols=usecols, compression="infer", dtype=str, low_memory=False)
        else:
            df = pd.read_parquet(path)

        cols = {c.lower(): c for c in df.columns}

        def pick(*names):
            for name in names:
                if name in cols:
                    return cols[name]
            return None

        gvkey_col = pick("gvkey", "gvkey_id")
        cusip_col = pick("cusip", "cusip8", "cusip_8", "cusip9", "cusip_9")
        isin_col = pick("isin", "isin_code")
        ticker_col = pick("ticker", "tic", "symbol", "ticker_symbol")

        if gvkey_col is None:
            return None

        out = pd.DataFrame()
        out["gvkey"] = df[gvkey_col]
        if cusip_col is not None:
            out["cusip_raw"] = df[cusip_col]
        if isin_col is not None:
            out["isin_raw"] = df[isin_col]
        if ticker_col is not None:
            out["ticker_raw"] = df[ticker_col]

        out["gvkey"] = clean_text(out["gvkey"].astype("string")).str.extract(r"(\d+)", expand=False)
        out["gvkey"] = out["gvkey"].where(~out["gvkey"].str.lower().isin(["", "nan", "none", "<na>"]))
        out["gvkey"] = out["gvkey"].str.zfill(6)

        if "cusip_raw" in out.columns:
            out["cusip8"] = (
                clean_text(out["cusip_raw"])
                .str.replace(r"[^0-9A-Za-z]", "", regex=True)
                .str.upper()
                .str.slice(0, 8)
            )
            out = out.drop(columns=["cusip_raw"])
        if "isin_raw" in out.columns:
            out["isin"] = clean_text(out["isin_raw"]).str.upper()
            out = out.drop(columns=["isin_raw"])
        if "ticker_raw" in out.columns:
            out["ticker"] = clean_text(out["ticker_raw"]).str.upper().str.strip()
            out = out.drop(columns=["ticker_raw"])

        out = out.dropna(subset=["gvkey"]).drop_duplicates()
        out.to_parquet(cached, index=False)
        return out

    if os.getenv("CIQ_BUILD_ONLY") == "1":
        load_ciq_identifiers()
        return None

    def flag_us_entity(frame, side):
        isin_col = f"{side}_isin"
        country_col = f"{side}_country"
        cusip_col = f"{side}_cusip"
        ric_col = f"{side}_ric"

        isin = frame[isin_col].astype("string") if isin_col in frame.columns else pd.Series([pd.NA] * len(frame))
        country = frame[country_col].astype("string") if country_col in frame.columns else pd.Series([pd.NA] * len(frame))
        cusip = frame[cusip_col].astype("string") if cusip_col in frame.columns else pd.Series([pd.NA] * len(frame))
        ric = frame[ric_col].astype("string") if ric_col in frame.columns else pd.Series([pd.NA] * len(frame))

        is_us_isin = isin.str.upper().str.startswith("US", na=False)
        is_us_country = country.str.upper().str.contains(r"UNITED STATES|USA", na=False)
        is_us_cusip = cusip.notna() & ~cusip.str.lower().isin(["", "nan", "none", "<na>"])
        ric_upper = ric.str.upper().fillna("")
        is_us_ric = ric_upper.str.contains(r"\.(N|OQ|A|B|PK|OB|ARCA|P|Q)(\^|$)")

        return is_us_isin | is_us_country | is_us_cusip | is_us_ric

    def validate_permno_date(frame, permno_col, date_col, permco_col=None, valid_flag_col=None):
        msenames_path = CRSP_DIR / "msenames_2000-01-01_to_2024-12-31.parquet"
        if not msenames_path.exists() or permno_col not in frame.columns:
            return frame

        names = pd.read_parquet(msenames_path, columns=["permno", "namedt", "nameendt"])
        names["namedt"] = pd.to_datetime(names["namedt"], errors="coerce")
        names["nameendt"] = pd.to_datetime(names["nameendt"], errors="coerce")
        names = names.dropna(subset=["permno"]).drop_duplicates()
        names = names.groupby("permno", as_index=False).agg(namedt=("namedt", "min"), nameendt=("nameendt", "max"))
        names = names.rename(columns={"permno": permno_col})

        out = frame.copy()
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
        out[permno_col] = pd.to_numeric(out[permno_col], errors="coerce")
        out = out.merge(names, on=permno_col, how="left")

        valid = (out[permno_col].notna()) & (out[date_col] >= out["namedt"]) & (out[date_col] <= out["nameendt"])
        if valid_flag_col is not None:
            out[valid_flag_col] = valid
        out.loc[~valid, permno_col] = pd.NA
        if permco_col is not None and permco_col in out.columns:
            out.loc[~valid, permco_col] = pd.NA
        out = out.drop(columns=["namedt", "nameendt"])
        return out

    def attach_gvkey_from_ciq(frame, side, ciq):
        if ciq is None or ciq.empty:
            return frame
        gvkey_col = f"{side}_gvkey"
        cusip_col = f"{side}_cusip"
        isin_col = f"{side}_isin"
        us_flag_col = f"{side}_us_flag"

        out = frame.copy()
        if gvkey_col not in out.columns:
            out[gvkey_col] = pd.NA

        if cusip_col in out.columns:
            out[f"{side}_cusip8"] = (
                out[cusip_col]
                .astype("string")
                .str.replace(r"[^0-9A-Za-z]", "", regex=True)
                .str.upper()
                .str.slice(0, 8)
            )
        if isin_col in out.columns:
            out[f"{side}_isin_clean"] = clean_text(out[isin_col]).str.upper()

        if us_flag_col in out.columns:
            if f"{side}_cusip8" in out.columns:
                out.loc[~out[us_flag_col], f"{side}_cusip8"] = pd.NA
            if f"{side}_isin_clean" in out.columns:
                out.loc[~out[us_flag_col], f"{side}_isin_clean"] = pd.NA

        if "cusip8" in ciq.columns and f"{side}_cusip8" in out.columns:
            keys = out[f"{side}_cusip8"].dropna().unique().tolist()
            if keys:
                ciq_cus = ciq.loc[ciq["cusip8"].isin(keys), ["cusip8", "gvkey"]].copy()
                if not ciq_cus.empty:
                    ciq_cus["cusip8"] = ciq_cus["cusip8"].astype("string")
                    ciq_cus["gvkey"] = ciq_cus["gvkey"].astype("string")
                    ciq_cus = ciq_cus.drop_duplicates("cusip8", keep="first")
                    cus_map = dict(zip(ciq_cus["cusip8"], ciq_cus["gvkey"]))
                    out[gvkey_col] = out[gvkey_col].fillna(out[f"{side}_cusip8"].map(cus_map))

        if "isin" in ciq.columns and f"{side}_isin_clean" in out.columns:
            keys = out[f"{side}_isin_clean"].dropna().unique().tolist()
            if keys:
                ciq_isin = ciq.loc[ciq["isin"].isin(keys), ["isin", "gvkey"]].copy()
                if not ciq_isin.empty:
                    ciq_isin["isin"] = ciq_isin["isin"].astype("string")
                    ciq_isin["gvkey"] = ciq_isin["gvkey"].astype("string")
                    ciq_isin = ciq_isin.drop_duplicates("isin", keep="first")
                    isin_map = dict(zip(ciq_isin["isin"], ciq_isin["gvkey"]))
                    out[gvkey_col] = out[gvkey_col].fillna(out[f"{side}_isin_clean"].map(isin_map))

        if "ticker" in ciq.columns and f"{side}_ticker" in out.columns:
            keys = out[f"{side}_ticker"].dropna().unique().tolist()
            if keys:
                ciq_tic = ciq.loc[ciq["ticker"].isin([k.upper() for k in keys]), ["ticker", "gvkey"]].copy()
                if not ciq_tic.empty:
                    ciq_tic["ticker"] = ciq_tic["ticker"].astype("string").str.upper()
                    ciq_tic["gvkey"] = ciq_tic["gvkey"].astype("string")
                    ciq_tic = ciq_tic.drop_duplicates("ticker", keep="first")
                    tic_map = dict(zip(ciq_tic["ticker"], ciq_tic["gvkey"]))
                    out[gvkey_col] = out[gvkey_col].fillna(out[f"{side}_ticker"].str.upper().map(tic_map))

        drop_cols = [c for c in [f"{side}_cusip8", f"{side}_isin_clean"] if c in out.columns]
        if drop_cols:
            out = out.drop(columns=drop_cols)
        return out

    def attach_permno_from_gvkey(frame, gvkey_col, date_col, permno_col, permco_col, link):
        if frame is None or frame.empty or link is None or link.empty or gvkey_col not in frame.columns:
            return frame
        link = link.copy()
        if "lpermno" not in link.columns and "permno" in link.columns:
            link = link.rename(columns={"permno": "lpermno"})
        if "lpermco" not in link.columns and "permco" in link.columns:
            link = link.rename(columns={"permco": "lpermco"})
        link["linkdt"] = pd.to_datetime(link["linkdt"], errors="coerce")
        link["linkenddt"] = pd.to_datetime(link["linkenddt"], errors="coerce").fillna(pd.Timestamp("2099-12-31"))

        tmp = frame[[gvkey_col, date_col]].copy()
        tmp = tmp.reset_index()
        tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
        tmp["gvkey"] = tmp[gvkey_col]

        merge = tmp.merge(link, on="gvkey", how="left")
        merge = merge[(merge[date_col] >= merge["linkdt"]) & (merge[date_col] <= merge["linkenddt"])]
        merge = merge.sort_values(["index", "linkdt"], ascending=[True, False])
        merge = merge.drop_duplicates("index", keep="first")

        out = frame.merge(
            merge[["index", "lpermno", "lpermco"]],
            left_index=True,
            right_on="index",
            how="left",
        )
        if permno_col in out.columns:
            out[permno_col] = out[permno_col].fillna(out["lpermno"])
        else:
            out[permno_col] = out["lpermno"]
        if permco_col in out.columns:
            out[permco_col] = out[permco_col].fillna(out["lpermco"])
        else:
            out[permco_col] = out["lpermco"]
        out = out.drop(columns=[c for c in ["index", "lpermno", "lpermco"] if c in out.columns])
        return out

    def attach_permno_from_cusip(frame, cusip_col, date_col, permno_col, permco_col):
        msenames_path = CRSP_DIR / "msenames_2000-01-01_to_2024-12-31.parquet"
        if not msenames_path.exists():
            return frame

        names = pd.read_parquet(
            msenames_path,
            columns=["permno", "permco", "namedt", "nameendt", "ncusip", "cusip"],
        )
        names["namedt"] = pd.to_datetime(names["namedt"], errors="coerce")
        names["nameendt"] = pd.to_datetime(names["nameendt"], errors="coerce")
        names["cusip8"] = (
            names["ncusip"].fillna(names["cusip"]).astype("string").str.replace(r"[^0-9A-Za-z]", "", regex=True).str.upper()
        )
        names["cusip8"] = names["cusip8"].where(
            ~names["cusip8"].str.lower().isin(["", "nan", "none", "<na>"])
        ).str[:8]
        names = names[names["cusip8"].notna()]

        frame = frame.copy()
        frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
        frame["cusip8"] = (
            frame[cusip_col]
            .astype("string")
            .str.replace(r"[^0-9A-Za-z]", "", regex=True)
            .str.upper()
        )
        frame["cusip8"] = frame["cusip8"].where(
            ~frame["cusip8"].str.lower().isin(["", "nan", "none", "<na>"])
        ).str[:8]

        needed = frame["cusip8"].dropna().unique().tolist()
        if not needed:
            return frame
        names = names[names["cusip8"].isin(needed)]
        if names.empty:
            return frame

        merge = (
            frame[["cusip8", date_col]]
            .reset_index()
            .merge(names, on="cusip8", how="left")
        )
        merge = merge[(merge[date_col] >= merge["namedt"]) & (merge[date_col] <= merge["nameendt"])]
        merge = merge.sort_values(["index", "namedt"], ascending=[True, False])
        merge = merge.drop_duplicates("index", keep="first")

        frame = frame.merge(
            merge[["index", "permno", "permco"]],
            left_index=True,
            right_on="index",
            how="left",
        )
        frame[permno_col] = frame[permno_col].fillna(frame["permno"])
        frame[permco_col] = frame[permco_col].fillna(frame["permco"])
        frame = frame.drop(columns=["index", "permno", "permco", "cusip8"])
        return frame

    def attach_permno_from_ticker(frame, ticker_col, date_col, permno_col, permco_col):
        msenames_path = CRSP_DIR / "msenames_2000-01-01_to_2024-12-31.parquet"
        if not msenames_path.exists():
            return frame

        names = pd.read_parquet(
            msenames_path,
            columns=["permno", "permco", "namedt", "nameendt", "ticker", "tsymbol"],
        )
        names["namedt"] = pd.to_datetime(names["namedt"], errors="coerce")
        names["nameendt"] = pd.to_datetime(names["nameendt"], errors="coerce")
        names["ticker_clean"] = (
            names["ticker"].fillna(names["tsymbol"]).astype("string").str.upper().str.strip()
        )
        names["ticker_clean"] = names["ticker_clean"].where(
            ~names["ticker_clean"].str.lower().isin(["", "nan", "none", "<na>"])
        )
        names = names[names["ticker_clean"].notna()]

        frame = frame.copy()
        frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
        frame["ticker_clean"] = (
            frame[ticker_col].astype("string").str.upper().str.strip()
        )
        frame["ticker_clean"] = frame["ticker_clean"].where(
            ~frame["ticker_clean"].str.lower().isin(["", "nan", "none", "<na>"])
        )

        needed = frame["ticker_clean"].dropna().unique().tolist()
        if not needed:
            return frame
        names = names[names["ticker_clean"].isin(needed)]
        if names.empty:
            return frame

        merge = (
            frame[["ticker_clean", date_col]]
            .reset_index()
            .merge(names, on="ticker_clean", how="left")
        )
        merge = merge[(merge[date_col] >= merge["namedt"]) & (merge[date_col] <= merge["nameendt"])]
        merge = merge.sort_values(["index", "namedt"], ascending=[True, False])
        merge = merge.drop_duplicates("index", keep="first")

        frame = frame.merge(
            merge[["index", "permno", "permco"]],
            left_index=True,
            right_on="index",
            how="left",
        )
        frame[permno_col] = frame[permno_col].fillna(frame["permno"])
        frame[permco_col] = frame[permco_col].fillna(frame["permco"])
        frame = frame.drop(columns=["index", "permno", "permco", "ticker_clean"])
        return frame

    def attach_permno_from_ric(frame, ric_col, date_col, permno_col, permco_col):
        msenames_path = CRSP_DIR / "msenames_2000-01-01_to_2024-12-31.parquet"
        if not msenames_path.exists():
            return frame

        names = pd.read_parquet(
            msenames_path,
            columns=["permno", "permco", "namedt", "nameendt", "ticker", "tsymbol"],
        )
        names["namedt"] = pd.to_datetime(names["namedt"], errors="coerce")
        names["nameendt"] = pd.to_datetime(names["nameendt"], errors="coerce")
        names["ticker_clean"] = (
            names["ticker"].fillna(names["tsymbol"]).astype("string").str.upper().str.strip()
        )
        names["ticker_clean"] = names["ticker_clean"].where(
            ~names["ticker_clean"].str.lower().isin(["", "nan", "none", "<na>"])
        )
        names["ticker_compact"] = names["ticker_clean"].str.replace(r"[^0-9A-Z]", "", regex=True)
        names = names[names["ticker_compact"].notna()]

        frame = frame.copy()
        frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
        ric = frame[ric_col].astype("string").str.upper().str.strip()
        ric = ric.where(~ric.str.lower().isin(["", "nan", "none", "<na>"]))
        ric_base = ric.str.split(r"[\\.^]").str[0]
        frame["ric_compact"] = ric_base.str.replace(r"[^0-9A-Z]", "", regex=True)
        frame["ric_compact"] = frame["ric_compact"].where(
            ~frame["ric_compact"].str.lower().isin(["", "nan", "none", "<na>"])
        )

        needed = frame["ric_compact"].dropna().unique().tolist()
        if not needed:
            return frame
        names = names[names["ticker_compact"].isin(needed)]
        if names.empty:
            return frame

        merge = (
            frame[["ric_compact", date_col]]
            .reset_index()
            .merge(names, left_on="ric_compact", right_on="ticker_compact", how="left")
        )
        merge = merge[(merge[date_col] >= merge["namedt"]) & (merge[date_col] <= merge["nameendt"])]
        merge = merge.sort_values(["index", "namedt"], ascending=[True, False])
        merge = merge.drop_duplicates("index", keep="first")

        frame = frame.merge(
            merge[["index", "permno", "permco"]],
            left_index=True,
            right_on="index",
            how="left",
        )
        frame[permno_col] = frame[permno_col].fillna(frame["permno"])
        frame[permco_col] = frame[permco_col].fillna(frame["permco"])
        frame = frame.drop(columns=["index", "permno", "permco", "ric_compact"])
        return frame

    def apply_permid_map(frame):
        map_path = DATA_DIR / "refinitiv" / "permid_map.parquet"
        if not map_path.exists():
            return frame

        perm_map = pd.read_parquet(map_path)
        if "permid" not in perm_map.columns:
            return frame
        perm_map = perm_map.copy()
        perm_map["permid"] = perm_map["permid"].astype(str)

        out = frame.copy()
        for side in ["target", "acquiror"]:
            id_col = f"{side}_id"
            if id_col not in out.columns:
                continue
            out[id_col] = out[id_col].astype(str)
            out.loc[out[id_col].isin(["nan", "None", "<NA>"]), id_col] = pd.NA
            side_map = perm_map.rename(columns={c: f"{side}_{c}_map" for c in perm_map.columns if c != "permid"})
            out = out.merge(side_map, left_on=id_col, right_on="permid", how="left")

            for field in ["ric", "cusip", "isin", "ticker", "name"]:
                existing = f"{side}_{field}"
                mapped = f"{side}_{field}_map"
                if mapped in out.columns:
                    out[existing] = out[existing].fillna(out[mapped])
            if "permid" in out.columns:
                drop_cols = [c for c in out.columns if c.endswith("_map")] + ["permid"]
                out = out.drop(columns=drop_cols)
        return out

    def clean_text(series: pd.Series) -> pd.Series:
        s = series.astype("string")
        s = s.str.strip()
        s = s.where(~s.str.lower().isin(["", "nan", "none", "<na>"]))
        return s

    def cusip_from_isin(series: pd.Series) -> pd.Series:
        s = clean_text(series).str.upper()
        us = s.str.startswith("US", na=False)
        out = pd.Series([pd.NA] * len(s), index=s.index, dtype="string")
        out.loc[us] = s.loc[us].str.slice(2, 11)
        out = out.where(~out.str.lower().isin(["", "nan", "none", "<na>"]))
        return out

    def apply_ric_cusip_map(frame):
        map_path = DATA_DIR / "refinitiv" / "ric_to_cusip_map.parquet"
        if not map_path.exists():
            return frame

        ric_map = pd.read_parquet(map_path)
        if "ric" not in ric_map.columns:
            return frame

        ric_map = ric_map.copy()
        ric_map["ric_clean"] = clean_text(ric_map["ric"]).str.upper()
        for col in ["cusip", "isin", "ticker"]:
            if col in ric_map.columns:
                ric_map[col] = clean_text(ric_map[col]).str.upper()
        ric_map = ric_map.dropna(subset=["ric_clean"]).drop_duplicates("ric_clean")

        out = frame.copy()
        for side in ["target", "acquiror"]:
            ric_col = f"{side}_ric"
            if ric_col not in out.columns:
                continue
            out[f"{side}_ric_clean"] = clean_text(out[ric_col]).str.upper()
            out = out.merge(
                ric_map.rename(
                    columns={c: f"{side}_{c}_map" for c in ric_map.columns if c != "ric_clean"}
                ),
                left_on=f"{side}_ric_clean",
                right_on="ric_clean",
                how="left",
            )
            for field in ["cusip", "isin", "ticker"]:
                mapped = f"{side}_{field}_map"
                if mapped in out.columns:
                    out[f"{side}_{field}"] = out[f"{side}_{field}"].fillna(out[mapped])
            drop_cols = [c for c in out.columns if c.endswith("_map")] + [f"{side}_ric_clean", "ric_clean"]
            out = out.drop(columns=[c for c in drop_cols if c in out.columns])
        return out

    def standardize_refinitiv(frame):
        out = pd.DataFrame(index=frame.index)
        out["source"] = "refinitiv"
        out["source_id"] = first_col(frame, "Instrument")
        out["deal_id"] = out["source_id"]

        ann = first_col(frame, "Date Announced", "Announcement Date", "Announced Date")
        out["event_date"] = pd.to_datetime(ann, errors="coerce")
        out["announce_date"] = out["event_date"]
        comp = first_col(frame, "Date Completed", "Date Effective", "Completion Date", "Completion/Effective Date")
        out["completion_date"] = pd.to_datetime(comp, errors="coerce")

        out["deal_status"] = first_col(frame, "Deal Status")
        out["deal_type"] = first_col(frame, "M&A Type", "Deal Type")
        out["deal_value"] = pd.to_numeric(first_col(frame, "Deal Value", "Deal Value (USD)"), errors="coerce")
        out["deal_value_currency"] = first_col(frame, "Deal Value Currency")
        out["payment_type"] = first_col(frame, "Payment Type")
        out["pct_cash"] = pd.to_numeric(first_col(frame, "Percent Cash", "Pct Cash"), errors="coerce")
        out["pct_stock"] = pd.to_numeric(first_col(frame, "Percent Stock", "Pct Stock"), errors="coerce")
        out["premium_1day"] = pd.to_numeric(first_col(frame, "Premium 1 Day"), errors="coerce")
        out["premium_1week"] = pd.to_numeric(first_col(frame, "Premium 1 Week"), errors="coerce")
        out["premium_4week"] = pd.to_numeric(first_col(frame, "Premium 4 Week", "Premium 4 Weeks"), errors="coerce")
        out["transaction_nature"] = first_col(frame, "Transaction Nature")
        out["deal_synopsis"] = first_col(frame, "Deal Synopsis")

        out["target_id"] = first_col(frame, "Target PermID", "Target ID")
        out["target_name"] = first_col(frame, "Target Full Name", "Target Name", "Target")
        out["target_ticker"] = first_col(frame, "Target Ticker", "Target RIC")
        out["target_ric"] = first_col(frame, "Target RIC")
        out["target_cusip"] = first_col(frame, "Target CUSIP", "Target Cusip", "Target CUSIP 9", "Target CUSIP9")
        out["target_isin"] = first_col(frame, "Target ISIN")
        out["target_gvkey"] = pd.NA
        out["target_permno"] = pd.NA
        out["target_permco"] = pd.NA
        out["target_sic"] = first_col(frame, "Target Primary SIC Code")
        out["target_country"] = first_col(frame, "Target Nation", "Target Country")

        out["acquiror_id"] = first_col(frame, "Acquiror PermID", "Acquiror ID", "Acquirer PermID", "Acquirer ID")
        out["acquiror_name"] = first_col(frame, "Acquiror Full Name", "Acquiror Name", "Acquirer Name", "Acquiror")
        out["acquiror_ticker"] = first_col(frame, "Acquiror Ticker", "Acquirer Ticker", "Acquiror RIC")
        out["acquiror_ric"] = first_col(frame, "Acquiror RIC", "Acquirer RIC")
        out["acquiror_cusip"] = first_col(frame, "Acquiror CUSIP", "Acquiror Cusip", "Acquirer CUSIP", "Acquirer Cusip")
        out["acquiror_isin"] = first_col(frame, "Acquiror ISIN", "Acquirer ISIN")
        out["acquiror_gvkey"] = pd.NA
        out["acquiror_permno"] = pd.NA
        out["acquiror_permco"] = pd.NA
        out["acquiror_sic"] = first_col(frame, "Acquiror Primary SIC Code", "Acquirer Primary SIC Code")
        out["acquiror_country"] = first_col(frame, "Acquiror Nation", "Acquirer Nation", "Acquiror Country")

        out["action_type"] = "mna"
        out["action_subtype"] = out["deal_type"]
        out["universe_filtered"] = False
        out["year"] = out["event_date"].dt.year
        out["source_id"] = out["source_id"].astype("string")
        out["deal_id"] = out["deal_id"].astype("string")
        out["target_id"] = out["target_id"].astype("string")
        out["acquiror_id"] = out["acquiror_id"].astype("string")
        return out

    def standardize_crsp(frame, universe_frame):
        frame = frame.copy()
        frame["action_date"] = pd.to_datetime(frame["action_date"], errors="coerce")
        frame["month_end"] = frame["action_date"].dt.to_period("M").dt.to_timestamp("M")

        u = universe_frame[["date", "permno"]].copy()
        u["date"] = pd.to_datetime(u["date"], errors="coerce")
        u = u.rename(columns={"date": "month_end"})
        u = u.dropna(subset=["month_end", "permno"]).drop_duplicates()
        merged = frame.merge(u, on=["month_end", "permno"], how="left", indicator=True, sort=False)
        in_universe = merged["_merge"].eq("both").to_numpy()

        out = pd.DataFrame(index=frame.index)
        out["source"] = "crsp_delist"
        out["source_id"] = pd.NA
        out["deal_id"] = (
            "crsp_delist:"
            + frame["permno"].astype(str)
            + ":"
            + frame["action_date"].dt.strftime("%Y-%m-%d")
        )

        out["event_date"] = frame["action_date"]
        out["announce_date"] = out["event_date"]
        out["completion_date"] = out["event_date"]

        out["deal_status"] = "Completed"
        out["deal_type"] = frame.get("action_type")
        out["deal_value"] = pd.to_numeric(frame.get("deal_amount"), errors="coerce")
        out["deal_value_currency"] = pd.NA

        out["target_id"] = "permno:" + frame["permno"].astype(str)
        out["target_name"] = frame.get("company_name")
        out["target_ticker"] = frame.get("ticker")
        out["target_permno"] = frame.get("permno")
        out["target_permco"] = frame.get("permco")
        out["target_sic"] = frame.get("sic")
        out["target_country"] = "United States"

        out["acquiror_id"] = pd.NA
        out["acquiror_name"] = pd.NA
        out["acquiror_ticker"] = pd.NA
        out["acquiror_permno"] = pd.NA
        out["acquiror_permco"] = pd.NA
        out["acquiror_sic"] = pd.NA
        out["acquiror_country"] = pd.NA

        out["action_type"] = "acquisition"
        out["action_subtype"] = frame.get("action_type")

        out["universe_filtered"] = in_universe
        out["year"] = out["event_date"].dt.year
        out["source_id"] = out["source_id"].astype("string")
        out["deal_id"] = out["deal_id"].astype("string")
        out["target_id"] = out["target_id"].astype("string")
        out["acquiror_id"] = out["acquiror_id"].astype("string")
        return out

    frames = []
    ciq = load_ciq_identifiers()
    link_path = CRSP_DIR / "ccmxpf_lnkhist.parquet"
    link = pd.read_parquet(link_path) if link_path.exists() else None

    # Refinitiv M&A (deal-level)
    ref_path = DATA_DIR / "refinitiv" / "ma_deals_all.parquet"
    if ref_path.exists():
        ref = pd.read_parquet(ref_path)
        ref = standardize_refinitiv(ref)
        ref = apply_permid_map(ref)
        for col in ["target_cusip", "target_ticker", "acquiror_cusip", "acquiror_ticker", "target_ric", "acquiror_ric"]:
            if col in ref.columns:
                ref[col] = clean_text(ref[col])
        ref = apply_ric_cusip_map(ref)
        for side in ["target", "acquiror"]:
            us_flag = flag_us_entity(ref, side)
            flag_col = f"{side}_us_flag"
            ref[flag_col] = us_flag
            for col in [f"{side}_ticker", f"{side}_ric"]:
                if col in ref.columns:
                    ref.loc[~us_flag, col] = pd.NA
        if "target_cusip" in ref.columns and "target_isin" in ref.columns:
            derived = cusip_from_isin(ref["target_isin"])
            ref["target_cusip"] = ref["target_cusip"].fillna(derived)
        if "acquiror_cusip" in ref.columns and "acquiror_isin" in ref.columns:
            derived = cusip_from_isin(ref["acquiror_isin"])
            ref["acquiror_cusip"] = ref["acquiror_cusip"].fillna(derived)
        ref = attach_permno_from_cusip(ref, "target_cusip", "event_date", "target_permno", "target_permco")
        ref = attach_permno_from_cusip(ref, "acquiror_cusip", "event_date", "acquiror_permno", "acquiror_permco")
        ref = attach_permno_from_ticker(ref, "target_ticker", "event_date", "target_permno", "target_permco")
        ref = attach_permno_from_ticker(ref, "acquiror_ticker", "event_date", "acquiror_permno", "acquiror_permco")
        ref = attach_permno_from_ric(ref, "target_ric", "event_date", "target_permno", "target_permco")
        ref = attach_permno_from_ric(ref, "acquiror_ric", "event_date", "acquiror_permno", "acquiror_permco")
        ref = attach_gvkey_from_ciq(ref, "target", ciq)
        ref = attach_gvkey_from_ciq(ref, "acquiror", ciq)
        if link is not None:
            ref = attach_permno_from_gvkey(ref, "target_gvkey", "event_date", "target_permno", "target_permco", link)
            ref = attach_permno_from_gvkey(ref, "acquiror_gvkey", "event_date", "acquiror_permno", "acquiror_permco", link)
        ref = validate_permno_date(ref, "target_permno", "event_date", "target_permco", "target_permno_valid")
        ref = validate_permno_date(ref, "acquiror_permno", "event_date", "acquiror_permco", "acquiror_permno_valid")
        if "target_permno" in ref.columns:
            mask = ref["target_cusip"].notna() | ref["target_ticker"].notna() | ref["target_ric"].notna()
            ref.loc[~mask, "target_permno"] = pd.NA
        if "acquiror_permno" in ref.columns:
            mask = ref["acquiror_cusip"].notna() | ref["acquiror_ticker"].notna() | ref["acquiror_ric"].notna()
            ref.loc[~mask, "acquiror_permno"] = pd.NA
        ref = mark_universe_flags(ref, "event_date", "target_permno", "target_in_universe")
        ref = mark_universe_flags(ref, "event_date", "acquiror_permno", "acquiror_in_universe")
        ref["universe_filtered"] = ref["target_in_universe"] | ref["acquiror_in_universe"]
        frames.append(ref)

    # CRSP delisting acquisitions (outcomes)
    acq_path = DATA_DIR / "acquisitions_clean.parquet"
    if acq_path.exists():
        acq = pd.read_parquet(acq_path)
        frames.append(standardize_crsp(acq, universe))

    if not frames:
        return None

    combined = pd.concat(frames, ignore_index=True, sort=False)
    out_path = CURATED_DIR / "mna_master.parquet"
    combined.to_parquet(out_path, index=False)
    return combined


def summarize_dataset(name, df, date_col):
    if df is None or df.empty:
        return None
    s = pd.to_datetime(df[date_col], errors="coerce") if date_col in df.columns else None
    if s is not None and s.notna().any():
        return {"dataset": name, "rows": len(df), "min_date": s.min(), "max_date": s.max()}
    return {"dataset": name, "rows": len(df), "min_date": None, "max_date": None}


def log(msg: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}")


def _log_stage(label: str, start_ts: float, df) -> None:
    elapsed = time.time() - start_ts
    if df is None:
        log(f"{label}: skipped (no data) in {elapsed:.1f}s")
        return
    rows = len(df)
    log(f"{label}: {rows:,} rows in {elapsed:.1f}s")


def main():
    if os.getenv("CIQ_BUILD_ONLY") == "1":
        log("Building CIQ identifiers map only...")
        build_mna_master(pd.DataFrame({"date": [], "permno": []}))
        return
    max_rank_raw = os.getenv("UNIVERSE_MAX_RANK", "3000")
    try:
        max_rank = int(max_rank_raw)
    except ValueError:
        max_rank = 3000
    if max_rank <= 0:
        max_rank = None
    log(f"Building universe (max_rank={max_rank_raw})...")
    t0 = time.time()
    universe = build_r3000_proxy(max_rank=max_rank)
    _log_stage("universe", t0, universe)

    log("Building corporate actions master...")
    t0 = time.time()
    corp_actions = build_corporate_actions_master(universe)
    _log_stage("corporate_actions_master", t0, corp_actions)

    log("Building prices master...")
    t0 = time.time()
    prices = build_prices_master(universe)
    _log_stage("prices_master", t0, prices)

    log("Building prices master (full)...")
    t0 = time.time()
    prices_full = build_prices_master_full(universe)
    _log_stage("prices_master_full", t0, prices_full)

    log("Building fundamentals master...")
    t0 = time.time()
    fundamentals = build_fundamentals_master(universe)
    _log_stage("fundamentals_master", t0, fundamentals)

    log("Building buybacks master...")
    t0 = time.time()
    buybacks = build_buybacks_master(universe)
    _log_stage("buybacks_master", t0, buybacks)

    log("Building M&A master...")
    t0 = time.time()
    mna = build_mna_master(universe)
    _log_stage("mna_master", t0, mna)

    summary = []
    summary.append(summarize_dataset("universe_r3000_proxy", universe, "date"))
    summary.append(summarize_dataset("corporate_actions_master", corp_actions, "action_date"))
    summary.append(summarize_dataset("prices_master", prices, "date"))
    summary.append(summarize_dataset("prices_master_full", prices_full, "date"))
    summary.append(summarize_dataset("fundamentals_master", fundamentals, "datadate"))
    summary.append(summarize_dataset("buybacks_master", buybacks, "action_date"))
    summary.append(summarize_dataset("mna_master", mna, "event_date"))

    summary = [s for s in summary if s is not None]
    summary_df = pd.DataFrame(summary)
    summary_path = CURATED_DIR / "master_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    log(f"Saved master summary -> {summary_path}")


if __name__ == "__main__":
    main()
