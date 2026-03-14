#!/usr/bin/env python
"""
Pull Refinitiv (RDP) monthly prices for 2025 and map to CRSP gvkey/permno.

Outputs:
  data/refinitiv/prices_monthly_rdp_2025.parquet   (raw, RIC-keyed)
  data/prices_monthly_rdp_2025.parquet             (mapped, gvkey/permno keyed)

Optional:
  MERGE_INTO_PRICES_MONTHLY=1 will append into data/prices_monthly.parquet
  (dedupe on gvkey + date).
"""

import os
import time
from pathlib import Path
import pandas as pd
import refinitiv.data as rd


DATA_DIR = Path(__file__).parent.parent / "data"
CRSP_DIR = DATA_DIR / "wrds" / "crsp"
REF_DIR = DATA_DIR / "refinitiv"
REF_DIR.mkdir(parents=True, exist_ok=True)

RDP_START = os.getenv("RDP_START", "2024-12-01")
RDP_END = os.getenv("RDP_END", "2025-12-31")
UNIVERSE_DATE = os.getenv("RDP_UNIVERSE_DATE", "2024-12-31")
BATCH_SIZE = int(os.getenv("RDP_BATCH", "50"))
SLEEP = float(os.getenv("RDP_SLEEP", "0.2"))
SPLIT_ON_ERROR = os.getenv("RDP_SPLIT_ON_ERROR", "1") == "1"
SKIP_PULL = os.getenv("RDP_SKIP_PULL", "0") == "1"

MERGE = os.getenv("MERGE_INTO_PRICES_MONTHLY", "0") == "1"


def log(msg: str) -> None:
    print(msg, flush=True)


def pick_best_ric(group: pd.DataFrame) -> pd.DataFrame:
    def rank_ric(ric: str) -> int:
        if not isinstance(ric, str):
            return 99
        ric = ric.upper()
        if ric.endswith(".N"):
            return 0
        if ric.endswith(".OQ"):
            return 1
        if ric.endswith(".Q"):
            return 2
        if ric.endswith(".A"):
            return 3
        if ric.endswith(".K"):
            return 4
        if ric.endswith(".P"):
            return 5
        return 9

    grp = group.copy()
    grp["ric_rank"] = grp["ric"].map(rank_ric)
    grp = grp.sort_values(["ric_rank", "ric"])
    return grp.head(1)


def build_permno_ric_map() -> pd.DataFrame:
    universe_path = DATA_DIR / "curated" / "universe_r3000_proxy.parquet"
    names_path = CRSP_DIR / "msenames_2000-01-01_to_2026-12-31.parquet"
    ric_map_path = REF_DIR / "ric_to_cusip_map.parquet"

    if not universe_path.exists():
        raise FileNotFoundError("Missing universe_r3000_proxy.parquet")
    if not names_path.exists():
        raise FileNotFoundError("Missing msenames_2000-01-01_to_2026-12-31.parquet")
    if not ric_map_path.exists():
        raise FileNotFoundError("Missing ric_to_cusip_map.parquet")

    universe = pd.read_parquet(universe_path, columns=["date", "permno"])
    universe["date"] = pd.to_datetime(universe["date"], errors="coerce")
    target_date = pd.to_datetime(UNIVERSE_DATE)
    u = universe[universe["date"] == target_date][["permno"]].dropna().drop_duplicates()
    log(f"Universe date {UNIVERSE_DATE}: {len(u):,} permnos")

    names = pd.read_parquet(
        names_path,
        columns=["permno", "namedt", "nameendt", "ncusip", "cusip", "ticker"],
    )
    names["namedt"] = pd.to_datetime(names["namedt"], errors="coerce")
    names["nameendt"] = pd.to_datetime(names["nameendt"], errors="coerce")
    names["cusip8"] = (
        names["ncusip"]
        .fillna(names["cusip"])
        .astype("string")
        .str.replace(r"[^0-9A-Za-z]", "", regex=True)
        .str.upper()
        .str[:8]
    )
    names = names[names["cusip8"].notna()]

    names = names.merge(u, on="permno", how="inner")
    names = names[(target_date >= names["namedt"]) & (target_date <= names["nameendt"])]
    names = names.drop_duplicates("permno", keep="first")

    ric_map = pd.read_parquet(ric_map_path)
    ric_map["cusip8"] = (
        ric_map["cusip"]
        .astype("string")
        .str.replace(r"[^0-9A-Za-z]", "", regex=True)
        .str.upper()
        .str[:8]
    )
    ric_map = ric_map[ric_map["cusip8"].notna()]

    merged = names.merge(ric_map[["ric", "cusip8", "ticker"]], on="cusip8", how="left")
    merged = merged.dropna(subset=["ric"])

    # Prefer primary exchange RICs; avoid groupby.apply deprecation
    def rank_ric(ric: str) -> int:
        if not isinstance(ric, str):
            return 99
        ric = ric.upper()
        if ric.endswith(".N"):
            return 0
        if ric.endswith(".OQ"):
            return 1
        if ric.endswith(".Q"):
            return 2
        if ric.endswith(".A"):
            return 3
        if ric.endswith(".K"):
            return 4
        if ric.endswith(".P"):
            return 5
        return 9

    merged["ric_rank"] = merged["ric"].map(rank_ric)
    merged = merged.sort_values(["permno", "ric_rank", "ric"])
    merged = merged.drop_duplicates("permno", keep="first")

    # Consolidate ticker columns (from names vs ric map)
    if "ticker" not in merged.columns:
        if "ticker_y" in merged.columns and "ticker_x" in merged.columns:
            merged["ticker"] = merged["ticker_y"].fillna(merged["ticker_x"])
        elif "ticker_y" in merged.columns:
            merged["ticker"] = merged["ticker_y"]
        elif "ticker_x" in merged.columns:
            merged["ticker"] = merged["ticker_x"]
    coverage = merged["permno"].nunique()
    log(f"Mapped permnos to RICs: {coverage:,}")

    cols = ["permno", "cusip8", "ric"]
    if "ticker" in merged.columns:
        cols.append("ticker")
    out = merged[cols].copy()
    out = out[~out["ric"].astype("string").str.contains(r"\\^", na=False)]
    return out


def attach_gvkey(map_df: pd.DataFrame) -> pd.DataFrame:
    link_path = CRSP_DIR / "ccmxpf_lnkhist.parquet"
    if not link_path.exists():
        return map_df

    link = pd.read_parquet(link_path, columns=["gvkey", "lpermno", "linkdt", "linkenddt"])
    link["linkdt"] = pd.to_datetime(link["linkdt"], errors="coerce")
    link["linkenddt"] = pd.to_datetime(link["linkenddt"], errors="coerce").fillna(pd.Timestamp("2099-12-31"))

    target_date = pd.to_datetime(UNIVERSE_DATE)
    tmp = map_df.merge(link, left_on="permno", right_on="lpermno", how="left")
    tmp = tmp[(target_date >= tmp["linkdt"]) & (target_date <= tmp["linkenddt"])]
    tmp = tmp.drop_duplicates("permno", keep="first")
    return tmp.drop(columns=["lpermno", "linkdt", "linkenddt"])


def extract_prices(tickers: list) -> pd.DataFrame:
    all_prices = []
    field_sets = [
        ["TR.CLOSEPRICE", "TR.TOTRETURN"],
        ["TR.PRICECLOSE", "TR.TOTRETURN"],
        ["TR.CLOSEPRICE"],
        ["TR.PRICECLOSE"],
    ]

    def try_fetch(batch, fields):
        try:
            data = rd.get_history(
                universe=batch,
                fields=fields,
                start=RDP_START,
                end=RDP_END,
                interval="monthly",
            )
            return data, None
        except Exception as e:
            return None, e

    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i:i + BATCH_SIZE]
        log(f"Pulling batch {i//BATCH_SIZE + 1}/{(len(tickers)-1)//BATCH_SIZE + 1} ...")
        data = None
        err = None
        for fields in field_sets:
            data, err = try_fetch(batch, fields)
            if data is not None and len(data) > 0:
                break
        if data is not None and len(data) > 0:
            all_prices.append(data.reset_index())
        else:
            if err:
                log(f"  Batch error: {err}")
            if SPLIT_ON_ERROR:
                for ric in batch:
                    for fields in field_sets:
                        single, _ = try_fetch([ric], fields)
                        if single is not None and len(single) > 0:
                            all_prices.append(single.reset_index())
                            break
        time.sleep(SLEEP)

    if not all_prices:
        return pd.DataFrame()

    combined = pd.concat(all_prices, ignore_index=True)
    return combined


