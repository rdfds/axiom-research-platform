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


def normalize_prices(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    # Some RDP outputs save columns as tuple-like strings: "('RIC', 'Field')"
    import ast

    def parse_tuple_str(val):
        if isinstance(val, str) and val.startswith("(") and val.endswith(")"):
            try:
                parsed = ast.literal_eval(val)
                if isinstance(parsed, tuple) and len(parsed) == 2:
                    return parsed
            except Exception:
                return val
        return val

    parsed_cols = [parse_tuple_str(c) for c in df.columns]
    has_tuple_cols = any(isinstance(c, tuple) for c in parsed_cols)

    if has_tuple_cols:
        # Convert to MultiIndex columns
        tuples = []
        for c in parsed_cols:
            if isinstance(c, tuple):
                tuples.append(c)
            else:
                tuples.append((str(c), ""))
        df = df.copy()
        df.columns = pd.MultiIndex.from_tuples(tuples)

        # Identify date column
        date_col = None
        for col in df.columns:
            if str(col[0]).lower() == "date":
                date_col = col
                break
        if date_col is None:
            raise ValueError("Could not find date column in Refinitiv output.")

        dates = pd.to_datetime(df[date_col], errors="coerce")
        wide = df.drop(columns=[date_col])
        wide.index = dates
        wide.index.name = "date"

        # Stack RIC level into rows => columns become field names
        long = wide.stack(level=0, future_stack=True).reset_index().rename(columns={"level_1": "ric"})

        close_col = None
        tr_col = None
        for c in long.columns:
            if isinstance(c, str) and c.upper() in ("TR.CLOSEPRICE", "CLOSEPRICE", "PRICE CLOSE", "CLOSE PRICE", "TR.PRICECLOSE"):
                close_col = c
            if isinstance(c, str) and c.upper() in ("TR.TOTRETURN", "TOTRETURN", "TOTAL RETURN", "TR.TOTALRETURN"):
                tr_col = c

        if close_col is None:
            raise ValueError("Close price column not found in Refinitiv output.")

        out = pd.DataFrame()
        out["date"] = pd.to_datetime(long["date"], errors="coerce")
        out["ric"] = long["ric"].astype("string")
        out["prc"] = pd.to_numeric(long[close_col], errors="coerce")

        if tr_col is not None:
            out["total_return_index"] = pd.to_numeric(long[tr_col], errors="coerce")
            out = out.sort_values(["ric", "date"])
            if out["total_return_index"].notna().any():
                out["ret"] = out.groupby("ric")["total_return_index"].pct_change()
            else:
                out["ret"] = out.groupby("ric")["prc"].pct_change()
        else:
            out = out.sort_values(["ric", "date"])
            out["ret"] = out.groupby("ric")["prc"].pct_change()

    else:
        cols = {str(c).lower(): c for c in df.columns}
        date_col = cols.get("date") or cols.get("datetime") or cols.get("index")
        inst_col = cols.get("instrument") or cols.get("ric")

        close_col = None
        tr_col = None
        for c in df.columns:
            if str(c).upper() in ("TR.CLOSEPRICE", "CLOSEPRICE", "PRICE CLOSE", "CLOSE PRICE", "TR.PRICECLOSE"):
                close_col = c
            if str(c).upper() in ("TR.TOTRETURN", "TOTRETURN", "TOTAL RETURN", "TR.TOTALRETURN"):
                tr_col = c

        if date_col is None or inst_col is None or close_col is None:
            raise ValueError("Unexpected price data columns from Refinitiv.")

        out = pd.DataFrame()
        out["date"] = pd.to_datetime(df[date_col], errors="coerce")
        out["ric"] = df[inst_col].astype("string")
        out["prc"] = pd.to_numeric(df[close_col], errors="coerce")

        if tr_col is not None:
            out["total_return_index"] = pd.to_numeric(df[tr_col], errors="coerce")
            out = out.sort_values(["ric", "date"])
            if out["total_return_index"].notna().any():
                out["ret"] = out.groupby("ric")["total_return_index"].pct_change()
            else:
                out["ret"] = out.groupby("ric")["prc"].pct_change()
        else:
            out = out.sort_values(["ric", "date"])
            out["ret"] = out.groupby("ric")["prc"].pct_change()

    # Keep only 2025
    out = out[(out["date"] >= "2025-01-01") & (out["date"] <= "2025-12-31")]
    return out


def merge_into_prices_monthly(mapped: pd.DataFrame) -> None:
    prices_path = DATA_DIR / "prices_monthly.parquet"
    if not prices_path.exists():
        mapped.to_parquet(prices_path, index=False)
        log(f"Created {prices_path}")
        return

    existing = pd.read_parquet(prices_path)
    combined = pd.concat([existing, mapped], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined = combined.drop_duplicates(subset=["gvkey", "date"], keep="first")
    combined.to_parquet(prices_path, index=False)
    log(f"Updated {prices_path}")


def main():
    log("Connecting to Refinitiv...")
    rd.open_session()

    try:
        mapping = build_permno_ric_map()
        mapping = attach_gvkey(mapping)
        mapping = mapping.dropna(subset=["ric"])

        tickers = mapping["ric"].dropna().unique().tolist()
        log(f"RICs to pull: {len(tickers):,}")

        raw_out = REF_DIR / "prices_monthly_rdp_2025.parquet"
        if SKIP_PULL and raw_out.exists():
            log(f"Using existing raw file -> {raw_out}")
            raw = pd.read_parquet(raw_out)
        else:
            raw = extract_prices(tickers)
        if raw is None or raw.empty:
            log("No price rows returned.")
            return

        raw.to_parquet(raw_out, index=False)
        log(f"Saved raw RDP prices -> {raw_out}")

        norm = normalize_prices(raw)
        mapped = norm.merge(mapping, left_on="ric", right_on="ric", how="left")
        mapped = mapped.dropna(subset=["gvkey"])
        mapped["source"] = "refinitiv_rdp"

        mapped_out = DATA_DIR / "prices_monthly_rdp_2025.parquet"
        mapped.to_parquet(mapped_out, index=False)
        log(f"Saved mapped prices -> {mapped_out}")

        if MERGE:
            merge_into_prices_monthly(mapped[["gvkey", "date", "prc", "ret", "source"]])

    finally:
        rd.close_session()
        log("Refinitiv session closed.")


if __name__ == "__main__":
    main()
