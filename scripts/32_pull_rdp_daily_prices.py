#!/usr/bin/env python
"""
Pull Refinitiv (RDP) daily prices and map to permno by date.

Writes partitioned parquet to:
  data/warehouse/warehouse_prices_daily_rdp/year=YYYY/part_*.parquet

Env:
  RDP_START=2000-01-01
  RDP_END=2025-12-31
  RDP_BATCH=50
  RDP_SLEEP=0.2
  RDP_SPLIT_ON_ERROR=1
"""

import os
import time
import uuid
import concurrent.futures as futures
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import refinitiv.data as rd

DATA_DIR = Path(__file__).parent.parent / "data"
CRSP_DIR = DATA_DIR / "wrds" / "crsp"
REF_DIR = DATA_DIR / "refinitiv"
OUT_DIR = DATA_DIR / "warehouse" / "warehouse_prices_daily_rdp"

RDP_START = os.getenv("RDP_START", "2000-01-01")
RDP_END = os.getenv("RDP_END", datetime.utcnow().strftime("%Y-%m-%d"))
RDP_UNIVERSE_DATE = os.getenv("RDP_UNIVERSE_DATE", RDP_END)
BATCH_SIZE = int(os.getenv("RDP_BATCH", "50"))
SLEEP = float(os.getenv("RDP_SLEEP", "0.2"))
SPLIT_ON_ERROR = os.getenv("RDP_SPLIT_ON_ERROR", "1") == "1"
RDP_TIMEOUT = float(os.getenv("RDP_TIMEOUT", "120"))
RDP_LIMIT = int(os.getenv("RDP_LIMIT", "0"))
RDP_DEBUG = os.getenv("RDP_DEBUG", "0") == "1"
RDP_RELAX_NAME_FILTER = os.getenv("RDP_RELAX_NAME_FILTER", "0") == "1"


def log(msg: str) -> None:
    print(msg, flush=True)


def year_chunks(start: str, end: str):
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    for year in range(start_dt.year, end_dt.year + 1):
        chunk_start = max(pd.Timestamp(year=year, month=1, day=1), start_dt)
        chunk_end = min(pd.Timestamp(year=year, month=12, day=31), end_dt)
        yield chunk_start, chunk_end


def load_ric_map() -> pd.DataFrame:
    ric_map_path = REF_DIR / "ric_to_cusip_map.parquet"
    if not ric_map_path.exists():
        raise FileNotFoundError("Missing ric_to_cusip_map.parquet")
    ric_map = pd.read_parquet(ric_map_path)
    ric_map["ric"] = ric_map["ric"].astype("string").str.upper().str.strip()
    ric_map["cusip8"] = (
        ric_map["cusip"]
        .astype("string")
        .str.replace(r"[^0-9A-Za-z]", "", regex=True)
        .str.upper()
        .str[:8]
    )
    ric_map = ric_map[ric_map["cusip8"].notna()]
    ric_map = ric_map.drop_duplicates("ric")
    return ric_map[["ric", "cusip8", "ticker"]]


def load_names() -> pd.DataFrame:
    names_path = CRSP_DIR / "msenames_2000-01-01_to_2026-12-31.parquet"
    if not names_path.exists():
        raise FileNotFoundError("Missing msenames_2000-01-01_to_2026-12-31.parquet")
    names = pd.read_parquet(
        names_path,
        columns=["permno", "namedt", "nameendt", "ncusip", "cusip"],
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
    # If CRSP coverage ends before the RDP pull end date, extend the last-known
    # nameendt to cover the pull range so 2025+ dates can still map.
    max_end = names["nameendt"].max()
    target_end = pd.to_datetime(RDP_END, errors="coerce")
    if pd.notna(max_end) and pd.notna(target_end) and target_end > max_end:
        names.loc[names["nameendt"] == max_end, "nameendt"] = target_end
    return names[["permno", "namedt", "nameendt", "cusip8"]]


def filter_ric_map(ric_map: pd.DataFrame, names: pd.DataFrame) -> pd.DataFrame:
    universe_date = pd.to_datetime(RDP_UNIVERSE_DATE, errors="coerce")
    if pd.isna(universe_date):
        return ric_map
    max_end = names["nameendt"].max()
    if pd.notna(max_end) and universe_date > max_end:
        universe_date = max_end
    active = names[(names["namedt"] <= universe_date) & (names["nameendt"] >= universe_date)]
    active_cusips = set(active["cusip8"].dropna().unique().tolist())
    filtered = ric_map[ric_map["cusip8"].isin(active_cusips)].copy()
    if filtered.empty:
        return ric_map
    return filtered


FIELD_SETS = [
    ["TR.OPENPRICE", "TR.HIGHPRICE", "TR.LOWPRICE", "TR.CLOSEPRICE", "TR.VOLUME", "TR.TOTRETURN"],
    ["TR.OPENPRICE", "TR.HIGHPRICE", "TR.LOWPRICE", "TR.PRICECLOSE", "TR.VOLUME", "TR.TOTRETURN"],
    ["TR.CLOSEPRICE", "TR.VOLUME", "TR.TOTRETURN"],
    ["TR.PRICECLOSE", "TR.VOLUME", "TR.TOTRETURN"],
    ["TR.CLOSEPRICE"],
    ["TR.PRICECLOSE"],
]


def normalize_prices(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    import ast
    import re

    ric_from_columns_name = None
    if getattr(df, "columns", None) is not None and df.columns.name:
        ric_from_columns_name = str(df.columns.name).upper().strip()
    if RDP_DEBUG:
        log(f"    raw columns name: {getattr(df.columns, 'name', None)}")
        try:
            log(f"    raw head:\n{df.head(2).to_string()}")
            log(f"    raw dtypes:\n{df.dtypes}")
        except Exception:
            pass

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
        tuples = []
        for c in parsed_cols:
            if isinstance(c, tuple):
                tuples.append(c)
            else:
                tuples.append((str(c), ""))
        df = df.copy()
        df.columns = pd.MultiIndex.from_tuples(tuples)

        # Determine which level is RIC vs field
        level0 = df.columns.get_level_values(0).unique()
        level1 = df.columns.get_level_values(1).unique()

        def ric_score(values):
            vals = [str(v) for v in values if v is not None]
            if not vals:
                return 0.0
            hits = 0
            for v in vals:
                v = v.upper()
                if re.match(r"^[A-Z0-9]{1,6}\\.[A-Z0-9]+$", v):
                    hits += 1
            return hits / len(vals)

        ric_level = 0 if ric_score(level0) >= ric_score(level1) else 1

        date_col = None
        for col in df.columns:
            if str(col[0]).lower() in ("date", "datetime"):
                date_col = col
                break

        if date_col is None:
            # Some outputs put date on the index (often as strings)
            idx_dates = pd.to_datetime(df.index, errors="coerce")
            if idx_dates.notna().any():
                dates = idx_dates
                wide = df.copy()
            else:
                raise ValueError("Could not find date column in Refinitiv output.")
        else:
            dates = pd.to_datetime(df[date_col], errors="coerce")
            wide = df.drop(columns=[date_col])

        wide.index = dates
        wide.index.name = "date"

        long = (
            wide.stack(level=ric_level, future_stack=True)
            .reset_index()
            .rename(columns={"level_1": "ric"})
        )
    else:
        cols = {str(c).lower(): c for c in df.columns}
        date_col = cols.get("date") or cols.get("datetime") or cols.get("index")
        inst_col = cols.get("instrument") or cols.get("ric")
        long = df.copy()
        single_ric_handled = False
        # Single-RIC response: columns.name holds the RIC; df is indexed by date.
        if inst_col is None and ric_from_columns_name and re.match(r"^[A-Z0-9]{1,6}\\.[A-Z0-9]+$", ric_from_columns_name):
            idx_dates = pd.to_datetime(df.index, errors="coerce")
            if idx_dates.notna().any():
                long = long.reset_index()
                # Ensure the first column (index) is named 'date'
                if long.columns.size > 0:
                    long = long.rename(columns={long.columns[0]: "date"})
                long["ric"] = ric_from_columns_name
                date_col = "date"
                inst_col = "ric"
                single_ric_handled = True
        if date_col is None:
            idx_dates = pd.to_datetime(df.index, errors="coerce")
            if idx_dates.notna().any():
                long = long.reset_index().rename(columns={"index": "date"})
                date_col = "date"
            else:
                raise ValueError(f"Unexpected Refinitiv output columns: {list(df.columns)}")
        if inst_col is None and not single_ric_handled:
            # Try to detect instrument column by name heuristic
            for c in df.columns:
                name = str(c).lower()
                if "ric" in name or "instrument" in name or "symbol" in name:
                    inst_col = c
                    break
            if inst_col is None and ric_from_columns_name:
                # Single-RIC response: columns.name often holds the RIC
                if re.match(r"^[A-Z0-9]{1,6}\.[A-Z0-9]+$", ric_from_columns_name):
                    long["ric"] = ric_from_columns_name
                else:
                    raise ValueError(f"Unexpected Refinitiv output columns: {list(df.columns)}")
            elif inst_col is None:
                raise ValueError(f"Unexpected Refinitiv output columns: {list(df.columns)}")
        if date_col not in long.columns:
            idx_dates = pd.to_datetime(df.index, errors="coerce")
            if idx_dates.notna().any():
                long = long.reset_index().rename(columns={"index": "date"})
                date_col = "date"
            else:
                raise ValueError(f"Unexpected Refinitiv output columns: {list(df.columns)}")
        long["date"] = pd.to_datetime(long[date_col], errors="coerce")
        if long["date"].notna().any() and long["date"].min() < pd.Timestamp("1980-01-01"):
            idx_dates = pd.to_datetime(df.index, errors="coerce")
            if idx_dates.notna().any():
                long["date"] = idx_dates.values
        if "ric" not in long.columns:
            long["ric"] = long[inst_col].astype("string").str.upper().str.strip()

    def pick_exact(colnames):
        for name in colnames:
            for c in long.columns:
                if str(c).upper() == name:
                    return c
        return None

    def pick_contains(tokens):
        for c in long.columns:
            name = str(c).upper()
            if all(tok in name for tok in tokens):
                return c
        return None

    def pick_close_fallback():
        exclude_tokens = ["OPEN", "HIGH", "LOW", "VOLUME", "TOTRETURN", "TOTALRETURN", "RETURN", "BID", "ASK"]
        candidates = []
        for c in long.columns:
            name = str(c).upper()
            if name in ("DATE", "DATETIME", "RIC", "INSTRUMENT"):
                continue
            if any(tok in name for tok in exclude_tokens):
                continue
            # Prefer numeric columns
            if pd.api.types.is_numeric_dtype(long[c]):
                candidates.append(c)
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        # Pick column with most non-null values
        counts = {c: long[c].notna().sum() for c in candidates}
        return max(counts, key=counts.get)

    close_col = pick_exact(["TR.CLOSEPRICE", "CLOSEPRICE", "PRICE CLOSE", "CLOSE PRICE", "TR.PRICECLOSE", "CLOSE", "PX_CLOSE"])
    if close_col is None:
        close_col = pick_contains(["CLOSE", "PRICE"]) or pick_contains(["CLOSE"])
    if close_col is None:
        close_col = pick_close_fallback()

    open_col = pick_exact(["TR.OPENPRICE", "OPENPRICE", "PRICE OPEN", "OPEN"])
    if open_col is None:
        open_col = pick_contains(["OPEN"])

    high_col = pick_exact(["TR.HIGHPRICE", "HIGHPRICE", "HIGH"])
    if high_col is None:
        high_col = pick_contains(["HIGH"])

    low_col = pick_exact(["TR.LOWPRICE", "LOWPRICE", "LOW"])
    if low_col is None:
        low_col = pick_contains(["LOW"])

    vol_col = pick_exact(["TR.VOLUME", "VOLUME"])
    if vol_col is None:
        vol_col = pick_contains(["VOLUME"])

    tr_col = pick_exact(["TR.TOTRETURN", "TOTAL RETURN", "TR.TOTALRETURN"])
    if tr_col is None:
        tr_col = pick_contains(["TOTRETURN"]) or pick_contains(["TOTAL", "RETURN"])

    if RDP_DEBUG:
        log(f"    normalize: columns={list(long.columns)}")
        log(f"    normalize: close_col={close_col} open_col={open_col} high_col={high_col} low_col={low_col} vol_col={vol_col} tr_col={tr_col}")

    if close_col is None:
        raise ValueError(f"Close price column not found. Columns: {list(long.columns)}")

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(long["date"], errors="coerce")
    out["ric"] = long["ric"].astype("string").str.upper().str.strip()
    if ric_from_columns_name:
        out["ric"] = ric_from_columns_name
    if RDP_DEBUG:
        sample_ric = out["ric"].dropna().astype("string").unique().tolist()[:5]
        log(f"    normalize: sample ric values={sample_ric}")
    out["open"] = pd.to_numeric(long[open_col], errors="coerce") if open_col else np.nan
    out["high"] = pd.to_numeric(long[high_col], errors="coerce") if high_col else np.nan
    out["low"] = pd.to_numeric(long[low_col], errors="coerce") if low_col else np.nan
    out["close"] = pd.to_numeric(long[close_col], errors="coerce")
    out["volume"] = pd.to_numeric(long[vol_col], errors="coerce") if vol_col else np.nan

    if tr_col is not None:
        out["total_return_index"] = pd.to_numeric(long[tr_col], errors="coerce")
    else:
        out["total_return_index"] = np.nan

    out = out.sort_values(["ric", "date"])
    if out["total_return_index"].notna().any():
        out["ret"] = out.groupby("ric")["total_return_index"].pct_change()
    else:
        out["ret"] = out.groupby("ric")["close"].pct_change()

    return out


def map_to_permno(prices: pd.DataFrame, ric_map: pd.DataFrame, names: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        return prices
    if RDP_DEBUG:
        sample_prices = prices["ric"].dropna().astype("string").str.upper().str.strip().unique().tolist()[:5]
        sample_map = ric_map["ric"].dropna().astype("string").str.upper().str.strip().unique().tolist()[:5]
        log(f"    map: sample prices rics={sample_prices}")
        log(f"    map: sample map rics={sample_map}")
    merged = prices.merge(ric_map, on="ric", how="left")
    if RDP_DEBUG:
        log(f"    map: start {len(prices):,} -> after ric_map {len(merged):,} | cusip8 missing: {merged['cusip8'].isna().mean():.2%}")
    merged = merged.dropna(subset=["cusip8"])
    merged = merged.merge(names, on="cusip8", how="left")
    if RDP_DEBUG:
        log(f"    map: after names merge {len(merged):,} | permno missing: {merged['permno'].isna().mean():.2%}")
        log(f"    map: name date null pct namedt={merged['namedt'].isna().mean():.2%} nameendt={merged['nameendt'].isna().mean():.2%}")
        log(f"    map: date sample min/max={merged['date'].min()} / {merged['date'].max()}")
        if "namedt" in merged.columns and "nameendt" in merged.columns:
            log(f"    map: namedt sample min/max={merged['namedt'].min()} / {merged['nameendt'].max()}")
    if not RDP_RELAX_NAME_FILTER:
        # Keep rows even if name date bounds are missing
        lower_ok = merged["namedt"].isna() | (merged["date"] >= merged["namedt"])
        upper_ok = merged["nameendt"].isna() | (merged["date"] <= merged["nameendt"])
        merged = merged[lower_ok & upper_ok]
        if RDP_DEBUG:
            log(f"    map: after date filter {len(merged):,}")
    merged = merged.sort_values(["ric", "date", "nameendt"])
    merged = merged.drop_duplicates(subset=["ric", "date"], keep="last")
    merged = merged.dropna(subset=["permno"])
    merged["permno"] = merged["permno"].astype("Int64")
    return merged


def write_partitioned(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    df = df.copy()
    df["event_time"] = pd.to_datetime(df["date"])
    df["available_time"] = df["event_time"] + pd.Timedelta(hours=16)
    df["entity_id"] = df["permno"].astype("Int64").astype("string")
    df["security_id"] = df["entity_id"]
    df["company_id"] = None
    df["adjusted_close"] = df["close"]
    df["source_system"] = "refinitiv_rdp"
    df["ingestion_time"] = datetime.utcnow()
    df["raw_payload_hash"] = pd.util.hash_pandas_object(df[["ric", "date", "close"]], index=False).map(lambda x: f"{x:016x}")
    df["version_id"] = pd.util.hash_pandas_object(df[["entity_id", "date", "raw_payload_hash"]], index=False).map(lambda x: f"{x:016x}")
    df["upstream_version_ids"] = None
    df["quality_flags"] = None

    out_cols = [
        "source_system",
        "entity_id",
        "company_id",
        "security_id",
        "event_time",
        "available_time",
        "ingestion_time",
        "version_id",
        "raw_payload_hash",
        "upstream_version_ids",
        "quality_flags",
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
        "total_return_index",
        "ret",
        "ric",
        "cusip8",
        "permno",
    ]
    for col in out_cols:
        if col not in df.columns:
            df[col] = np.nan
    df = df[out_cols]

    df["year"] = df["event_time"].dt.year.astype("Int64")
    rows = 0
    for year, ydf in df.groupby("year"):
        if pd.isna(year):
            continue
        year_dir = OUT_DIR / f"year={int(year)}"
        year_dir.mkdir(parents=True, exist_ok=True)
        part_path = year_dir / f"part_{uuid.uuid4().hex}.parquet"
        ydf.drop(columns=["year"]).to_parquet(part_path, index=False)
        rows += len(ydf)
    return rows


def extract_prices_batch(batch, start, end):
    last_err = None
    for fields in FIELD_SETS:
        try:
            with futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(
                    rd.get_history,
                    universe=batch,
                    fields=fields,
                    start=start,
                    end=end,
                    interval="daily",
                )
                data = fut.result(timeout=RDP_TIMEOUT)
            if data is not None and len(data) > 0:
                return data
        except Exception as e:
            last_err = e
            continue
    raise last_err if last_err is not None else RuntimeError("No data returned")


def main():
    ric_map = load_ric_map()
    names = load_names()
    ric_map = filter_ric_map(ric_map, names)
    rics = ric_map["ric"].dropna().unique().tolist()
    if RDP_LIMIT and RDP_LIMIT > 0:
        rics = rics[:RDP_LIMIT]
    log(f"RICs to pull: {len(rics):,}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rd.open_session()

    try:
        for start, end in year_chunks(RDP_START, RDP_END):
            log(f"Year {start.year}: {start.date()} -> {end.date()}")
            for i in range(0, len(rics), BATCH_SIZE):
                batch = rics[i:i + BATCH_SIZE]
                log(f"  Batch {i//BATCH_SIZE + 1}/{(len(rics)-1)//BATCH_SIZE + 1}")
                try:
                    raw = extract_prices_batch(batch, start, end)
                except Exception as e:
                    log(f"    Batch error: {e}")
                    if not SPLIT_ON_ERROR:
                        continue
                    for ric in batch:
                        try:
                            raw = extract_prices_batch([ric], start, end)
                            norm = normalize_prices(raw)
                            mapped = map_to_permno(norm, ric_map, names)
                            rows = write_partitioned(mapped)
                            log(f"    {ric}: {rows:,} rows")
                        except Exception:
                            continue
                    time.sleep(SLEEP)
                    continue

                norm = normalize_prices(raw)
                if RDP_DEBUG:
                    ric_count = norm["ric"].nunique() if "ric" in norm.columns else 0
                    log(f"    Raw rows: {len(raw):,} | Norm rows: {len(norm):,} | RICs: {ric_count:,} | date NA pct: {norm['date'].isna().mean() if 'date' in norm.columns else 'n/a'}")
                mapped = map_to_permno(norm, ric_map, names)
                if RDP_DEBUG:
                    overlap = set(norm["ric"].unique()) & set(ric_map["ric"].unique()) if "ric" in norm.columns else set()
                    log(f"    Mapped rows: {len(mapped):,} | Overlap RICs: {len(overlap):,}")
                rows = write_partitioned(mapped)
                log(f"    Wrote {rows:,} rows")
                time.sleep(SLEEP)
    finally:
        rd.close_session()
        log("Refinitiv session closed.")


if __name__ == "__main__":
    main()
