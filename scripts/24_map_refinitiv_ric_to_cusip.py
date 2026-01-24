#!/usr/bin/env python
"""
Map Refinitiv RICs to identifiers via Symbology (Discovery Convert Symbols).
===========================================================================
Builds a RIC -> CUSIP/ISIN/Ticker map using Refinitiv Symbology.

Outputs:
  data/refinitiv/ric_to_cusip_map.parquet

Run:
  python -u scripts/24_map_refinitiv_ric_to_cusip.py
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd
import refinitiv.data as rd
import refinitiv.data.discovery as disc
from refinitiv.data.content import symbol_conversion as sc


DATA_DIR = Path(__file__).parent.parent / "data" / "refinitiv"
MAP_PATH = DATA_DIR / "ric_to_cusip_map.parquet"
PARTS_DIR = DATA_DIR / "ric_map_parts"
PARTS_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = int(os.getenv("RIC_MAP_BATCH", "200"))
SLEEP_SECONDS = float(os.getenv("RIC_MAP_SLEEP", "0.2"))
SAVE_PARTS = os.getenv("RIC_MAP_SAVE_PARTS", "1") == "1"
SKIP_EXISTING = os.getenv("RIC_MAP_SKIP_EXISTING", "1") == "1"
RETRIES = int(os.getenv("RIC_MAP_RETRIES", "3"))
RETRY_SLEEP = float(os.getenv("RIC_MAP_RETRY_SLEEP", "1.5"))
US_ONLY = os.getenv("RIC_MAP_US_ONLY", "1") == "1"
ASSET_STATE = os.getenv("RIC_MAP_ASSET_STATE", "ACTIVE").upper()
APPEND = os.getenv("RIC_MAP_APPEND", "1") == "1"


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def ensure_session() -> bool:
    try:
        _ = rd.get_data(universe="0#.SPX", fields=["TR.CommonName"])
        return True
    except Exception as e:
        log(f"Refinitiv session check failed: {e}")
        return False


def clean_text(series: pd.Series) -> pd.Series:
    s = series.astype("string")
    s = s.str.strip()
    s = s.where(~s.str.lower().isin(["", "nan", "none", "<na>"]))
    return s


class Progress:
    def __init__(self, total: int):
        self.total = max(total, 1)
        self.start = time.time()
        self.last_print = 0.0

    def update(self, current: int, note: str = ""):
        now = time.time()
        if now - self.last_print < 0.2 and current < self.total:
            return
        elapsed = now - self.start
        rate = current / elapsed if elapsed > 0 else 0
        remaining = (self.total - current) / rate if rate > 0 else 0
        percent = current / self.total * 100
        bar_len = 28
        filled = int(bar_len * percent / 100)
        bar = "#" * filled + "-" * (bar_len - filled)
        msg = (
            f"[{percent:5.1f}%] [{bar}] {current}/{self.total} "
            f"| {elapsed/60:5.1f}m elapsed | ETA {remaining/60:5.1f}m {note}"
        )
        sys.stdout.write("\r" + msg.ljust(120))
        sys.stdout.flush()
        self.last_print = now

    def finish(self):
        sys.stdout.write("\n")
        sys.stdout.flush()


def _pick_col(df: pd.DataFrame, *names):
    for name in names:
        if name in df.columns:
            return name
        for col in df.columns:
            if col.lower() == name.lower():
                return col
    return None


def main():
    mna_path = Path(__file__).parent.parent / "data" / "curated" / "mna_master.parquet"
    if not mna_path.exists():
        raise FileNotFoundError("Missing mna_master.parquet; run scripts/22_build_master_datasets.py first.")

    log("Connecting to Refinitiv...")
    rd.open_session()
    if not ensure_session():
        rd.close_session()
        return
    log("Connected.")

    df = pd.read_parquet(mna_path, columns=["source", "target_ric", "acquiror_ric", "target_country", "acquiror_country"])
    df = df[df["source"] == "refinitiv"]
    if US_ONLY:
        df = df[(df["target_country"] == "United States") | (df["acquiror_country"] == "United States")]

    rics = pd.concat([df["target_ric"], df["acquiror_ric"]], ignore_index=True)
    rics = clean_text(rics).dropna().unique().tolist()
    if not rics:
        log("No RICs found to map.")
        rd.close_session()
        return

    log(f"RICs to map: {len(rics):,}")

    if ASSET_STATE not in ["ACTIVE", "INACTIVE", "BOTH"]:
        raise ValueError("RIC_MAP_ASSET_STATE must be ACTIVE, INACTIVE, or BOTH.")

    states = [sc.AssetState.ACTIVE] if ASSET_STATE == "ACTIVE" else [sc.AssetState.INACTIVE]
    if ASSET_STATE == "BOTH":
        states = [sc.AssetState.ACTIVE, sc.AssetState.INACTIVE]

    total_batches = (len(rics) + BATCH_SIZE - 1) // BATCH_SIZE
    progress = Progress(total_batches)
    rows = []

    for i in range(0, len(rics), BATCH_SIZE):
        batch_idx = i // BATCH_SIZE + 1
        batch = rics[i : i + BATCH_SIZE]
        part_path = PARTS_DIR / f"ric_part_{batch_idx:05d}.parquet"
        if SAVE_PARTS and SKIP_EXISTING and part_path.exists():
            if batch_idx % 25 == 0 or batch_idx == 1:
                log(f"Batch {batch_idx}/{total_batches} skipped (already saved).")
            progress.update(batch_idx, "skipped")
            continue

        df_batch = None
        for attempt in range(1, RETRIES + 1):
            try:
                frames = []
                for state in states:
                    frames.append(
                        disc.convert_symbols(
                            symbols=batch,
                            from_symbol_type=sc.SymbolTypes.RIC,
                            to_symbol_types=[
                                sc.SymbolTypes.RIC,
                                sc.SymbolTypes.CUSIP,
                                sc.SymbolTypes.ISIN,
                                sc.SymbolTypes.TICKER_SYMBOL,
                            ],
                            preferred_country_code=sc.CountryCode.USA if US_ONLY else None,
                            asset_class=sc.AssetClass.EQUITIES,
                            asset_state=state,
                        )
                    )
                df_batch = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
                break
            except Exception as e:
                log(f"Batch {batch_idx} attempt {attempt} failed: {e}")
                if attempt < RETRIES:
                    time.sleep(RETRY_SLEEP * attempt)

        if df_batch is not None and len(df_batch) > 0:
            if SAVE_PARTS:
                df_batch.to_parquet(part_path, index=False)
            else:
                rows.append(df_batch)

        progress.update(batch_idx, "ok" if df_batch is not None and len(df_batch) > 0 else "empty")
        if batch_idx % 25 == 0 or batch_idx == 1:
            log(f"Batch {batch_idx}/{total_batches} completed.")

        if SLEEP_SECONDS:
            time.sleep(SLEEP_SECONDS)

    progress.finish()

    if SAVE_PARTS:
        part_files = sorted(PARTS_DIR.glob("ric_part_*.parquet"))
        if not part_files:
            log("No mapping rows returned.")
            rd.close_session()
            return
        mapped = pd.concat((pd.read_parquet(p) for p in part_files), ignore_index=True)
    else:
        if not rows:
            log("No mapping rows returned.")
            rd.close_session()
            return
        mapped = pd.concat(rows, ignore_index=True)

    ric_col = _pick_col(mapped, "RIC")
    cusip_col = _pick_col(mapped, "CUSIP")
    isin_col = _pick_col(mapped, "ISIN")
    ticker_col = _pick_col(mapped, "TickerSymbol", "Ticker")

    out = pd.DataFrame()
    out["ric"] = mapped[ric_col] if ric_col else pd.NA
    out["cusip"] = mapped[cusip_col] if cusip_col else pd.NA
    out["isin"] = mapped[isin_col] if isin_col else pd.NA
    out["ticker"] = mapped[ticker_col] if ticker_col else pd.NA

    out["ric"] = clean_text(out["ric"]).str.upper()
    for col in ["cusip", "isin", "ticker"]:
        out[col] = clean_text(out[col]).str.upper()

    out = out.dropna(subset=["ric"]).drop_duplicates("ric", keep="first")
    if APPEND and MAP_PATH.exists():
        existing = pd.read_parquet(MAP_PATH)
        existing["ric"] = clean_text(existing["ric"]).str.upper()
        for col in ["cusip", "isin", "ticker"]:
            existing[col] = clean_text(existing[col]).str.upper()
        out = pd.concat([existing, out], ignore_index=True)
        out = out.dropna(subset=["ric"]).drop_duplicates("ric", keep="first")

    out.to_parquet(MAP_PATH, index=False)
    log(f"Saved RIC map -> {MAP_PATH} ({len(out):,} rows)")

    rd.close_session()


if __name__ == "__main__":
    main()
