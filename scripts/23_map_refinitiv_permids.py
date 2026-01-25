#!/usr/bin/env python
"""
Map Refinitiv PermIDs to identifiers (RIC/CUSIP/ISIN/Ticker).
============================================================
Reads Refinitiv M&A deals and builds a PermID -> identifier lookup table.

Outputs:
  data/refinitiv/permid_map.parquet

Run:
  python -u scripts/23_map_refinitiv_permids.py
"""

import os
from pathlib import Path
from datetime import datetime
from typing import List
import sys
import time

import pandas as pd
import refinitiv.data as rd


DATA_DIR = Path(__file__).parent.parent / "data" / "refinitiv"
MAP_PATH = DATA_DIR / "permid_map.parquet"
PARTS_DIR = DATA_DIR / "permid_map_parts"
PARTS_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = int(os.getenv("PERMID_MAP_BATCH", "200"))
SLEEP_SECONDS = float(os.getenv("PERMID_MAP_SLEEP", "0.3"))
US_ONLY = os.getenv("PERMID_MAP_US_ONLY", "1") == "1"
SAVE_PARTS = os.getenv("PERMID_MAP_SAVE_PARTS", "1") == "1"
SKIP_EXISTING = os.getenv("PERMID_MAP_SKIP_EXISTING", "1") == "1"
RETRIES = int(os.getenv("PERMID_MAP_RETRIES", "3"))
RETRY_SLEEP = float(os.getenv("PERMID_MAP_RETRY_SLEEP", "1.5"))

FIELD_CANDIDATES = [
    "TR.CommonName",
    "TR.RIC",
    "TR.PrimaryRIC",
    "TR.TickerSymbol",
    "TR.ExchangeTicker",
    "TR.CUSIP",
    "TR.CUSIP9",
    "TR.ISIN",
]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def ensure_session() -> bool:
    try:
        _ = rd.get_data(universe="0#.SPX", fields=["TR.CommonName"])
        return True
    except Exception as e:
        log(f"Refinitiv session check failed: {e}")
        return False


def _resolve_permid_format(sample_permid: str) -> str:
    candidates = [
        "{pid}",
        "PermID:{pid}",
        "PERMID:{pid}",
    ]
    for fmt in candidates:
        try:
            _ = rd.get_data(
                universe=[fmt.format(pid=sample_permid)],
                fields=["TR.CommonName"],
            )
            return fmt
        except Exception:
            continue
    return ""


def _probe_fields(universe_sample: List[str]) -> List[str]:
    working = []
    for field in FIELD_CANDIDATES:
        try:
            _ = rd.get_data(universe=universe_sample, fields=[field])
            working.append(field)
        except Exception as e:
            log(f"Field not available: {field} ({e})")
    return working


def _first_col(df: pd.DataFrame, *names):
    for name in names:
        if name in df.columns:
            return df[name]
    return None


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


def main():
    ma_path = DATA_DIR / "ma_deals_all.parquet"
    if not ma_path.exists():
        raise FileNotFoundError("Missing ma_deals_all.parquet; run pull_refinitiv_all.py first.")

    log("Connecting to Refinitiv...")
    rd.open_session()
    if not ensure_session():
        rd.close_session()
        return
    log("Connected.")

    columns = ["Target PermID", "Acquiror PermID"]
    if US_ONLY:
        columns += ["Target Nation", "Acquiror Nation"]

    deals = pd.read_parquet(ma_path, columns=columns)
    if US_ONLY:
        deals = deals[
            (deals["Target Nation"] == "United States") | (deals["Acquiror Nation"] == "United States")
        ]
    permids = pd.concat([deals["Target PermID"], deals["Acquiror PermID"]], ignore_index=True)
    permids = permids.dropna().astype(str)
    permids = permids[permids.str.len() > 3].unique().tolist()
    if not permids:
        log("No PermIDs found in M&A deals.")
        rd.close_session()
        return

    sample_pid = permids[0]
    fmt = _resolve_permid_format(sample_pid)
    if not fmt:
        log("Unable to resolve a PermID universe format. Aborting.")
        rd.close_session()
        return

    sample_universe = [fmt.format(pid=p) for p in permids[:5]]
    log("Probing PermID fields...")
    working_fields = _probe_fields(sample_universe)
    if not working_fields:
        log("No working fields for PermID mapping.")
        rd.close_session()
        return
    log(f"Working PermID fields: {working_fields}")

    rows = []
    total_batches = (len(permids) + BATCH_SIZE - 1) // BATCH_SIZE
    progress = Progress(total_batches)

    for i in range(0, len(permids), BATCH_SIZE):
        batch_idx = i // BATCH_SIZE + 1
        batch = permids[i : i + BATCH_SIZE]
        part_path = PARTS_DIR / f"permid_part_{batch_idx:05d}.parquet"
        if SAVE_PARTS and SKIP_EXISTING and part_path.exists():
            if batch_idx % 25 == 0 or batch_idx == 1:
                log(f"Batch {batch_idx}/{total_batches} skipped (already saved).")
            progress.update(batch_idx, "skipped")
            continue

        universe = [fmt.format(pid=p) for p in batch]
        df = None
        for attempt in range(1, RETRIES + 1):
            try:
                df = rd.get_data(universe=universe, fields=working_fields)
                break
            except Exception as e:
                log(f"Batch {batch_idx} attempt {attempt} failed: {e}")
                if attempt < RETRIES:
                    time.sleep(RETRY_SLEEP * attempt)

        if df is not None and len(df) > 0:
            if SAVE_PARTS:
                df.to_parquet(part_path, index=False)
            else:
                rows.append(df)

        progress.update(batch_idx, "ok" if df is not None and len(df) > 0 else "empty")
        if batch_idx % 25 == 0 or batch_idx == 1:
            log(f"Batch {batch_idx}/{total_batches} completed.")

        if SLEEP_SECONDS:
            time.sleep(SLEEP_SECONDS)

    progress.finish()

    if SAVE_PARTS:
        part_files = sorted(PARTS_DIR.glob("permid_part_*.parquet"))
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
    mapped["permid"] = mapped["Instrument"].astype(str).str.replace("PERMID:", "", case=False).str.replace("PermID:", "")

    out = pd.DataFrame()
    out["permid"] = mapped["permid"]
    out["name"] = _first_col(mapped, "Common Name", "Organization Name", "Instrument Name")
    out["ric"] = _first_col(mapped, "RIC", "Primary RIC")
    out["cusip"] = _first_col(mapped, "CUSIP", "CUSIP 9", "CUSIP9")
    out["isin"] = _first_col(mapped, "ISIN")
    out["ticker"] = _first_col(mapped, "Ticker Symbol", "Exchange Ticker", "Ticker")

    out = out.dropna(subset=["permid"]).drop_duplicates("permid", keep="first")
    out.to_parquet(MAP_PATH, index=False)
    log(f"Saved PermID map -> {MAP_PATH} ({len(out):,} rows)")

    rd.close_session()


if __name__ == "__main__":
    main()
