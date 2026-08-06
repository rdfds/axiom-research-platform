#!/usr/bin/env python
"""
Probe + Pull Refinitiv Corporate Actions (US Active, 2000-present)
==================================================================
This script:
1) Probes which corporate-action fields are available in your entitlement.
2) Builds a US active equity universe (attempts full universe; falls back to indices).
3) Pulls corporate actions for that universe from 2000-01-01 through 2026-02-02.

Outputs:
  data/refinitiv/universe_us_active.parquet
  data/refinitiv/corporate_actions/ca_<year>_part_<batch>.parquet
  data/refinitiv/corporate_actions/manifest.jsonl

Run:
  python -u scripts/18_probe_and_pull_refinitiv_corp_actions.py
"""

import json
import time
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from pandas.api.types import is_string_dtype
import refinitiv.data as rd


DATA_DIR = Path(__file__).parent.parent / "data" / "refinitiv"
DATA_DIR.mkdir(parents=True, exist_ok=True)

UNIVERSE_PATH = DATA_DIR / "universe_us_active.parquet"
CA_DIR = DATA_DIR / "corporate_actions"
CA_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = CA_DIR / "manifest.jsonl"

START_DATE = "2000-01-01"
END_DATE = "2026-02-02"  # Current date in this workspace

BATCH_SIZE = 75
SLEEP_SECONDS = 0.4

CA_FIELD_CANDIDATES = [
    "TR.CAActionType",
    "TR.CAEventType",
    "TR.CAType",
    "TR.CAAnnouncementDate",
    "TR.CAEffectiveDate",
    "TR.CAExDate",
    "TR.CARecordDate",
    "TR.CAPayDate",
    "TR.CAAmount",
    "TR.CAAdjustmentFactor",
    "TR.CAAdjustmentType",
    "TR.CACurrency",
    "TR.CARatio",
    "TR.CAStatus",
    "TR.CAAmount",
    "TR.CAAmountGross",
    "TR.CAAmountNet",
    "TR.CAExAmount",
    "TR.CAShareFactor",
    "TR.CAValue",
]

# If CAType is required, we will try these. Add/remove as needed.
CA_TYPE_CANDIDATES = [
    "DIV",  # Dividends (may or may not be valid)
    "DVD",
    "SSP",  # Stock splits
    "RHT",  # Rights
    "SPI",  # Spinoff
    "BON",  # Bonus issue
    "CAP",  # Capital change
    "MER",  # Merger
    "TND",  # Tender
    "LIQ",  # Liquidation
]

# Safety / quality controls
ABORT_IF_EMPTY_PROBE = True
MIN_EVENT_ROWS = 5
MIN_EVENT_RATIO = 0.01
MIN_DATE_EVENT_ROWS = 3
MIN_DATE_EVENT_RATIO = 0.01
MAX_CONSECUTIVE_EMPTY_BATCHES = 10


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def batched(items: List[str], batch_size: int) -> List[List[str]]:
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def try_screen_universe() -> Tuple[List[str], Dict[str, str]]:
    """
    Attempt to build a full US active equity universe using Refinitiv Screen.
    Returns (tickers, meta). meta is empty if no success.
    """
    screen_queries = [
        ("SCREEN(U(IN(Equity)), TR.ExchangeCountry='United States' AND TR.Status='Active')", "exchange_country_status"),
        ("SCREEN(U(IN(Equity)), TR.ExchangeCountry='United States')", "exchange_country"),
        ("SCREEN(U(IN(Equity)), TR.CountryOfIncorporation='United States' AND TR.Status='Active')", "incorporation_country_status"),
        ("SCREEN(U(IN(Equity)), TR.CountryOfIncorporation='United States')", "incorporation_country"),
    ]

    for screen, label in screen_queries:
        try:
            df = rd.get_data(universe=screen, fields=["TR.CommonName"])
            if df is None or len(df) == 0:
                log(f"Screen {label} returned no rows.")
                continue
            tickers = df["Instrument"].dropna().unique().tolist()
            if len(tickers) < 500:
                log(f"Screen {label} returned only {len(tickers)} tickers (too small).")
                continue
            return tickers, {"method": "screen", "label": label}
        except Exception as e:
            log(f"Screen {label} failed: {e}")

    return [], {}


def universe_from_indices() -> Tuple[List[str], Dict[str, str]]:
    index_universes = [
        "0#.SPX",
        "0#.MID",
        "0#.SML",
        "0#.RUI",
        "0#.RUT",
        "0#.RUA",
        "0#.NDX",
    ]
    tickers: List[str] = []

    for idx in index_universes:
        try:
            df = rd.get_data(universe=idx, fields=["TR.CommonName"])
            if df is None or len(df) == 0:
                continue
            tickers.extend(df["Instrument"].dropna().tolist())
        except Exception as e:
            log(f"Index universe {idx} failed: {e}")

    tickers = sorted(list(set(tickers)))
    return tickers, {"method": "indices", "label": ",".join(index_universes)}


def build_universe() -> List[str]:
    if UNIVERSE_PATH.exists():
        log(f"Loading cached universe from {UNIVERSE_PATH.name}...")
        df = pd.read_parquet(UNIVERSE_PATH)
        return df["ric"].dropna().unique().tolist()

    log("Building US active equity universe...")
    tickers, meta = try_screen_universe()

    if not tickers:
        log("Full screen universe failed. Falling back to index constituents.")
        tickers, meta = universe_from_indices()

    if not tickers:
        raise RuntimeError("Failed to build any universe from Refinitiv.")

    df = pd.DataFrame({
        "ric": sorted(list(set(tickers))),
        "source_method": meta.get("method", "unknown"),
        "source_label": meta.get("label", "unknown"),
        "pulled_at": datetime.now().isoformat(),
    })
    df.to_parquet(UNIVERSE_PATH, index=False)
    log(f"Saved universe: {len(df):,} tickers -> {UNIVERSE_PATH.name}")
    return df["ric"].tolist()


def probe_ca_fields(sample_tickers: List[str], start_date: str, end_date: str) -> List[str]:
    log("Probing corporate action fields...")
    working_fields = []
    for field in CA_FIELD_CANDIDATES:
        try:
            _ = rd.get_data(
                universe=sample_tickers,
                fields=[field],
                parameters={"SDate": start_date, "EDate": end_date},
            )
            working_fields.append(field)
        except Exception as e:
            log(f"Field not available: {field} ({e})")

    if not working_fields:
        raise RuntimeError("No corporate-action fields are available with current entitlements.")

    log(f"Working CA fields: {working_fields}")
    return working_fields


def probe_ca_type_requirement(sample_tickers: List[str], fields: List[str], start_date: str, end_date: str) -> bool:
    log("Probing whether CAType parameter is required...")
    try:
        _ = rd.get_data(
            universe=sample_tickers,
            fields=fields,
            parameters={"SDate": start_date, "EDate": end_date},
        )
        log("CAType not required.")
        return False
    except Exception as e:
        msg = str(e).lower()
        if "catype" in msg or "ca type" in msg:
            log("CAType appears to be required.")
            return True
        log(f"CAType probe failed for other reason: {e}")
        raise


def _event_columns(df: pd.DataFrame) -> List[str]:
    keywords = [
        "announcement date",
        "effective date",
        "ex date",
        "record date",
        "pay date",
        "adjustment factor",
        "adjustment type",
        "action type",
        "event type",
        "amount",
        "ratio",
        "currency",
        "status",
    ]
    cols = []
    for c in df.columns:
        cl = c.lower()
        if any(k in cl for k in keywords):
            cols.append(c)
    return cols


def _event_signal_columns(df: pd.DataFrame, event_cols: List[str]) -> List[str]:
    date_cols = [c for c in event_cols if "date" in c.lower()]
    value_keywords = ["amount", "ratio", "value", "currency", "factor", "share factor"]
    value_cols = [c for c in event_cols if any(k in c.lower() for k in value_keywords)]
    return date_cols + value_cols


def _clean_event_fields(df: pd.DataFrame, event_cols: List[str]) -> pd.DataFrame:
    cleaned = df.copy()
    placeholders = {
        "Capital Change Type": pd.NA,
        "Not Available": pd.NA,
        "N/A": pd.NA,
        "NA": pd.NA,
    }
    for col in event_cols:
        if is_string_dtype(cleaned[col]):
            cleaned[col] = cleaned[col].astype("string")
            cleaned[col] = cleaned[col].replace(r"^\s*$", pd.NA, regex=True)
            cleaned[col] = cleaned[col].replace(placeholders)
    return cleaned


def filter_event_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    event_cols = _event_columns(df)
    if not event_cols:
        return df
    cleaned = _clean_event_fields(df, event_cols)
    signal_cols = _event_signal_columns(cleaned, event_cols)
    if not signal_cols:
        signal_cols = event_cols
    mask = cleaned[signal_cols].notna().any(axis=1)
    return cleaned[mask].copy()


def probe_event_density(
    sample_tickers: List[str],
    fields: List[str],
    start_date: str,
    end_date: str,
    use_ca_type: bool,
) -> Tuple[int, int, float, int, float]:
    params = {"SDate": start_date, "EDate": end_date}
    if use_ca_type:
        # Try the first CAType as a probe
        params["CAType"] = CA_TYPE_CANDIDATES[0]

    df = rd.get_data(universe=sample_tickers, fields=fields, parameters=params)
    if df is None:
        return 0, 0, 0.0, 0, 0.0
    df = df.reset_index()
    event_cols = _event_columns(df)
    cleaned = _clean_event_fields(df, event_cols) if event_cols else df
    signal_cols = _event_signal_columns(cleaned, event_cols) if event_cols else []
    if not signal_cols:
        signal_cols = event_cols
    filtered = cleaned[cleaned[signal_cols].notna().any(axis=1)] if signal_cols else cleaned
    total = len(df)
    events = len(filtered) if filtered is not None else 0
    ratio = events / total if total else 0.0
    date_cols = [c for c in event_cols if "date" in c.lower()]
    date_events = 0
    date_ratio = 0.0
    if date_cols:
        date_events = int(cleaned[date_cols].notna().any(axis=1).sum())
        date_ratio = date_events / total if total else 0.0
    return events, total, ratio, date_events, date_ratio


def save_manifest(entry: Dict) -> None:
    with open(MANIFEST_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def pull_corporate_actions(
    tickers: List[str],
    fields: List[str],
    start_date: str,
    end_date: str,
    use_ca_type: bool,
) -> None:
    start_year = int(start_date[:4])
    end_year = int(end_date[:4])

    batches = batched(tickers, BATCH_SIZE)
    log(f"Pulling CA data for {len(tickers):,} tickers in {len(batches)} batches...")

    for year in range(start_year, end_year + 1):
        year_start = f"{year}-01-01"
        year_end = f"{year}-12-31"
        if year == end_year:
            year_end = end_date

        log(f"Year {year}: {year_start} -> {year_end}")
        empty_batches = 0

        for b_idx, batch in enumerate(batches):
            if use_ca_type:
                for ca_type in CA_TYPE_CANDIDATES:
                    out_path = CA_DIR / f"ca_{ca_type}_{year}_part_{b_idx:04d}.parquet"
                    if out_path.exists():
                        continue
                    try:
                        df = rd.get_data(
                            universe=batch,
                            fields=fields,
                            parameters={"SDate": year_start, "EDate": year_end, "CAType": ca_type},
                        )
                        if df is None or len(df) == 0:
                            continue
                        df = df.reset_index()
                        df = filter_event_rows(df)
                        if df is None or df.empty:
                            empty_batches += 1
                            if empty_batches >= MAX_CONSECUTIVE_EMPTY_BATCHES:
                                log(f"Detected {MAX_CONSECUTIVE_EMPTY_BATCHES} consecutive empty batches; skipping remainder of year {year}.")
                                break
                            continue
                        df["ca_type"] = ca_type
                        df["pull_start"] = year_start
                        df["pull_end"] = year_end
                        df["batch_index"] = b_idx
                        df.to_parquet(out_path, index=False)
                        save_manifest({
                            "file": out_path.name,
                            "rows": len(df),
                            "year": year,
                            "ca_type": ca_type,
                            "batch_index": b_idx,
                            "timestamp": datetime.now().isoformat(),
                        })
                        log(f"Saved {len(df):,} rows -> {out_path.name}")
                    except Exception as e:
                        log(f"Batch {b_idx} CAType {ca_type} failed: {e}")
                    time.sleep(SLEEP_SECONDS)
                if empty_batches >= MAX_CONSECUTIVE_EMPTY_BATCHES:
                    break
            else:
                out_path = CA_DIR / f"ca_{year}_part_{b_idx:04d}.parquet"
                if out_path.exists():
                    continue
                try:
                    df = rd.get_data(
                        universe=batch,
                        fields=fields,
                        parameters={"SDate": year_start, "EDate": year_end},
                    )
                    if df is None or len(df) == 0:
                        continue
                    df = df.reset_index()
                    df = filter_event_rows(df)
                    if df is None or df.empty:
                        empty_batches += 1
                        if empty_batches >= MAX_CONSECUTIVE_EMPTY_BATCHES:
                            log(f"Detected {MAX_CONSECUTIVE_EMPTY_BATCHES} consecutive empty batches; skipping remainder of year {year}.")
                            break
                        continue
                    df["pull_start"] = year_start
                    df["pull_end"] = year_end
                    df["batch_index"] = b_idx
                    df.to_parquet(out_path, index=False)
                    save_manifest({
                        "file": out_path.name,
                        "rows": len(df),
                        "year": year,
                        "batch_index": b_idx,
                        "timestamp": datetime.now().isoformat(),
                    })
                    log(f"Saved {len(df):,} rows -> {out_path.name}")
                except Exception as e:
                    log(f"Batch {b_idx} failed: {e}")
                time.sleep(SLEEP_SECONDS)


def main() -> None:
    log("Opening Refinitiv session...")
    rd.open_session()

    try:
        tickers = build_universe()

        sample_tickers = tickers[:50] if len(tickers) >= 50 else tickers
        log(f"Probe sample tickers: {sample_tickers}")

        probe_start = "2024-01-01"
        probe_end = "2024-12-31"

        fields = probe_ca_fields(sample_tickers, probe_start, probe_end)
        use_ca_type = probe_ca_type_requirement(sample_tickers, fields, probe_start, probe_end)

        events, total, ratio, date_events, date_ratio = probe_event_density(
            sample_tickers, fields, probe_start, probe_end, use_ca_type
        )
        log(
            "Probe event density (signal/date): "
            f"{events}/{total} ({ratio:.2%}) rows look like events; "
            f"{date_events}/{total} ({date_ratio:.2%}) rows have dates"
        )
        if ABORT_IF_EMPTY_PROBE and (
            events < MIN_EVENT_ROWS
            or ratio < MIN_EVENT_RATIO
            or date_events < MIN_DATE_EVENT_ROWS
            or date_ratio < MIN_DATE_EVENT_RATIO
        ):
            raise RuntimeError(
                "Corporate action probe returned too few event rows. "
                "This likely indicates a snapshot feed or missing entitlement. "
                "Aborting to avoid empty data."
            )

        pull_corporate_actions(
            tickers=tickers,
            fields=fields,
            start_date=START_DATE,
            end_date=END_DATE,
            use_ca_type=use_ca_type,
        )

    finally:
        rd.close_session()
        log("Refinitiv session closed.")


if __name__ == "__main__":
    main()
