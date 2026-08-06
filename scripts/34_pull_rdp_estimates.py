#!/usr/bin/env python
"""
Pull Refinitiv Estimates with publish time + period end (A4 spec) and
ingest into warehouse_estimates.

This script probes for available date fields (publish time and period end)
and aborts if it cannot find both (to keep A4 spec-compliant).

Env:
  EST_START=2000-01-01
  EST_END=YYYY-MM-DD (default: today UTC)
  EST_PERIODS=FY1,FY2,NTM
  EST_SAMPLE=50
  EST_BATCH=75
  EST_SLEEP=0.2
  EST_PROBE_ONLY=0
  EST_TICKERS=comma,separated,override list
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import refinitiv.data as rd

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion import (
    append_canonical_records,
    compute_raw_payload_hash,
    compute_version_id,
    write_raw_records,
)


DATA_DIR = Path(__file__).parent.parent / "data"
CRSP_DIR = DATA_DIR / "wrds" / "crsp"
REF_DIR = DATA_DIR / "refinitiv"

EST_START = os.getenv("EST_START", "2000-01-01")
EST_END = os.getenv("EST_END", datetime.utcnow().strftime("%Y-%m-%d"))
EST_PERIODS = [p.strip() for p in os.getenv("EST_PERIODS", "FY1,FY2,NTM").split(",") if p.strip()]
EST_SAMPLE = int(os.getenv("EST_SAMPLE", "50"))
EST_BATCH = int(os.getenv("EST_BATCH", "75"))
EST_SLEEP = float(os.getenv("EST_SLEEP", "0.2"))
EST_PROBE_ONLY = os.getenv("EST_PROBE_ONLY", "0") == "1"
EST_TICKERS = os.getenv("EST_TICKERS")


VALUE_FIELDS = {
    "eps": ["TR.EPSMean"],
    "revenue": ["TR.RevenueMean"],
    "ebitda": ["TR.EBITDAMean"],
}

NUM_FIELDS = [
    "TR.NumberOfEstimates",
    "TR.NumOfEstimates",
]

PUBLISH_FIELDS = [
    "TR.EstimateDate",
    "TR.EstimateDateTime",
    "TR.EPSMeanDate",
    "TR.EPSMeanDateTime",
    "TR.EPSMeanLastUpdated",
    "TR.LastUpdateDate",
    "TR.LastUpdateDateTime",
]

PERIOD_FIELDS = [
    "TR.EPSMeanPeriodEndDate",
    "TR.EPSPeriodEndDate",
    "TR.PeriodEndDate",
    "TR.FiscalPeriodEndDate",
    "TR.FiscalPeriodEnd",
    "TR.FYEndDate",
]


def log(msg: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def load_universe() -> List[str]:
    if EST_TICKERS:
        return [t.strip() for t in EST_TICKERS.split(",") if t.strip()]
    path = REF_DIR / "universe_us_active.parquet"
    if not path.exists():
        raise FileNotFoundError("Missing universe_us_active.parquet")
    df = pd.read_parquet(path)
    if "ric" in df.columns:
        rics = df["ric"].dropna().astype("string").str.upper().tolist()
    else:
        rics = df.iloc[:, 0].dropna().astype("string").str.upper().tolist()
    return rics


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
        columns=["permno", "permco", "namedt", "nameendt", "ncusip", "cusip"],
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
    # Extend CRSP end date to cover estimate end date for mapping
    max_end = names["nameendt"].max()
    target_end = pd.to_datetime(EST_END, errors="coerce")
    if pd.notna(max_end) and pd.notna(target_end) and target_end > max_end:
        names.loc[names["nameendt"] == max_end, "nameendt"] = target_end
    return names[["permno", "permco", "namedt", "nameendt", "cusip8"]]


def probe_fields(sample: List[str], period: str) -> Dict[str, str]:
    """
    Try to find publish time + period end fields that return non-null data.
    Returns a dict with keys: publish_field, period_field, num_field (optional).
    """
    base_fields = [VALUE_FIELDS["eps"][0]]
    found_publish = None
    found_period = None
    found_num = None

    params = {"Period": period}
    for field in PUBLISH_FIELDS:
        try:
            df = rd.get_data(universe=sample, fields=base_fields + [field], parameters=params)
            if field in df.columns and df[field].notna().any():
                found_publish = field
                break
        except Exception:
            continue

    for field in PERIOD_FIELDS:
        try:
            df = rd.get_data(universe=sample, fields=base_fields + [field], parameters=params)
            if field in df.columns and df[field].notna().any():
                found_period = field
                break
        except Exception:
            continue

    for field in NUM_FIELDS:
        try:
            df = rd.get_data(universe=sample, fields=base_fields + [field], parameters=params)
            if field in df.columns and df[field].notna().any():
                found_num = field
                break
        except Exception:
            continue

    return {
        "publish_field": found_publish,
        "period_field": found_period,
        "num_field": found_num,
    }


def map_to_permco(df: pd.DataFrame, ric_map: pd.DataFrame, names: pd.DataFrame) -> pd.DataFrame:
    merged = df.merge(ric_map, on="ric", how="left")
    merged = merged.dropna(subset=["cusip8"])
    merged = merged.merge(names, on="cusip8", how="left")
    merged = merged[(merged["available_time"] >= merged["namedt"]) & (merged["available_time"] <= merged["nameendt"])]
    merged = merged.sort_values(["ric", "available_time", "nameendt"])
    merged = merged.drop_duplicates(subset=["ric", "available_time"], keep="last")
    merged = merged.dropna(subset=["permno"])
    merged["permno"] = merged["permno"].astype("Int64")
    merged["permco"] = merged["permco"].astype("Int64")
    return merged


def main() -> None:
    rics = load_universe()
    if EST_SAMPLE and len(rics) > EST_SAMPLE:
        sample = rics[:EST_SAMPLE]
    else:
        sample = rics

    log(f"Universe size: {len(rics):,} | sample: {len(sample):,}")

    rd.open_session()

    try:
        # Probe for fields
        period = EST_PERIODS[0] if EST_PERIODS else "FY1"
        log(f"Probing estimate fields for period {period}...")
        probe = probe_fields(sample, period)
        log(f"Probe results: {probe}")

        if not probe["publish_field"] or not probe["period_field"]:
            log("Missing publish_time or period_end fields. Cannot build spec-compliant A4.")
            return

        publish_field = probe["publish_field"]
        period_field = probe["period_field"]
        num_field = probe["num_field"]

        if EST_PROBE_ONLY:
            return

        ric_map = load_ric_map()
        names = load_names()

        ingestion_time = datetime.utcnow()
        source_system = "refinitiv_estimates"

        records = []
        raw_records = []

        for period in EST_PERIODS:
            log(f"Pulling estimates for {period}...")
            fields = [publish_field, period_field]
            for metric, candidates in VALUE_FIELDS.items():
                fields.extend(candidates)
            if num_field:
                fields.append(num_field)

            for i in range(0, len(rics), EST_BATCH):
                batch = rics[i:i+EST_BATCH]
                try:
                    df = rd.get_data(universe=batch, fields=fields, parameters={"Period": period})
                except Exception as exc:
                    log(f"  Batch error: {exc}")
                    time.sleep(EST_SLEEP)
                    continue

                if df is None or df.empty:
                    time.sleep(EST_SLEEP)
                    continue

                if "Instrument" not in df.columns:
                    time.sleep(EST_SLEEP)
                    continue

                df = df.rename(columns={"Instrument": "ric"})
                df["ric"] = df["ric"].astype("string").str.upper().str.strip()
                df["available_time"] = pd.to_datetime(df[publish_field], errors="coerce")
                df["event_time"] = pd.to_datetime(df[period_field], errors="coerce")

                df = df.dropna(subset=["available_time", "event_time"])
                if df.empty:
                    time.sleep(EST_SLEEP)
                    continue

                df = map_to_permco(df, ric_map, names)
                if df.empty:
                    time.sleep(EST_SLEEP)
                    continue

                for _, row in df.iterrows():
                    entity_id = str(row["permco"]) if not pd.isna(row["permco"]) else str(row["permno"])
                    if entity_id is None:
                        continue

                    for metric, candidates in VALUE_FIELDS.items():
                        value_field = None
                        for c in candidates:
                            if c in row.index:
                                value_field = c
                                break
                        if value_field is None:
                            continue
                        consensus_value = row.get(value_field)
                        if consensus_value is None or pd.isna(consensus_value):
                            continue

                        payload = {
                            "ric": row.get("ric"),
                            "metric": metric,
                            "period": period,
                            "consensus_value": float(consensus_value),
                            "num_estimates": row.get(num_field) if num_field else None,
                            "publish_time": row.get(publish_field),
                            "period_end": row.get(period_field),
                        }
                        raw_payload_hash = compute_raw_payload_hash(payload)
                        version_id = compute_version_id(
                            source_system=source_system,
                            entity_id=entity_id,
                            event_time=row["event_time"].to_pydatetime(),
                            available_time=row["available_time"].to_pydatetime(),
                            raw_payload_hash=raw_payload_hash,
                        )

                        raw_records.append(
                            {
                                "entity_id": entity_id,
                                "company_id": entity_id,
                                "security_id": str(row["permno"]) if not pd.isna(row["permno"]) else None,
                                "event_time": row["event_time"],
                                "available_time": row["available_time"],
                                "payload": payload,
                            }
                        )

                        records.append(
                            {
                                "source_system": source_system,
                                "entity_id": entity_id,
                                "company_id": entity_id,
                                "security_id": str(row["permno"]) if not pd.isna(row["permno"]) else None,
                                "event_time": row["event_time"],
                                "available_time": row["available_time"],
                                "ingestion_time": ingestion_time,
                                "version_id": version_id,
                                "raw_payload_hash": raw_payload_hash,
                                "upstream_version_ids": [version_id],
                                "quality_flags": [],
                                "metric": metric,
                                "period": period,
                                "consensus_value": float(consensus_value),
                                "num_estimates": row.get(num_field) if num_field else None,
                                "revision_direction": None,
                                "revision_magnitude": None,
                            }
                        )

                time.sleep(EST_SLEEP)

        if raw_records:
            write_raw_records(source_system=source_system, records=raw_records)
        if records:
            append_canonical_records("warehouse_estimates", records)
            log(f"Ingested {len(records):,} estimate records")
    finally:
        rd.close_session()
        log("Refinitiv session closed.")


if __name__ == "__main__":
    main()
