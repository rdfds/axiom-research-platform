#!/usr/bin/env python
"""
Build canonical Entity + EntityIdentifier tables from the ID mapping file.
Also emits stub EntityRelationship + EntityCorrectionLog tables.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_dt(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def build_entity_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["namedt"] = parse_dt(df.get("namedt"))
    df["nameendt"] = parse_dt(df.get("nameendt"))

    # Prefer "title" (SEC) then "comnam" (CRSP) for legal name
    df["legal_name"] = df.get("title").combine_first(df.get("comnam"))

    grouped = df.sort_values(["company_id", "nameendt"]).groupby("company_id", as_index=False)
    latest = grouped.tail(1)

    entity = pd.DataFrame(
        {
            "entity_id": latest["company_id"].astype("string"),
            "entity_type": "company",
            "legal_name": latest["legal_name"].astype("string"),
            "inception_date": grouped["namedt"].min()["namedt"].dt.date.astype("string"),
            "termination_date": grouped["nameendt"].max()["nameendt"].dt.date.astype("string"),
            "current_status": None,
        }
    )
    return entity


