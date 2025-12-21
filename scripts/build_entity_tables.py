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


def build_identifier_table(df: pd.DataFrame, source_path: Path) -> pd.DataFrame:
    df = df.copy()
    df["namedt"] = parse_dt(df.get("namedt"))
    df["nameendt"] = parse_dt(df.get("nameendt"))
    df = df.reset_index().rename(columns={"index": "row_id"})

    fields: Dict[str, str] = {
        "cik": "cik",
        "ticker": "ticker",
        "permno": "permno",
        "permco": "permco",
        "cusip": "cusip",
    }

    parts: List[pd.DataFrame] = []
    for field, id_type in fields.items():
        if field not in df.columns:
            continue
        part = df[["company_id", field, "namedt", "nameendt", "row_id"]].copy()
        part = part[part[field].notna()]
        if part.empty:
            continue
        out = pd.DataFrame(
            {
                "entity_id": part["company_id"].astype("string"),
                "identifier_type": id_type,
                "identifier_value": part[field].astype("string"),
                "valid_from": part["namedt"],
                "valid_to": part["nameendt"],
                "source_id": "entity_id_map",
                "source_type": "mapping",
                "published_at": part["namedt"].combine_first(part["nameendt"]),
                "effective_at": part["namedt"],
                "ingested_at": utc_now(),
                "confidence_score": 1.0,
                "raw_pointer": (
                    f"{source_path.as_posix()}#row=" + part["row_id"].astype("string") + f":{field}"
                ),
            }
        )
        parts.append(out)

    if not parts:
        return pd.DataFrame(
            columns=[
                "entity_id",
                "identifier_type",
                "identifier_value",
                "valid_from",
                "valid_to",
                "source_id",
                "source_type",
                "published_at",
                "effective_at",
                "ingested_at",
                "confidence_score",
                "raw_pointer",
            ]
        )

    return pd.concat(parts, ignore_index=True)


def empty_entity_relationship() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "parent_entity_id",
            "child_entity_id",
            "relationship_type",
            "valid_from",
            "valid_to",
            "source_id",
            "source_type",
            "published_at",
            "effective_at",
            "ingested_at",
            "confidence_score",
            "raw_pointer",
        ]
    )


