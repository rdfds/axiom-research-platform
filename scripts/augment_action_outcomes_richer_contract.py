#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.config import load_config


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    raw = df.get(column)
    if isinstance(raw, pd.Series):
        return pd.to_numeric(raw, errors="coerce")
    return pd.Series([pd.NA] * len(df), index=df.index, dtype="Float64")


def _parse_args() :
    parser = argparse.ArgumentParser(
        description="Augment an existing action outcomes parquet with richer contract-era macro and debt/liquidity columns."
    )
    parser.add_argument(
        "--in-path",
        default=str(ROOT / "data" / "curated" / "action_outcomes_with_credit_ratings.normalized_full.parquet"),
    )
    parser.add_argument(
        "--out-path",
        default=str(ROOT / "data" / "curated" / "action_outcomes_with_credit_ratings.normalized_full.rich_contract_v3.parquet"),
    )
    parser.add_argument(
        "--fundamentals-path",
        default=str(ROOT / "data" / "curated" / "fundamentals_master.parquet"),
    )
    parser.add_argument(
        "--raw-timeseries-path",
        default=str(ROOT / "data" / "inputs_layer" / "raw_timeseries.parquet"),
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "pipeline_v1.json"),
    )
    parser.add_argument("--duckdb-memory", default="4GB")
    parser.add_argument("--duckdb-threads", type=int, default=1)
    return parser.parse_args()


def _macro_subquery(*, macro_path: Path, series_id: Optional[str], value_alias: str) -> str:
    if not series_id:
        return f"SELECT CAST(NULL AS DOUBLE) AS {value_alias}, CAST(NULL AS TIMESTAMP) AS event_time WHERE FALSE"
    return f"""
        SELECT
            CAST(value AS DOUBLE) AS {value_alias},
            CAST(event_time AS TIMESTAMP) AS event_time
        FROM read_parquet('{macro_path.as_posix()}')
        WHERE entity_id = '{series_id}'
        ORDER BY event_time
    """


