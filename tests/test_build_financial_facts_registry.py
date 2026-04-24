from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _write_financial_rows(root: Path, year: int, rows: list[dict]) -> Path:
    year_dir = root / f"year={year}"
    year_dir.mkdir(parents=True, exist_ok=True)
    out = year_dir / "part_test.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    return out


def _row(
    *,
    entity_id: str,
    line_item: str,
    value: float,
    fiscal_period_end: str = "2025-09-30",
    statement_type: str = "balance_sheet",
    available_time: str = "2025-11-05T00:00:00Z",
    ingestion_time: str = "2025-11-05T00:00:00Z",
) -> dict:
    return {
        "source_system": "sec_edgar_xbrl",
        "entity_id": entity_id,
        "company_id": entity_id,
        "event_time": fiscal_period_end,
        "available_time": available_time,
        "ingestion_time": ingestion_time,
        "version_id": f"{entity_id}:{line_item}:{fiscal_period_end}",
        "fiscal_period_end": fiscal_period_end,
        "fiscal_year": 2025,
        "fiscal_quarter": 3,
        "statement_type": statement_type,
        "line_item": line_item,
        "value": value,
        "currency": "USD",
        "units": "USD",
    }


