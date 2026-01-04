from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from scripts.backfill_statement_direct_optional_metrics import (
    _fact_parquet_source_arg,
    _interest_expense_repair_from_companyfacts,
    _iter_row_batches,
    _parse_iso_date,
    _repair_total_debt_from_single_statement_component,
)


def _statement_node(value: Optional[float], support_mode: str, end: Optional[str] = "2024-09-30"):
    return {
        "support_mode": support_mode,
        "value": value,
        "component_breakdown": None
        if end is None
        else {
            "fact_type": "financial.debt_component",
            "end": end,
            "formula": "statement_direct_fact",
        },
    }


