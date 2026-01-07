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


def test_repair_total_debt_from_single_statement_component_promotes_exact_long_term_match():
    repaired = _repair_total_debt_from_single_statement_component(
        current_node={
            "support_mode": "proxy_missing_component",
            "value": 62_000_000.0,
            "missing_reason": "debt_component_missing",
            "component_breakdown": {
                "mode": "partial_debt_stack",
                "formula": "partial_debt_stack_with_short_term_borrowings",
            },
        },
        current_debt_statement_node=_statement_node(None, "unsupported"),
        long_term_debt_statement_node=_statement_node(62_000_000.0, "exact"),
        as_of_time="2024-12-31T00:00:00+00:00",
        computed_at="2026-03-22T00:00:00+00:00",
        provenance_source="/tmp/facts.parquet",
    )

    assert repaired is not None
    assert repaired["support_mode"] == "exact"
    assert repaired["value"] == 62_000_000.0
    assert repaired["component_breakdown"]["mode"] == "statement_direct_single_component_total_debt"
    assert repaired["component_breakdown"]["inferred_zero_component"] == "capital_structure.current_debt_statement_direct"


