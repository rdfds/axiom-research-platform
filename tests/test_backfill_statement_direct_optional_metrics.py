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


def test_repair_total_debt_from_single_statement_component_marks_stale_match_as_proxy():
    repaired = _repair_total_debt_from_single_statement_component(
        current_node={
            "support_mode": "proxy_missing_component",
            "value": 76_402_000.0,
            "missing_reason": "debt_component_missing",
            "component_breakdown": {
                "mode": "partial_debt_stack",
                "formula": "partial_debt_stack_with_short_term_borrowings",
            },
        },
        current_debt_statement_node=_statement_node(76_402_000.0, "exact", end="2024-01-31"),
        long_term_debt_statement_node=_statement_node(None, "unsupported"),
        as_of_time="2024-12-31T00:00:00+00:00",
        computed_at="2026-03-22T00:00:00+00:00",
        provenance_source="/tmp/facts.parquet",
    )

    assert repaired is not None
    assert repaired["support_mode"] == "proxy_missing_component"
    assert repaired["missing_reason"] == "statement_debt_pair_stale"


def test_parse_iso_date_accepts_datetime_objects():
    value = datetime(2024, 9, 30, 0, 0, tzinfo=timezone.utc)

    assert _parse_iso_date(value).isoformat() == "2024-09-30"


def test_interest_expense_repair_from_companyfacts_builds_ttm_from_interest_expense_concept():
    repaired = _interest_expense_repair_from_companyfacts(
        current_node={
            "support_mode": "unsupported",
            "value": None,
            "missing_reason": "statement_fact_unavailable",
            "provenance": [],
        },
        companyfacts={
            "facts": {
                "us-gaap": {
                    "InterestExpense": {
                        "units": {
                            "USD": [
                                {
                                    "start": "2024-01-01",
                                    "end": "2024-09-30",
                                    "filed": "2024-10-31",
                                    "val": 75.0,
                                    "fy": 2024,
                                    "fp": "Q3",
                                    "form": "10-Q",
                                },
                                {
                                    "start": "2023-01-01",
                                    "end": "2023-12-31",
                                    "filed": "2024-02-20",
                                    "val": 110.0,
                                    "fy": 2023,
                                    "fp": "FY",
                                    "form": "10-K",
                                },
                                {
                                    "start": "2023-01-01",
                                    "end": "2023-09-30",
                                    "filed": "2023-11-01",
                                    "val": 70.0,
                                    "fy": 2023,
                                    "fp": "Q3",
                                    "form": "10-Q",
                                },
                            ]
                        }
                    }
                }
            }
        },
        companyfacts_path=Path("/tmp/companyfacts.json"),
        as_of_time="2024-12-31T00:00:00+00:00",
        computed_at="2026-03-22T00:00:00+00:00",
    )

    assert repaired is not None
    assert repaired["support_mode"] == "exact"
    assert repaired["value"] == 115.0
    assert repaired["component_breakdown"]["concept"] == "InterestExpense"
    assert repaired["component_breakdown"]["ttm_context"]["mode"] == "ytd_plus_prior_fy_minus_prior_ytd"


def test_iter_row_batches_yields_incremental_batches():
    rows = [{"company_id": str(i)} for i in range(5)]

    batches = list(_iter_row_batches(rows, 2))

    assert [len(batch) for batch in batches] == [2, 2, 1]
    assert batches[0][0]["company_id"] == "0"
    assert batches[-1][0]["company_id"] == "4"


def test_fact_parquet_source_arg_prunes_partition_years(tmp_path: Path):
    facts_root = tmp_path / "facts_asof_2026"
    for year in (2021, 2022, 2023, 2024, 2025):
        part = facts_root / f"year={year}" / "part.parquet"
        part.parent.mkdir(parents=True, exist_ok=True)
        part.write_text("placeholder")

    source_arg = _fact_parquet_source_arg(facts_root, "2024-12-31T00:00:00+00:00")

    assert "year=2022" in source_arg
    assert "year=2023" in source_arg
    assert "year=2024" in source_arg
    assert "year=2021" not in source_arg
    assert "year=2025" not in source_arg
