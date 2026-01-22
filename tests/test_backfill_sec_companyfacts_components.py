from pathlib import Path

import pytest

from scripts.backfill_sec_companyfacts_components import (
    _cash_sti_proxy_represents_cash_only,
    _extract_lease_liabilities,
    _extract_revolver_undrawn,
    _extract_restricted_cash,
    _repair_cash_sti_from_statement_cash,
    _repair_restricted_cash_from_total_cash_reconciliation,
    _statement_fact_node_is_fresh_enough,
)


def _companyfacts_with_entries(entries):
    facts = {}
    for concept, value in entries.items():
        facts[concept] = {
            "units": {
                "USD": [
                    {
                        "val": value,
                        "end": "2024-12-31",
                        "filed": "2024-12-31",
                        "fy": 2024,
                        "fp": "FY",
                        "form": "10-K",
                    }
                ]
            }
        }
    return {"facts": {"us-gaap": facts}}


def _companyfacts_with_fact_rows(entries):
    facts = {}
    for concept, rows in entries.items():
        facts[concept] = {"units": {"USD": rows}}
    return {"facts": {"us-gaap": facts}}


def test_extract_restricted_cash_uses_mixed_fallback_when_exact_concepts_absent():
    companyfacts = _companyfacts_with_entries(
        {
            "RestrictedCashAndCashEquivalentsAtCarryingValue": 42.0,
        }
    )

    value, meta = _extract_restricted_cash(companyfacts, "2024-12-31")

    assert value == 42.0
    assert meta is not None
    assert meta["mode"] == "mixed_total_restricted_cash_fallback"
    assert meta["chosen"]["concept"] == "RestrictedCashAndCashEquivalentsAtCarryingValue"


def test_extract_restricted_cash_does_not_use_cash_plus_restricted_total():
    companyfacts = _companyfacts_with_entries(
        {
            "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents": 125.0,
        }
    )

    value, meta = _extract_restricted_cash(companyfacts, "2024-12-31")

    assert value is None
    assert meta is None


def test_extract_restricted_cash_does_not_double_count_synonymous_current_concepts():
    companyfacts = _companyfacts_with_entries(
        {
            "RestrictedCash": 2.0,
            "RestrictedCashCurrent": 2.0,
        }
    )

    value, meta = _extract_restricted_cash(companyfacts, "2024-12-31")

    assert value == 2.0
    assert meta is not None
    assert meta.get("mode") != "sum_current_components"


def test_extract_revolver_undrawn_respects_filing_date_cutoff():
    companyfacts = _companyfacts_with_fact_rows(
        {
            "LineOfCreditFacilityRemainingBorrowingCapacity": [
                {
                    "val": 9742.0,
                    "end": "2023-12-31",
                    "filed": "2024-02-16",
                    "fy": 2023,
                    "fp": "FY",
                    "form": "10-K",
                    "frame": "CY2023Q4I",
                },
                {
                    "val": 9929.0,
                    "end": "2024-12-31",
                    "filed": "2025-02-14",
                    "fy": 2024,
                    "fp": "FY",
                    "form": "10-K",
                    "frame": "CY2024Q4I",
                },
            ]
        }
    )

    value, meta = _extract_revolver_undrawn(companyfacts, "2024-12-31")

    assert value == 9742.0
    assert meta is not None
    assert meta["concept"] == "LineOfCreditFacilityRemainingBorrowingCapacity"
    assert meta["end"] == "2023-12-31"
    assert meta["filed"] == "2024-02-16"


def test_repair_restricted_cash_from_total_cash_reconciliation():
    companyfacts = _companyfacts_with_entries(
        {
            "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents": 125.0,
        }
    )
    cash_eq_node = {
        "support_mode": "exact",
        "value": 100.0,
        "component_breakdown": {"formula": "cash_and_equivalents_statement_direct"},
        "provenance": [{"artifact_type": "StatementDirect"}],
    }
    restricted_node = {
        "support_mode": "unsupported",
        "value": None,
        "missing_reason": "sec_concept_unavailable",
        "component_breakdown": {},
        "provenance": [],
    }

    repaired = _repair_restricted_cash_from_total_cash_reconciliation(
        restricted_node=restricted_node,
        cash_eq_node=cash_eq_node,
        cash_sti_node=None,
        marketable_node=None,
        companyfacts=companyfacts,
        companyfacts_path=Path("/tmp/CIK0000000000.json"),
        as_of_date="2024-12-31",
        as_of_time="2024-12-31T00:00:00+00:00",
        computed_at="2026-03-22T00:00:00+00:00",
    )

    assert repaired is not None
    assert repaired["support_mode"] == "exact"
    assert repaired["value"] == 25.0
    assert repaired["component_breakdown"]["mode"] == "cash_plus_restricted_total_minus_cash_equivalents"


