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


def test_cash_sti_proxy_represents_cash_only_when_short_term_investments_are_absent():
    assert _cash_sti_proxy_represents_cash_only(
        cash_sti_node={
            "support_mode": "proxy_missing_component",
            "value": 98.0,
            "missing_reason": "cash_or_sti_component_missing",
            "component_breakdown": {
                "mode": "partial_cash_stack",
                "cash": {"concept": "CashAndCashEquivalentsAtCarryingValue"},
                "short_term_investments": None,
            },
        },
        marketable_node={
            "support_mode": "unsupported",
            "value": None,
            "missing_reason": "sec_concept_absent",
        },
    )


def test_repair_restricted_cash_from_grouped_cash_proxy_reconciliation():
    companyfacts = _companyfacts_with_entries(
        {
            "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents": 125.0,
        }
    )
    grouped_cash_node = {
        "support_mode": "proxy_missing_component",
        "value": 100.0,
        "missing_reason": "cash_or_sti_component_missing",
        "component_breakdown": {
            "mode": "partial_cash_stack",
            "cash": {"concept": "CashAndCashEquivalentsAtCarryingValue"},
            "short_term_investments": None,
        },
        "provenance": [{"artifact_type": "SecCompanyFacts"}],
    }
    repaired = _repair_restricted_cash_from_total_cash_reconciliation(
        restricted_node={
            "support_mode": "unsupported",
            "value": None,
            "missing_reason": "sec_concept_unavailable",
            "component_breakdown": {},
            "provenance": [],
        },
        cash_eq_node={
            "support_mode": "unsupported",
            "value": None,
            "missing_reason": "statement_fact_unavailable",
            "component_breakdown": {},
            "provenance": [],
        },
        cash_sti_node=grouped_cash_node,
        marketable_node={
            "support_mode": "unsupported",
            "value": None,
            "missing_reason": "sec_concept_absent",
        },
        companyfacts=companyfacts,
        companyfacts_path=Path("/tmp/CIK0000000000.json"),
        as_of_date="2024-12-31",
        as_of_time="2024-12-31T00:00:00+00:00",
        computed_at="2026-03-22T00:00:00+00:00",
    )

    assert repaired is not None
    assert repaired["support_mode"] == "exact"
    assert repaired["value"] == 25.0
    assert repaired["component_breakdown"]["mode"] == "cash_plus_restricted_total_minus_grouped_cash_cash_only_proxy"


def test_statement_fact_freshness_rejects_stale_cash_fact():
    assert not _statement_fact_node_is_fresh_enough(
        {
            "support_mode": "exact",
            "component_breakdown": {
                "effective_at": "2022-10-31",
            },
        },
        "2024-12-31",
    )


def test_repair_cash_sti_from_statement_cash_rejects_stale_statement_cash():
    repaired = _repair_cash_sti_from_statement_cash(
        cash_sti_node={
            "support_mode": "proxy_missing_component",
            "value": 100.0,
            "missing_reason": "cash_or_sti_component_missing",
            "component_breakdown": {
                "mode": "partial_cash_stack",
            },
            "provenance": [],
        },
        cash_eq_node={
            "support_mode": "exact",
            "value": 125.0,
            "component_breakdown": {
                "effective_at": "2022-10-31",
                "formula": "statement_direct_fact",
            },
            "provenance": [{"artifact_type": "StatementDirect"}],
        },
        marketable_node={
            "support_mode": "unsupported",
            "value": None,
            "missing_reason": "sec_concept_absent",
            "component_breakdown": {},
            "provenance": [],
        },
        companyfacts_path=Path("/tmp/CIK0000000000.json"),
        as_of_date="2024-12-31",
        as_of_time="2024-12-31T00:00:00+00:00",
        computed_at="2026-03-25T00:00:00+00:00",
    )

    assert repaired is None


def test_extract_lease_liabilities_promotes_stale_operating_total_with_fresh_rou_when_finance_absent():
    companyfacts = _companyfacts_with_fact_rows(
        {
            "OperatingLeaseLiability": [
                {
                    "val": 15.0,
                    "end": "2023-12-31",
                    "filed": "2024-03-05",
                    "fy": 2023,
                    "fp": "FY",
                    "form": "10-K",
                }
            ],
            "OperatingLeaseRightOfUseAsset": [
                {
                    "val": 24.0,
                    "end": "2024-09-30",
                    "filed": "2024-10-29",
                    "fy": 2024,
                    "fp": "Q3",
                    "form": "10-Q",
                }
            ],
        }
    )

    value, meta = _extract_lease_liabilities(companyfacts, "2024-12-31")

    assert value == 15.0
    assert meta is not None
    assert meta["mode"] == "operating_only_no_finance_concepts"
    assert (
        meta["operating_component"]["support_override"]
        == "stale_liability_total_corroborated_by_fresh_rou_asset"
    )


def test_extract_lease_liabilities_promotes_stale_operating_and_finance_totals_with_fresh_rou():
    companyfacts = _companyfacts_with_fact_rows(
        {
            "OperatingLeaseLiability": [
                {
                    "val": 15.0,
                    "end": "2023-12-31",
                    "filed": "2024-03-05",
                    "fy": 2023,
                    "fp": "FY",
                    "form": "10-K",
                }
            ],
            "FinanceLeaseLiability": [
                {
                    "val": 5.0,
                    "end": "2023-12-31",
                    "filed": "2024-03-05",
                    "fy": 2023,
                    "fp": "FY",
                    "form": "10-K",
                }
            ],
            "OperatingLeaseRightOfUseAsset": [
                {
                    "val": 24.0,
                    "end": "2024-09-30",
                    "filed": "2024-10-29",
                    "fy": 2024,
                    "fp": "Q3",
                    "form": "10-Q",
                }
            ],
            "FinanceLeaseRightOfUseAsset": [
                {
                    "val": 6.0,
                    "end": "2024-09-30",
                    "filed": "2024-10-29",
                    "fy": 2024,
                    "fp": "Q3",
                    "form": "10-Q",
                }
            ],
        }
    )

    value, meta = _extract_lease_liabilities(companyfacts, "2024-12-31")

    assert value == 20.0
    assert meta is not None
    assert meta["mode"] == "sum_operating_finance"
    assert (
        meta["operating_component"]["support_override"]
        == "stale_liability_total_corroborated_by_fresh_rou_asset"
    )
    assert (
        meta["finance_component"]["support_override"]
        == "stale_liability_total_corroborated_by_fresh_rou_asset"
    )


def test_extract_lease_liabilities_promotes_fresh_noncurrent_plus_stale_current_when_finance_support_is_only_stale():
    companyfacts = _companyfacts_with_fact_rows(
        {
            "OperatingLeaseLiabilityCurrent": [
                {
                    "val": 4.278,
                    "end": "2023-12-31",
                    "filed": "2024-03-05",
                    "fy": 2023,
                    "fp": "FY",
                    "form": "10-K",
                }
            ],
            "LesseeOperatingLeaseLiabilityPaymentsDueNextTwelveMonths": [
                {
                    "val": 4.363,
                    "end": "2023-12-31",
                    "filed": "2024-03-05",
                    "fy": 2023,
                    "fp": "FY",
                    "form": "10-K",
                }
            ],
            "OperatingLeaseLiability": [
                {
                    "val": 15.695,
                    "end": "2023-12-31",
                    "filed": "2024-03-05",
                    "fy": 2023,
                    "fp": "FY",
                    "form": "10-K",
                }
            ],
            "OperatingLeaseLiabilityNoncurrent": [
                {
                    "val": 19.696,
                    "end": "2024-09-30",
                    "filed": "2024-10-29",
                    "fy": 2024,
                    "fp": "Q3",
                    "form": "10-Q",
                }
            ],
            "OperatingLeaseRightOfUseAsset": [
                {
                    "val": 16.256,
                    "end": "2023-12-31",
                    "filed": "2024-03-05",
                    "fy": 2023,
                    "fp": "FY",
                    "form": "10-K",
                },
                {
                    "val": 24.690,
                    "end": "2024-09-30",
                    "filed": "2024-10-29",
                    "fy": 2024,
                    "fp": "Q3",
                    "form": "10-Q",
                }
            ],
            "FinanceLeaseLiability": [
                {
                    "val": 5.537,
                    "end": "2023-12-31",
                    "filed": "2024-03-05",
                    "fy": 2023,
                    "fp": "FY",
                    "form": "10-K",
                }
            ],
            "FinanceLeaseRightOfUseAsset": [
                {
                    "val": 2.097,
                    "end": "2023-12-31",
                    "filed": "2024-03-05",
                    "fy": 2023,
                    "fp": "FY",
                    "form": "10-K",
                }
            ],
        }
    )

    value, meta = _extract_lease_liabilities(companyfacts, "2024-12-31")

    assert value == 24.059
    assert meta is not None
    assert meta["mode"] == "operating_only_no_fresh_finance_support"
    assert (
        meta["operating_component"]["support_override"]
        == "mixed_fresh_and_stale_components_rebased_by_rou_basis_delta"
    )
    assert meta["operating_component"]["stale_component"] == "current"
    assert meta["operating_component"]["stale_basis_delta"] == pytest.approx(-0.561)
    assert meta["operating_component"]["stale_current_due_cap_component"]["concept"] == "LesseeOperatingLeaseLiabilityPaymentsDueNextTwelveMonths"


def test_extract_lease_liabilities_prefers_fresh_operating_components_over_stale_direct_total():
    companyfacts = _companyfacts_with_fact_rows(
        {
            "OperatingLeaseLiability": [
                {
                    "val": 20.187,
                    "end": "2023-12-30",
                    "filed": "2024-02-21",
                    "fy": 2023,
                    "fp": "FY",
                    "form": "10-K",
                }
            ],
            "OperatingLeaseLiabilityCurrent": [
                {
                    "val": 16.194,
                    "end": "2024-09-28",
                    "filed": "2024-10-30",
                    "fy": 2024,
                    "fp": "Q3",
                    "form": "10-Q",
                }
            ],
            "OperatingLeaseLiabilityNoncurrent": [
                {
                    "val": 47.753,
                    "end": "2024-09-28",
                    "filed": "2024-10-30",
                    "fy": 2024,
                    "fp": "Q3",
                    "form": "10-Q",
                }
            ],
            "OperatingLeaseRightOfUseAsset": [
                {
                    "val": 57.753,
                    "end": "2024-09-28",
                    "filed": "2024-10-30",
                    "fy": 2024,
                    "fp": "Q3",
                    "form": "10-Q",
                }
            ],
        }
    )

    value, meta = _extract_lease_liabilities(companyfacts, "2024-12-31")

    assert value == pytest.approx(63.947)
    assert meta is not None
    assert meta["mode"] == "operating_only_no_finance_concepts"
    assert meta["operating_component"]["mode"] == "operating_sum_current_noncurrent"


def test_extract_lease_liabilities_prefers_fresh_finance_components_over_stale_direct_total():
    companyfacts = _companyfacts_with_fact_rows(
        {
            "OperatingLeaseLiabilityCurrent": [
                {
                    "val": 4.2,
                    "end": "2024-09-30",
                    "filed": "2024-11-07",
                    "fy": 2024,
                    "fp": "Q3",
                    "form": "10-Q",
                }
            ],
            "OperatingLeaseLiabilityNoncurrent": [
                {
                    "val": 14.5,
                    "end": "2024-09-30",
                    "filed": "2024-11-07",
                    "fy": 2024,
                    "fp": "Q3",
                    "form": "10-Q",
                }
            ],
            "OperatingLeaseLiability": [
                {
                    "val": 14.5,
                    "end": "2023-12-31",
                    "filed": "2024-04-15",
                    "fy": 2023,
                    "fp": "FY",
                    "form": "10-K",
                }
            ],
            "OperatingLeaseRightOfUseAsset": [
                {
                    "val": 18.2,
                    "end": "2024-09-30",
                    "filed": "2024-11-07",
                    "fy": 2024,
                    "fp": "Q3",
                    "form": "10-Q",
                }
            ],
            "FinanceLeaseLiabilityCurrent": [
                {
                    "val": 4.4,
                    "end": "2024-09-30",
                    "filed": "2024-11-07",
                    "fy": 2024,
                    "fp": "Q3",
                    "form": "10-Q",
                }
            ],
            "FinanceLeaseLiabilityNoncurrent": [
                {
                    "val": 20.0,
                    "end": "2024-09-30",
                    "filed": "2024-11-07",
                    "fy": 2024,
                    "fp": "Q3",
                    "form": "10-Q",
                }
            ],
            "FinanceLeaseLiability": [
                {
                    "val": 2.4,
                    "end": "2023-12-31",
                    "filed": "2024-04-15",
                    "fy": 2023,
                    "fp": "FY",
                    "form": "10-K",
                }
            ],
        }
    )

    value, meta = _extract_lease_liabilities(companyfacts, "2024-12-31")

    assert value == pytest.approx(43.1)
    assert meta is not None
    assert meta["mode"] == "sum_operating_finance"
    assert meta["finance_component"]["mode"] == "finance_sum_current_noncurrent"


def test_extract_lease_liabilities_promotes_stale_internally_consistent_operating_and_finance_totals():
    companyfacts = _companyfacts_with_fact_rows(
        {
            "OperatingLeaseLiability": [
                {
                    "val": 617.0,
                    "end": "2023-12-30",
                    "filed": "2024-02-15",
                    "fy": 2023,
                    "fp": "FY",
                    "form": "10-K",
                }
            ],
            "OperatingLeaseLiabilityCurrent": [
                {
                    "val": 136.0,
                    "end": "2023-12-30",
                    "filed": "2024-02-15",
                    "fy": 2023,
                    "fp": "FY",
                    "form": "10-K",
                }
            ],
            "OperatingLeaseLiabilityNoncurrent": [
                {
                    "val": 481.0,
                    "end": "2023-12-30",
                    "filed": "2024-02-15",
                    "fy": 2023,
                    "fp": "FY",
                    "form": "10-K",
                }
            ],
            "LesseeOperatingLeaseLiabilityPaymentsDue": [
                {
                    "val": 700.0,
                    "end": "2023-12-30",
                    "filed": "2024-02-15",
                    "fy": 2023,
                    "fp": "FY",
                    "form": "10-K",
                }
            ],
            "LesseeOperatingLeaseLiabilityUndiscountedExcessAmount": [
                {
                    "val": 83.0,
                    "end": "2023-12-30",
                    "filed": "2024-02-15",
                    "fy": 2023,
                    "fp": "FY",
                    "form": "10-K",
                }
            ],
            "FinanceLeaseLiability": [
                {
                    "val": 145.0,
                    "end": "2023-12-30",
                    "filed": "2024-02-15",
                    "fy": 2023,
                    "fp": "FY",
                    "form": "10-K",
                }
            ],
            "FinanceLeaseLiabilityCurrent": [
                {
                    "val": 32.0,
                    "end": "2023-12-30",
                    "filed": "2024-02-15",
                    "fy": 2023,
                    "fp": "FY",
                    "form": "10-K",
                }
            ],
            "FinanceLeaseLiabilityNoncurrent": [
                {
                    "val": 113.0,
                    "end": "2023-12-30",
                    "filed": "2024-02-15",
                    "fy": 2023,
                    "fp": "FY",
                    "form": "10-K",
                }
            ],
            "FinanceLeaseLiabilityPaymentsDue": [
                {
                    "val": 180.0,
                    "end": "2023-12-30",
                    "filed": "2024-02-15",
                    "fy": 2023,
                    "fp": "FY",
                    "form": "10-K",
                }
            ],
            "FinanceLeaseLiabilityUndiscountedExcessAmount": [
                {
                    "val": 35.0,
                    "end": "2023-12-30",
                    "filed": "2024-02-15",
                    "fy": 2023,
                    "fp": "FY",
                    "form": "10-K",
                }
            ],
        }
    )

    value, meta = _extract_lease_liabilities(companyfacts, "2024-12-31")

    assert value == pytest.approx(762.0)
    assert meta is not None
    assert meta["mode"] == "sum_operating_finance"
    assert (
        meta["operating_component"]["support_override"]
        == "stale_internally_consistent_lease_carry_forward"
    )
    assert (
        meta["finance_component"]["support_override"]
        == "stale_internally_consistent_lease_carry_forward"
    )


def test_extract_lease_liabilities_does_not_promote_stale_only_references_outside_carry_forward_window():
    companyfacts = _companyfacts_with_fact_rows(
        {
            "OperatingLeaseLiability": [
                {
                    "val": 20.0,
                    "end": "2020-12-31",
                    "filed": "2021-02-15",
                    "fy": 2020,
                    "fp": "FY",
                    "form": "10-K",
                }
            ],
            "OperatingLeaseLiabilityCurrent": [
                {
                    "val": 5.0,
                    "end": "2020-12-31",
                    "filed": "2021-02-15",
                    "fy": 2020,
                    "fp": "FY",
                    "form": "10-K",
                }
            ],
            "OperatingLeaseLiabilityNoncurrent": [
                {
                    "val": 15.0,
                    "end": "2020-12-31",
                    "filed": "2021-02-15",
                    "fy": 2020,
                    "fp": "FY",
                    "form": "10-K",
                }
            ],
        }
    )

    value, meta = _extract_lease_liabilities(companyfacts, "2024-12-31")

    assert value is None
    assert meta is not None
    assert meta["mode"] == "lease_total_unavailable"
