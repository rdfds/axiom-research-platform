import json
from pathlib import Path

import scripts.backfill_smart_normalized_metrics_v1 as smart_mod
from scripts.backfill_smart_normalized_metrics_v1 import (
    _build_fail_open_smart_metrics,
    _effective_cash_equivalents_value,
    _effective_liquidity_component_values,
    _effective_lease_liability_value,
    _extract_retirement_note_components_from_html,
    _effective_total_debt_baseline,
    _grouped_cash_proxy_can_complete_with_exact_marketable_securities,
    _grouped_cash_proxy_can_promote_to_exact_cash_baseline,
    _load_completed_company_ids,
    _load_companyfacts,
    _load_retirement_note_components,
    _market_availability_adjustment,
    _summarize_output_rows,
    materialize_smart_metrics_for_row,
)


def test_effective_liquidity_components_infers_zero_restricted_cash_from_grouped_reconciliation():
    resolved = _effective_liquidity_component_values(
        cash_grouped={"support_mode": "exact", "value": 120.0},
        cash_exact={"support_mode": "exact", "value": 100.0},
        restricted_cash_sec={"support_mode": "unsupported", "value": None, "missing_reason": "sec_concept_unavailable"},
        marketable_sec={"support_mode": "exact", "value": 20.0},
        restricted_cash={"support_mode": "unsupported", "value": None},
        marketable={"support_mode": "unsupported", "value": None},
    )

    assert resolved["restricted_cash_value"] == 0.0
    assert resolved["restricted_cash_inferred_zero"] is True
    assert resolved["restricted_cash_zero_reconciled"] is True


def test_effective_liquidity_components_market_defaults_restricted_cash_when_grouped_cash_is_exact():
    resolved = _effective_liquidity_component_values(
        cash_grouped={"support_mode": "exact", "value": 120.0},
        cash_exact={"support_mode": "unsupported", "value": None, "missing_reason": "statement_fact_unavailable"},
        restricted_cash_sec={"support_mode": "unsupported", "value": None, "missing_reason": "sec_concept_unavailable"},
        marketable_sec={"support_mode": "unsupported", "value": None, "missing_reason": "sec_concept_absent"},
        restricted_cash={"support_mode": "unsupported", "value": None},
        marketable={"support_mode": "unsupported", "value": None},
    )

    assert resolved["restricted_cash_value"] == 0.0
    assert resolved["restricted_cash_inferred_zero"] is True
    assert resolved["restricted_cash_market_default_zero"] is True


def test_effective_liquidity_components_infers_zero_marketable_from_grouped_reconciliation():
    resolved = _effective_liquidity_component_values(
        cash_grouped={"support_mode": "proxy_missing_component", "value": 46_699_000.0},
        cash_exact={"support_mode": "exact", "value": 45_300_000.0},
        restricted_cash_sec={"support_mode": "exact", "value": 1_399_000.0},
        marketable_sec={"support_mode": "unsupported", "value": None, "missing_reason": "sec_concept_unavailable"},
        restricted_cash={"support_mode": "unsupported", "value": None},
        marketable={"support_mode": "unsupported", "value": None},
    )

    assert resolved["marketable_value"] == 0.0
    assert resolved["marketable_inferred_zero"] is True
    assert resolved["marketable_zero_reconciled"] is True


def test_effective_total_debt_baseline_promotes_single_statement_component_match():
    resolved = _effective_total_debt_baseline(
        total_debt={
            "support_mode": "proxy_missing_component",
            "value": 62_000_000.0,
            "missing_reason": "debt_component_missing",
            "component_breakdown": {"mode": "partial_debt_stack"},
        },
        current_debt={"support_mode": "unsupported", "value": None},
        long_term_debt={"support_mode": "exact", "value": 62_000_000.0},
    )

    assert resolved["value"] == 62_000_000.0
    assert resolved["exact"] is True
    assert resolved["source_metric"] == "capital_structure.long_term_debt_statement_direct"
    assert resolved["override_reason"] == "single_statement_component_matches_total_debt"


def test_effective_total_debt_baseline_uses_long_term_debt_when_short_term_borrowings_duplicate_current():
    resolved = _effective_total_debt_baseline(
        total_debt={
            "support_mode": "exact",
            "value": 439_600_000.0,
            "component_breakdown": {
                "mode": "current_plus_noncurrent_debt_plus_short_term_borrowings",
            },
        },
        current_debt={
            "support_mode": "exact",
            "value": 39_700_000.0,
            "component_breakdown": {"concept": "LongTermDebtCurrent"},
        },
        long_term_debt={
            "support_mode": "exact",
            "value": 410_600_000.0,
            "component_breakdown": {"concept": "LongTermDebt"},
        },
    )

    assert resolved["value"] == 410_600_000.0
    assert resolved["exact"] is True
    assert resolved["source_metric"] == "capital_structure.long_term_debt_statement_direct"
    assert resolved["override_reason"] == "short_term_borrowings_overlap_current_debt"


def test_effective_total_debt_baseline_keeps_total_debt_when_overlap_delta_is_not_material():
    resolved = _effective_total_debt_baseline(
        total_debt={
            "support_mode": "exact",
            "value": 193_294_000.0,
            "component_breakdown": {
                "mode": "current_plus_noncurrent_debt_plus_short_term_borrowings",
                "current": {"concept": "LongTermDebtCurrent"},
            },
        },
        current_debt={
            "support_mode": "unsupported",
            "value": None,
            "component_breakdown": {"concept": "LongTermDebtCurrent"},
        },
        long_term_debt={
            "support_mode": "exact",
            "value": 193_750_000.0,
            "component_breakdown": {"concept": "LongTermDebt"},
        },
    )

    assert resolved["value"] == 193_294_000.0
    assert resolved["exact"] is True
    assert resolved["source_metric"] == "capital_structure.total_debt_provider_direct"
    assert resolved["override_reason"] is None


def test_effective_total_debt_baseline_falls_back_to_partial_statement_components():
    resolved = _effective_total_debt_baseline(
        total_debt={"support_mode": "unsupported", "value": None},
        current_debt={"support_mode": "exact", "value": 12_000_000.0},
        long_term_debt={"support_mode": "unsupported", "value": None},
    )

    assert resolved["value"] == 12_000_000.0
    assert resolved["exact"] is False
    assert resolved["source_metric"] == (
        "capital_structure.current_debt_statement_direct + "
        "capital_structure.long_term_debt_statement_direct"
    )
    assert resolved["formula"] == "sum_available_statement_debt_components"


def test_effective_cash_equivalents_value_prefers_fresher_companyfacts_cash():
    resolved = _effective_cash_equivalents_value(
        {
            "support_mode": "exact",
            "value": 8_432_000_000.0,
            "provenance": [
                {
                    "artifact_type": "ExtractedFact",
                    "artifact_id": "cash_q3",
                    "source": "facts",
                    "published_at": "2025-11-04",
                    "ingested_at": "2025-11-05",
                    "hash": None,
                }
            ],
        },
        companyfacts={
            "facts": {
                "us-gaap": {
                    "CashAndCashEquivalentsAtCarryingValue": {
                        "units": {
                            "USD": [
                                {
                                    "val": 7_105_000_000.0,
                                    "end": "2025-12-31",
                                    "filed": "2026-02-25",
                                    "fy": 2025,
                                    "fp": "FY",
                                    "form": "10-K",
                                }
                            ]
                        }
                    }
                }
            }
        },
        as_of_time="2026-02-28T00:00:00Z",
    )

    assert resolved["value"] == 7_105_000_000.0
    assert resolved["exact"] is True
    assert resolved["source_metric"] == "liquidity.cash_and_equivalents_companyfacts_exact"
    assert resolved["support_override"] == "companyfacts_cash_exact_newer_than_provider_direct"


