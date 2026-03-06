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


def test_effective_total_debt_baseline_prefers_fresher_companyfacts_debt():
    resolved = _effective_total_debt_baseline(
        total_debt={
            "support_mode": "exact",
            "value": 9_575_000_000.0,
            "provenance": [
                {
                    "artifact_type": "ExtractedFact",
                    "artifact_id": "debt_q3",
                    "source": "facts",
                    "published_at": "2025-11-04",
                    "ingested_at": "2025-11-05",
                    "hash": None,
                }
            ],
        },
        current_debt={"support_mode": "unsupported", "value": None},
        long_term_debt={"support_mode": "unsupported", "value": None},
        companyfacts={
            "facts": {
                "us-gaap": {
                    "DebtLongtermAndShorttermCombinedAmount": {
                        "units": {
                            "USD": [
                                {
                                    "val": 10_600_000_000.0,
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

    assert resolved["value"] == 10_600_000_000.0
    assert resolved["exact"] is True
    assert resolved["source_metric"] == "capital_structure.total_debt_companyfacts_exact"
    assert resolved["override_reason"] == "fresher_companyfacts_total_debt"


def test_effective_total_debt_baseline_preserves_supported_total_debt_floor_against_lower_companyfacts():
    resolved = _effective_total_debt_baseline(
        total_debt={
            "support_mode": "proxy_missing_component",
            "value": 9_575_000_000.0,
            "provenance": [
                {
                    "artifact_type": "ExtractedFact",
                    "artifact_id": "debt_q3",
                    "source": "facts",
                    "published_at": "2025-11-04",
                    "ingested_at": "2025-11-05",
                    "hash": None,
                }
            ],
        },
        current_debt={"support_mode": "unsupported", "value": None},
        long_term_debt={"support_mode": "unsupported", "value": None},
        companyfacts={
            "facts": {
                "us-gaap": {
                    "DebtLongtermAndShorttermCombinedAmount": {
                        "units": {
                            "USD": [
                                {
                                    "val": 8_100_000_000.0,
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

    assert resolved["value"] == 9_575_000_000.0
    assert resolved["exact"] is False
    assert resolved["source_metric"] == "capital_structure.total_debt_provider_direct"
    assert resolved["override_reason"] == "preserve_supported_total_debt_floor"


def test_effective_liquidity_components_infers_zero_restricted_and_marketable_from_cash_reconciliation():
    resolved = _effective_liquidity_component_values(
        cash_grouped={"support_mode": "proxy_missing_component", "value": 22_900_000.0},
        cash_exact={"support_mode": "exact", "value": 22_900_000.0},
        restricted_cash_sec={"support_mode": "unsupported", "value": None, "missing_reason": "sec_concept_unavailable"},
        marketable_sec={"support_mode": "unsupported", "value": None, "missing_reason": "sec_concept_unavailable"},
        restricted_cash={"support_mode": "unsupported", "value": None},
        marketable={"support_mode": "unsupported", "value": None},
    )

    assert resolved["restricted_cash_value"] == 0.0
    assert resolved["marketable_value"] == 0.0
    assert resolved["restricted_cash_zero_reconciled"] is True
    assert resolved["marketable_zero_reconciled"] is True


def test_grouped_cash_proxy_can_promote_exact_when_short_term_investments_are_absent():
    can_promote = _grouped_cash_proxy_can_promote_to_exact_cash_baseline(
        cash_grouped={
            "support_mode": "proxy_missing_component",
            "value": 270_300_000.0,
            "missing_reason": "cash_or_sti_component_missing",
            "component_breakdown": {
                "mode": "partial_cash_stack",
                "cash": {"concept": "CashAndCashEquivalentsAtCarryingValue"},
                "short_term_investments": None,
            },
        },
        cash_exact={"support_mode": "unsupported", "value": None, "missing_reason": "statement_fact_unavailable"},
        marketable_sec={"support_mode": "unsupported", "value": None, "missing_reason": "sec_concept_absent"},
        marketable={"support_mode": "unsupported", "value": None, "missing_reason": "not_disclosed"},
        marketable_inferred_zero=True,
    )

    assert can_promote is True


def test_grouped_cash_proxy_does_not_promote_when_short_term_investments_are_only_unavailable():
    can_promote = _grouped_cash_proxy_can_promote_to_exact_cash_baseline(
        cash_grouped={
            "support_mode": "proxy_missing_component",
            "value": 419_000_000.0,
            "missing_reason": "cash_or_sti_component_missing",
            "component_breakdown": {
                "mode": "partial_cash_stack",
                "cash": {"concept": "CashAndCashEquivalentsAtCarryingValue"},
                "short_term_investments": None,
            },
        },
        cash_exact={"support_mode": "unsupported", "value": None, "missing_reason": "statement_fact_unavailable"},
        marketable_sec={"support_mode": "unsupported", "value": None, "missing_reason": "sec_concept_unavailable"},
        marketable={"support_mode": "unsupported", "value": None, "missing_reason": "not_disclosed"},
        marketable_inferred_zero=False,
    )

    assert can_promote is False


def test_grouped_cash_proxy_can_complete_with_exact_marketable_securities():
    can_complete = _grouped_cash_proxy_can_complete_with_exact_marketable_securities(
        cash_grouped={
            "support_mode": "proxy_missing_component",
            "value": 834_000_000.0,
            "missing_reason": "cash_or_sti_component_missing",
            "component_breakdown": {
                "mode": "partial_cash_stack",
                "cash": {"concept": "Cash"},
                "short_term_investments": None,
            },
        },
        marketable_value=7_638_000_000.0,
    )

    assert can_complete is True


def test_effective_lease_liability_value_infers_zero_when_sec_concept_absent():
    resolved = _effective_lease_liability_value(
        {
            "support_mode": "unsupported",
            "value": None,
            "missing_reason": "sec_concept_absent",
        }
    )

    assert resolved["value"] == 0.0
    assert resolved["exact"] is True
    assert resolved["inferred_zero"] is True


def test_effective_lease_liability_value_promotes_fresh_reference_without_rou_override():
    resolved = _effective_lease_liability_value(
        {
            "support_mode": "unsupported",
            "value": None,
            "missing_reason": "sec_concept_unavailable",
            "component_breakdown": {
                "operating_reference": {
                    "present": True,
                    "direct_total_reference": {
                        "value": 33.0,
                        "components": [{"end": "2024-09-30"}],
                    },
                },
                "finance_reference": {
                    "present": False,
                },
            },
        },
        as_of_time="2024-12-31T00:00:00Z",
    )

    assert resolved["value"] == 33.0
    assert resolved["exact"] is True
    assert resolved["support_override"] == "fresh_liability_total_reference"


def test_effective_lease_liability_value_infers_zero_when_no_lease_references_are_present():
    resolved = _effective_lease_liability_value(
        {
            "support_mode": "unsupported",
            "value": None,
            "missing_reason": "sec_concept_unavailable",
            "component_breakdown": {
                "operating_reference": {"present": False},
                "finance_reference": {"present": False},
            },
        },
        as_of_time="2024-12-31T00:00:00Z",
    )

    assert resolved["value"] == 0.0
    assert resolved["exact"] is True
    assert resolved["inferred_zero"] is True
    assert resolved["support_override"] == "no_lease_references_present"


def test_effective_lease_liability_value_promotes_stale_operating_total_with_fresh_rou_asset():
    resolved = _effective_lease_liability_value(
        {
            "support_mode": "unsupported",
            "value": None,
            "missing_reason": "sec_concept_unavailable",
            "component_breakdown": {
                "operating_reference": {
                    "present": True,
                    "direct_total_reference": {
                        "value": 162_200_000.0,
                        "components": [{"end": "2023-12-31"}],
                    },
                    "right_of_use_asset_reference": {
                        "components": [{"end": "2024-09-28"}],
                    },
                },
                "finance_reference": {
                    "present": False,
                },
            },
        },
        as_of_time="2024-12-31T00:00:00Z",
    )

    assert resolved["value"] == 162_200_000.0
    assert resolved["exact"] is True
    assert resolved["support_override"] == "stale_liability_total_corroborated_by_fresh_rou_asset"


def test_effective_lease_liability_value_prefers_fresher_partial_reference_over_stale_direct_total():
    resolved = _effective_lease_liability_value(
        {
            "support_mode": "unsupported",
            "value": None,
            "missing_reason": "sec_concept_unavailable",
            "component_breakdown": {
                "operating_reference": {
                    "present": True,
                    "direct_total_reference": {
                        "value": 15.0,
                        "components": [{"end": "2023-12-31"}],
                    },
                    "partial_component_reference": {
                        "value": 24.0,
                        "current_components": [{"end": "2023-12-31"}],
                        "noncurrent_components": [{"end": "2024-09-30"}],
                    },
                },
                "finance_reference": {"present": False},
            },
        },
        as_of_time="2024-12-31T00:00:00Z",
    )

    assert resolved["value"] == 24.0
    assert resolved["exact"] is True
    assert resolved["support_override"] == "fresh_liability_total_reference"


def test_effective_lease_liability_value_combines_fresh_and_stale_rou_corroborated_classes():
    resolved = _effective_lease_liability_value(
        {
            "support_mode": "unsupported",
            "value": None,
            "missing_reason": "sec_concept_unavailable",
            "component_breakdown": {
                "operating_reference": {
                    "present": True,
                    "partial_component_reference": {
                        "value": 150.0,
                        "current_components": [{"end": "2024-09-30"}],
                        "noncurrent_components": [{"end": "2024-09-30"}],
                    },
                },
                "finance_reference": {
                    "present": True,
                    "direct_total_reference": {
                        "value": 20.0,
                        "components": [{"end": "2023-12-31"}],
                    },
                    "right_of_use_asset_reference": {
                        "components": [{"end": "2024-09-30"}],
                    },
                },
            },
        },
        as_of_time="2024-12-31T00:00:00Z",
    )

    assert resolved["value"] == 170.0
    assert resolved["exact"] is True
    assert resolved["support_override"] == "hybrid_fresh_and_stale_liability_total_reference"


def test_effective_lease_liability_value_promotes_immaterial_stale_finance_tail():
    resolved = _effective_lease_liability_value(
        {
            "support_mode": "unsupported",
            "value": None,
            "missing_reason": "sec_concept_unavailable",
            "component_breakdown": {
                "operating_reference": {
                    "present": True,
                    "partial_component_reference": {
                        "value": 150_000_000.0,
                        "current_components": [{"end": "2024-09-30"}],
                        "noncurrent_components": [{"end": "2024-09-30"}],
                    },
                },
                "finance_reference": {
                    "present": True,
                    "direct_total_reference": {
                        "value": 4_000_000.0,
                        "components": [{"end": "2023-12-31"}],
                    },
                },
            },
        },
        as_of_time="2024-12-31T00:00:00Z",
        total_debt_value=2_000_000_000.0,
    )

    assert resolved["value"] == 154_000_000.0
    assert resolved["exact"] is True
    assert resolved["support_override"] == "hybrid_fresh_and_immaterial_stale_liability_total_reference"


def test_effective_lease_liability_value_does_not_promote_material_stale_finance_tail():
    resolved = _effective_lease_liability_value(
        {
            "support_mode": "unsupported",
            "value": None,
            "missing_reason": "sec_concept_unavailable",
            "component_breakdown": {
                "operating_reference": {
                    "present": True,
                    "partial_component_reference": {
                        "value": 150_000_000.0,
                        "current_components": [{"end": "2024-09-30"}],
                        "noncurrent_components": [{"end": "2024-09-30"}],
                    },
                },
                "finance_reference": {
                    "present": True,
                    "direct_total_reference": {
                        "value": 60_000_000.0,
                        "components": [{"end": "2023-12-31"}],
                    },
                },
            },
        },
        as_of_time="2024-12-31T00:00:00Z",
        total_debt_value=2_000_000_000.0,
    )

    assert resolved["value"] is None
    assert resolved["exact"] is False
    assert resolved["support_override"] is None


def test_effective_lease_liability_value_requires_fresh_rou_for_each_present_class():
    resolved = _effective_lease_liability_value(
        {
            "support_mode": "unsupported",
            "value": None,
            "missing_reason": "sec_concept_unavailable",
            "component_breakdown": {
                "operating_reference": {
                    "present": True,
                    "direct_total_reference": {
                        "value": 150_000_000.0,
                        "components": [{"end": "2024-09-30"}],
                    },
                    "right_of_use_asset_reference": {
                        "components": [{"end": "2024-09-30"}],
                    },
                },
                "finance_reference": {
                    "present": True,
                    "direct_total_reference": {
                        "value": 20_000_000.0,
                        "components": [{"end": "2023-12-31"}],
                    },
                    "right_of_use_asset_reference": {
                        "components": [{"end": "2023-12-31"}],
                    },
                },
            },
        },
        as_of_time="2024-12-31T00:00:00Z",
    )

    assert resolved["value"] is None
    assert resolved["exact"] is False
    assert resolved["support_override"] is None


