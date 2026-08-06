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


def test_effective_lease_liability_value_ignores_partial_reference_without_total_value():
    resolved = _effective_lease_liability_value(
        {
            "support_mode": "unsupported",
            "value": None,
            "missing_reason": "sec_concept_unavailable",
            "component_breakdown": {
                "operating_reference": {
                    "present": True,
                    "partial_component_reference": {
                        "value": None,
                        "current_components": [],
                        "noncurrent_components": [{"end": "2024-09-30"}],
                    },
                    "right_of_use_asset_reference": {
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

    assert resolved["value"] is None
    assert resolved["exact"] is False


def test_effective_lease_liability_value_ignores_negative_exact_value():
    resolved = _effective_lease_liability_value(
        {
            "support_mode": "exact",
            "value": -25.0,
        },
        as_of_time="2024-12-31T00:00:00Z",
    )

    assert resolved["value"] is None
    assert resolved["exact"] is False
    assert resolved["support_override"] == "negative_lease_liability_exact_value_ignored"


def test_materialize_smart_metrics_for_row_promotes_stale_lease_and_updates_debt_metrics():
    registry = {
        "metrics": {
            "debt_like_obligations_normalized": {"status": "partially_feasible", "promotion_rule": "test"},
            "available_liquidity_normalized": {"status": "partially_feasible", "promotion_rule": "test"},
            "operating_earnings_normalized": {"status": "partially_feasible", "promotion_rule": "test"},
            "net_debt_normalized": {"status": "partially_feasible", "promotion_rule": "test"},
            "gross_leverage_normalized": {"status": "partially_feasible", "promotion_rule": "test"},
            "net_leverage_normalized": {"status": "partially_feasible", "promotion_rule": "test"},
        }
    }
    row = {
        "company_id": "123",
        "as_of_time": "2024-12-31T00:00:00Z",
        "features": {
            "capital_structure.total_debt_provider_direct": {
                "support_mode": "exact",
                "value": 100.0,
                "component_breakdown": {"mode": "current_plus_noncurrent_debt"},
            },
            "capital_structure.current_debt_statement_direct": {"support_mode": "unsupported", "value": None},
            "capital_structure.long_term_debt_statement_direct": {"support_mode": "unsupported", "value": None},
            "capital_structure.lease_liabilities_sec_exact": {
                "support_mode": "unsupported",
                "value": None,
                "missing_reason": "sec_concept_unavailable",
                "component_breakdown": {
                    "operating_reference": {
                        "present": True,
                        "direct_total_reference": {
                            "value": 25.0,
                            "components": [{"end": "2023-12-31"}],
                        },
                        "right_of_use_asset_reference": {
                            "components": [{"end": "2024-09-30"}],
                        },
                    },
                    "finance_reference": {
                        "present": False,
                    },
                },
            },
            "liquidity.cash_and_short_term_investments_provider_direct": {
                "support_mode": "exact",
                "value": 20.0,
            },
            "liquidity.cash_and_equivalents_statement_direct": {"support_mode": "exact", "value": 20.0},
            "liquidity.restricted_cash_sec_exact": {
                "support_mode": "unsupported",
                "value": None,
                "missing_reason": "sec_concept_absent",
            },
            "liquidity.marketable_securities_sec_exact": {
                "support_mode": "unsupported",
                "value": None,
                "missing_reason": "sec_concept_absent",
            },
            "liquidity.restricted_cash": {"support_mode": "unsupported", "value": None},
            "liquidity.marketable_securities": {"support_mode": "unsupported", "value": None},
            "liquidity.revolver_undrawn_sec_exact": {"support_mode": "unsupported", "value": None},
            "liquidity.revolver_undrawn": {"support_mode": "unsupported", "value": None},
            "operating.ebitda_ltm_provider_direct": {"support_mode": "exact", "value": 50.0},
        },
    }

    repaired = materialize_smart_metrics_for_row(
        row=row,
        registry=registry,
        computed_at="2026-03-22T00:00:00Z",
        provenance_sources=["registry.json"],
    )

    assert repaired["features"]["capital_structure.debt_like_obligations_normalized"]["support_mode"] == "exact"
    assert repaired["features"]["capital_structure.debt_like_obligations_normalized"]["value"] == 125.0
    assert (
        repaired["features"]["capital_structure.debt_like_obligations_normalized"]["component_breakdown"][
            "lease_support_override"
        ]
        == "stale_liability_total_corroborated_by_fresh_rou_asset"
    )
    assert repaired["features"]["capital_structure.net_debt_normalized"]["value"] == 105.0
    assert repaired["features"]["capital_structure.gross_leverage_normalized"]["value"] == 2.5
    assert repaired["features"]["capital_structure.net_leverage_normalized"]["value"] == 2.1


def test_materialize_smart_metrics_for_row_floors_debt_like_at_total_debt_when_lease_input_is_negative():
    registry = {
        "metrics": {
            "debt_like_obligations_normalized": {"status": "partially_feasible", "promotion_rule": "test"},
            "available_liquidity_normalized": {"status": "partially_feasible", "promotion_rule": "test"},
            "operating_earnings_normalized": {"status": "partially_feasible", "promotion_rule": "test"},
            "net_debt_normalized": {"status": "partially_feasible", "promotion_rule": "test"},
            "gross_leverage_normalized": {"status": "partially_feasible", "promotion_rule": "test"},
            "net_leverage_normalized": {"status": "partially_feasible", "promotion_rule": "test"},
        }
    }
    row = {
        "company_id": "negative-lease-probe",
        "as_of_time": "2024-12-31T00:00:00Z",
        "features": {
            "capital_structure.total_debt_provider_direct": {"support_mode": "exact", "value": 100.0},
            "capital_structure.current_debt_statement_direct": {"support_mode": "unsupported", "value": None},
            "capital_structure.long_term_debt_statement_direct": {"support_mode": "unsupported", "value": None},
            "capital_structure.lease_liabilities_sec_exact": {"support_mode": "exact", "value": -20.0},
            "liquidity.cash_and_short_term_investments_provider_direct": {"support_mode": "exact", "value": 20.0},
            "liquidity.cash_and_equivalents_statement_direct": {"support_mode": "exact", "value": 20.0},
            "liquidity.restricted_cash_sec_exact": {
                "support_mode": "unsupported",
                "value": None,
                "missing_reason": "sec_concept_absent",
            },
            "liquidity.marketable_securities_sec_exact": {
                "support_mode": "unsupported",
                "value": None,
                "missing_reason": "sec_concept_absent",
            },
            "liquidity.restricted_cash": {"support_mode": "unsupported", "value": None},
            "liquidity.marketable_securities": {"support_mode": "unsupported", "value": None},
            "liquidity.revolver_undrawn_sec_exact": {"support_mode": "unsupported", "value": None},
            "liquidity.revolver_undrawn": {"support_mode": "unsupported", "value": None},
            "operating.ebitda_ltm_provider_direct": {"support_mode": "exact", "value": 40.0},
        },
    }

    repaired = materialize_smart_metrics_for_row(
        row=row,
        registry=registry,
        computed_at="2026-04-01T00:00:00Z",
        provenance_sources=["registry.json"],
    )

    debt_like = repaired["features"]["capital_structure.debt_like_obligations_normalized"]
    assert debt_like["support_mode"] == "proxy_missing_component"
    assert debt_like["value"] == 100.0
    assert debt_like["component_breakdown"]["lease_negative_input_ignored"] is True
    assert repaired["features"]["capital_structure.net_debt_normalized"]["value"] == 80.0
    assert repaired["features"]["capital_structure.gross_leverage_normalized"]["value"] == 2.5
    assert repaired["features"]["capital_structure.net_leverage_normalized"]["value"] == 2.0


def test_materialize_smart_metrics_for_row_repairs_operating_earnings_from_net_income_tax_interest_and_dna():
    registry = {
        "metrics": {
            "debt_like_obligations_normalized": {"status": "partially_feasible", "promotion_rule": "test"},
            "available_liquidity_normalized": {"status": "partially_feasible", "promotion_rule": "test"},
            "operating_earnings_normalized": {"status": "partially_feasible", "promotion_rule": "test"},
            "net_debt_normalized": {"status": "partially_feasible", "promotion_rule": "test"},
            "gross_leverage_normalized": {"status": "partially_feasible", "promotion_rule": "test"},
            "net_leverage_normalized": {"status": "partially_feasible", "promotion_rule": "test"},
        }
    }
    companyfacts = {
        "facts": {
            "us-gaap": {
                "InterestExpense": {
                    "units": {
                        "USD": [
                            {"start": "2024-01-01", "end": "2024-12-31", "filed": "2024-12-31", "val": 10.0, "fy": 2024, "fp": "FY"}
                        ]
                    }
                },
                "IncomeTaxExpenseBenefit": {
                    "units": {
                        "USD": [
                            {"start": "2024-01-01", "end": "2024-12-31", "filed": "2024-12-31", "val": 15.0, "fy": 2024, "fp": "FY"}
                        ]
                    }
                },
                "DepreciationAndAmortization": {
                    "units": {
                        "USD": [
                            {"start": "2024-01-01", "end": "2024-12-31", "filed": "2024-12-31", "val": 25.0, "fy": 2024, "fp": "FY"}
                        ]
                    }
                },
            }
        }
    }
    row = {
        "company_id": "456",
        "as_of_time": "2024-12-31T00:00:00Z",
        "features": {
            "capital_structure.total_debt_provider_direct": {
                "support_mode": "exact",
                "value": 100.0,
                "component_breakdown": {"mode": "current_plus_noncurrent_debt"},
            },
            "capital_structure.current_debt_statement_direct": {"support_mode": "unsupported", "value": None},
            "capital_structure.long_term_debt_statement_direct": {"support_mode": "unsupported", "value": None},
            "capital_structure.lease_liabilities_sec_exact": {
                "support_mode": "unsupported",
                "value": None,
                "missing_reason": "sec_concept_absent",
            },
            "liquidity.cash_and_short_term_investments_provider_direct": {"support_mode": "exact", "value": 20.0},
            "liquidity.cash_and_equivalents_statement_direct": {"support_mode": "exact", "value": 20.0},
            "liquidity.restricted_cash_sec_exact": {
                "support_mode": "unsupported",
                "value": None,
                "missing_reason": "sec_concept_absent",
            },
            "liquidity.marketable_securities_sec_exact": {
                "support_mode": "unsupported",
                "value": None,
                "missing_reason": "sec_concept_absent",
            },
            "liquidity.restricted_cash": {"support_mode": "unsupported", "value": None},
            "liquidity.marketable_securities": {"support_mode": "unsupported", "value": None},
            "liquidity.revolver_undrawn_sec_exact": {"support_mode": "unsupported", "value": None},
            "liquidity.revolver_undrawn": {"support_mode": "unsupported", "value": None},
            "operating.ebitda_ltm_provider_direct": {
                "support_mode": "unsupported",
                "value": None,
                "missing_reason": "sec_operating_income_ttm_unavailable",
            },
            "earnings.net_income_ttm_provider_direct": {"support_mode": "exact", "value": 50.0},
            "capital_structure.interest_expense_statement_direct": {"support_mode": "unsupported", "value": None},
        },
    }

    repaired = materialize_smart_metrics_for_row(
        row=row,
        registry=registry,
        computed_at="2026-03-22T00:00:00Z",
        provenance_sources=["registry.json"],
        companyfacts=companyfacts,
    )

    earnings = repaired["features"]["operating.operating_earnings_normalized"]
    assert earnings["support_mode"] == "exact"
    assert earnings["value"] == 100.0
    assert (
        earnings["component_breakdown"]["earnings_support_override"]
        == "net_income_plus_interest_tax_depreciation_amortization"
    )
    assert repaired["features"]["capital_structure.gross_leverage_normalized"]["support_mode"] == "exact"
    assert repaired["features"]["capital_structure.gross_leverage_normalized"]["value"] == 1.0
    assert repaired["features"]["capital_structure.net_leverage_normalized"]["value"] == 0.8


def test_materialize_smart_metrics_for_row_promotes_grouped_cash_market_baseline_when_restricted_cash_is_undisclosed():
    registry = {
        "metrics": {
            "debt_like_obligations_normalized": {"status": "partially_feasible", "promotion_rule": "test"},
            "available_liquidity_normalized": {"status": "partially_feasible", "promotion_rule": "test"},
            "operating_earnings_normalized": {"status": "partially_feasible", "promotion_rule": "test"},
            "net_debt_normalized": {"status": "partially_feasible", "promotion_rule": "test"},
            "gross_leverage_normalized": {"status": "partially_feasible", "promotion_rule": "test"},
            "net_leverage_normalized": {"status": "partially_feasible", "promotion_rule": "test"},
        }
    }
    row = {
        "company_id": "market-liquidity-probe",
        "as_of_time": "2024-12-31T00:00:00Z",
        "features": {
            "capital_structure.total_debt_provider_direct": {"support_mode": "exact", "value": 100.0},
            "capital_structure.current_debt_statement_direct": {"support_mode": "unsupported", "value": None},
            "capital_structure.long_term_debt_statement_direct": {"support_mode": "unsupported", "value": None},
            "capital_structure.lease_liabilities_sec_exact": {
                "support_mode": "unsupported",
                "value": None,
                "missing_reason": "sec_concept_absent",
            },
            "liquidity.cash_and_short_term_investments_provider_direct": {"support_mode": "exact", "value": 20.0},
            "liquidity.cash_and_equivalents_statement_direct": {
                "support_mode": "unsupported",
                "value": None,
                "missing_reason": "statement_fact_unavailable",
            },
            "liquidity.restricted_cash_sec_exact": {
                "support_mode": "unsupported",
                "value": None,
                "missing_reason": "sec_concept_unavailable",
            },
            "liquidity.marketable_securities_sec_exact": {
                "support_mode": "unsupported",
                "value": None,
                "missing_reason": "sec_concept_absent",
            },
            "liquidity.restricted_cash": {"support_mode": "unsupported", "value": None},
            "liquidity.marketable_securities": {"support_mode": "unsupported", "value": None},
            "liquidity.revolver_undrawn_sec_exact": {"support_mode": "unsupported", "value": None},
            "liquidity.revolver_undrawn": {"support_mode": "unsupported", "value": None},
            "operating.ebitda_ltm_provider_direct": {"support_mode": "exact", "value": 40.0},
        },
    }

    repaired = materialize_smart_metrics_for_row(
        row=row,
        registry=registry,
        computed_at="2026-03-24T00:00:00Z",
        provenance_sources=["registry.json"],
    )

    available_liquidity = repaired["features"]["liquidity.available_liquidity_normalized"]
    assert available_liquidity["support_mode"] == "exact"
    assert available_liquidity["value"] == 20.0
    assert (
        available_liquidity["component_breakdown"]["restricted_cash_support_override"]
        == "grouped_cash_market_baseline_without_restricted_cash_disclosure"
    )
    assert repaired["features"]["capital_structure.net_debt_normalized"]["support_mode"] == "exact"
    assert repaired["features"]["capital_structure.net_debt_normalized"]["value"] == 80.0
    assert repaired["features"]["capital_structure.net_leverage_normalized"]["support_mode"] == "exact"
    assert repaired["features"]["capital_structure.net_leverage_normalized"]["value"] == 2.0


def test_market_availability_adjustment_matches_effective_window():
    overrides = {
        "0000104169": [
            {
                "effective_start": "2024-12-06",
                "effective_end": "2025-03-31",
                "not_freely_transferable_cash": 3_600_000_000.0,
                "source": "Walmart 10-Q filed 2024-12-06",
            }
        ]
    }

    adjustment = _market_availability_adjustment(
        overrides,
        company_id="0000104169",
        as_of_time="2024-12-31T00:00:00Z",
    )

    assert adjustment is not None
    assert adjustment["value"] == 3_600_000_000.0
    assert adjustment["source"] == "Walmart 10-Q filed 2024-12-06"


def test_materialize_smart_metrics_for_row_applies_market_availability_adjustment():
    registry = {
        "metrics": {
            "debt_like_obligations_normalized": {"status": "partially_feasible", "promotion_rule": "test"},
            "available_liquidity_normalized": {"status": "partially_feasible", "promotion_rule": "test"},
            "operating_earnings_normalized": {"status": "partially_feasible", "promotion_rule": "test"},
            "net_debt_normalized": {"status": "partially_feasible", "promotion_rule": "test"},
            "gross_leverage_normalized": {"status": "partially_feasible", "promotion_rule": "test"},
            "net_leverage_normalized": {"status": "partially_feasible", "promotion_rule": "test"},
        }
    }
    row = {
        "company_id": "0000104169",
        "as_of_time": "2024-12-31T00:00:00Z",
        "features": {
            "capital_structure.total_debt_provider_direct": {"support_mode": "exact", "value": 100.0},
            "capital_structure.current_debt_statement_direct": {"support_mode": "unsupported", "value": None},
            "capital_structure.long_term_debt_statement_direct": {"support_mode": "unsupported", "value": None},
            "capital_structure.lease_liabilities_sec_exact": {
                "support_mode": "unsupported",
                "value": None,
                "missing_reason": "sec_concept_absent",
            },
            "liquidity.cash_and_short_term_investments_provider_direct": {"support_mode": "exact", "value": 20.0},
            "liquidity.cash_and_equivalents_statement_direct": {
                "support_mode": "unsupported",
                "value": None,
                "missing_reason": "statement_fact_unavailable",
            },
            "liquidity.restricted_cash_sec_exact": {
                "support_mode": "unsupported",
                "value": None,
                "missing_reason": "sec_concept_unavailable",
            },
            "liquidity.marketable_securities_sec_exact": {
                "support_mode": "unsupported",
                "value": None,
                "missing_reason": "sec_concept_absent",
            },
            "liquidity.restricted_cash": {"support_mode": "unsupported", "value": None},
            "liquidity.marketable_securities": {"support_mode": "unsupported", "value": None},
            "liquidity.revolver_undrawn_sec_exact": {"support_mode": "unsupported", "value": None},
            "liquidity.revolver_undrawn": {"support_mode": "unsupported", "value": None},
            "operating.ebitda_ltm_provider_direct": {"support_mode": "exact", "value": 40.0},
        },
    }
    overrides = {
        "0000104169": [
            {
                "effective_start": "2024-12-06",
                "effective_end": "2025-03-31",
                "not_freely_transferable_cash": 3.6,
                "source": "Walmart 10-Q filed 2024-12-06",
                "reported_period_end": "2024-10-31",
            }
        ]
    }

    repaired = materialize_smart_metrics_for_row(
        row=row,
        registry=registry,
        computed_at="2026-03-24T00:00:00Z",
        provenance_sources=["registry.json"],
        market_availability_overrides=overrides,
    )

    available_liquidity = repaired["features"]["liquidity.available_liquidity_normalized"]
    assert available_liquidity["support_mode"] == "exact"
    assert available_liquidity["value"] == 16.4
    assert available_liquidity["component_breakdown"]["not_freely_transferable_cash_disclosed"] == 3.6
    assert (
        available_liquidity["component_breakdown"]["market_availability_adjustment"]["source"]
        == "Walmart 10-Q filed 2024-12-06"
    )
    assert repaired["features"]["capital_structure.net_debt_normalized"]["value"] == 83.6
    assert repaired["features"]["capital_structure.net_leverage_normalized"]["value"] == 2.09


def test_materialize_smart_metrics_for_row_prefers_sec_cash_plus_marketable_when_grouped_cash_is_inconsistent():
    registry = {
        "metrics": {
            "debt_like_obligations_normalized": {"status": "partially_feasible", "promotion_rule": "test"},
            "available_liquidity_normalized": {"status": "partially_feasible", "promotion_rule": "test"},
            "operating_earnings_normalized": {"status": "partially_feasible", "promotion_rule": "test"},
            "net_debt_normalized": {"status": "partially_feasible", "promotion_rule": "test"},
            "gross_leverage_normalized": {"status": "partially_feasible", "promotion_rule": "test"},
            "net_leverage_normalized": {"status": "partially_feasible", "promotion_rule": "test"},
        }
    }
    companyfacts = {
        "facts": {
            "us-gaap": {
                "CashAndCashEquivalentsAtCarryingValue": {
                    "units": {
                        "USD": [
                            {"end": "2024-09-30", "filed": "2024-10-31", "val": 6.15, "fy": 2024, "fp": "Q3", "form": "10-Q"}
                        ]
                    }
                }
            }
        }
    }
    row = {
        "company_id": "0001543151",
        "as_of_time": "2024-12-31T00:00:00Z",
        "features": {
            "capital_structure.total_debt_provider_direct": {"support_mode": "exact", "value": 9.807},
            "capital_structure.current_debt_statement_direct": {"support_mode": "unsupported", "value": None},
            "capital_structure.long_term_debt_statement_direct": {"support_mode": "unsupported", "value": None},
            "capital_structure.lease_liabilities_sec_exact": {
                "support_mode": "unsupported",
                "value": None,
                "missing_reason": "sec_concept_absent",
            },
            "liquidity.cash_and_short_term_investments_provider_direct": {"support_mode": "exact", "value": 6.977},
            "liquidity.cash_and_equivalents_statement_direct": {
                "support_mode": "unsupported",
                "value": None,
                "missing_reason": "statement_fact_unavailable",
            },
            "liquidity.restricted_cash_sec_exact": {"support_mode": "exact", "value": 1.92},
            "liquidity.marketable_securities_sec_exact": {"support_mode": "exact", "value": 2.913},
            "liquidity.restricted_cash": {"support_mode": "unsupported", "value": None},
            "liquidity.marketable_securities": {"support_mode": "unsupported", "value": None},
            "liquidity.revolver_undrawn_sec_exact": {"support_mode": "unsupported", "value": None},
            "liquidity.revolver_undrawn": {"support_mode": "unsupported", "value": None},
            "operating.ebitda_ltm_provider_direct": {"support_mode": "exact", "value": 4.714},
        },
    }

    repaired = materialize_smart_metrics_for_row(
        row=row,
        registry=registry,
        computed_at="2026-03-24T00:00:00Z",
        provenance_sources=["registry.json"],
        companyfacts=companyfacts,
    )

    available_liquidity = repaired["features"]["liquidity.available_liquidity_normalized"]
    assert available_liquidity["support_mode"] == "exact"
    assert round(available_liquidity["value"], 3) == 9.063
    assert (
        available_liquidity["component_breakdown"]["cash_basis_support_override"]
        == "sec_cash_and_marketable_override_provider_grouped_cash"
    )
    assert available_liquidity["component_breakdown"]["restricted_cash_already_excluded_from_cash_basis"] is True
    assert repaired["features"]["capital_structure.net_debt_normalized"]["support_mode"] == "exact"
    assert round(repaired["features"]["capital_structure.net_debt_normalized"]["value"], 3) == 0.744
    assert round(repaired["features"]["capital_structure.net_leverage_normalized"]["value"], 3) == 0.158


def test_materialize_smart_metrics_for_row_includes_exact_marketable_securities_when_grouped_cash_is_cash_only_proxy():
    registry = {
        "metrics": {
            "debt_like_obligations_normalized": {"status": "partially_feasible", "promotion_rule": "test"},
            "available_liquidity_normalized": {"status": "partially_feasible", "promotion_rule": "test"},
            "operating_earnings_normalized": {"status": "partially_feasible", "promotion_rule": "test"},
            "net_debt_normalized": {"status": "partially_feasible", "promotion_rule": "test"},
            "gross_leverage_normalized": {"status": "partially_feasible", "promotion_rule": "test"},
            "net_leverage_normalized": {"status": "partially_feasible", "promotion_rule": "test"},
        }
    }
    row = {
        "company_id": "aal-probe",
        "as_of_time": "2024-12-31T00:00:00Z",
        "features": {
            "capital_structure.total_debt_provider_direct": {"support_mode": "exact", "value": 31_110_000_000.0},
            "capital_structure.current_debt_statement_direct": {"support_mode": "unsupported", "value": None},
            "capital_structure.long_term_debt_statement_direct": {"support_mode": "unsupported", "value": None},
            "capital_structure.lease_liabilities_sec_exact": {
                "support_mode": "unsupported",
                "value": None,
                "missing_reason": "sec_concept_unavailable",
            },
            "liquidity.cash_and_short_term_investments_provider_direct": {
                "support_mode": "proxy_missing_component",
                "value": 834_000_000.0,
                "missing_reason": "cash_or_sti_component_missing",
                "component_breakdown": {
                    "mode": "partial_cash_stack",
                    "cash": {"concept": "Cash"},
                    "short_term_investments": None,
                },
            },
            "liquidity.cash_and_equivalents_statement_direct": {
                "support_mode": "unsupported",
                "value": None,
                "missing_reason": "statement_fact_unavailable",
            },
            "liquidity.restricted_cash_sec_exact": {"support_mode": "exact", "value": 99_000_000.0},
            "liquidity.marketable_securities_sec_exact": {"support_mode": "exact", "value": 7_638_000_000.0},
            "liquidity.restricted_cash": {"support_mode": "unsupported", "value": None},
            "liquidity.marketable_securities": {"support_mode": "unsupported", "value": None},
            "liquidity.revolver_undrawn_sec_exact": {"support_mode": "exact", "value": 2_868_000_000.0},
            "liquidity.revolver_undrawn": {"support_mode": "unsupported", "value": None},
            "operating.ebitda_ltm_provider_direct": {"support_mode": "exact", "value": 4_436_000_000.0},
        },
    }

    repaired = materialize_smart_metrics_for_row(
        row=row,
        registry=registry,
        computed_at="2026-03-22T00:00:00Z",
        provenance_sources=["registry.json"],
    )

    available_liquidity = repaired["features"]["liquidity.available_liquidity_normalized"]
    assert available_liquidity["support_mode"] == "exact"
    assert available_liquidity["value"] == 11_340_000_000.0
    assert (
        available_liquidity["component_breakdown"]["cash_basis_support_override"]
        == "partial_cash_stack_completed_by_marketable_securities"
    )
    assert available_liquidity["component_breakdown"]["restricted_cash_already_excluded_from_cash_basis"] is True
    assert (
        available_liquidity["component_breakdown"]["formula"]
        == "cash_and_short_term_investments_provider_direct_cash_component + marketable_securities_sec_exact + revolver_undrawn_exact"
    )


def test_materialize_smart_metrics_adds_pension_metric_and_inclusive_debt_views_exact():
    registry = {"metrics": {}}
    row = {
        "company_id": "pension-exact-probe",
        "as_of_time": "2024-12-31T00:00:00Z",
        "features": {
            "capital_structure.total_debt_provider_direct": {"support_mode": "exact", "value": 100.0},
            "capital_structure.current_debt_statement_direct": {"support_mode": "unsupported", "value": None},
            "capital_structure.long_term_debt_statement_direct": {"support_mode": "unsupported", "value": None},
            "capital_structure.lease_liabilities_sec_exact": {"support_mode": "exact", "value": 20.0},
            "liquidity.cash_and_short_term_investments_provider_direct": {"support_mode": "exact", "value": 30.0},
            "liquidity.cash_and_equivalents_statement_direct": {"support_mode": "exact", "value": 30.0},
            "liquidity.restricted_cash_sec_exact": {
                "support_mode": "unsupported",
                "value": None,
                "missing_reason": "sec_concept_absent",
            },
            "liquidity.marketable_securities_sec_exact": {
                "support_mode": "unsupported",
                "value": None,
                "missing_reason": "sec_concept_absent",
            },
            "liquidity.restricted_cash": {"support_mode": "unsupported", "value": None},
            "liquidity.marketable_securities": {"support_mode": "unsupported", "value": None},
            "liquidity.revolver_undrawn_sec_exact": {"support_mode": "unsupported", "value": None},
            "liquidity.revolver_undrawn": {"support_mode": "unsupported", "value": None},
            "operating.ebitda_ltm_provider_direct": {"support_mode": "exact", "value": 60.0},
        },
    }
    companyfacts = {
        "facts": {
            "us-gaap": {
                "DefinedBenefitPensionPlanLiabilitiesNoncurrent": {
                    "units": {
                        "USD": [
                                {
                                    "end": "2024-12-31",
                                    "filed": "2024-12-31",
                                    "val": 15.0,
                                    "fy": 2024,
                                    "fp": "FY",
                                    "form": "10-K",
                            }
                        ]
                    }
                }
            }
        }
    }

    repaired = materialize_smart_metrics_for_row(
        row=row,
        registry=registry,
        computed_at="2026-04-01T00:00:00Z",
        provenance_sources=["registry.json"],
        companyfacts=companyfacts,
    )

    pension = repaired["features"]["capital_structure.net_pension_liability"]
    assert pension["support_mode"] == "exact"
    assert pension["value"] == 15.0
    combined_retirement = repaired["features"]["capital_structure.combined_retirement_liability"]
    assert combined_retirement["support_mode"] == "proxy_missing_component"
    assert combined_retirement["value"] == 15.0
    assert combined_retirement["component_breakdown"]["support_override"] == (
        "combined_retirement_uses_pension_only;single_pension_liability_component"
    )
    debt_including_pension = repaired["features"]["capital_structure.debt_like_obligations_including_pension"]
    assert debt_including_pension["support_mode"] == "exact"
    assert debt_including_pension["value"] == 135.0
    assert repaired["features"]["capital_structure.net_debt_including_pension"]["value"] == 105.0
    assert repaired["features"]["capital_structure.gross_leverage_including_pension"]["value"] == 2.25
    assert repaired["features"]["capital_structure.net_leverage_including_pension"]["value"] == 1.75
    debt_including_retirement = repaired["features"]["capital_structure.debt_like_obligations_including_retirement"]
    assert debt_including_retirement["support_mode"] == "proxy_missing_component"
    assert debt_including_retirement["value"] == 135.0
    assert (
        debt_including_retirement["component_breakdown"]["combined_retirement_support_override"]
        == "combined_retirement_uses_pension_only;single_pension_liability_component"
    )
    assert repaired["features"]["capital_structure.net_debt_including_retirement"]["value"] == 105.0
    assert repaired["features"]["capital_structure.gross_leverage_including_retirement"]["value"] == 2.25
    assert repaired["features"]["capital_structure.net_leverage_including_retirement"]["value"] == 1.75
    assert repaired["features"]["capital_structure.retirement_obligation_regime"]["value"] == "pension_exact"


def test_extract_retirement_note_components_from_html_splits_pension_from_other_postretirement():
    filing = {
        "cik": "0000000001",
        "filing_date": "2024-12-01",
        "form": "10-K",
        "accession_number": "0000000001-24-000001",
        "primary_document": "test.htm",
    }
    html = """
    <html>
      <body>
        <table>
          <tr><th></th><th>Pension benefits 2024</th><th>Other benefits 2024</th></tr>
          <tr><td>Funded status at end of year</td><td>(15)</td><td>(4)</td></tr>
        </table>
      </body>
    </html>
    """

    parsed = _extract_retirement_note_components_from_html(
        filing=filing,
        html=html,
        as_of_time="2024-12-31T00:00:00Z",
    )

    assert parsed is not None
    assert parsed["pension_value"] == 15.0
    assert parsed["other_postretirement_value"] == 4.0
    assert parsed["component_meta"]["pension"]["source_meta"]["mode"] == "funded_status_row"
    assert parsed["component_meta"]["other_postretirement"]["source_meta"]["mode"] == "funded_status_row"


def test_load_retirement_note_components_carries_forward_prior_filing_split(monkeypatch, tmp_path):
    filings = [
        {
            "cik": "0000000001",
            "filing_date": "2024-11-01",
            "form": "10-Q",
            "accession_number": "0000000001-24-000002",
            "primary_document": "q3.htm",
        },
        {
            "cik": "0000000001",
            "filing_date": "2024-02-20",
            "form": "10-K",
            "accession_number": "0000000001-24-000001",
            "primary_document": "annual.htm",
        },
    ]
    latest_html = """
    <html>
      <body>
        <table>
          <tr><th></th><th>Revenue 2024</th></tr>
          <tr><td>Net sales</td><td>10</td></tr>
        </table>
      </body>
    </html>
    """
    annual_html = """
    <html>
      <body>
        <table>
          <tr><th></th><th>Pension benefits 2024</th><th>Other benefits 2024</th></tr>
          <tr><td>Funded status at end of year</td><td>(15)</td><td>(4)</td></tr>
        </table>
      </body>
    </html>
    """

    monkeypatch.setattr(smart_mod, "_recent_sec_filings", lambda **kwargs: filings)
    monkeypatch.setattr(
        smart_mod,
        "_fetch_sec_primary_document",
        lambda filing, **kwargs: annual_html if filing["form"] == "10-K" else latest_html,
    )

    parsed = _load_retirement_note_components(
        cik="0000000001",
        as_of_time="2024-12-31T00:00:00Z",
        session=object(),
        cache_dir=tmp_path,
    )

    assert parsed is not None
    assert parsed["pension_value"] == 15.0
    assert parsed["other_postretirement_value"] == 4.0
    assert parsed["regime_hint"] == "pension_proxy_split_note"
    assert parsed["component_meta"]["carryforward_used"] is True
    assert parsed["component_meta"]["carryforward_source"] == "prior_filing_note"
    assert parsed["component_meta"]["filing"]["form"] == "10-K"


def test_materialize_smart_metrics_keeps_pension_unsupported_when_only_combined_proxy_exists():
    registry = {"metrics": {}}
    row = {
        "company_id": "pension-proxy-probe",
        "as_of_time": "2024-12-31T00:00:00Z",
        "features": {
            "capital_structure.total_debt_provider_direct": {"support_mode": "exact", "value": 100.0},
            "capital_structure.current_debt_statement_direct": {"support_mode": "unsupported", "value": None},
            "capital_structure.long_term_debt_statement_direct": {"support_mode": "unsupported", "value": None},
            "capital_structure.lease_liabilities_sec_exact": {"support_mode": "unsupported", "value": None},
            "liquidity.cash_and_short_term_investments_provider_direct": {"support_mode": "exact", "value": 25.0},
            "liquidity.cash_and_equivalents_statement_direct": {"support_mode": "exact", "value": 25.0},
            "liquidity.restricted_cash_sec_exact": {
                "support_mode": "unsupported",
                "value": None,
                "missing_reason": "sec_concept_absent",
            },
            "liquidity.marketable_securities_sec_exact": {
                "support_mode": "unsupported",
                "value": None,
                "missing_reason": "sec_concept_absent",
            },
            "liquidity.restricted_cash": {"support_mode": "unsupported", "value": None},
            "liquidity.marketable_securities": {"support_mode": "unsupported", "value": None},
            "liquidity.revolver_undrawn_sec_exact": {"support_mode": "unsupported", "value": None},
            "liquidity.revolver_undrawn": {"support_mode": "unsupported", "value": None},
            "operating.ebitda_ltm_provider_direct": {"support_mode": "exact", "value": 50.0},
        },
    }
    companyfacts = {
        "facts": {
            "us-gaap": {
                "PensionAndOtherPostretirementAndPostemploymentBenefitPlansLiabilitiesNoncurrent": {
                    "units": {
                        "USD": [
                                {
                                    "end": "2024-12-31",
                                    "filed": "2024-12-31",
                                    "val": 12.0,
                                    "fy": 2024,
                                    "fp": "FY",
                                    "form": "10-K",
                            }
                        ]
                    }
                }
            }
        }
    }

    repaired = materialize_smart_metrics_for_row(
        row=row,
        registry=registry,
        computed_at="2026-04-01T00:00:00Z",
        provenance_sources=["registry.json"],
        companyfacts=companyfacts,
    )

    pension = repaired["features"]["capital_structure.net_pension_liability"]
    assert pension["support_mode"] == "unsupported"
    assert pension["value"] is None
    assert pension["component_breakdown"]["support_override"] == "combined_pension_and_postretirement_liability_not_separable"
    combined_retirement = repaired["features"]["capital_structure.combined_retirement_liability"]
    assert combined_retirement["support_mode"] == "proxy_missing_component"
    assert combined_retirement["value"] == 12.0
    assert combined_retirement["component_breakdown"]["support_override"] == (
        "combined_retirement_proxy_companyfacts_unseparated"
    )
    debt_including_pension = repaired["features"]["capital_structure.debt_like_obligations_including_pension"]
    assert debt_including_pension["support_mode"] == "proxy_missing_component"
    assert debt_including_pension["value"] == 100.0
    assert debt_including_pension["component_breakdown"]["pension_missing_assumed_zero"] is True
    assert (
        debt_including_pension["component_breakdown"]["pension_support_override"]
        == "combined_pension_and_postretirement_liability_not_separable"
    )
    assert repaired["features"]["capital_structure.net_debt_including_pension"]["support_mode"] == "proxy_missing_component"
    assert repaired["features"]["capital_structure.net_debt_including_pension"]["value"] == 75.0
    debt_including_retirement = repaired["features"]["capital_structure.debt_like_obligations_including_retirement"]
    assert debt_including_retirement["support_mode"] == "proxy_missing_component"
    assert debt_including_retirement["value"] == 112.0
    assert (
        debt_including_retirement["component_breakdown"]["combined_retirement_support_override"]
        == "combined_retirement_proxy_companyfacts_unseparated"
    )
    assert repaired["features"]["capital_structure.net_debt_including_retirement"]["value"] == 87.0
    assert repaired["features"]["capital_structure.gross_leverage_including_retirement"]["value"] == 2.24
    assert repaired["features"]["capital_structure.net_leverage_including_retirement"]["value"] == 1.74
    assert repaired["features"]["capital_structure.retirement_obligation_regime"]["value"] == "combined_retirement_only"


def test_materialize_smart_metrics_uses_filing_note_split_to_isolate_pension_from_combined_proxy():
    registry = {"metrics": {}}
    row = {
        "company_id": "pension-filing-note-probe",
        "as_of_time": "2024-12-31T00:00:00Z",
        "features": {
            "capital_structure.total_debt_provider_direct": {"support_mode": "exact", "value": 100.0},
            "capital_structure.current_debt_statement_direct": {"support_mode": "unsupported", "value": None},
            "capital_structure.long_term_debt_statement_direct": {"support_mode": "unsupported", "value": None},
            "capital_structure.lease_liabilities_sec_exact": {"support_mode": "unsupported", "value": None},
            "liquidity.cash_and_short_term_investments_provider_direct": {"support_mode": "exact", "value": 25.0},
            "liquidity.cash_and_equivalents_statement_direct": {"support_mode": "exact", "value": 25.0},
            "liquidity.restricted_cash_sec_exact": {
                "support_mode": "unsupported",
                "value": None,
                "missing_reason": "sec_concept_absent",
            },
            "liquidity.marketable_securities_sec_exact": {
                "support_mode": "unsupported",
                "value": None,
                "missing_reason": "sec_concept_absent",
            },
            "liquidity.restricted_cash": {"support_mode": "unsupported", "value": None},
            "liquidity.marketable_securities": {"support_mode": "unsupported", "value": None},
            "liquidity.revolver_undrawn_sec_exact": {"support_mode": "unsupported", "value": None},
            "liquidity.revolver_undrawn": {"support_mode": "unsupported", "value": None},
            "operating.ebitda_ltm_provider_direct": {"support_mode": "exact", "value": 50.0},
        },
    }
    companyfacts = {
        "facts": {
            "us-gaap": {
                "PensionAndOtherPostretirementAndPostemploymentBenefitPlansLiabilitiesNoncurrent": {
                    "units": {
                        "USD": [
                            {
                                "end": "2024-12-31",
                                "filed": "2024-12-31",
                                "val": 12.0,
                                "fy": 2024,
                                "fp": "FY",
                                "form": "10-K",
                            }
                        ]
                    }
                }
            }
        }
    }
    retirement_note = {
        "pension_value": 9.0,
        "other_postretirement_value": 3.0,
        "component_meta": {
            "mode": "filing_note_retirement_split",
            "pension": {"source_meta": {"mode": "funded_status_row"}},
            "other_postretirement": {"source_meta": {"mode": "funded_status_row"}},
        },
    }

    repaired = materialize_smart_metrics_for_row(
        row=row,
        registry=registry,
        computed_at="2026-04-01T00:00:00Z",
        provenance_sources=["registry.json"],
        companyfacts=companyfacts,
        retirement_note_loader=lambda: retirement_note,
    )

    pension = repaired["features"]["capital_structure.net_pension_liability"]
    assert pension["support_mode"] == "proxy_missing_component"
    assert pension["value"] == 9.0
    assert pension["component_breakdown"]["support_override"] == (
        "filing_note_defined_benefit_pension_proxy;other_postretirement_excluded_from_pension_metric"
    )
    other_postretirement = repaired["features"]["capital_structure.other_postretirement_benefit_liability"]
    assert other_postretirement["support_mode"] == "proxy_missing_component"
    assert other_postretirement["value"] == 3.0
    assert other_postretirement["component_breakdown"]["support_override"] == "filing_note_other_postretirement_proxy"
    combined_retirement = repaired["features"]["capital_structure.combined_retirement_liability"]
    assert combined_retirement["support_mode"] == "proxy_missing_component"
    assert combined_retirement["value"] == 12.0
    assert combined_retirement["component_breakdown"]["support_override"] == "filing_note_combined_retirement_proxy"
    debt_including_pension = repaired["features"]["capital_structure.debt_like_obligations_including_pension"]
    assert debt_including_pension["support_mode"] == "proxy_missing_component"
    assert debt_including_pension["value"] == 109.0
    assert (
        debt_including_pension["component_breakdown"]["pension_support_override"]
        == "filing_note_defined_benefit_pension_proxy;other_postretirement_excluded_from_pension_metric"
    )
    assert repaired["features"]["capital_structure.net_debt_including_pension"]["value"] == 84.0
    debt_including_retirement = repaired["features"]["capital_structure.debt_like_obligations_including_retirement"]
    assert debt_including_retirement["support_mode"] == "proxy_missing_component"
    assert debt_including_retirement["value"] == 112.0
    assert (
        debt_including_retirement["component_breakdown"]["combined_retirement_support_override"]
        == "filing_note_combined_retirement_proxy"
    )
    assert repaired["features"]["capital_structure.net_debt_including_retirement"]["value"] == 87.0
    assert repaired["features"]["capital_structure.gross_leverage_including_retirement"]["value"] == 2.24
    assert repaired["features"]["capital_structure.net_leverage_including_retirement"]["value"] == 1.74
    regime = repaired["features"]["capital_structure.retirement_obligation_regime"]
    assert regime["value"] == "pension_proxy_split_note"
    assert regime["component_breakdown"]["classification_reference"]["mode"] == "filing_note_split"


def test_materialize_smart_metrics_keeps_other_postretirement_unsupported_when_split_is_unavailable():
    registry = {"metrics": {}}
    row = {
        "company_id": "other-postretirement-unavailable-probe",
        "as_of_time": "2024-12-31T00:00:00Z",
        "features": {
            "capital_structure.total_debt_provider_direct": {"support_mode": "exact", "value": 100.0},
            "capital_structure.current_debt_statement_direct": {"support_mode": "unsupported", "value": None},
            "capital_structure.long_term_debt_statement_direct": {"support_mode": "unsupported", "value": None},
            "capital_structure.lease_liabilities_sec_exact": {"support_mode": "unsupported", "value": None},
            "liquidity.cash_and_short_term_investments_provider_direct": {"support_mode": "exact", "value": 25.0},
            "liquidity.cash_and_equivalents_statement_direct": {"support_mode": "exact", "value": 25.0},
            "liquidity.restricted_cash_sec_exact": {"support_mode": "unsupported", "value": None},
            "liquidity.marketable_securities_sec_exact": {"support_mode": "unsupported", "value": None},
            "liquidity.restricted_cash": {"support_mode": "unsupported", "value": None},
            "liquidity.marketable_securities": {"support_mode": "unsupported", "value": None},
            "liquidity.revolver_undrawn_sec_exact": {"support_mode": "unsupported", "value": None},
            "liquidity.revolver_undrawn": {"support_mode": "unsupported", "value": None},
            "operating.ebitda_ltm_provider_direct": {"support_mode": "exact", "value": 50.0},
        },
    }
    repaired = materialize_smart_metrics_for_row(
        row=row,
        registry=registry,
        computed_at="2026-04-02T00:00:00Z",
        provenance_sources=["registry.json"],
    )

    other_postretirement = repaired["features"]["capital_structure.other_postretirement_benefit_liability"]
    assert other_postretirement["support_mode"] == "unsupported"
    assert other_postretirement["value"] is None
    combined_retirement = repaired["features"]["capital_structure.combined_retirement_liability"]
    assert combined_retirement["support_mode"] == "unsupported"
    assert combined_retirement["value"] is None
    debt_including_retirement = repaired["features"]["capital_structure.debt_like_obligations_including_retirement"]
    assert debt_including_retirement["support_mode"] == "proxy_missing_component"
    assert debt_including_retirement["value"] == 100.0
    assert debt_including_retirement["component_breakdown"]["combined_retirement_missing_assumed_zero"] is True
    assert repaired["features"]["capital_structure.net_debt_including_retirement"]["value"] == 75.0
    assert repaired["features"]["capital_structure.gross_leverage_including_retirement"]["value"] == 2.0
    assert repaired["features"]["capital_structure.net_leverage_including_retirement"]["value"] == 1.5
    assert repaired["features"]["capital_structure.retirement_obligation_regime"]["value"] == "retirement_not_surfaced"


def test_materialize_smart_metrics_marks_defined_contribution_only_regime_from_filing_hint():
    registry = {"metrics": {}}
    row = {
        "company_id": "defined-contribution-probe",
        "as_of_time": "2024-12-31T00:00:00Z",
        "features": {
            "capital_structure.total_debt_provider_direct": {"support_mode": "exact", "value": 100.0},
            "capital_structure.current_debt_statement_direct": {"support_mode": "unsupported", "value": None},
            "capital_structure.long_term_debt_statement_direct": {"support_mode": "unsupported", "value": None},
            "capital_structure.lease_liabilities_sec_exact": {"support_mode": "unsupported", "value": None},
            "liquidity.cash_and_short_term_investments_provider_direct": {"support_mode": "exact", "value": 25.0},
            "liquidity.cash_and_equivalents_statement_direct": {"support_mode": "exact", "value": 25.0},
            "liquidity.restricted_cash_sec_exact": {"support_mode": "unsupported", "value": None},
            "liquidity.marketable_securities_sec_exact": {"support_mode": "unsupported", "value": None},
            "liquidity.revolver_undrawn_sec_exact": {"support_mode": "unsupported", "value": None},
            "operating.ebitda_ltm_provider_direct": {"support_mode": "exact", "value": 60.0},
            "operating.ebit_statement_direct": {"support_mode": "exact", "value": 50.0},
            "capital_structure.interest_expense_statement_direct": {"support_mode": "unsupported", "value": None},
        },
    }

    repaired = materialize_smart_metrics_for_row(
        row=row,
        registry=registry,
        computed_at="2026-04-02T00:00:00Z",
        provenance_sources=["registry.json"],
        companyfacts=None,
        retirement_note_loader=lambda: {
            "regime_hint": "defined_contribution_only",
            "component_meta": {
                "mode": "defined_contribution_only_filing_text",
                "carryforward_used": False,
            },
        },
    )

    assert repaired["features"]["capital_structure.net_pension_liability"]["support_mode"] == "unsupported"
    assert repaired["features"]["capital_structure.combined_retirement_liability"]["support_mode"] == "unsupported"
    regime = repaired["features"]["capital_structure.retirement_obligation_regime"]
    assert regime["support_mode"] == "exact"
    assert regime["value"] == "defined_contribution_only"
    assert regime["component_breakdown"]["regime_source"] == "filing_text_hint"


def test_materialize_smart_metrics_falls_back_when_retirement_note_loader_times_out():
    registry = {"metrics": {}}
    row = {
        "company_id": "retirement-timeout-probe",
        "as_of_time": "2024-12-31T00:00:00Z",
        "features": {
            "capital_structure.total_debt_provider_direct": {"support_mode": "exact", "value": 100.0},
            "capital_structure.current_debt_statement_direct": {"support_mode": "unsupported", "value": None},
            "capital_structure.long_term_debt_statement_direct": {"support_mode": "unsupported", "value": None},
            "capital_structure.lease_liabilities_sec_exact": {"support_mode": "unsupported", "value": None},
            "liquidity.cash_and_short_term_investments_provider_direct": {"support_mode": "exact", "value": 25.0},
            "liquidity.cash_and_equivalents_statement_direct": {"support_mode": "exact", "value": 25.0},
            "liquidity.restricted_cash_sec_exact": {"support_mode": "unsupported", "value": None},
            "liquidity.marketable_securities_sec_exact": {"support_mode": "unsupported", "value": None},
            "liquidity.revolver_undrawn_sec_exact": {"support_mode": "unsupported", "value": None},
            "operating.ebitda_ltm_provider_direct": {"support_mode": "exact", "value": 50.0},
            "operating.ebit_statement_direct": {"support_mode": "exact", "value": 40.0},
            "capital_structure.interest_expense_statement_direct": {"support_mode": "unsupported", "value": None},
        },
    }
    companyfacts = {
        "facts": {
            "us-gaap": {
                "PensionAndOtherPostretirementAndPostemploymentBenefitPlansLiabilitiesNoncurrent": {
                    "units": {
                        "USD": [
                            {
                                "end": "2024-12-31",
                                "val": 12.0,
                                "filed": "2025-02-01",
                                "form": "10-K",
                                "fy": 2024,
                                "fp": "FY",
                            }
                        ]
                    }
                }
            }
        }
    }

    def _timeout_loader():
        raise smart_mod._CompanyProcessingTimeout("retirement_note_timeout")

    repaired = materialize_smart_metrics_for_row(
        row=row,
        registry=registry,
        computed_at="2026-04-02T00:00:00Z",
        provenance_sources=["registry.json"],
        companyfacts=companyfacts,
        retirement_note_loader=_timeout_loader,
    )

    assert repaired["features"]["capital_structure.net_pension_liability"]["support_mode"] == "unsupported"
    combined = repaired["features"]["capital_structure.combined_retirement_liability"]
    assert combined["support_mode"] == "proxy_missing_component"
    assert combined["value"] == 12.0
    assert repaired["features"]["capital_structure.retirement_obligation_regime"]["value"] == "combined_retirement_only"


def test_load_companyfacts_returns_none_on_timeout(monkeypatch, tmp_path):
    path = tmp_path / "companyfacts.json"
    path.write_text("{}")

    def _raise_timeout(*args, **kwargs):  # noqa: ARG001
        raise smart_mod.subprocess.TimeoutExpired(cmd="/bin/cat", timeout=1.5)

    monkeypatch.setattr(smart_mod.subprocess, "run", _raise_timeout)

    assert _load_companyfacts(path) is None


def test_load_completed_company_ids_reads_partial_output(tmp_path):
    path = tmp_path / "partial.jsonl"
    rows = [
        {"company_id": "0001", "features": {}},
        {"company_id": "0002", "features": {}},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    assert _load_completed_company_ids(path) == {"0001", "0002"}


def test_summarize_output_rows_counts_fail_open(tmp_path):
    path = tmp_path / "smart.jsonl"
    row = {
        "company_id": "0001",
        "features": _build_fail_open_smart_metrics(
            as_of_time="2024-12-31T00:00:00Z",
            computed_at="2026-03-31T00:00:00Z",
            provenance_sources=["registry.json"],
            error_type="company_processing_timeout",
            error_message="boom",
        ),
    }
    path.write_text(json.dumps(row) + "\n")

    summary = _summarize_output_rows(path)

    assert summary["capital_structure.debt_like_obligations_normalized"]["unsupported"] == 1
    assert summary["capital_structure.net_leverage_normalized"]["unsupported"] == 1
    assert summary["row_fail_open"]["company_processing_timeout"] == 1
