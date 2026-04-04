import time

import pytest

from scripts.backfill_input_layer_v1_metrics import (
    _CompanyProcessingTimeout,
    _build_legacy_provider_metric,
    _build_sec_core_metric,
    _build_fail_open_metric_set,
    _company_processing_guard,
    _compute_ttm_from_concept,
    _select_preferred_direct_metric,
)


def _instant_fact(value: float):
    return {
        "val": value,
        "end": "2024-09-30",
        "filed": "2024-10-31",
        "fy": 2024,
        "fp": "Q3",
        "form": "10-Q",
        "frame": "CY2024Q3I",
    }


def _duration_fact(value: float, *, start: str, end: str, filed: str, fy: int, fp: str, form: str = "10-Q", frame=None):
    fact = {
        "val": value,
        "start": start,
        "end": end,
        "filed": filed,
        "fy": fy,
        "fp": fp,
        "form": form,
    }
    if frame is not None:
        fact["frame"] = frame
    return fact


def _metric_node(*, value, support_mode, primary_source_basis, missing_reason=None, quality_flags=None, breakdown=None):
    return {
        "value": value,
        "support_mode": support_mode,
        "primary_source_basis": primary_source_basis,
        "missing_reason": missing_reason,
        "quality_flags": quality_flags,
        "component_breakdown": breakdown or {},
    }


def test_selection_prefers_sec_revenue_over_provider_direct():
    sec_node = _metric_node(
        value=258_805_000_000.0,
        support_mode="exact",
        primary_source_basis="sec_companyfacts",
    )
    provider_node = _metric_node(
        value=275_235_000_000.0,
        support_mode="exact",
        primary_source_basis="provider_direct",
    )

    selected = _select_preferred_direct_metric(
        metric_name="operating.revenue_ttm_provider_direct",
        sec_or_market_node=sec_node,
        provider_node=provider_node,
    )

    assert selected["primary_source_basis"] == "sec_companyfacts"
    assert selected["value"] == 258_805_000_000.0
    assert "provider_direct_superseded_by_sec_companyfacts" in (selected.get("quality_flags") or [])
    assert selected["component_breakdown"]["selection_policy"] == "prefer_sec_companyfacts_reconstruction"


def test_revenue_ttm_lag_1y_uses_prior_year_ttm_asof():
    companyfacts = {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            _duration_fact(
                                80.0,
                                start="2022-01-01",
                                end="2022-12-31",
                                filed="2023-02-15",
                                fy=2022,
                                fp="FY",
                                form="10-K",
                            ),
                            _duration_fact(
                                60.0,
                                start="2022-01-01",
                                end="2022-09-30",
                                filed="2022-11-01",
                                fy=2022,
                                fp="Q3",
                                form="10-Q",
                            ),
                            _duration_fact(
                                72.0,
                                start="2023-01-01",
                                end="2023-09-30",
                                filed="2023-11-01",
                                fy=2023,
                                fp="Q3",
                                form="10-Q",
                            ),
                            _duration_fact(
                                81.0,
                                start="2024-01-01",
                                end="2024-09-30",
                                filed="2024-11-01",
                                fy=2024,
                                fp="Q3",
                                form="10-Q",
                            ),
                        ]
                    }
                }
            }
        }
    }

    value, support_mode, missing_reason, component_breakdown, quality_flags = _build_sec_core_metric(
        "operating.revenue_ttm_lag_1y",
        companyfacts,
        "2024-12-31",
    )

    assert value == 92.0
    assert support_mode == "exact"
    assert missing_reason is None
    assert quality_flags is None
    assert component_breakdown["lagged_as_of_date"] == "2023-12-31"
    assert component_breakdown["mode"] == "ytd_plus_prior_fy_minus_prior_ytd"
    assert component_breakdown["concept"] == "Revenues"


def test_legacy_provider_metric_is_cleanly_unsupported_when_no_provider_field_exists():
    node = _build_legacy_provider_metric(
        metric_name="operating.revenue_ttm_lag_1y",
        provider_row=None,
        as_of_time="2024-12-31T00:00:00+00:00",
        computed_at="2026-04-05T00:00:00+00:00",
        provenance_source="/tmp/provider.parquet",
        unit="usd",
    )

    assert node["support_mode"] == "unsupported"
    assert node["missing_reason"] == "provider_direct_field_not_defined_for_metric"
    assert "provider_direct_field_not_defined_for_metric" in (node.get("quality_flags") or [])


def test_net_income_falls_back_to_profit_loss_when_net_income_loss_is_stale_or_missing():
    companyfacts = {
        "facts": {
            "us-gaap": {
                "NetIncomeLoss": {
                    "units": {
                        "USD": [
                            _duration_fact(
                                2_700_000_000.0,
                                start="2010-01-01",
                                end="2010-12-31",
                                filed="2011-02-22",
                                fy=2010,
                                fp="FY",
                                form="10-K",
                            )
                        ]
                    }
                },
                "ProfitLoss": {
                    "units": {
                        "USD": [
                            _duration_fact(
                                6_493_000_000.0,
                                start="2023-01-01",
                                end="2023-12-31",
                                filed="2024-02-15",
                                fy=2023,
                                fp="FY",
                                form="10-K",
                            ),
                            _duration_fact(
                                7_659_000_000.0,
                                start="2023-01-01",
                                end="2023-09-30",
                                filed="2023-11-01",
                                fy=2023,
                                fp="Q3",
                                form="10-Q",
                            ),
                            _duration_fact(
                                7_998_000_000.0,
                                start="2024-01-01",
                                end="2024-09-30",
                                filed="2024-11-06",
                                fy=2024,
                                fp="Q3",
                                form="10-Q",
                            ),
                            _duration_fact(
                                2_463_000_000.0,
                                start="2024-07-01",
                                end="2024-09-30",
                                filed="2024-11-06",
                                fy=2024,
                                fp="Q3",
                                form="10-Q",
                                frame="CY2024Q3",
                            ),
                            _duration_fact(
                                5_535_000_000.0,
                                start="2024-01-01",
                                end="2024-06-30",
                                filed="2024-08-07",
                                fy=2024,
                                fp="Q2",
                                form="10-Q",
                            ),
                            _duration_fact(
                                2_681_000_000.0,
                                start="2024-04-01",
                                end="2024-06-30",
                                filed="2024-08-07",
                                fy=2024,
                                fp="Q2",
                                form="10-Q",
                                frame="CY2024Q2",
                            ),
                            _duration_fact(
                                2_854_000_000.0,
                                start="2024-01-01",
                                end="2024-03-31",
                                filed="2024-05-01",
                                fy=2024,
                                fp="Q1",
                                form="10-Q",
                                frame="CY2024Q1",
                            ),
                        ]
                    }
                },
            }
        }
    }

    value, support_mode, missing_reason, component_breakdown, quality_flags = _build_sec_core_metric(
        "earnings.net_income_ttm_provider_direct",
        companyfacts,
        "2024-12-31",
    )

    assert value == 6_832_000_000.0
    assert support_mode == "exact"
    assert missing_reason is None
    assert quality_flags is None
    assert component_breakdown["concept"] == "ProfitLoss"


def test_selection_keeps_provider_ebitda_when_sec_bridge_is_only_partial():
    sec_node = _metric_node(
        value=11_391_000_000.0,
        support_mode="proxy_missing_component",
        primary_source_basis="sec_companyfacts",
        quality_flags=["partial_depreciation_without_full_amortization"],
    )
    provider_node = _metric_node(
        value=14_668_000_000.0,
        support_mode="exact",
        primary_source_basis="provider_direct",
    )

    selected = _select_preferred_direct_metric(
        metric_name="operating.ebitda_ltm_provider_direct",
        sec_or_market_node=sec_node,
        provider_node=provider_node,
    )

    assert selected["primary_source_basis"] == "provider_direct"
    assert selected["value"] == 14_668_000_000.0
    assert "provider_direct_retained_due_to_partial_sec_ebitda_bridge" in (selected.get("quality_flags") or [])


def test_selection_keeps_provider_total_debt_when_sec_stack_is_unavailable():
    provider_node = _metric_node(
        value=158_522_000_000.0,
        support_mode="exact",
        primary_source_basis="provider_direct",
    )
    sec_node = _metric_node(
        value=None,
        support_mode="unsupported",
        primary_source_basis="sec_companyfacts",
        missing_reason="sec_debt_components_unavailable",
    )

    selected = _select_preferred_direct_metric(
        metric_name="capital_structure.total_debt_provider_direct",
        sec_or_market_node=sec_node,
        provider_node=provider_node,
    )

    assert selected["primary_source_basis"] == "provider_direct"
    assert selected["value"] == 158_522_000_000.0
    assert "provider_direct_retained_due_to_partial_sec_debt_stack" in (selected.get("quality_flags") or [])


def test_selection_prefers_pit_market_cap_over_provider_direct():
    market_node = _metric_node(
        value=947_000_000_000.0,
        support_mode="exact",
        primary_source_basis="sec_companyfacts",
        breakdown={"formula": "price_spot * shares_outstanding"},
    )
    provider_node = _metric_node(
        value=949_565_692_090.96,
        support_mode="exact",
        primary_source_basis="provider_direct",
    )

    selected = _select_preferred_direct_metric(
        metric_name="market.market_cap_provider_direct",
        sec_or_market_node=market_node,
        provider_node=provider_node,
    )

    assert selected["primary_source_basis"] == "sec_companyfacts"
    assert selected["value"] == 947_000_000_000.0
    assert "provider_direct_superseded_by_pit_market_cap" in (selected.get("quality_flags") or [])


def test_selection_keeps_provider_market_cap_when_pit_cap_is_only_proxy():
    market_node = _metric_node(
        value=725_816_444_525.25,
        support_mode="proxy_missing_component",
        primary_source_basis="sec_companyfacts",
        missing_reason="price_component_not_exact",
    )
    provider_node = _metric_node(
        value=949_565_692_090.96,
        support_mode="exact",
        primary_source_basis="provider_direct",
    )

    selected = _select_preferred_direct_metric(
        metric_name="market.market_cap_provider_direct",
        sec_or_market_node=market_node,
        provider_node=provider_node,
    )

    assert selected["primary_source_basis"] == "provider_direct"
    assert selected["value"] == 949_565_692_090.96
    assert "provider_direct_retained_due_to_proxy_pit_market_cap" in (selected.get("quality_flags") or [])


def test_total_debt_does_not_double_count_short_term_borrowings_when_they_overlap_current_debt():
    companyfacts = {
        "facts": {
            "us-gaap": {
                "LongTermDebtCurrent": {"units": {"USD": [_instant_fact(39_700_000.0)]}},
                "LongTermDebtNoncurrent": {"units": {"USD": [_instant_fact(360_200_000.0)]}},
                "LongTermDebt": {"units": {"USD": [_instant_fact(410_600_000.0)]}},
                "ShortTermBorrowings": {"units": {"USD": [_instant_fact(39_700_000.0)]}},
            }
        }
    }

    value, support_mode, missing_reason, component_breakdown, quality_flags = _build_sec_core_metric(
        "capital_structure.total_debt_provider_direct",
        companyfacts,
        "2024-12-31",
    )

    assert value == 360_200_000.0
    assert support_mode == "exact"
    assert missing_reason is None
    assert quality_flags is None
    assert component_breakdown["mode"] == "long_term_debt_with_overlapping_short_term_borrowings"
    assert component_breakdown["formula"] == "exact_long_term_debt_total_due_to_current_short_term_overlap"


def test_total_debt_uses_long_term_debt_total_when_it_is_the_only_exact_debt_total():
    companyfacts = {
        "facts": {
            "us-gaap": {
                "LongTermDebt": {"units": {"USD": [_instant_fact(6_794_502_000.0)]}},
            }
        }
    }

    value, support_mode, missing_reason, component_breakdown, quality_flags = _build_sec_core_metric(
        "capital_structure.total_debt_provider_direct",
        companyfacts,
        "2024-12-31",
    )

    assert value == 6_794_502_000.0
    assert support_mode == "exact"
    assert missing_reason is None
    assert quality_flags is None
    assert component_breakdown["mode"] == "long_term_debt_total_only"


def test_total_debt_uses_noncurrent_debt_total_when_it_is_the_only_exact_debt_total():
    companyfacts = {
        "facts": {
            "us-gaap": {
                "LongTermDebtNoncurrent": {"units": {"USD": [_instant_fact(360_200_000.0)]}},
            }
        }
    }

    value, support_mode, missing_reason, component_breakdown, quality_flags = _build_sec_core_metric(
        "capital_structure.total_debt_provider_direct",
        companyfacts,
        "2024-12-31",
    )

    assert value == 360_200_000.0
    assert support_mode == "exact"
    assert missing_reason is None
    assert quality_flags is None
    assert component_breakdown["mode"] == "noncurrent_debt_total_only"
    assert component_breakdown["noncurrent_debt_total"]["concept"] == "LongTermDebtNoncurrent"


def test_total_debt_uses_convertible_debt_total_when_it_is_the_only_exact_debt_total():
    companyfacts = {
        "facts": {
            "us-gaap": {
                "ConvertibleDebt": {"units": {"USD": [_instant_fact(393_588_000.0)]}},
            }
        }
    }

    value, support_mode, missing_reason, component_breakdown, quality_flags = _build_sec_core_metric(
        "capital_structure.total_debt_provider_direct",
        companyfacts,
        "2024-12-31",
    )

    assert value == 393_588_000.0
    assert support_mode == "exact"
    assert missing_reason is None
    assert quality_flags is None
    assert component_breakdown["mode"] == "noncurrent_debt_total_only"
    assert component_breakdown["noncurrent_debt_total"]["concept"] == "ConvertibleDebt"


def test_total_debt_uses_short_term_borrowings_plus_noncurrent_debt_when_aligned_current_is_missing():
    companyfacts = {
        "facts": {
            "us-gaap": {
                "LinesOfCreditCurrent": {"units": {"USD": [_instant_fact(50_000_000.0)]}},
                "LongTermLineOfCredit": {"units": {"USD": [_instant_fact(200_000_000.0)]}},
            }
        }
    }

    value, support_mode, missing_reason, component_breakdown, quality_flags = _build_sec_core_metric(
        "capital_structure.total_debt_provider_direct",
        companyfacts,
        "2024-12-31",
    )

    assert value == 250_000_000.0
    assert support_mode == "exact"
    assert missing_reason is None
    assert quality_flags is None
    assert component_breakdown["mode"] == "short_term_borrowings_plus_noncurrent_debt"


def test_total_debt_adds_secured_borrowings_to_generic_current_and_noncurrent_debt_stack():
    companyfacts = {
        "facts": {
            "us-gaap": {
                "DebtCurrent": {"units": {"USD": [_instant_fact(14_392_000_000.0)]}},
                "SecuredDebt": {"units": {"USD": [_instant_fact(6_283_000_000.0)]}},
                "LongTermDebtNoncurrent": {"units": {"USD": [_instant_fact(41_804_000_000.0)]}},
            }
        }
    }

    value, support_mode, missing_reason, component_breakdown, quality_flags = _build_sec_core_metric(
        "capital_structure.total_debt_provider_direct",
        companyfacts,
        "2026-03-28",
    )

    assert value == 62_479_000_000.0
    assert support_mode == "exact"
    assert missing_reason is None
    assert quality_flags is None
    assert component_breakdown["mode"] == "current_plus_noncurrent_debt_plus_short_term_borrowings"
    assert component_breakdown["short_term_borrowings"]["concept"] == "SecuredDebt"


def test_company_processing_guard_times_out():
    with pytest.raises(_CompanyProcessingTimeout):
        with _company_processing_guard(0.05):
            time.sleep(0.2)


def test_fail_open_metric_set_marks_all_metrics_unsupported():
    metrics = _build_fail_open_metric_set(
        as_of_time="2024-12-31T00:00:00+00:00",
        computed_at="2026-03-29T22:30:00+00:00",
        provenance_source="/tmp/companyfacts/CIK0000000001.json",
        error_type="company_processing_timeout",
        error_message="timed out on issuer parse",
    )

    assert metrics["operating.revenue_ttm_provider_direct"]["support_mode"] == "unsupported"
    assert metrics["capital_structure.net_debt_standardized"]["support_mode"] == "unsupported"
    assert metrics["capital_structure.gross_leverage_standardized"]["missing_reason"] == "company_processing_timeout"
    assert metrics["earnings.net_margin_standardized"]["component_breakdown"]["error_type"] == "company_processing_timeout"
    assert "company_processing_fail_open" in (metrics["market.market_cap_provider_direct"]["quality_flags"] or [])


def test_cash_and_short_term_investments_uses_current_afs_debt_securities_when_marketable_concept_is_company_specific():
    companyfacts = {
        "facts": {
            "us-gaap": {
                "CashAndCashEquivalentsAtCarryingValue": {"units": {"USD": [_instant_fact(20_945_000_000.0)]}},
                "AvailableForSaleSecuritiesDebtSecuritiesCurrent": {"units": {"USD": [_instant_fact(6_724_000_000.0)]}},
            }
        }
    }

    value, support_mode, missing_reason, component_breakdown, quality_flags = _build_sec_core_metric(
        "liquidity.cash_and_short_term_investments_provider_direct",
        companyfacts,
        "2026-03-28",
    )

    assert value == 27_669_000_000.0
    assert support_mode == "exact"
    assert missing_reason is None
    assert quality_flags is None
    assert component_breakdown["mode"] == "cash_plus_short_term_investments"
    assert component_breakdown["short_term_investments"]["concept"] == "AvailableForSaleSecuritiesDebtSecuritiesCurrent"


def test_cash_and_short_term_investments_uses_within_one_year_afs_maturity_as_short_term_investments():
    companyfacts = {
        "facts": {
            "us-gaap": {
                "CashAndCashEquivalentsAtCarryingValue": {"units": {"USD": [_instant_fact(9_980_000_000.0)]}},
                "AvailableForSaleSecuritiesDebtMaturitiesWithinOneYearFairValue": {
                    "units": {"USD": [_instant_fact(748_000_000.0)]}
                },
            }
        }
    }

    value, support_mode, missing_reason, component_breakdown, quality_flags = _build_sec_core_metric(
        "liquidity.cash_and_short_term_investments_provider_direct",
        companyfacts,
        "2026-03-28",
    )

    assert value == 10_728_000_000.0
    assert support_mode == "exact"
    assert missing_reason is None
    assert quality_flags is None
    assert component_breakdown["mode"] == "cash_plus_short_term_investments"
    assert (
        component_breakdown["short_term_investments"]["concept"]
        == "AvailableForSaleSecuritiesDebtMaturitiesWithinOneYearFairValue"
    )


def test_cash_and_short_term_investments_uses_combined_cash_restricted_total_when_only_cash_is_current():
    current_fact = {
        "val": 4_121_000_000.0,
        "end": "2025-12-31",
        "filed": "2026-01-29",
        "fy": 2025,
        "fp": "FY",
        "form": "10-K",
        "frame": "CY2025Q4I",
    }
    companyfacts = {
        "facts": {
            "us-gaap": {
                "CashAndCashEquivalentsAtCarryingValue": {"units": {"USD": [current_fact]}},
                "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents": {"units": {"USD": [current_fact]}},
            }
        }
    }

    value, support_mode, missing_reason, component_breakdown, quality_flags = _build_sec_core_metric(
        "liquidity.cash_and_short_term_investments_provider_direct",
        companyfacts,
        "2026-03-29",
    )

    assert value == 4_121_000_000.0
    assert support_mode == "exact"
    assert missing_reason is None
    assert quality_flags is None
    assert component_breakdown["mode"] == "combined_cash_restricted_less_restricted_plus_short_term_investments"
    assert component_breakdown["restricted_cash_adjustment"]["mode"] == "infer_zero_restricted_cash_due_to_absent_current_restricted_cash_concept"


def test_cash_and_short_term_investments_uses_combined_cash_restricted_total_plus_short_term_investments_when_cash_current_missing():
    current_combined = {
        "val": 670_000_000.0,
        "end": "2025-12-31",
        "filed": "2026-02-12",
        "fy": 2025,
        "fp": "FY",
        "form": "10-K",
        "frame": "CY2025Q4I",
    }
    current_sti = {
        "val": 5_000_000.0,
        "end": "2025-12-31",
        "filed": "2026-02-12",
        "fy": 2025,
        "fp": "FY",
        "form": "10-K",
        "frame": "CY2025Q4I",
    }
    companyfacts = {
        "facts": {
            "us-gaap": {
                "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents": {"units": {"USD": [current_combined]}},
                "ShortTermInvestments": {"units": {"USD": [current_sti]}},
            }
        }
    }

    value, support_mode, missing_reason, component_breakdown, quality_flags = _build_sec_core_metric(
        "liquidity.cash_and_short_term_investments_provider_direct",
        companyfacts,
        "2026-03-29",
    )

    assert value == 675_000_000.0
    assert support_mode == "exact"
    assert missing_reason is None
    assert quality_flags is None
    assert component_breakdown["mode"] == "combined_cash_restricted_less_restricted_plus_short_term_investments"
    assert component_breakdown["short_term_investments"]["concept"] == "ShortTermInvestments"


def test_cash_and_short_term_investments_subtracts_current_restricted_cash_from_combined_cash_restricted_total():
    current_combined = {
        "val": 4_501_000_000.0,
        "end": "2025-12-31",
        "filed": "2026-02-11",
        "fy": 2025,
        "fp": "FY",
        "form": "10-K",
        "frame": "CY2025Q4I",
    }
    current_restricted = {
        "val": 135_000_000.0,
        "end": "2025-12-31",
        "filed": "2026-02-11",
        "fy": 2025,
        "fp": "FY",
        "form": "10-K",
        "frame": "CY2025Q4I",
    }
    companyfacts = {
        "facts": {
            "us-gaap": {
                "CashAndCashEquivalentsAtCarryingValue": {"units": {"USD": [{"val": 4_310_000_000.0, **{k: v for k, v in current_combined.items() if k != 'val'}}]}},
                "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents": {"units": {"USD": [current_combined]}},
                "RestrictedCashAndCashEquivalentsAtCarryingValue": {"units": {"USD": [current_restricted]}},
            }
        }
    }

    value, support_mode, missing_reason, component_breakdown, quality_flags = _build_sec_core_metric(
        "liquidity.cash_and_short_term_investments_provider_direct",
        companyfacts,
        "2026-03-29",
    )

    assert value == 4_366_000_000.0
    assert support_mode == "exact"
    assert missing_reason is None
    assert quality_flags is None
    assert component_breakdown["restricted_cash_adjustment"]["concept"] == "RestrictedCashAndCashEquivalentsAtCarryingValue"


def test_total_debt_uses_combined_including_current_maturities_when_available():
    companyfacts = {
        "facts": {
            "us-gaap": {
                "LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities": {
                    "units": {"USD": [_instant_fact(310_000_000.0)]}
                },
                "FinanceLeaseLiability": {"units": {"USD": [_instant_fact(30_000_000.0)]}},
                "FinanceLeaseLiabilityCurrent": {"units": {"USD": [_instant_fact(10_000_000.0)]}},
                "FinanceLeaseLiabilityNoncurrent": {"units": {"USD": [_instant_fact(20_000_000.0)]}},
            }
        }
    }

    value, support_mode, missing_reason, component_breakdown, quality_flags = _build_sec_core_metric(
        "capital_structure.total_debt_provider_direct",
        companyfacts,
        "2024-12-31",
    )

    assert value == 280_000_000.0
    assert support_mode == "exact"
    assert missing_reason is None
    assert quality_flags is None
    assert component_breakdown["mode"] == "combined_debt"


def test_total_debt_keeps_combined_capital_lease_total_exact_when_finance_lease_concepts_are_absent():
    companyfacts = {
        "facts": {
            "us-gaap": {
                "LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities": {
                    "units": {"USD": [_instant_fact(310_000_000.0)]}
                },
            }
        }
    }

    value, support_mode, missing_reason, component_breakdown, quality_flags = _build_sec_core_metric(
        "capital_structure.total_debt_provider_direct",
        companyfacts,
        "2024-12-31",
    )

    assert value == 310_000_000.0
    assert support_mode == "exact"
    assert missing_reason is None
    assert quality_flags is None
    assert component_breakdown["mode"] == "combined_debt"
    assert component_breakdown["finance_lease_adjustment"]["mode"] == "infer_zero_finance_lease_adjustment_due_to_absent_finance_lease_concepts"
    assert component_breakdown["finance_lease_adjustment_value"] == 0.0


def test_total_debt_infers_zero_proxy_when_no_balance_sheet_debt_concepts_are_present():
    companyfacts = {
        "facts": {
            "us-gaap": {
                "ProceedsFromIssuanceOfLongTermDebt": {"units": {"USD": [_instant_fact(50_000_000.0)]}},
                "RepaymentsOfDebt": {"units": {"USD": [_instant_fact(50_000_000.0)]}},
            }
        }
    }

    value, support_mode, missing_reason, component_breakdown, quality_flags = _build_sec_core_metric(
        "capital_structure.total_debt_provider_direct",
        companyfacts,
        "2024-12-31",
    )

    assert value == 0.0
    assert support_mode == "proxy_missing_component"
    assert missing_reason == "no_debt_balance_concepts_present"
    assert quality_flags == ["no_debt_balance_concepts_present"]
    assert component_breakdown["mode"] == "no_debt_balance_concepts_present"


def test_compute_ttm_prefers_comparative_prior_period_from_newer_filing_even_when_fy_matches_latest():
    companyfacts = {
        "facts": {
            "us-gaap": {
                "DepreciationDepletionAndAmortization": {
                    "units": {
                        "USD": [
                            _duration_fact(
                                45_139.0,
                                start="2023-01-01",
                                end="2023-09-30",
                                filed="2023-11-06",
                                fy=2023,
                                fp="Q3",
                            ),
                            _duration_fact(
                                45_139_000.0,
                                start="2023-01-01",
                                end="2023-09-30",
                                filed="2024-11-04",
                                fy=2024,
                                fp="Q3",
                            ),
                            _duration_fact(
                                58_169_000.0,
                                start="2023-01-01",
                                end="2023-12-31",
                                filed="2024-02-26",
                                fy=2023,
                                fp="FY",
                                form="10-K",
                            ),
                            _duration_fact(
                                51_936_000.0,
                                start="2024-01-01",
                                end="2024-09-30",
                                filed="2024-11-04",
                                fy=2024,
                                fp="Q3",
                            ),
                        ]
                    }
                }
            }
        }
    }

    value, meta = _compute_ttm_from_concept(
        companyfacts,
        "DepreciationDepletionAndAmortization",
        "2024-12-31",
    )

    assert value == 64_966_000.0
    assert meta is not None
    assert meta["prior_same_period"]["filed"] == "2024-11-04"
    assert meta["prior_same_period"]["value"] == 45_139_000.0


def test_ebitda_ttm_downgrades_when_depreciation_bridge_is_stale_latest_fy_only():
    companyfacts = {
        "facts": {
            "us-gaap": {
                "OperatingIncomeLoss": {
                    "units": {
                        "USD": [
                            _duration_fact(
                                1_480_000_000.0,
                                start="2024-01-01",
                                end="2024-09-30",
                                filed="2024-10-24",
                                fy=2024,
                                fp="Q3",
                            ),
                            _duration_fact(
                                3_034_000_000.0,
                                start="2023-01-01",
                                end="2023-12-31",
                                filed="2024-02-21",
                                fy=2023,
                                fp="FY",
                                form="10-K",
                            ),
                            _duration_fact(
                                2_378_000_000.0,
                                start="2023-01-01",
                                end="2023-09-30",
                                filed="2023-10-19",
                                fy=2023,
                                fp="Q3",
                            ),
                        ]
                    }
                },
                "DepreciationDepletionAndAmortization": {
                    "units": {
                        "USD": [
                            _duration_fact(
                                2_300_000_000.0,
                                start="2023-01-01",
                                end="2023-12-31",
                                filed="2024-02-21",
                                fy=2023,
                                fp="FY",
                                form="10-K",
                            ),
                        ]
                    }
                },
            }
        }
    }

    value, support_mode, missing_reason, component_breakdown, quality_flags = _build_sec_core_metric(
        "operating.ebitda_ltm_provider_direct",
        companyfacts,
        "2024-12-31",
    )

    assert value == 4_436_000_000.0
    assert support_mode == "proxy_missing_component"
    assert missing_reason is None
    assert "stale_depreciation_amortization_bridge" in (quality_flags or [])


def test_ebitda_ttm_prefers_fresher_direct_depreciation_and_amortization_concept():
    companyfacts = {
        "facts": {
            "us-gaap": {
                "OperatingIncomeLoss": {
                    "units": {
                        "USD": [
                            _duration_fact(
                                1_480_000_000.0,
                                start="2024-01-01",
                                end="2024-09-30",
                                filed="2024-10-24",
                                fy=2024,
                                fp="Q3",
                            ),
                            _duration_fact(
                                3_034_000_000.0,
                                start="2023-01-01",
                                end="2023-12-31",
                                filed="2024-02-21",
                                fy=2023,
                                fp="FY",
                                form="10-K",
                            ),
                            _duration_fact(
                                2_378_000_000.0,
                                start="2023-01-01",
                                end="2023-09-30",
                                filed="2024-10-24",
                                fy=2024,
                                fp="Q3",
                            ),
                        ]
                    }
                },
                "DepreciationDepletionAndAmortization": {
                    "units": {
                        "USD": [
                            _duration_fact(
                                2_300_000_000.0,
                                start="2023-01-01",
                                end="2023-12-31",
                                filed="2024-02-21",
                                fy=2023,
                                fp="FY",
                                form="10-K",
                            ),
                        ]
                    }
                },
                "DepreciationAndAmortization": {
                    "units": {
                        "USD": [
                            _duration_fact(
                                1_424_000_000.0,
                                start="2024-01-01",
                                end="2024-09-30",
                                filed="2024-10-24",
                                fy=2024,
                                fp="Q3",
                            ),
                            _duration_fact(
                                1_936_000_000.0,
                                start="2023-01-01",
                                end="2023-12-31",
                                filed="2024-02-21",
                                fy=2023,
                                fp="FY",
                                form="10-K",
                            ),
                            _duration_fact(
                                1_456_000_000.0,
                                start="2023-01-01",
                                end="2023-09-30",
                                filed="2024-10-24",
                                fy=2024,
                                fp="Q3",
                            ),
                        ]
                    }
                },
            }
        }
    }

    value, support_mode, missing_reason, component_breakdown, quality_flags = _build_sec_core_metric(
        "operating.ebitda_ltm_provider_direct",
        companyfacts,
        "2024-12-31",
    )

    assert value == 4_040_000_000.0
    assert support_mode == "exact"
    assert missing_reason is None
    assert quality_flags is None
    depreciation_meta = component_breakdown["depreciation_amortization"]
    assert depreciation_meta["concept"] == "DepreciationAndAmortization"
    assert depreciation_meta["mode"] == "ytd_plus_prior_fy_minus_prior_ytd"


def test_ebitda_ttm_prefers_depreciation_plus_amortization_sum_over_depreciation_only():
    companyfacts = {
        "facts": {
            "us-gaap": {
                "OperatingIncomeLoss": {
                    "units": {
                        "USD": [
                            _duration_fact(
                                403_800_000.0,
                                start="2024-01-01",
                                end="2024-09-30",
                                filed="2024-10-31",
                                fy=2024,
                                fp="Q3",
                            ),
                            _duration_fact(
                                342_800_000.0,
                                start="2023-01-01",
                                end="2023-12-31",
                                filed="2024-02-23",
                                fy=2023,
                                fp="FY",
                                form="10-K",
                            ),
                            _duration_fact(
                                267_500_000.0,
                                start="2023-01-01",
                                end="2023-09-30",
                                filed="2024-10-31",
                                fy=2024,
                                fp="Q3",
                            ),
                        ]
                    }
                },
                "Depreciation": {
                    "units": {
                        "USD": [
                            _duration_fact(
                                124_400_000.0,
                                start="2023-01-01",
                                end="2023-12-31",
                                filed="2024-02-23",
                                fy=2023,
                                fp="FY",
                                form="10-K",
                            ),
                        ]
                    }
                },
                "AmortizationOfIntangibleAssets": {
                    "units": {
                        "USD": [
                            _duration_fact(
                                13_600_000.0,
                                start="2023-01-01",
                                end="2023-12-31",
                                filed="2024-02-23",
                                fy=2023,
                                fp="FY",
                                form="10-K",
                            ),
                        ]
                    }
                },
            }
        }
    }

    value, support_mode, missing_reason, component_breakdown, quality_flags = _build_sec_core_metric(
        "operating.ebitda_ltm_provider_direct",
        companyfacts,
        "2024-12-31",
    )

    assert value == 617_100_000.0
    assert support_mode == "proxy_missing_component"
    assert missing_reason is None
    assert "partial_depreciation_without_full_amortization" not in (quality_flags or [])
    assert "stale_depreciation_amortization_bridge" in (quality_flags or [])
    depreciation_meta = component_breakdown["depreciation_amortization"]
    assert depreciation_meta["mode"] == "sum_concepts"


def test_ebitda_ttm_prefers_fresher_depreciation_plus_amortization_sum_over_depreciation_only():
    companyfacts = {
        "facts": {
            "us-gaap": {
                "OperatingIncomeLoss": {
                    "units": {
                        "USD": [
                            _duration_fact(
                                172_733_000.0,
                                start="2024-02-04",
                                end="2024-11-02",
                                filed="2024-12-11",
                                fy=2024,
                                fp="Q3",
                            ),
                            _duration_fact(
                                209_636_000.0,
                                start="2023-02-04",
                                end="2024-02-03",
                                filed="2024-04-02",
                                fy=2023,
                                fp="FY",
                                form="10-K",
                            ),
                            _duration_fact(
                                148_217_000.0,
                                start="2023-01-29",
                                end="2023-10-28",
                                filed="2024-12-11",
                                fy=2024,
                                fp="Q3",
                            ),
                        ]
                    }
                },
                "DepreciationDepletionAndAmortization": {
                    "units": {
                        "USD": [
                            _duration_fact(
                                53_280_000.0,
                                start="2023-02-05",
                                end="2024-02-03",
                                filed="2024-04-02",
                                fy=2023,
                                fp="FY",
                                form="10-K",
                            ),
                        ]
                    }
                },
                "Depreciation": {
                    "units": {
                        "USD": [
                            _duration_fact(
                                29_456_000.0,
                                start="2024-02-04",
                                end="2024-11-02",
                                filed="2024-12-11",
                                fy=2024,
                                fp="Q3",
                            ),
                            _duration_fact(
                                36_172_000.0,
                                start="2023-02-05",
                                end="2024-02-03",
                                filed="2024-04-02",
                                fy=2023,
                                fp="FY",
                                form="10-K",
                            ),
                            _duration_fact(
                                25_575_000.0,
                                start="2023-01-29",
                                end="2023-10-28",
                                filed="2024-12-11",
                                fy=2024,
                                fp="Q3",
                            ),
                        ]
                    }
                },
                "AmortizationOfIntangibleAssets": {
                    "units": {
                        "USD": [
                            _duration_fact(
                                8_277_000.0,
                                start="2024-02-04",
                                end="2024-11-02",
                                filed="2024-12-11",
                                fy=2024,
                                fp="Q3",
                            ),
                            _duration_fact(
                                10_541_000.0,
                                start="2023-02-05",
                                end="2024-02-03",
                                filed="2024-04-02",
                                fy=2023,
                                fp="FY",
                                form="10-K",
                            ),
                            _duration_fact(
                                7_576_000.0,
                                start="2023-01-29",
                                end="2023-10-28",
                                filed="2024-12-11",
                                fy=2024,
                                fp="Q3",
                            ),
                        ]
                    }
                },
            }
        }
    }

    value, support_mode, missing_reason, component_breakdown, quality_flags = _build_sec_core_metric(
        "operating.ebitda_ltm_provider_direct",
        companyfacts,
        "2024-12-31",
    )

    assert value == 285_447_000.0
    assert support_mode == "exact"
    assert missing_reason is None
    assert quality_flags is None
    depreciation_meta = component_breakdown["depreciation_amortization"]
    assert depreciation_meta["mode"] == "sum_concepts"


def test_ebitda_ttm_uses_fresh_other_depreciation_and_amortization_when_standard_direct_concepts_are_stale():
    companyfacts = {
        "facts": {
            "us-gaap": {
                "OperatingIncomeLoss": {
                    "units": {
                        "USD": [
                            _duration_fact(
                                100_000_000.0,
                                start="2024-01-01",
                                end="2024-09-30",
                                filed="2024-11-07",
                                fy=2024,
                                fp="Q3",
                            ),
                            _duration_fact(
                                120_000_000.0,
                                start="2023-01-01",
                                end="2023-12-31",
                                filed="2024-02-20",
                                fy=2023,
                                fp="FY",
                                form="10-K",
                            ),
                            _duration_fact(
                                80_000_000.0,
                                start="2023-01-01",
                                end="2023-09-30",
                                filed="2024-11-07",
                                fy=2024,
                                fp="Q3",
                            ),
                        ]
                    }
                },
                "DepreciationDepletionAndAmortization": {
                    "units": {
                        "USD": [
                            _duration_fact(
                                30_000_000.0,
                                start="2023-01-01",
                                end="2023-12-31",
                                filed="2024-02-20",
                                fy=2023,
                                fp="FY",
                                form="10-K",
                            ),
                        ]
                    }
                },
                "OtherDepreciationAndAmortization": {
                    "units": {
                        "USD": [
                            _duration_fact(
                                25_000_000.0,
                                start="2024-01-01",
                                end="2024-09-30",
                                filed="2024-11-07",
                                fy=2024,
                                fp="Q3",
                            ),
                            _duration_fact(
                                32_000_000.0,
                                start="2023-01-01",
                                end="2023-12-31",
                                filed="2024-02-20",
                                fy=2023,
                                fp="FY",
                                form="10-K",
                            ),
                            _duration_fact(
                                24_000_000.0,
                                start="2023-01-01",
                                end="2023-09-30",
                                filed="2024-11-07",
                                fy=2024,
                                fp="Q3",
                            ),
                        ]
                    }
                },
            }
        }
    }

    value, support_mode, missing_reason, component_breakdown, quality_flags = _build_sec_core_metric(
        "operating.ebitda_ltm_provider_direct",
        companyfacts,
        "2024-12-31",
    )

    assert value == 173_000_000.0
    assert support_mode == "exact"
    assert missing_reason is None
    assert quality_flags is None
    depreciation_meta = component_breakdown["depreciation_amortization"]
    assert depreciation_meta["concept"] == "OtherDepreciationAndAmortization"
    assert depreciation_meta["mode"] == "ytd_plus_prior_fy_minus_prior_ytd"


def test_ebitda_ttm_ignores_zero_valued_stale_latest_fy_amortization_component():
    companyfacts = {
        "facts": {
            "us-gaap": {
                "OperatingIncomeLoss": {
                    "units": {
                        "USD": [
                            _duration_fact(
                                178_773_000.0,
                                start="2023-10-01",
                                end="2024-09-30",
                                filed="2024-11-20",
                                fy=2024,
                                fp="FY",
                                form="10-K",
                            ),
                        ]
                    }
                },
                "Depreciation": {
                    "units": {
                        "USD": [
                            _duration_fact(
                                6_871_000.0,
                                start="2023-10-01",
                                end="2024-09-30",
                                filed="2024-11-20",
                                fy=2024,
                                fp="FY",
                                form="10-K",
                            ),
                        ]
                    }
                },
                "AmortizationOfIntangibleAssets": {
                    "units": {
                        "USD": [
                            _duration_fact(
                                0.0,
                                start="2022-10-01",
                                end="2023-09-30",
                                filed="2023-12-06",
                                fy=2023,
                                fp="FY",
                                form="10-K",
                            ),
                        ]
                    }
                },
            }
        }
    }

    value, support_mode, missing_reason, component_breakdown, quality_flags = _build_sec_core_metric(
        "operating.ebitda_ltm_provider_direct",
        companyfacts,
        "2024-12-31",
    )

    assert value == 185_644_000.0
    assert support_mode == "exact"
    assert missing_reason is None
    assert quality_flags is None
    depreciation_meta = component_breakdown["depreciation_amortization"]
    assert depreciation_meta["mode"] == "sum_concepts"


def test_ebitda_ttm_uses_fresh_finance_lease_amortization_with_fresh_depreciation():
    companyfacts = {
        "facts": {
            "us-gaap": {
                "OperatingIncomeLoss": {
                    "units": {
                        "USD": [
                            _duration_fact(
                                20_000_000.0,
                                start="2024-04-01",
                                end="2024-09-30",
                                filed="2024-11-12",
                                fy=2025,
                                fp="Q2",
                            ),
                            _duration_fact(
                                30_000_000.0,
                                start="2023-04-01",
                                end="2024-03-31",
                                filed="2024-06-11",
                                fy=2024,
                                fp="FY",
                                form="10-K",
                            ),
                            _duration_fact(
                                18_000_000.0,
                                start="2023-04-01",
                                end="2023-09-30",
                                filed="2024-11-12",
                                fy=2025,
                                fp="Q2",
                            ),
                        ]
                    }
                },
                "Depreciation": {
                    "units": {
                        "USD": [
                            _duration_fact(
                                5_330_000.0,
                                start="2024-04-01",
                                end="2024-09-30",
                                filed="2024-11-12",
                                fy=2025,
                                fp="Q2",
                            ),
                            _duration_fact(
                                10_544_000.0,
                                start="2023-04-01",
                                end="2024-03-31",
                                filed="2024-06-11",
                                fy=2024,
                                fp="FY",
                                form="10-K",
                            ),
                            _duration_fact(
                                5_966_000.0,
                                start="2023-04-01",
                                end="2023-09-30",
                                filed="2024-11-12",
                                fy=2025,
                                fp="Q2",
                            ),
                        ]
                    }
                },
                "FinanceLeaseRightOfUseAssetAmortization": {
                    "units": {
                        "USD": [
                            _duration_fact(
                                1_367_000.0,
                                start="2024-04-01",
                                end="2024-09-30",
                                filed="2024-11-12",
                                fy=2025,
                                fp="Q2",
                            ),
                            _duration_fact(
                                2_300_000.0,
                                start="2023-04-01",
                                end="2024-03-31",
                                filed="2024-06-11",
                                fy=2024,
                                fp="FY",
                                form="10-K",
                            ),
                            _duration_fact(
                                1_100_000.0,
                                start="2023-04-01",
                                end="2023-09-30",
                                filed="2024-11-12",
                                fy=2025,
                                fp="Q2",
                            ),
                        ]
                    }
                },
            }
        }
    }

    value, support_mode, missing_reason, component_breakdown, quality_flags = _build_sec_core_metric(
        "operating.ebitda_ltm_provider_direct",
        companyfacts,
        "2024-12-31",
    )

    assert value == 44_475_000.0
    assert support_mode == "exact"
    assert missing_reason is None
    assert quality_flags is None
    depreciation_meta = component_breakdown["depreciation_amortization"]
    assert depreciation_meta["mode"] == "sum_concepts"


def test_ebitda_ttm_uses_capitalized_software_amortization_when_direct_intangible_amortization_is_missing():
    companyfacts = {
        "facts": {
            "us-gaap": {
                "OperatingIncomeLoss": {
                    "units": {
                        "USD": [
                            _duration_fact(
                                20_000_000.0,
                                start="2024-01-01",
                                end="2024-09-30",
                                filed="2024-11-06",
                                fy=2024,
                                fp="Q3",
                            ),
                            _duration_fact(
                                24_000_000.0,
                                start="2023-01-01",
                                end="2023-12-31",
                                filed="2024-02-15",
                                fy=2023,
                                fp="FY",
                                form="10-K",
                            ),
                            _duration_fact(
                                18_000_000.0,
                                start="2023-01-01",
                                end="2023-09-30",
                                filed="2024-11-06",
                                fy=2024,
                                fp="Q3",
                            ),
                        ]
                    }
                },
                "Depreciation": {
                    "units": {
                        "USD": [
                            _duration_fact(
                                2_000_000.0,
                                start="2024-01-01",
                                end="2024-09-30",
                                filed="2024-11-06",
                                fy=2024,
                                fp="Q3",
                            ),
                            _duration_fact(
                                2_900_000.0,
                                start="2023-01-01",
                                end="2023-12-31",
                                filed="2024-02-15",
                                fy=2023,
                                fp="FY",
                                form="10-K",
                            ),
                            _duration_fact(
                                2_200_000.0,
                                start="2023-01-01",
                                end="2023-09-30",
                                filed="2024-11-06",
                                fy=2024,
                                fp="Q3",
                            ),
                        ]
                    }
                },
                "CapitalizedComputerSoftwareAmortization1": {
                    "units": {
                        "USD": [
                            _duration_fact(
                                10_400_000.0,
                                start="2024-01-01",
                                end="2024-09-30",
                                filed="2024-11-06",
                                fy=2024,
                                fp="Q3",
                            ),
                            _duration_fact(
                                11_600_000.0,
                                start="2023-01-01",
                                end="2023-12-31",
                                filed="2024-02-15",
                                fy=2023,
                                fp="FY",
                                form="10-K",
                            ),
                            _duration_fact(
                                9_000_000.0,
                                start="2023-01-01",
                                end="2023-09-30",
                                filed="2024-11-06",
                                fy=2024,
                                fp="Q3",
                            ),
                        ]
                    }
                },
            }
        }
    }

    value, support_mode, missing_reason, component_breakdown, quality_flags = _build_sec_core_metric(
        "operating.ebitda_ltm_provider_direct",
        companyfacts,
        "2024-12-31",
    )

    assert value == 41_700_000.0
    assert support_mode == "exact"
    assert missing_reason is None
    assert quality_flags is None
    depreciation_meta = component_breakdown["depreciation_amortization"]
    assert depreciation_meta["mode"] == "sum_concepts"
