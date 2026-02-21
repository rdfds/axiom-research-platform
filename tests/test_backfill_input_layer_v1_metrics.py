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


