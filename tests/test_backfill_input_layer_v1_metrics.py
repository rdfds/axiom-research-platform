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


