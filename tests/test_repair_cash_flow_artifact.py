from pathlib import Path

from scripts.repair_cash_flow_artifact import (
    repair_market_fcf_yield,
    repair_operating_fcf_conversion,
)


def _node(name, value, *, support_mode="unsupported", unit="ratio"):
    return {
        "name": name,
        "value": value,
        "unit": unit,
        "computed_at": "2026-03-23T00:00:00+00:00",
        "as_of_time": "2024-12-31T00:00:00+00:00",
        "window": None,
        "confidence": None,
        "provenance": [],
        "missing_reason": None if value is not None else "unavailable",
        "fallback_used": None,
        "support_mode": support_mode,
        "component_breakdown": None,
        "quality_flags": None,
    }


def _companyfacts_with_cash_flow(*, operating_cash_flow=300.0, capex=100.0):
    return {
        "facts": {
            "us-gaap": {
                "NetCashProvidedByUsedInOperatingActivities": {
                    "units": {
                        "USD": [
                            {
                                "start": "2024-01-01",
                                "end": "2024-12-31",
                                "val": operating_cash_flow,
                                "fy": 2024,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2025-02-20",
                                "frame": "CY2024",
                            }
                        ]
                    }
                },
                "PaymentsToAcquirePropertyPlantAndEquipment": {
                    "units": {
                        "USD": [
                            {
                                "start": "2024-01-01",
                                "end": "2024-12-31",
                                "val": capex,
                                "fy": 2024,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2025-02-20",
                                "frame": "CY2024",
                            }
                        ]
                    }
                },
            }
        }
    }


def test_repair_market_fcf_yield_from_companyfacts_cash_flow():
    features = {
        "market.fcf_yield": _node("market.fcf_yield", None),
        "market.market_cap_provider_direct": _node(
            "market.market_cap_provider_direct",
            1000.0,
            support_mode="exact",
            unit="usd",
        ),
    }

    repaired = repair_market_fcf_yield(
        features=features,
        companyfacts=_companyfacts_with_cash_flow(operating_cash_flow=300.0, capex=100.0),
        companyfacts_path=Path("/tmp/CIK0000000004.json"),
        computed_at="2026-03-23T00:00:00+00:00",
        as_of_time="2025-12-31T00:00:00+00:00",
    )

    assert repaired is True
    node = features["market.fcf_yield"]
    assert node["value"] == 0.2
    assert node["support_mode"] == "exact"
    assert node["fallback_used"] == "sec_companyfacts_free_cash_flow_ttm"


def test_repair_operating_fcf_conversion_uses_normalized_operating_earnings():
    features = {
        "operating.fcf_conversion": _node("operating.fcf_conversion", None),
        "operating.ebitda_ltm_provider_direct": _node(
            "operating.ebitda_ltm_provider_direct",
            None,
            support_mode="unsupported",
            unit="usd",
        ),
        "operating.operating_earnings_normalized": _node(
            "operating.operating_earnings_normalized",
            90.0,
            support_mode="exact",
            unit="usd",
        ),
    }

    repaired = repair_operating_fcf_conversion(
        features=features,
        companyfacts=_companyfacts_with_cash_flow(operating_cash_flow=240.0, capex=60.0),
        companyfacts_path=Path("/tmp/CIK0000000005.json"),
        computed_at="2026-03-23T00:00:00+00:00",
        as_of_time="2025-12-31T00:00:00+00:00",
    )

    assert repaired is True
    node = features["operating.fcf_conversion"]
    assert node["value"] == 2.0
    assert node["support_mode"] == "exact"
    assert node["fallback_used"] == "sec_companyfacts_fcf_plus_normalized_operating_earnings"


def test_repair_cash_flow_normalizes_negative_capex_sign():
    features = {
        "market.fcf_yield": _node("market.fcf_yield", None),
        "market.market_cap_provider_direct": _node(
            "market.market_cap_provider_direct",
            1000.0,
            support_mode="exact",
            unit="usd",
        ),
    }

    repaired = repair_market_fcf_yield(
        features=features,
        companyfacts=_companyfacts_with_cash_flow(operating_cash_flow=300.0, capex=-100.0),
        companyfacts_path=Path("/tmp/CIK0000000006.json"),
        computed_at="2026-03-23T00:00:00+00:00",
        as_of_time="2025-12-31T00:00:00+00:00",
    )

    assert repaired is True
    node = features["market.fcf_yield"]
    assert node["value"] == 0.2
    assert node["component_breakdown"]["capex_ttm"] == 100.0
