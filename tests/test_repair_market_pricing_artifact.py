import copy

from scripts.repair_market_pricing_artifact import (
    repair_ebitda_margin_ttm,
    repair_enterprise_value,
    repair_ev_ebitda,
    repair_pe_ratio,
)


def _node(name, value, *, support_mode="exact", unit="usd"):
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


def test_repair_enterprise_value_from_input_layer_components():
    features = {
        "market.enterprise_value": _node("market.enterprise_value", None),
        "market.market_cap_provider_direct": _node("market.market_cap_provider_direct", 1000.0),
        "capital_structure.total_debt_provider_direct": _node("capital_structure.total_debt_provider_direct", 250.0),
        "liquidity.cash_and_short_term_investments_provider_direct": _node(
            "liquidity.cash_and_short_term_investments_provider_direct",
            125.0,
        ),
    }

    assert repair_enterprise_value(features=features, computed_at="2026-03-23T00:00:00+00:00") is True
    repaired = features["market.enterprise_value"]
    assert repaired["value"] == 1125.0
    assert repaired["fallback_used"] == "input_layer_market_cap_plus_total_debt_minus_cash"
    assert repaired["support_mode"] == "exact"


def test_repair_enterprise_value_overwrites_stale_existing_value():
    features = {
        "market.enterprise_value": _node("market.enterprise_value", 900.0),
        "market.market_cap_provider_direct": _node("market.market_cap_provider_direct", 1000.0),
        "capital_structure.total_debt_provider_direct": _node("capital_structure.total_debt_provider_direct", 250.0),
        "liquidity.cash_and_short_term_investments_provider_direct": _node(
            "liquidity.cash_and_short_term_investments_provider_direct",
            125.0,
        ),
    }

    assert repair_enterprise_value(features=features, computed_at="2026-03-23T00:00:00+00:00") is True
    repaired = features["market.enterprise_value"]
    assert repaired["value"] == 1125.0
    assert repaired["fallback_used"] == "input_layer_market_cap_plus_total_debt_minus_cash"


def test_repair_ebitda_margin_uses_normalized_operating_earnings_when_provider_ebitda_missing():
    features = {
        "operating.ebitda_margin_ttm": _node("operating.ebitda_margin_ttm", None, unit="ratio"),
        "operating.revenue_ttm_provider_direct": _node("operating.revenue_ttm_provider_direct", 200.0),
        "operating.ebitda_ltm_provider_direct": _node(
            "operating.ebitda_ltm_provider_direct",
            None,
            support_mode="unsupported",
        ),
        "operating.operating_earnings_normalized": _node(
            "operating.operating_earnings_normalized",
            50.0,
            unit="usd",
        ),
    }

    assert repair_ebitda_margin_ttm(features=features, computed_at="2026-03-23T00:00:00+00:00") is True
    repaired = features["operating.ebitda_margin_ttm"]
    assert repaired["value"] == 0.25
    assert repaired["fallback_used"] == "provider_revenue_plus_normalized_operating_earnings"
    assert repaired["support_mode"] == "exact"


