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


def test_repair_ebitda_margin_overwrites_stale_period_specific_value_with_ttm_inputs():
    stale_target = _node("operating.ebitda_margin_ttm", 0.23, unit="ratio")
    stale_target["component_breakdown"] = {
        "revenue": 6411.0,
        "ebitda": 1485.0,
        "revenue_period": "2024-03-31 00:00:00+00:00",
        "ebitda_period": "2024-03-31 00:00:00+00:00",
        "period_match_type": "exact_period_match",
    }
    features = {
        "operating.ebitda_margin_ttm": stale_target,
        "operating.revenue_ttm_provider_direct": _node("operating.revenue_ttm_provider_direct", 26_130.0),
        "operating.ebitda_ltm_provider_direct": _node("operating.ebitda_ltm_provider_direct", 3_988.0),
        "operating.operating_earnings_normalized": _node("operating.operating_earnings_normalized", 3_988.0),
    }

    assert repair_ebitda_margin_ttm(features=features, computed_at="2026-03-23T00:00:00+00:00") is True
    repaired = features["operating.ebitda_margin_ttm"]
    assert round(repaired["value"], 12) == round(3_988.0 / 26_130.0, 12)
    assert repaired["fallback_used"] == "provider_direct_revenue_and_ebitda"
    assert repaired["component_breakdown"]["revenue_source_metric"] == "operating.revenue_ttm_provider_direct"


def test_repair_ev_ebitda_uses_repaired_enterprise_value():
    features = {
        "market.enterprise_value": _node("market.enterprise_value", None, support_mode="unsupported"),
        "market.market_cap_provider_direct": _node("market.market_cap_provider_direct", 1000.0),
        "capital_structure.total_debt_provider_direct": _node("capital_structure.total_debt_provider_direct", 250.0),
        "liquidity.cash_and_equivalents_statement_direct": _node(
            "liquidity.cash_and_equivalents_statement_direct",
            125.0,
        ),
        "market.ev_ebitda": _node("market.ev_ebitda", None, unit="x"),
        "operating.ebitda_ltm_provider_direct": _node("operating.ebitda_ltm_provider_direct", 100.0),
        "operating.operating_earnings_normalized": _node("operating.operating_earnings_normalized", None),
    }

    assert repair_enterprise_value(features=features, computed_at="2026-03-23T00:00:00+00:00") is True
    assert repair_ev_ebitda(features=features, computed_at="2026-03-23T00:00:00+00:00") is True
    repaired = features["market.ev_ebitda"]
    assert repaired["value"] == 11.25
    assert repaired["fallback_used"] == "repaired_enterprise_value_plus_provider_ebitda"


def test_repair_ev_ebitda_overwrites_reference_value_with_current_ttm_inputs():
    features = {
        "market.enterprise_value": _node("market.enterprise_value", 1_000.0),
        "market.ev_ebitda": _node("market.ev_ebitda", 7.0, unit="x"),
        "operating.ebitda_ltm_provider_direct": _node("operating.ebitda_ltm_provider_direct", 100.0),
        "operating.operating_earnings_normalized": _node("operating.operating_earnings_normalized", 100.0),
    }
    features["market.ev_ebitda"]["component_breakdown"] = {
        "enterprise_value": 1_000.0,
        "ebitda_ttm": 142.857,
        "reference_instrument": "TEST.OQ",
    }

    assert repair_ev_ebitda(features=features, computed_at="2026-03-23T00:00:00+00:00") is True
    repaired = features["market.ev_ebitda"]
    assert repaired["value"] == 10.0
    assert repaired["fallback_used"] == "repaired_enterprise_value_plus_provider_ebitda"


def test_repair_pe_ratio_from_market_cap_and_net_income():
    features = {
        "market.pe_ratio": _node("market.pe_ratio", None, unit="x"),
        "market.market_cap_provider_direct": _node("market.market_cap_provider_direct", 300.0),
        "earnings.net_income_ttm_provider_direct": _node(
            "earnings.net_income_ttm_provider_direct",
            20.0,
        ),
        "market.price_spot": _node("market.price_spot", 15.0, unit="usd_per_share"),
    }

    assert repair_pe_ratio(features=features, computed_at="2026-03-23T00:00:00+00:00") is True
    repaired = features["market.pe_ratio"]
    assert repaired["value"] == 15.0
    assert repaired["fallback_used"] == "market_cap_plus_net_income_ttm"
    assert repaired["support_mode"] == "exact"


def test_repair_pe_ratio_marks_non_positive_net_income_unsupported():
    features = {
        "market.pe_ratio": _node("market.pe_ratio", None, unit="x"),
        "market.market_cap_provider_direct": _node("market.market_cap_provider_direct", 300.0),
        "earnings.net_income_ttm_provider_direct": _node(
            "earnings.net_income_ttm_provider_direct",
            -5.0,
        ),
    }

    assert repair_pe_ratio(features=features, computed_at="2026-03-23T00:00:00+00:00") is True
    repaired = features["market.pe_ratio"]
    assert repaired["value"] is None
    assert repaired["support_mode"] == "unsupported"
    assert repaired["missing_reason"] == "non_positive_net_income_ttm"
