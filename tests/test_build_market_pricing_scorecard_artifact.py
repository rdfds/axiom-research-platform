from scripts.build_market_pricing_scorecard_artifact import (
    _collect_cross_section,
    _overall_score_node,
    _score_from_components,
    _valuation_gap_node,
)


def _node(name, value, *, support_mode="exact", unit="ratio"):
    return {
        "name": name,
        "value": value,
        "unit": unit,
        "computed_at": "2026-03-23T00:00:00+00:00",
        "as_of_time": "2024-12-31T00:00:00+00:00",
        "window": None,
        "confidence": None,
        "provenance": [],
        "missing_reason": None if value is not None else "unsupported",
        "fallback_used": None,
        "support_mode": support_mode,
        "component_breakdown": None,
        "quality_flags": None,
    }


def _row(company_id, features):
    return {
        "company_id": company_id,
        "as_of_time": "2024-12-31T00:00:00+00:00",
        "features": features,
    }


def test_score_from_components_ranks_value_and_quality():
    rows = [
        _row(
            "0001",
            {
                "market.ev_ebitda": _node("market.ev_ebitda", 8.0),
                "market.fcf_yield": _node("market.fcf_yield", 0.08),
                "operating.ebitda_margin_ttm": _node("operating.ebitda_margin_ttm", 0.25),
                "operating.revenue_yoy_last_q": _node("operating.revenue_yoy_last_q", 0.12),
                "operating.revenue_cagr_3y": _node("operating.revenue_cagr_3y", 0.09),
                "operating.ebitda_margin_trend_8q": _node("operating.ebitda_margin_trend_8q", 0.01),
                "operating.margin_volatility_8q": _node("operating.margin_volatility_8q", 0.01),
                "operating.fcf_conversion": _node("operating.fcf_conversion", 0.60),
            },
        ),
        _row(
            "0002",
            {
                "market.ev_ebitda": _node("market.ev_ebitda", 14.0),
                "market.fcf_yield": _node("market.fcf_yield", 0.03),
                "operating.ebitda_margin_ttm": _node("operating.ebitda_margin_ttm", 0.11),
                "operating.revenue_yoy_last_q": _node("operating.revenue_yoy_last_q", 0.01),
                "operating.revenue_cagr_3y": _node("operating.revenue_cagr_3y", 0.02),
                "operating.ebitda_margin_trend_8q": _node("operating.ebitda_margin_trend_8q", -0.01),
                "operating.margin_volatility_8q": _node("operating.margin_volatility_8q", 0.05),
                "operating.fcf_conversion": _node("operating.fcf_conversion", 0.20),
            },
        ),
    ]
    percentile_maps = _collect_cross_section(rows)

    value_top = _score_from_components(
        "market.value_score",
        row=rows[0],
        percentile_maps=percentile_maps,
        computed_at="2026-03-23T00:00:00+00:00",
    )
    value_bottom = _score_from_components(
        "market.value_score",
        row=rows[1],
        percentile_maps=percentile_maps,
        computed_at="2026-03-23T00:00:00+00:00",
    )
    quality_top = _score_from_components(
        "market.quality_score",
        row=rows[0],
        percentile_maps=percentile_maps,
        computed_at="2026-03-23T00:00:00+00:00",
    )

    assert value_top["value"] > value_bottom["value"]
    assert value_top["support_mode"] == "exact"
    assert quality_top["value"] > 50.0


