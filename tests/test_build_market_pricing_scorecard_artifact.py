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


