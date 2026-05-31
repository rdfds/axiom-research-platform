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


