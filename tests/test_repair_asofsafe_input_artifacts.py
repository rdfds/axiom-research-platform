from scripts.repair_asofsafe_input_artifacts import _apply_market_metric_repairs


def _feature(name: str, value, *, support_mode="proxy_missing_component", missing_reason=None):
    return {
        "name": name,
        "value": value,
        "unit": "ratio",
        "computed_at": "2026-03-29T00:00:00+00:00",
        "as_of_time": "2024-12-31T00:00:00+00:00",
        "window": None,
        "confidence": 0.5 if value is not None else None,
        "provenance": [],
        "missing_reason": missing_reason,
        "fallback_used": "monthly_price_history_proxy",
        "support_mode": support_mode,
        "component_breakdown": {"formula": "monthly_proxy"},
        "quality_flags": ["monthly_price_history_proxy"],
    }


