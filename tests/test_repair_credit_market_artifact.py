import pandas as pd

from scripts.repair_credit_market_artifact import (
    repair_credit_spread_level,
    repair_credit_spread_percentile_2y,
    repair_credit_window_proxy,
    repair_macro_us_ig_oas,
    repair_macro_us_ig_oas_percentile_history,
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


def _spread_history(start, values):
    dates = pd.date_range(start=start, periods=len(values), freq="ME", tz="UTC")
    return pd.DataFrame({"time": dates, "value": values})


