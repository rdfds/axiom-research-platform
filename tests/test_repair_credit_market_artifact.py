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


def test_repair_macro_us_ig_oas_aliases_existing_macro_ig_oas():
    features = {
        "macro.ig_oas": _node("macro.ig_oas", 0.82, support_mode="exact", unit="spread"),
        "macro.us_ig_oas": _node("macro.us_ig_oas", None, unit="spread"),
    }

    repaired = repair_macro_us_ig_oas(features=features, computed_at="2026-03-23T00:00:00+00:00")

    assert repaired is True
    node = features["macro.us_ig_oas"]
    assert node["value"] == 0.82
    assert node["support_mode"] == "exact"
    assert node["fallback_used"] == "macro_ig_oas_alias"


def test_repair_macro_us_ig_oas_percentile_history_from_monthly_history():
    features = {
        "macro.us_ig_oas_percentile_history": _node("macro.us_ig_oas_percentile_history", None, unit="percentile"),
        "macro.us_ig_oas": _node("macro.us_ig_oas", 1.10, support_mode="exact", unit="spread"),
    }
    history = _spread_history("2015-01-31", [0.90 + (i * 0.01) for i in range(120)])

    repaired = repair_macro_us_ig_oas_percentile_history(
        features=features,
        ig_history=history,
        as_of=pd.Timestamp("2024-12-31", tz="UTC"),
        computed_at="2026-03-23T00:00:00+00:00",
    )

    assert repaired is True
    node = features["macro.us_ig_oas_percentile_history"]
    assert node["value"] > 90.0
    assert node["support_mode"] == "exact"


