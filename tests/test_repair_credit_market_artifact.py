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


def test_repair_credit_spread_and_credit_window_from_macro_and_company_risk():
    features = {
        "macro.us_ig_oas": _node("macro.us_ig_oas", 0.80, support_mode="exact", unit="spread"),
        "macro.hy_oas": _node("macro.hy_oas", 3.20, support_mode="exact", unit="spread"),
        "market.volatility_30d": _node("market.volatility_30d", 0.30, support_mode="exact"),
        "market.drawdown_90d": _node("market.drawdown_90d", -0.25, support_mode="exact"),
        "capital_structure.gross_leverage_normalized": _node(
            "capital_structure.gross_leverage_normalized",
            4.0,
            support_mode="exact",
            unit="x",
        ),
        "liquidity.available_liquidity_normalized": _node(
            "liquidity.available_liquidity_normalized",
            200.0,
            support_mode="exact",
            unit="usd",
        ),
        "capital_structure.debt_like_obligations_normalized": _node(
            "capital_structure.debt_like_obligations_normalized",
            400.0,
            support_mode="exact",
            unit="usd",
        ),
        "operating.fcf_conversion": _node("operating.fcf_conversion", 0.20, support_mode="exact"),
        "market.credit_spread_level": _node("market.credit_spread_level", None, unit="spread"),
        "market.credit_spread_percentile_2y": _node("market.credit_spread_percentile_2y", None, unit="percentile"),
        "market.credit_window_proxy": _node("market.credit_window_proxy", None),
    }
    ig_history = _spread_history("2022-01-31", [0.70 + (i * 0.02) for i in range(36)])
    hy_history = _spread_history("2022-01-31", [3.00 + (i * 0.03) for i in range(36)])

    repaired_spread = repair_credit_spread_level(features=features, computed_at="2026-03-23T00:00:00+00:00")
    repaired_pct = repair_credit_spread_percentile_2y(
        features=features,
        ig_history=ig_history,
        hy_history=hy_history,
        as_of=pd.Timestamp("2024-12-31", tz="UTC"),
        computed_at="2026-03-23T00:00:00+00:00",
    )
    repaired_window = repair_credit_window_proxy(features=features, computed_at="2026-03-23T00:00:00+00:00")

    assert repaired_spread is True
    spread_node = features["market.credit_spread_level"]
    assert 0.008 < spread_node["value"] < 0.032
    assert spread_node["support_mode"] == "proxy_missing_component"
    assert spread_node["fallback_used"] == "macro_oas_plus_company_risk_heuristic"

    assert repaired_pct is True
    pct_node = features["market.credit_spread_percentile_2y"]
    assert 50.0 < pct_node["value"] <= 100.0
    assert pct_node["support_mode"] == "proxy_missing_component"

    assert repaired_window is True
    window_node = features["market.credit_window_proxy"]
    assert 0.0 <= window_node["value"] <= 1.0
    assert window_node["fallback_used"] == "credit_spread_plus_price_volatility_heuristic"
