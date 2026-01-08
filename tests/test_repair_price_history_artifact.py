import json
from pathlib import Path

import pandas as pd

from scripts.repair_price_history_artifact import (
    _needs_exact_price_history_repair,
    _load_market_cache,
    _price_metrics,
    repair_price_history_metrics,
)


def _node(name, value, *, support_mode="unsupported", unit="ratio", provenance=None):
    return {
        "name": name,
        "value": value,
        "unit": unit,
        "computed_at": "2026-03-23T00:00:00+00:00",
        "as_of_time": "2024-12-31T00:00:00+00:00",
        "window": None,
        "confidence": None,
        "provenance": provenance or [],
        "missing_reason": None if value is not None else "unavailable",
        "fallback_used": None,
        "support_mode": support_mode,
        "component_breakdown": {"formula": "placeholder"},
        "quality_flags": None,
    }


def test_price_metrics_computes_volatility_and_drawdown():
    dates = pd.date_range("2024-10-01", periods=70, freq="B", tz="UTC")
    prices = pd.Series(range(100, 170), dtype=float)
    frame = pd.DataFrame({"trade_date": dates, "price": prices})
    metrics = _price_metrics(frame)
    assert "market.volatility_30d" in metrics
    assert "market.volatility_90d" in metrics
    assert "market.drawdown_90d" in metrics
    assert metrics["market.volatility_90d"]["value"] > 0
    assert metrics["market.drawdown_90d"]["value"] <= 0


