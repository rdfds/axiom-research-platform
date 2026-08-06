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


def test_load_market_cache_prefers_price_proxy(tmp_path: Path):
    path = tmp_path / "market_cache.parquet"
    df = pd.DataFrame(
        {
            "permno": [10001, 10001],
            "trade_date": pd.to_datetime(["2024-12-30", "2024-12-31"]),
            "price_proxy": [10.0, 11.0],
            "close_price": [9.5, 10.5],
        }
    )
    df.to_parquet(path, index=False)
    loaded = _load_market_cache(path)
    frame = loaded["10001"]
    assert frame["price"].iloc[-1] == 11.0


def test_repair_price_history_metrics_uses_total_return_provenance():
    provenance = [
        {
            "artifact_type": "MarketTimeseries",
            "artifact_id": "market_timeseries:test.parquet",
            "source": "/tmp/test.parquet",
            "published_at": "2024-12-31T00:00:00+00:00",
            "ingested_at": "2026-03-23T00:00:00+00:00",
            "hash": None,
        }
    ]
    features = {
        "market.total_return_3m_standardized": _node(
            "market.total_return_3m_standardized",
            0.1,
            support_mode="exact",
            provenance=provenance,
        ),
        "market.total_return_12m_standardized": _node(
            "market.total_return_12m_standardized",
            0.2,
            support_mode="exact",
            provenance=provenance,
        ),
        "market.volatility_30d": _node("market.volatility_30d", None, unit="annualized"),
        "market.volatility_90d": _node("market.volatility_90d", None, unit="annualized"),
        "market.drawdown_90d": _node("market.drawdown_90d", None),
    }
    metrics = {
        "market.volatility_30d": {"value": 0.25, "component_breakdown": {"formula": "stddev(daily_returns_30d) * sqrt(252)"}},
        "market.volatility_90d": {"value": 0.3, "component_breakdown": {"formula": "stddev(daily_returns_90d) * sqrt(252)"}},
        "market.drawdown_90d": {"value": -0.15, "component_breakdown": {"formula": "min(price_window_90d) / max(price_window_90d) - 1"}},
    }

    changed = repair_price_history_metrics(
        features=features,
        price_metrics=metrics,
        permno="12345",
        computed_at="2026-03-23T00:00:00+00:00",
    )

    assert changed is True
    assert features["market.volatility_30d"]["value"] == 0.25
    assert features["market.volatility_30d"]["support_mode"] == "exact"
    assert features["market.volatility_30d"]["fallback_used"] == "crsp_market_cache_price_history"
    assert features["market.volatility_30d"]["provenance"] == provenance
    assert features["market.volatility_30d"]["component_breakdown"]["selected_price_series"]["group_value"] == "12345"


def test_repair_price_history_metrics_overwrites_monthly_proxy_values():
    provenance = [
        {
            "artifact_type": "MarketTimeseries",
            "artifact_id": "market_timeseries:test.parquet",
            "source": "/tmp/test.parquet",
            "published_at": "2024-12-31T00:00:00+00:00",
            "ingested_at": "2026-03-23T00:00:00+00:00",
            "hash": None,
        }
    ]
    features = {
        "market.total_return_3m_standardized": _node(
            "market.total_return_3m_standardized",
            0.1,
            support_mode="exact",
            provenance=provenance,
        ),
        "market.total_return_12m_standardized": _node(
            "market.total_return_12m_standardized",
            0.2,
            support_mode="exact",
            provenance=provenance,
        ),
        "market.volatility_30d": _node(
            "market.volatility_30d",
            0.12,
            support_mode="exact",
            unit="annualized",
        ),
    }
    features["market.volatility_30d"]["fallback_used"] = "monthly_price_history_proxy"
    features["market.volatility_30d"]["quality_flags"] = ["monthly_price_history_proxy"]
    features["market.volatility_30d"]["component_breakdown"] = {
        "formula": "stddev(monthly_returns_6m) * sqrt(12)",
        "source_kind": "monthly_price_proxy",
    }
    metrics = {
        "market.volatility_30d": {
            "value": 0.25,
            "component_breakdown": {"formula": "stddev(daily_returns_30d) * sqrt(252)"},
        },
    }

    changed = repair_price_history_metrics(
        features=features,
        price_metrics=metrics,
        permno="12345",
        computed_at="2026-03-23T00:00:00+00:00",
    )

    assert changed is True
    assert features["market.volatility_30d"]["value"] == 0.25
    assert features["market.volatility_30d"]["support_mode"] == "exact"
    assert features["market.volatility_30d"]["fallback_used"] == "crsp_market_cache_price_history"
    assert features["market.volatility_30d"]["quality_flags"] is None
    assert features["market.volatility_30d"]["component_breakdown"]["selected_price_series"]["group_value"] == "12345"


def test_needs_exact_price_history_repair_keeps_true_exact_daily_series():
    node = _node("market.drawdown_90d", -0.1, support_mode="exact")
    node["component_breakdown"] = {
        "formula": "min(price_window_90d) / max(price_window_90d) - 1",
        "source_kind": "crsp_market_cache",
        "selected_price_series": {"source_kind": "crsp_market_cache"},
    }
    assert _needs_exact_price_history_repair(node) is False
