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


def test_apply_market_metric_repairs_overwrites_proxy_with_exact():
    features = {
        "market.price_spot": _feature("market.price_spot", 10.0),
        "market.total_return_1m_standardized": _feature("market.total_return_1m_standardized", 0.05),
    }
    repairs = {
        "market.price_spot": {
            "value": 11.0,
            "support_mode": "exact",
            "missing_reason": None,
            "component_breakdown": {"formula": "latest_crsp_close_on_or_before_asof"},
            "fallback_used": None,
            "quality_flags": None,
            "provenance": [{"source": "/tmp/crsp"}],
        },
        "market.total_return_1m_standardized": {
            "value": 0.06,
            "support_mode": "exact",
            "missing_reason": None,
            "component_breakdown": {"formula": "compound_total_return_from_crsp_daily_window"},
            "fallback_used": None,
            "quality_flags": None,
            "provenance": [{"source": "/tmp/crsp"}],
        },
    }

    _apply_market_metric_repairs(features=features, repaired_metrics=repairs, exact_only=True)

    assert features["market.price_spot"]["value"] == 11.0
    assert features["market.price_spot"]["support_mode"] == "exact"
    assert features["market.price_spot"]["fallback_used"] is None
    assert features["market.price_spot"]["quality_flags"] is None
    assert features["market.total_return_1m_standardized"]["value"] == 0.06
    assert features["market.total_return_1m_standardized"]["support_mode"] == "exact"


def test_apply_market_metric_repairs_demotes_non_exact_when_exact_required():
    features = {
        "market.total_return_12m_standardized": _feature("market.total_return_12m_standardized", 0.12),
    }
    repairs = {
        "market.total_return_12m_standardized": {
            "value": 0.11,
            "support_mode": "proxy_missing_component",
            "missing_reason": "total_return_component_unavailable",
            "component_breakdown": {"formula": "compound_price_return_from_crsp_daily_window"},
            "fallback_used": None,
            "quality_flags": ["price_return_only"],
            "provenance": [{"source": "/tmp/crsp"}],
        }
    }

    _apply_market_metric_repairs(features=features, repaired_metrics=repairs, exact_only=True)

    assert features["market.total_return_12m_standardized"]["value"] is None
    assert features["market.total_return_12m_standardized"]["support_mode"] == "unsupported"
    assert (
        features["market.total_return_12m_standardized"]["missing_reason"]
        == "total_return_component_unavailable"
    )
