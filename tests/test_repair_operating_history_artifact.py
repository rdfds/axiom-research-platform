from datetime import date
from pathlib import Path

from scripts.repair_operating_history_artifact import (
    repair_margin_history_metrics,
    repair_revenue_cagr_3y,
    repair_revenue_yoy_last_q,
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


def _companyfacts_with_revenue_quarters():
    return {
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {
                        "USD": [
                            {
                                "start": "2023-07-01",
                                "end": "2023-09-30",
                                "val": 90.0,
                                "fy": 2023,
                                "fp": "Q3",
                                "form": "10-Q",
                                "filed": "2023-10-30",
                                "frame": "CY2023Q3",
                            },
                            {
                                "start": "2024-07-01",
                                "end": "2024-09-30",
                                "val": 110.0,
                                "fy": 2024,
                                "fp": "Q3",
                                "form": "10-Q",
                                "filed": "2024-10-30",
                                "frame": "CY2024Q3",
                            },
                        ]
                    }
                }
            }
        }
    }


def _companyfacts_with_revenue_fy_series():
    return {
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {
                        "USD": [
                            {
                                "start": "2021-01-01",
                                "end": "2021-12-31",
                                "val": 100.0,
                                "fy": 2021,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2022-02-20",
                                "frame": "CY2021",
                            },
                            {
                                "start": "2022-01-01",
                                "end": "2022-12-31",
                                "val": 110.0,
                                "fy": 2022,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2023-02-20",
                                "frame": "CY2022",
                            },
                            {
                                "start": "2023-01-01",
                                "end": "2023-12-31",
                                "val": 121.0,
                                "fy": 2023,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2024-02-20",
                                "frame": "CY2023",
                            },
                            {
                                "start": "2024-01-01",
                                "end": "2024-12-31",
                                "val": 133.1,
                                "fy": 2024,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2025-02-20",
                                "frame": "CY2024",
                            },
                        ]
                    }
                }
            }
        }
    }


def _companyfacts_with_negative_revenue_anchor():
    return {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {
                                "start": "2022-01-01",
                                "end": "2022-12-31",
                                "val": -50.0,
                                "fy": 2022,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2023-02-20",
                                "frame": "CY2022",
                            },
                            {
                                "start": "2024-01-01",
                                "end": "2024-12-31",
                                "val": 100.0,
                                "fy": 2024,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2025-02-20",
                                "frame": "CY2024",
                            },
                        ]
                    }
                }
            }
        }
    }


def _companyfacts_with_stale_legacy_and_fresh_current_revenue_concepts():
    return {
        "facts": {
            "us-gaap": {
                "SalesRevenueNet": {
                    "units": {
                        "USD": [
                            {
                                "start": "2017-07-01",
                                "end": "2017-09-30",
                                "val": 100.0,
                                "fy": 2017,
                                "fp": "Q3",
                                "form": "10-Q",
                                "filed": "2017-10-30",
                                "frame": "CY2017Q3",
                            },
                            {
                                "start": "2018-07-01",
                                "end": "2018-09-30",
                                "val": 125.0,
                                "fy": 2018,
                                "fp": "Q3",
                                "form": "10-Q",
                                "filed": "2018-10-30",
                                "frame": "CY2018Q3",
                            },
                            {
                                "start": "2015-01-01",
                                "end": "2015-12-31",
                                "val": 400.0,
                                "fy": 2015,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2016-02-20",
                                "frame": "CY2015",
                            },
                            {
                                "start": "2018-01-01",
                                "end": "2018-12-31",
                                "val": 500.0,
                                "fy": 2018,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2019-02-20",
                                "frame": "CY2018",
                            },
                        ]
                    }
                },
                "Revenues": {
                    "units": {
                        "USD": [
                            {
                                "start": "2023-07-01",
                                "end": "2023-09-30",
                                "val": 200.0,
                                "fy": 2023,
                                "fp": "Q3",
                                "form": "10-Q",
                                "filed": "2023-10-30",
                                "frame": "CY2023Q3",
                            },
                            {
                                "start": "2024-07-01",
                                "end": "2024-09-30",
                                "val": 220.0,
                                "fy": 2024,
                                "fp": "Q3",
                                "form": "10-Q",
                                "filed": "2024-10-30",
                                "frame": "CY2024Q3",
                            },
                            {
                                "start": "2021-01-01",
                                "end": "2021-12-31",
                                "val": 800.0,
                                "fy": 2021,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2022-02-20",
                                "frame": "CY2021",
                            },
                            {
                                "start": "2022-01-01",
                                "end": "2022-12-31",
                                "val": 880.0,
                                "fy": 2022,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2023-02-20",
                                "frame": "CY2022",
                            },
                            {
                                "start": "2023-01-01",
                                "end": "2023-12-31",
                                "val": 968.0,
                                "fy": 2023,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2024-02-20",
                                "frame": "CY2023",
                            },
                            {
                                "start": "2024-01-01",
                                "end": "2024-12-31",
                                "val": 1064.8,
                                "fy": 2024,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2025-02-20",
                                "frame": "CY2024",
                            },
                        ]
                    }
                },
            }
        }
    }


def test_repair_revenue_yoy_from_sec_companyfacts_quarter_history():
    features = {
        "operating.revenue_yoy_last_q": _node("operating.revenue_yoy_last_q", None),
    }

    repaired = repair_revenue_yoy_last_q(
        features=features,
        companyfacts=_companyfacts_with_revenue_quarters(),
        companyfacts_path=Path("/tmp/CIK0000000001.json"),
        computed_at="2026-03-23T00:00:00+00:00",
        as_of_time="2024-12-31T00:00:00+00:00",
    )

    assert repaired is True
    node = features["operating.revenue_yoy_last_q"]
    assert round(node["value"], 6) == round((110.0 - 90.0) / 90.0, 6)
    assert node["support_mode"] == "exact"
    assert node["fallback_used"] == "sec_companyfacts_quarterly_revenue_history"
    assert node["component_breakdown"]["source_concept"] == "RevenueFromContractWithCustomerExcludingAssessedTax"


def test_repair_revenue_yoy_prefers_fresh_revenue_concept_over_stale_legacy_series():
    features = {
        "operating.revenue_yoy_last_q": _node("operating.revenue_yoy_last_q", None),
    }

    repaired = repair_revenue_yoy_last_q(
        features=features,
        companyfacts=_companyfacts_with_stale_legacy_and_fresh_current_revenue_concepts(),
        companyfacts_path=Path("/tmp/CIK0000000004.json"),
        computed_at="2026-03-23T00:00:00+00:00",
        as_of_time="2024-12-31T00:00:00+00:00",
    )

    assert repaired is True
    node = features["operating.revenue_yoy_last_q"]
    assert round(node["value"], 6) == round((220.0 - 200.0) / 200.0, 6)
    assert node["component_breakdown"]["source_concept"] == "Revenues"
    assert node["component_breakdown"]["latest_period"] == "2024-09-30 00:00:00+00:00"
    assert node["component_breakdown"]["prior_period"] == "2023-09-30 00:00:00+00:00"


def test_repair_revenue_yoy_refreshes_stale_existing_companyfacts_value():
    features = {
        "operating.revenue_yoy_last_q": _node("operating.revenue_yoy_last_q", 0.25, support_mode="exact"),
    }
    features["operating.revenue_yoy_last_q"]["fallback_used"] = "sec_companyfacts_quarterly_revenue_history"
    features["operating.revenue_yoy_last_q"]["component_breakdown"] = {
        "latest_period": "2018-09-30 00:00:00+00:00",
        "prior_period": "2017-09-30 00:00:00+00:00",
        "source_concept": "SalesRevenueNet",
    }

    repaired = repair_revenue_yoy_last_q(
        features=features,
        companyfacts=_companyfacts_with_stale_legacy_and_fresh_current_revenue_concepts(),
        companyfacts_path=Path("/tmp/CIK0000000007.json"),
        computed_at="2026-03-23T00:00:00+00:00",
        as_of_time="2024-12-31T00:00:00+00:00",
    )

    assert repaired is True
    node = features["operating.revenue_yoy_last_q"]
    assert round(node["value"], 6) == round((220.0 - 200.0) / 200.0, 6)
    assert node["component_breakdown"]["source_concept"] == "Revenues"
    assert node["component_breakdown"]["latest_period"] == "2024-09-30 00:00:00+00:00"


def test_repair_revenue_cagr_from_sec_companyfacts_ttm_history():
    features = {
        "operating.revenue_cagr_3y": _node("operating.revenue_cagr_3y", None),
    }

    repaired = repair_revenue_cagr_3y(
        features=features,
        companyfacts=_companyfacts_with_revenue_fy_series(),
        companyfacts_path=Path("/tmp/CIK0000000002.json"),
        computed_at="2026-03-23T00:00:00+00:00",
        as_of_time="2025-12-31T00:00:00+00:00",
    )

    assert repaired is True
    node = features["operating.revenue_cagr_3y"]
    assert round(node["value"], 4) == round((121.0 / 100.0) ** (1.0 / 2.001368925393566) - 1.0, 4)
    assert node["support_mode"] == "exact"
    assert node["fallback_used"] == "sec_companyfacts_ttm_revenue_history"


def test_repair_revenue_cagr_prefers_fresh_revenue_concept_over_stale_legacy_series():
    features = {
        "operating.revenue_cagr_3y": _node("operating.revenue_cagr_3y", None),
    }

    repaired = repair_revenue_cagr_3y(
        features=features,
        companyfacts=_companyfacts_with_stale_legacy_and_fresh_current_revenue_concepts(),
        companyfacts_path=Path("/tmp/CIK0000000005.json"),
        computed_at="2026-03-23T00:00:00+00:00",
        as_of_time="2025-12-31T00:00:00+00:00",
    )

    assert repaired is True
    node = features["operating.revenue_cagr_3y"]
    assert round(node["value"], 6) == round((988.0 / 800.0) ** (1.0 / 2.001368925393566) - 1.0, 6)
    assert node["component_breakdown"]["source_concept"] == "Revenues"
    assert node["component_breakdown"]["latest_period"] == "2024-12-31 00:00:00+00:00"
    assert node["component_breakdown"]["prior_period"] == "2022-12-31 00:00:00+00:00"


def test_repair_revenue_cagr_refreshes_stale_existing_companyfacts_value():
    features = {
        "operating.revenue_cagr_3y": _node("operating.revenue_cagr_3y", -0.04, support_mode="exact"),
    }
    features["operating.revenue_cagr_3y"]["fallback_used"] = "sec_companyfacts_ttm_revenue_history"
    features["operating.revenue_cagr_3y"]["component_breakdown"] = {
        "latest_period": "2018-12-31 00:00:00+00:00",
        "prior_period": "2015-12-31 00:00:00+00:00",
        "source_concept": "SalesRevenueNet",
    }

    repaired = repair_revenue_cagr_3y(
        features=features,
        companyfacts=_companyfacts_with_stale_legacy_and_fresh_current_revenue_concepts(),
        companyfacts_path=Path("/tmp/CIK0000000008.json"),
        computed_at="2026-03-23T00:00:00+00:00",
        as_of_time="2025-12-31T00:00:00+00:00",
    )

    assert repaired is True
    node = features["operating.revenue_cagr_3y"]
    assert node["component_breakdown"]["source_concept"] == "Revenues"
    assert node["component_breakdown"]["latest_period"] == "2024-12-31 00:00:00+00:00"


def test_repair_revenue_cagr_skips_non_positive_revenue_anchors():
    features = {
        "operating.revenue_cagr_3y": _node("operating.revenue_cagr_3y", None),
    }

    repaired = repair_revenue_cagr_3y(
        features=features,
        companyfacts=_companyfacts_with_negative_revenue_anchor(),
        companyfacts_path=Path("/tmp/CIK0000000006.json"),
        computed_at="2026-03-23T00:00:00+00:00",
        as_of_time="2025-12-31T00:00:00+00:00",
    )

    assert repaired is False
    assert features["operating.revenue_cagr_3y"]["value"] is None


def test_repair_margin_history_metrics_from_ttm_margin_series(monkeypatch):
    features = {
        "operating.ebitda_margin_trend_8q": _node("operating.ebitda_margin_trend_8q", None, unit="slope"),
        "operating.margin_volatility_8q": _node("operating.margin_volatility_8q", None, unit="stddev"),
    }

    monkeypatch.setattr(
        "scripts.repair_operating_history_artifact._build_ttm_margin_series",
        lambda companyfacts, as_of_date: (
            [
                {"period_end": date(2023, 3, 31), "margin": 0.10, "exact": True},
                {"period_end": date(2023, 6, 30), "margin": 0.11, "exact": True},
                {"period_end": date(2023, 9, 30), "margin": 0.12, "exact": True},
                {"period_end": date(2023, 12, 31), "margin": 0.13, "exact": True},
            ],
            "RevenueFromContractWithCustomerExcludingAssessedTax",
        ),
    )

    repaired = repair_margin_history_metrics(
        features=features,
        companyfacts={},
        companyfacts_path=Path("/tmp/CIK0000000003.json"),
        computed_at="2026-03-23T00:00:00+00:00",
        as_of_time="2024-12-31T00:00:00+00:00",
    )

    assert repaired is True
    trend = features["operating.ebitda_margin_trend_8q"]
    vol = features["operating.margin_volatility_8q"]
    assert trend["value"] > 0
    assert vol["value"] > 0
    assert trend["support_mode"] == "exact"
    assert vol["fallback_used"] == "sec_companyfacts_ttm_margin_history"


def test_repair_margin_history_metrics_refreshes_stale_existing_companyfacts_values(monkeypatch):
    features = {
        "operating.ebitda_margin_trend_8q": _node("operating.ebitda_margin_trend_8q", 0.01, support_mode="exact", unit="slope"),
        "operating.margin_volatility_8q": _node("operating.margin_volatility_8q", 0.03, support_mode="exact", unit="stddev"),
    }
    features["operating.ebitda_margin_trend_8q"]["fallback_used"] = "sec_companyfacts_ttm_margin_history"
    features["operating.margin_volatility_8q"]["fallback_used"] = "sec_companyfacts_ttm_margin_history"
    features["operating.ebitda_margin_trend_8q"]["component_breakdown"] = {"window_end": "2018-06-30 00:00:00+00:00"}
    features["operating.margin_volatility_8q"]["component_breakdown"] = {"window_end": "2018-06-30 00:00:00+00:00"}

    monkeypatch.setattr(
        "scripts.repair_operating_history_artifact._build_ttm_margin_series",
        lambda companyfacts, as_of_date: (
            [
                {"period_end": date(2023, 3, 31), "margin": 0.10, "exact": True},
                {"period_end": date(2023, 6, 30), "margin": 0.11, "exact": True},
                {"period_end": date(2023, 9, 30), "margin": 0.12, "exact": True},
                {"period_end": date(2023, 12, 31), "margin": 0.13, "exact": True},
            ],
            "Revenues",
        ),
    )

    repaired = repair_margin_history_metrics(
        features=features,
        companyfacts={},
        companyfacts_path=Path("/tmp/CIK0000000009.json"),
        computed_at="2026-03-23T00:00:00+00:00",
        as_of_time="2024-12-31T00:00:00+00:00",
    )

    assert repaired is True
    assert features["operating.ebitda_margin_trend_8q"]["component_breakdown"]["window_end"] == "2023-12-31 00:00:00+00:00"
    assert features["operating.margin_volatility_8q"]["component_breakdown"]["window_end"] == "2023-12-31 00:00:00+00:00"
