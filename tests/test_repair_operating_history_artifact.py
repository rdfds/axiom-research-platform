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


