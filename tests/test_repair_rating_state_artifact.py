import pandas as pd

from scripts.repair_rating_state_artifact import (
    build_rating_index,
    repair_rating_state,
)


def _node(name, value, *, support_mode="unsupported", unit="rating"):
    return {
        "name": name,
        "value": value,
        "unit": unit,
        "computed_at": "2026-03-23T00:00:00+00:00",
        "as_of_time": "2024-12-31T00:00:00+00:00",
        "window": None,
        "confidence": None,
        "provenance": [],
        "missing_reason": None if value is not None else "not_disclosed",
        "fallback_used": None,
        "support_mode": support_mode,
        "component_breakdown": None,
        "quality_flags": None,
    }


