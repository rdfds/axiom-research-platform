import pandas as pd

from scripts.repair_maturity_artifact import (
    _load_private_debt_schedule,
    repair_debt_due_0_12m,
    repair_debt_due_12_24m,
    repair_maturity_and_refi_metrics,
)


def _node(name, value, *, support_mode="unsupported", unit="usd"):
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


def test_repair_due_0_12_from_current_debt_exact():
    features = {
        "capital_structure.current_debt_statement_direct": _node(
            "capital_structure.current_debt_statement_direct",
            120.0,
            support_mode="exact",
        ),
        "capital_structure.debt_due_0_12m": _node("capital_structure.debt_due_0_12m", None),
    }

    repaired = repair_debt_due_0_12m(
        features=features,
        schedule_entry=None,
        computed_at="2026-03-23T00:00:00+00:00",
    )

    assert repaired is True
    node = features["capital_structure.debt_due_0_12m"]
    assert node["value"] == 120.0
    assert node["support_mode"] == "exact"
    assert node["fallback_used"] == "current_debt_statement_direct_as_due_0_12m"


