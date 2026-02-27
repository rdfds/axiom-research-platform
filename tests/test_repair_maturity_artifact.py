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


def test_repair_due_12_24_and_maturity_wall_from_private_schedule():
    features = {
        "capital_structure.debt_due_0_12m": _node("capital_structure.debt_due_0_12m", None),
        "capital_structure.debt_due_12_24m": _node("capital_structure.debt_due_12_24m", None),
        "capital_structure.total_debt_provider_direct": _node(
            "capital_structure.total_debt_provider_direct",
            400.0,
            support_mode="exact",
        ),
        "capital_structure.debt_like_obligations_normalized": _node(
            "capital_structure.debt_like_obligations_normalized",
            500.0,
            support_mode="exact",
        ),
        "capital_structure.maturity_wall_ratio_24m_reported": _node(
            "capital_structure.maturity_wall_ratio_24m_reported",
            None,
            unit="ratio",
        ),
        "capital_structure.maturity_wall_ratio_24m_market": _node(
            "capital_structure.maturity_wall_ratio_24m_market",
            None,
            unit="ratio",
        ),
        "capital_structure.maturity_wall_ratio_24m": _node(
            "capital_structure.maturity_wall_ratio_24m",
            None,
            unit="ratio",
        ),
        "capital_structure.refi_pressure_flag_reported": _node(
            "capital_structure.refi_pressure_flag_reported",
            None,
            unit="bool",
        ),
        "capital_structure.refi_pressure_flag_market": _node(
            "capital_structure.refi_pressure_flag_market",
            None,
            unit="bool",
        ),
        "capital_structure.refi_pressure_flag": _node(
            "capital_structure.refi_pressure_flag",
            None,
            unit="bool",
        ),
    }
    schedule_entry = {"due_0_12": 120.0, "due_12_24": 80.0}

    assert repair_debt_due_0_12m(
        features=features,
        schedule_entry=schedule_entry,
        computed_at="2026-03-23T00:00:00+00:00",
    )
    assert repair_debt_due_12_24m(
        features=features,
        schedule_entry=schedule_entry,
        computed_at="2026-03-23T00:00:00+00:00",
    )

    repair_count = repair_maturity_and_refi_metrics(
        features=features,
        computed_at="2026-03-23T00:00:00+00:00",
    )

    assert repair_count == 6
    assert features["capital_structure.maturity_wall_ratio_24m_reported"]["value"] == 0.5
    assert features["capital_structure.maturity_wall_ratio_24m_reported"]["support_mode"] == "exact"
    assert features["capital_structure.maturity_wall_ratio_24m_market"]["value"] == 0.4
    assert features["capital_structure.maturity_wall_ratio_24m_market"]["support_mode"] == "exact"
    assert features["capital_structure.maturity_wall_ratio_24m"]["value"] == 0.4
    assert features["capital_structure.refi_pressure_flag"]["value"] == 1.0


def test_repair_maturity_wall_lower_bound_from_current_debt_only():
    features = {
        "capital_structure.current_debt_statement_direct": _node(
            "capital_structure.current_debt_statement_direct",
            90.0,
            support_mode="exact",
        ),
        "capital_structure.debt_due_0_12m": _node("capital_structure.debt_due_0_12m", None),
        "capital_structure.debt_due_12_24m": _node("capital_structure.debt_due_12_24m", None),
        "capital_structure.total_debt_provider_direct": _node(
            "capital_structure.total_debt_provider_direct",
            300.0,
            support_mode="exact",
        ),
        "capital_structure.debt_like_obligations_normalized": _node(
            "capital_structure.debt_like_obligations_normalized",
            360.0,
            support_mode="exact",
        ),
        "capital_structure.maturity_wall_ratio_24m_reported": _node(
            "capital_structure.maturity_wall_ratio_24m_reported",
            None,
            unit="ratio",
        ),
        "capital_structure.maturity_wall_ratio_24m_market": _node(
            "capital_structure.maturity_wall_ratio_24m_market",
            None,
            unit="ratio",
        ),
        "capital_structure.maturity_wall_ratio_24m": _node(
            "capital_structure.maturity_wall_ratio_24m",
            None,
            unit="ratio",
        ),
    }

    assert repair_debt_due_0_12m(
        features=features,
        schedule_entry=None,
        computed_at="2026-03-23T00:00:00+00:00",
    )
    repair_count = repair_maturity_and_refi_metrics(
        features=features,
        computed_at="2026-03-23T00:00:00+00:00",
    )

    assert repair_count == 3
    reported = features["capital_structure.maturity_wall_ratio_24m_reported"]
    market = features["capital_structure.maturity_wall_ratio_24m_market"]
    decision = features["capital_structure.maturity_wall_ratio_24m"]
    assert reported["value"] == 0.3
    assert market["value"] == 0.25
    assert reported["support_mode"] == "proxy_missing_component"
    assert decision["support_mode"] == "proxy_missing_component"
    assert reported["quality_flags"] == ["lower_bound_only"]


def test_load_private_debt_schedule_accepts_company_id_and_metric_names(tmp_path):
    path = tmp_path / "private_debt_schedule.parquet"
    pd.DataFrame(
        {
            "company_id": ["1750"],
            "debt_due_0_12m": [10.0],
            "debt_due_12_24m": [20.0],
        }
    ).to_parquet(path, index=False)

    loaded = _load_private_debt_schedule(path)

    assert loaded["0000001750"]["due_0_12"] == 10.0
    assert loaded["0000001750"]["due_12_24"] == 20.0
