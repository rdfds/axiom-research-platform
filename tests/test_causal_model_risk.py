from __future__ import annotations

from types import SimpleNamespace

from src.causal_model_risk import build_causal_model_risk_report


def _row(action_id: str, quality: float, support: float, mode: float, oos: bool, feasible: bool = True):
    drivers = [
        {"driver_name": "causal_model_mode", "contribution": mode},
        {"driver_name": "causal_model_quality", "contribution": quality},
        {"driver_name": "causal_model_support_score", "contribution": support},
    ]
    if oos:
        drivers.append({"driver_name": "causal_model_oos_penalty", "contribution": -0.25})
    return {
        "feasible": feasible,
        "action_candidate": {
            "action_id": action_id,
            "impact_distribution": {
                "uncertainty_score": 0.4 if not oos else 0.8,
                "key_drivers": drivers,
            },
        },
    }


def test_build_causal_model_risk_report_summary():
    run = SimpleNamespace(
        run_id="r1",
        company_id="0000320193",
        as_of_time="2026-02-28T00:00:00+00:00",
        frozen_state=SimpleNamespace(snapshot_hash="abc"),
        model_versions=SimpleNamespace(mechanism_model_version="mechanism_model_v2_causal+mode_standalone"),
    )
    rows = [
        _row("capital_return.open_market_buyback", quality=0.2, support=0.7, mode=1.0, oos=False, feasible=True),
        _row("capital_return.open_market_buyback", quality=-0.4, support=0.1, mode=0.0, oos=True, feasible=False),
    ]
    report = build_causal_model_risk_report(run=run, snapshot={"regime": {"credit_regime": "neutral"}}, feasibility_results=rows)
    s = report["summary"]
    assert s["total_candidates"] == 2
    assert s["feasible_candidates"] == 1
    assert s["causal_present_rate"] == 1.0
    assert s["standalone_applied_rate"] == 0.5
    assert s["standalone_fallback_rate"] == 0.5
    assert s["oos_penalty_rate"] == 0.5
    assert report["causal_mode"] == "standalone"
    assert isinstance(report["action_breakdown"], list) and report["action_breakdown"]


def test_build_causal_model_risk_report_with_previous():
    run = SimpleNamespace(
        run_id="r2",
        company_id="0000320193",
        as_of_time="2026-02-28T00:00:00+00:00",
        frozen_state=SimpleNamespace(snapshot_hash="def"),
        model_versions=SimpleNamespace(mechanism_model_version="mechanism_model_v2_causal+mode_standalone"),
    )
    rows = [_row("capital_structure.refinancing", quality=0.1, support=0.6, mode=1.0, oos=False)]
    prev = {"generated_at": "2026-03-07T00:00:00Z", "summary": {"low_quality_rate": 0.5, "low_support_rate": 0.5, "oos_penalty_rate": 0.5}}
    report = build_causal_model_risk_report(run=run, snapshot={}, feasibility_results=rows, previous_report=prev)
    mon = report["challenger_monitoring"]
    assert mon["has_previous_report"] is True
    assert mon["delta_low_quality_rate"] < 0
