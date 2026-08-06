from __future__ import annotations

from src.causal_benchmark import compare_summaries, evaluate_summary_thresholds, summarize_model_card


def _card() -> dict:
    return {
        "version": "causal_impact_model_v2",
        "trained_at": "2026-03-09T00:00:00+00:00",
        "dataset_rows": 10000,
        "model_family": "hgb",
        "cell_level": "action_subtype",
        "objectives": {
            "value_creation": {
                "actions": {
                    "a::x": {"enabled": True, "oos_r2": 0.10},
                    "a::y": {"enabled": True, "oos_r2": 0.08},
                    "a::z": {"enabled": False, "oos_r2": -0.20},
                }
            },
            "risk_reduction": {
                "actions": {
                    "b::x": {"enabled": True, "oos_r2": 0.20},
                    "b::y": {"enabled": True, "oos_r2": 0.05},
                }
            },
            "growth": {
                "actions": {
                    "c::x": {"enabled": False, "oos_r2": -0.30},
                }
            },
        },
    }


def test_summarize_model_card_counts_and_oos_stats():
    summary = summarize_model_card(_card())
    totals = summary["totals"]
    assert totals["total_cells"] == 6
    assert totals["enabled_cells"] == 4
    assert abs(float(totals["enabled_rate"]) - (4 / 6)) < 1e-6
    assert totals["enabled_oos_r2_mean"] > 0.10
    assert summary["objectives"]["growth"]["enabled_cells"] == 0


def test_evaluate_summary_thresholds_detects_failures():
    summary = summarize_model_card(_card())
    gates = evaluate_summary_thresholds(
        summary,
        min_enabled_cells=5,
        min_enabled_rate=0.80,
        min_enabled_oos_r2_mean=0.15,
        required_objectives=["value_creation", "growth"],
        required_objective_min_enabled=1,
        required_objective_min_oos_r2_mean=0.05,
    )
    assert gates["pass"] is False
    assert "enabled_cells<5" in gates["failures"]
    assert "growth.enabled<1" in gates["failures"]


def test_compare_summaries_reports_improvement():
    champ = summarize_model_card(_card())
    upgraded = _card()
    upgraded["objectives"]["value_creation"]["actions"]["a::y"]["oos_r2"] = 0.18
    upgraded["objectives"]["growth"]["actions"]["c::x"] = {"enabled": True, "oos_r2": 0.12}
    chall = summarize_model_card(upgraded)
    cmp = compare_summaries(champ, chall)
    assert cmp["totals"]["delta_enabled_cells"] == 1
    assert cmp["totals"]["delta_enabled_oos_r2_mean"] > 0.0
    assert cmp["challenger_better_objective_fraction"] is not None
