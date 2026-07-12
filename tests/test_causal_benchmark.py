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


