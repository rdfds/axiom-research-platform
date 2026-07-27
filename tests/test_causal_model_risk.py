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


