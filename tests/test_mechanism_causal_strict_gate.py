from __future__ import annotations

import json

from src.mechanism_brain import MechanismBrain
from src.causal_impact_model import load_causal_routing_config


class _DummyRegistry:
    def get_action(self, action_id: str):  # noqa: D401
        return {"action_id": action_id}


class _StubCausalModel:
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, **kwargs):  # noqa: D401
        self.calls += 1
        return {"unexpected": True, "kwargs": kwargs}


def _brain() -> MechanismBrain:
    return MechanismBrain(
        action_registry=_DummyRegistry(),
        causal_model=None,
        causal_quality_floor=0.10,
        causal_support_floor=0.35,
        causal_min_train_rows=1000,
        causal_min_oos_r2=0.0,
        causal_min_treated_rows=1000,
        causal_min_control_rows=5000,
    )


def test_strict_causal_gate_passes_when_all_thresholds_met():
    brain = _brain()
    ok, reason = brain._passes_strict_causal_gate(
        {
            "model_quality": 0.20,
            "support_score": 0.60,
            "n_train": 5000,
            "out_of_sample_flag": False,
            "min_oos_r2": 0.05,
            "min_treated_rows": 1500,
            "min_control_rows": 8000,
        }
    )
    assert ok is True
    assert reason == "pass"


def test_strict_causal_gate_fails_on_oos_and_support_and_counts():
    brain = _brain()
    ok, reason = brain._passes_strict_causal_gate(
        {
            "model_quality": 0.08,
            "support_score": 0.10,
            "n_train": 600,
            "out_of_sample_flag": True,
            "min_oos_r2": -0.02,
            "min_treated_rows": 200,
            "min_control_rows": 3000,
        }
    )
    assert ok is False
    assert "out_of_support" in reason
    assert "quality<0.10" in reason
    assert "support<0.35" in reason
    assert "n_train<1000" in reason
    assert "oos_r2<0.00" in reason
    assert "treated<1000" in reason
    assert "control<5000" in reason


def test_strict_causal_gate_fails_when_oos_unavailable():
    brain = _brain()
    ok, reason = brain._passes_strict_causal_gate(
        {
            "model_quality": 0.30,
            "support_score": 0.70,
            "n_train": 3000,
            "out_of_sample_flag": False,
            "min_oos_r2": None,
            "min_treated_rows": 2000,
            "min_control_rows": 9000,
        }
    )
    assert ok is False
    assert "oos_unavailable" in reason


