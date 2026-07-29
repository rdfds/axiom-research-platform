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


