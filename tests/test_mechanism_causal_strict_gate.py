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


