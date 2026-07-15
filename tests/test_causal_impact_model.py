from __future__ import annotations

import json
import pickle
import __main__
from pathlib import Path

import numpy as np

from src.causal_impact_model import (
    CausalImpactModel,
    action_id_to_outcomes_action_type,
    action_subtype_to_outcomes_subtype,
    get_causal_action_policy,
    load_causal_routing_config,
)


class _DummyPredictor:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def predict(self, X):  # noqa: N803
        return [self.value for _ in X]


class _TreeStub:
    def __init__(self, nodes) -> None:
        self.nodes = nodes


