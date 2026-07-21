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


class _FailingHGBPredictor:
    def __init__(self) -> None:
        self.learning_rate = 0.1
        self._baseline_prediction = 1.0
        dtype = [
            ("value", "<f8"),
            ("count", "<u4"),
            ("feature_idx", "<u4"),
            ("num_threshold", "<f8"),
            ("missing_go_to_left", "u1"),
            ("left", "<u4"),
            ("right", "<u4"),
            ("gain", "<f8"),
            ("depth", "<u4"),
            ("is_leaf", "u1"),
            ("bin_threshold", "u1"),
            ("is_categorical", "u1"),
            ("bitset_idx", "<u4"),
        ]
        nodes = np.zeros(3, dtype=dtype)
        nodes[0] = (0.0, 10, 0, 0.5, 0, 1, 2, 0.0, 0, 0, 0, 0, 0)
        nodes[1] = (2.0, 5, 0, 0.0, 0, 1, 1, 0.0, 1, 1, 0, 0, 0)
        nodes[2] = (-1.0, 5, 0, 0.0, 0, 2, 2, 0.0, 1, 1, 0, 0, 0)
        self._predictors = [[_TreeStub(nodes)]]

    def predict(self, X):  # noqa: N803
        raise ValueError("synthetic omp failure")


def _payload(oos_r2: float) -> dict:
    return {
        "version": "causal_test_v2",
        "feature_order": [
            "base_market_cap",
            "action_size",
            "funding_mix_cash",
        ],
        "feature_stats": {
            "base_market_cap": {"mean": 1_000_000_000.0, "std": 500_000_000.0, "median": 1_000_000_000.0},
            "action_size": {"mean": 100_000_000.0, "std": 100_000_000.0, "median": 100_000_000.0},
            "funding_mix_cash": {"mean": 0.5, "std": 0.25, "median": 0.5},
        },
        "objectives": {
            "value_creation": {
                "models": {
                    "__global__": {
                        "intercept": 0.05,
                        "coefficients": {
                            "base_market_cap": 0.01,
                            "action_size": 0.01,
                            "funding_mix_cash": 0.01,
                        },
                        "residual_std": 0.05,
                        "n_train": 5000,
                        "n_valid": 600,
                        "r2": 0.45,
                        "oos_r2": oos_r2,
                    }
                }
            }
        },
    }


