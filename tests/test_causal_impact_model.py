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


def _predict_with_oos_r2(oos_r2: float):
    model = CausalImpactModel(_payload(oos_r2))
    pred = model.predict(
        action_id="capital_return.open_market_buyback",
        action_type="capital_return",
        params={"size_pct_market_cap": 0.05, "funding_mix": {"cash": 1.0, "debt": 0.0, "equity": 0.0}},
        features={"market.market_cap": {"value": 2_000_000_000.0}},
        regime={"credit_regime": "neutral", "vol_regime": "normal"},
    )
    assert pred is not None
    return pred


def test_predict_uses_debt_aliases_only_for_capital_structure_actions(monkeypatch):
    monkeypatch.setenv("AXIOM_ENABLE_RUNTIME_FEATURE_ADAPTER", "1")
    monkeypatch.setenv("AXIOM_RUNTIME_FEATURE_ADAPTER_RULES", "normalized_net_debt")
    payload = {
        "version": "causal_test_context_gate",
        "feature_order": ["base_net_debt"],
        "feature_stats": {
            "base_net_debt": {"mean": 420.0, "std": 1.0, "median": 420.0},
        },
        "objectives": {
            "value_creation": {
                "models": {
                    "__global__": {
                        "intercept": 0.0,
                        "coefficients": {"base_net_debt": 1.0},
                        "residual_std": 0.01,
                        "n_train": 5000,
                        "n_valid": 600,
                        "treated_rows": 1800,
                        "control_rows": 3200,
                        "r2": 0.25,
                        "oos_r2": 0.10,
                    }
                }
            }
        },
    }
    features = {
        "capital_structure.net_debt": {"value": 500.0, "support_mode": "exact"},
        "capital_structure.net_debt_normalized": {"value": 420.0, "support_mode": "exact"},
    }
    model = CausalImpactModel(payload)

    capital_return = model.predict(
        action_id="capital_return.open_market_buyback",
        action_type="capital_return",
        params={"size_pct_market_cap": 0.05, "funding_mix": {"cash": 1.0, "debt": 0.0, "equity": 0.0}},
        features=features,
        regime={"credit_regime": "neutral", "vol_regime": "normal"},
    )
    capital_structure = model.predict(
        action_id="capital_structure.refinancing",
        action_type="capital_structure",
        params={"size_pct_market_cap": 0.05, "funding_mix": {"cash": 0.0, "debt": 1.0, "equity": 0.0}},
        features=features,
        regime={"credit_regime": "neutral", "vol_regime": "normal"},
    )

    assert capital_return is not None
    assert capital_structure is not None
    assert capital_return.objectives["value_creation"]["median"] > 70.0
    assert abs(capital_structure.objectives["value_creation"]["median"]) < 1e-6


def test_negative_oos_quality_limits_blend_weight():
    pred = _predict_with_oos_r2(-0.30)
    assert pred.blend_weight <= 0.12
    assert pred.model_quality < 0.0
    assert pred.model_version == "causal_test_v2"
    assert 0.0 <= pred.support_score <= 1.0
    assert pred.min_oos_r2 is not None
    assert pred.min_oos_r2 < 0.0
    assert isinstance(pred.selected_model_keys, list)


def test_positive_oos_quality_increases_blend_weight():
    bad = _predict_with_oos_r2(-0.30)
    good = _predict_with_oos_r2(0.35)
    assert good.blend_weight > bad.blend_weight
    assert good.coverage_score > bad.coverage_score


def test_diagnose_returns_routing_metadata_without_prediction():
    payload = _payload(0.25)
    payload["objectives"]["risk_reduction"] = {
        "models": {
            "__global__": {
                "intercept": 0.03,
                "coefficients": {
                    "base_market_cap": 0.0,
                    "action_size": 0.0,
                    "funding_mix_cash": 0.0,
                },
                "residual_std": 0.03,
                "n_train": 7000,
                "n_valid": 800,
                "treated_rows": 1800,
                "control_rows": 5200,
                "r2": 0.25,
                "oos_r2": 0.12,
            }
        }
    }
    model = CausalImpactModel(payload)
    diag = model.diagnose(
        action_id="capital_return.open_market_buyback",
        action_type="capital_return",
        params={"size_pct_market_cap": 0.05, "funding_mix": {"cash": 1.0, "debt": 0.0, "equity": 0.0}},
        features={"market.market_cap": {"value": 2_000_000_000.0}},
        regime={"credit_regime": "neutral", "vol_regime": "normal"},
    )
    assert diag is not None
    assert diag.model_version == "causal_test_v2"
    assert diag.n_train == 5000
    assert diag.min_treated_rows == 1800
    assert diag.min_control_rows == 5200
    assert diag.min_oos_r2 is not None and abs(diag.min_oos_r2 - 0.12) < 1e-6
    assert diag.action_alias == "buyback"
    assert diag.selected_models_by_objective["value_creation"]["selected_key"] == "__global__"
    assert diag.selected_models_by_objective["risk_reduction"]["selected_key"] == "__global__"
    assert 0.0 <= diag.support_score <= 1.0


def test_dr_model_preferred_over_legacy_model():
    payload = _payload(0.25)
    payload["objectives"]["value_creation"]["models"]["buyback"] = {
        "intercept": 0.99,
        "coefficients": {
            "base_market_cap": 0.0,
            "action_size": 0.0,
            "funding_mix_cash": 0.0,
        },
        "residual_std": 0.001,
        "n_train": 800,
        "n_valid": 100,
        "r2": 0.10,
        "oos_r2": -0.10,
    }
    payload["objectives"]["value_creation"]["dr_models"] = {
        "buyback": {
            "method": "dr_aipw_ridge_v1",
            "intercept": 0.07,
            "coefficients": {
                "base_market_cap": 0.0,
                "action_size": 0.0,
                "funding_mix_cash": 0.0,
            },
            "residual_std": 0.01,
            "n_train": 5000,
            "n_valid": 600,
            "treated_rows": 1400,
            "control_rows": 3600,
            "crossfit_folds": 2,
            "propensity_clip": 0.05,
            "r2": 0.20,
            "oos_r2": 0.25,
        }
    }
    model = CausalImpactModel(payload)
    pred = model.predict(
        action_id="capital_return.open_market_buyback",
        action_type="capital_return",
        params={"size_pct_market_cap": 0.05, "funding_mix": {"cash": 1.0, "debt": 0.0, "equity": 0.0}},
        features={"market.market_cap": {"value": 2_000_000_000.0}},
        regime={"credit_regime": "neutral", "vol_regime": "normal"},
    )
    assert pred is not None
    # If DR model is selected, median should stay close to DR intercept, not legacy 0.99.
    assert pred.objectives["value_creation"]["median"] < 0.20
    assert pred.min_treated_rows >= 1000
    assert pred.min_control_rows >= 3000


def test_out_of_sample_flag_for_extreme_feature_distance():
    model = CausalImpactModel(_payload(0.25))
    pred = model.predict(
        action_id="capital_return.open_market_buyback",
        action_type="capital_return",
        params={"size_absolute_usd": 10_000_000_000_000_000.0, "funding_mix": {"cash": 1.0, "debt": 0.0, "equity": 0.0}},
        features={"market.market_cap": {"value": 9_999_999_999_999.0}},
        regime={"credit_regime": "neutral", "vol_regime": "normal"},
    )
    assert pred is not None
    assert pred.out_of_sample_flag is True


def test_capital_routing_config_overrides_action_policy(tmp_path, monkeypatch):
    routing_path = tmp_path / "causal_capital_routing_v1.json"
    routing_path.write_text(
        json.dumps(
            {
                "status_max_blend_weight": {"weak_prior_only": 0.07},
                "actions": {
                    "capital_structure.refinancing": {
                        "status": "weak_prior_only",
                        "model_action_alias": "bond_issuance",
                        "model_subtype_alias": "unknown",
                        "objective_allowlist": ["risk_reduction"],
                    }
                },
            }
        )
    )
    monkeypatch.setenv("CAUSAL_ROUTING_CONFIG_PATH", str(routing_path))
    load_causal_routing_config.cache_clear()

    payload = _payload(0.25)
    payload["objectives"]["risk_reduction"] = {
        "models": {
            "__global__": {
                "intercept": 0.03,
                "coefficients": {
                    "base_market_cap": 0.0,
                    "action_size": 0.0,
                    "funding_mix_cash": 0.0,
                },
                "residual_std": 0.03,
                "n_train": 7000,
                "n_valid": 800,
                "treated_rows": 1800,
                "control_rows": 5200,
                "r2": 0.25,
                "oos_r2": 0.12,
            }
        }
    }
    model = CausalImpactModel(payload)
    pred = model.predict(
        action_id="capital_structure.refinancing",
        action_type="capital_structure",
        params={"size_pct_market_cap": 0.05, "funding_mix": {"cash": 0.0, "debt": 1.0, "equity": 0.0}},
        features={"market.market_cap": {"value": 2_000_000_000.0}},
        regime={"credit_regime": "neutral", "vol_regime": "normal"},
    )
    assert pred is not None
    assert pred.action_status == "weak_prior_only"
    assert pred.max_blend_weight is not None and abs(pred.max_blend_weight - 0.07) < 1e-6
    assert pred.objective_allowlist == ["risk_reduction"]
    assert list(pred.objectives) == ["risk_reduction"]
    assert action_id_to_outcomes_action_type("capital_structure.refinancing", "capital_structure") == "bond_issuance"
    assert action_subtype_to_outcomes_subtype(
        "capital_structure.refinancing",
        "capital_structure",
        "",
    ) == "unknown"
    policy = get_causal_action_policy("capital_structure.refinancing", "capital_structure", "")
    assert policy.status == "weak_prior_only"

    load_causal_routing_config.cache_clear()


def test_predict_falls_back_to_canonical_action_cells_when_routing_alias_misses(tmp_path, monkeypatch):
    routing_path = tmp_path / "causal_capital_routing_v1.json"
    routing_path.write_text(
        json.dumps(
            {
                "actions": {
                    "capital_structure.equity_issuance": {
                        "status": "enabled",
                        "model_action_alias": "equity_offering_public_proxy",
                        "model_subtype_alias": "share_issuance_proxy",
                        "future_action_alias": "equity_issuance",
                        "objective_allowlist": ["risk_reduction"],
                    }
                },
            }
        )
    )
    monkeypatch.setenv("CAUSAL_ROUTING_CONFIG_PATH", str(routing_path))
    load_causal_routing_config.cache_clear()

    payload = _payload(0.25)
    payload["objectives"] = {
        "risk_reduction": {
            "models": {
                "capital_structure::equity_issuance": {
                    "intercept": 0.08,
                    "coefficients": {
                        "base_market_cap": 0.0,
                        "action_size": 0.0,
                        "funding_mix_cash": 0.0,
                    },
                    "residual_std": 0.02,
                    "n_train": 6000,
                    "n_valid": 700,
                    "treated_rows": 1500,
                    "control_rows": 4500,
                    "r2": 0.20,
                    "oos_r2": 0.11,
                }
            }
        }
    }
    model = CausalImpactModel(payload)
    diag = model.diagnose(
        action_id="capital_structure.equity_issuance",
        action_type="capital_structure",
        action_subtype="equity_issuance",
        params={"size_pct_market_cap": 0.05, "funding_mix": {"cash": 0.0, "debt": 0.0, "equity": 1.0}},
        features={"market.market_cap": {"value": 2_000_000_000.0}},
        regime={"credit_regime": "neutral", "vol_regime": "normal"},
    )
    assert diag is not None
    assert diag.selected_models_by_objective["risk_reduction"]["selected_key"] == "capital_structure::equity_issuance"

    pred = model.predict(
        action_id="capital_structure.equity_issuance",
        action_type="capital_structure",
        action_subtype="equity_issuance",
        params={"size_pct_market_cap": 0.05, "funding_mix": {"cash": 0.0, "debt": 0.0, "equity": 1.0}},
        features={"market.market_cap": {"value": 2_000_000_000.0}},
        regime={"credit_regime": "neutral", "vol_regime": "normal"},
    )
    assert pred is not None
    assert list(pred.objectives) == ["risk_reduction"]

    load_causal_routing_config.cache_clear()


def test_predict_uses_future_action_alias_list_for_buyback_cells(tmp_path, monkeypatch):
    routing_path = tmp_path / "causal_capital_routing_v1.json"
    routing_path.write_text(
        json.dumps(
            {
                "actions": {
                    "capital_return.open_market_buyback": {
                        "status": "enabled",
                        "model_action_alias": "buyback",
                        "model_subtype_alias": "buyback",
                        "future_action_aliases": ["buyback"],
                        "objective_allowlist": ["value_creation"],
                    }
                },
            }
        )
    )
    monkeypatch.setenv("CAUSAL_ROUTING_CONFIG_PATH", str(routing_path))
    load_causal_routing_config.cache_clear()

    payload = _payload(0.25)
    payload["objectives"] = {
        "value_creation": {
            "models": {
                "capital_return::buyback": {
                    "intercept": 0.12,
                    "coefficients": {
                        "base_market_cap": 0.0,
                        "action_size": 0.0,
                        "funding_mix_cash": 0.0,
                    },
                    "residual_std": 0.02,
                    "n_train": 7000,
                    "n_valid": 800,
                    "treated_rows": 1800,
                    "control_rows": 5200,
                    "r2": 0.18,
                    "oos_r2": 0.09,
                }
            }
        }
    }
    model = CausalImpactModel(payload)
    diag = model.diagnose(
        action_id="capital_return.open_market_buyback",
        action_type="capital_return",
        action_subtype="open_market_buyback",
        params={"size_pct_market_cap": 0.05, "funding_mix": {"cash": 1.0, "debt": 0.0, "equity": 0.0}},
        features={"market.market_cap": {"value": 2_000_000_000.0}},
        regime={"credit_regime": "neutral", "vol_regime": "normal"},
    )
    assert diag is not None
    assert diag.selected_models_by_objective["value_creation"]["selected_key"] == "capital_return::buyback"

    pred = model.predict(
        action_id="capital_return.open_market_buyback",
        action_type="capital_return",
        action_subtype="open_market_buyback",
        params={"size_pct_market_cap": 0.05, "funding_mix": {"cash": 1.0, "debt": 0.0, "equity": 0.0}},
        features={"market.market_cap": {"value": 2_000_000_000.0}},
        regime={"credit_regime": "neutral", "vol_regime": "normal"},
    )
    assert pred is not None
    assert pred.future_action_aliases == ["buyback"]
    assert list(pred.objectives) == ["value_creation"]

    load_causal_routing_config.cache_clear()


def test_dividend_initiate_policy_can_expose_rating_preservation(tmp_path, monkeypatch):
    routing_path = tmp_path / "causal_capital_routing_v1.json"
    routing_path.write_text(
        json.dumps(
            {
                "actions": {
                    "capital_return.dividend_initiate": {
                        "status": "enabled",
                        "model_action_alias": "dividend_initiate",
                        "model_subtype_alias": "dividend_initiate",
                        "objective_allowlist": ["rating_preservation"],
                        "max_blend_weight": 0.22,
                        "strict_gate_overrides": {"min_treated_rows": 750},
                    }
                }
            }
        )
    )
    monkeypatch.setenv("CAUSAL_ROUTING_CONFIG_PATH", str(routing_path))
    load_causal_routing_config.cache_clear()

    payload = _payload(0.25)
    payload["objectives"] = {
        "rating_preservation": {
            "models": {
                "capital_return::dividend_initiate": {
                    "intercept": 0.04,
                    "coefficients": {
                        "base_market_cap": 0.0,
                        "action_size": 0.0,
                        "funding_mix_cash": 0.0,
                    },
                    "residual_std": 0.02,
                    "n_train": 1200,
                    "n_valid": 80,
                    "treated_rows": 900,
                    "control_rows": 40000,
                    "r2": 0.18,
                    "oos_r2": 0.12,
                }
            }
        }
    }
    model = CausalImpactModel(payload)
    pred = model.predict(
        action_id="capital_return.dividend_initiate",
        action_type="capital_return",
        action_subtype="dividend_initiate",
        params={"size_pct_market_cap": 0.01, "funding_mix": {"cash": 1.0, "debt": 0.0, "equity": 0.0}},
        features={"market.market_cap": {"value": 2_000_000_000.0}},
        regime={"credit_regime": "neutral", "vol_regime": "normal"},
    )
    assert pred is not None
    assert pred.action_status == "enabled"
    assert pred.objective_allowlist == ["rating_preservation"]
    assert pred.max_blend_weight is not None and abs(pred.max_blend_weight - 0.22) < 1e-6
    assert pred.min_treated_rows_override == 750
    assert list(pred.objectives) == ["rating_preservation"]

    policy = get_causal_action_policy("capital_return.dividend_initiate", "capital_return", "dividend_initiate")
    assert policy.model_action_alias == "dividend_initiate"
    assert policy.model_subtype_alias == "dividend_initiate"
    assert policy.min_treated_rows_override == 750

    load_causal_routing_config.cache_clear()


def test_dividend_initiate_strict_gate_can_anchor_on_primary_objective(tmp_path, monkeypatch):
    routing_path = tmp_path / "causal_capital_routing_v1.json"
    routing_path.write_text(
        json.dumps(
            {
                "actions": {
                    "capital_return.dividend_initiate": {
                        "status": "enabled",
                        "model_action_alias": "dividend_initiate",
                        "model_subtype_alias": "dividend_initiate",
                        "objective_allowlist": ["growth", "rating_preservation"],
                        "strict_gate_primary_objectives": ["rating_preservation"],
                        "max_blend_weight": 0.22,
                        "strict_gate_overrides": {"min_treated_rows": 750, "min_oos_r2": 0.08},
                    }
                }
            }
        )
    )
    monkeypatch.setenv("CAUSAL_ROUTING_CONFIG_PATH", str(routing_path))
    load_causal_routing_config.cache_clear()

    payload = _payload(0.25)
    payload["objectives"] = {
        "growth": {
            "models": {
                "capital_return::dividend_initiate": {
                    "intercept": 0.03,
                    "coefficients": {
                        "base_market_cap": 0.0,
                        "action_size": 0.0,
                        "funding_mix_cash": 0.0,
                    },
                    "residual_std": 0.02,
                    "n_train": 1500,
                    "n_valid": 80,
                    "treated_rows": 900,
                    "control_rows": 40000,
                    "r2": 0.05,
                    "oos_r2": 0.02,
                }
            }
        },
        "rating_preservation": {
            "models": {
                "capital_return::dividend_initiate": {
                    "intercept": 0.04,
                    "coefficients": {
                        "base_market_cap": 0.0,
                        "action_size": 0.0,
                        "funding_mix_cash": 0.0,
                    },
                    "residual_std": 0.02,
                    "n_train": 1200,
                    "n_valid": 80,
                    "treated_rows": 900,
                    "control_rows": 40000,
                    "r2": 0.18,
                    "oos_r2": 0.12,
                }
            }
        },
    }
    model = CausalImpactModel(payload)
    pred = model.predict(
        action_id="capital_return.dividend_initiate",
        action_type="capital_return",
        action_subtype="dividend_initiate",
        params={"size_pct_market_cap": 0.01, "funding_mix": {"cash": 1.0, "debt": 0.0, "equity": 0.0}},
        features={"market.market_cap": {"value": 2_000_000_000.0}},
        regime={"credit_regime": "neutral", "vol_regime": "normal"},
    )
    assert pred is not None
    assert set(pred.objectives) == {"growth", "rating_preservation"}
    assert pred.strict_gate_primary_objectives == ["rating_preservation"]
    assert pred.min_oos_r2 is not None and abs(pred.min_oos_r2 - 0.12) < 1e-6
    assert abs(pred.model_quality - 0.12) < 1e-6

    policy = get_causal_action_policy("capital_return.dividend_initiate", "capital_return", "dividend_initiate")
    assert policy.strict_gate_primary_objectives == ("rating_preservation",)

    load_causal_routing_config.cache_clear()


