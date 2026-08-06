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


def test_predict_can_override_model_artifact_per_action(tmp_path, monkeypatch):
    base_model_path = tmp_path / "base_model.json"
    override_model_path = tmp_path / "override_model.json"
    routing_path = tmp_path / "causal_capital_routing_v1.json"

    base_payload = _payload(0.25)
    base_payload["version"] = "base_model_v1"
    base_payload["objectives"] = {
        "value_creation": base_payload["objectives"]["value_creation"],
    }
    base_model_path.write_text(json.dumps(base_payload))

    override_payload = _payload(0.25)
    override_payload["version"] = "override_model_v1"
    override_payload["objectives"] = {
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
    override_model_path.write_text(json.dumps(override_payload))

    routing_path.write_text(
        json.dumps(
            {
                "actions": {
                    "capital_return.dividend_initiate": {
                        "status": "enabled",
                        "model_action_alias": "dividend_initiate",
                        "model_subtype_alias": "dividend_initiate",
                        "objective_allowlist": ["rating_preservation"],
                        "strict_gate_primary_objectives": ["rating_preservation"],
                        "model_artifact_path_override": str(override_model_path),
                    }
                }
            }
        )
    )
    monkeypatch.setenv("CAUSAL_ROUTING_CONFIG_PATH", str(routing_path))
    load_causal_routing_config.cache_clear()

    model = CausalImpactModel.from_path(base_model_path)
    pred = model.predict(
        action_id="capital_return.dividend_initiate",
        action_type="capital_return",
        action_subtype="dividend_initiate",
        params={"size_pct_market_cap": 0.01, "funding_mix": {"cash": 1.0, "debt": 0.0, "equity": 0.0}},
        features={"market.market_cap": {"value": 2_000_000_000.0}},
        regime={"credit_regime": "neutral", "vol_regime": "normal"},
    )
    assert pred is not None
    assert pred.model_version == "override_model_v1"
    assert pred.model_artifact_path_override == str(override_model_path)
    assert list(pred.objectives) == ["rating_preservation"]

    diag = model.diagnose(
        action_id="capital_return.dividend_initiate",
        action_type="capital_return",
        action_subtype="dividend_initiate",
        params={"size_pct_market_cap": 0.01, "funding_mix": {"cash": 1.0, "debt": 0.0, "equity": 0.0}},
        features={"market.market_cap": {"value": 2_000_000_000.0}},
        regime={"credit_regime": "neutral", "vol_regime": "normal"},
    )
    assert diag is not None
    assert diag.model_version == "override_model_v1"
    assert diag.model_artifact_path_override == str(override_model_path)

    load_causal_routing_config.cache_clear()


def test_disabled_subtype_cell_falls_back_to_enabled_all_cell():
    payload = _payload(0.25)
    payload["objectives"]["value_creation"]["dr_models"] = {
        "buyback::buyback": {
            "method": "dr_aipw_hgb_v1",
            "model_family": "linear",
            "intercept": 0.90,
            "coefficients": {
                "base_market_cap": 0.0,
                "action_size": 0.0,
                "funding_mix_cash": 0.0,
            },
            "residual_std": 0.01,
            "n_train": 10000,
            "n_valid": 1000,
            "treated_rows": 2000,
            "control_rows": 8000,
            "r2": 0.3,
            "oos_r2": 0.2,
            "enabled": False,
        },
        "buyback::all": {
            "method": "dr_aipw_hgb_v1",
            "model_family": "linear",
            "intercept": 0.07,
            "coefficients": {
                "base_market_cap": 0.0,
                "action_size": 0.0,
                "funding_mix_cash": 0.0,
            },
            "residual_std": 0.01,
            "n_train": 10000,
            "n_valid": 1000,
            "treated_rows": 2000,
            "control_rows": 8000,
            "r2": 0.3,
            "oos_r2": 0.2,
            "enabled": True,
        },
    }
    model = CausalImpactModel(payload)
    pred = model.predict(
        action_id="capital_return.open_market_buyback",
        action_type="capital_return",
        action_subtype="open_market_buyback",
        params={"size_pct_market_cap": 0.05, "funding_mix": {"cash": 1.0, "debt": 0.0, "equity": 0.0}},
        features={"market.market_cap": {"value": 2_000_000_000.0}},
        regime={"credit_regime": "neutral", "vol_regime": "normal"},
    )
    assert pred is not None
    assert pred.objectives["value_creation"]["median"] < 0.2


def test_hgb_bundle_model_prediction(tmp_path: Path):
    payload = _payload(0.2)
    payload["model_bundle_path"] = "bundle.pkl"
    payload["objectives"]["value_creation"]["dr_models"] = {
        "buyback::buyback": {
            "method": "dr_aipw_hgb_v1",
            "model_family": "hgb",
            "bundle_key": "value_creation::buyback::buyback",
            "residual_std": 0.02,
            "n_train": 10000,
            "n_valid": 1000,
            "treated_rows": 2000,
            "control_rows": 8000,
            "r2": 0.3,
            "oos_r2": 0.2,
            "enabled": True,
        }
    }
    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps(payload))
    bundle = {"value_creation::buyback::buyback": _DummyPredictor(0.12)}
    with open(tmp_path / "bundle.pkl", "wb") as fh:
        pickle.dump(bundle, fh, protocol=pickle.HIGHEST_PROTOCOL)

    model = CausalImpactModel.from_path(model_path)
    pred = model.predict(
        action_id="capital_return.open_market_buyback",
        action_type="capital_return",
        action_subtype="open_market_buyback",
        params={"size_pct_market_cap": 0.05, "funding_mix": {"cash": 1.0, "debt": 0.0, "equity": 0.0}},
        features={"market.market_cap": {"value": 2_000_000_000.0}},
        regime={"credit_regime": "neutral", "vol_regime": "normal"},
    )
    assert pred is not None
    assert abs(pred.objectives["value_creation"]["median"] - 0.12) < 1e-6


def test_hgb_bundle_model_prediction_falls_back_when_predict_raises(tmp_path: Path):
    payload = _payload(0.2)
    payload["model_bundle_path"] = "bundle.pkl"
    payload["objectives"]["value_creation"]["dr_models"] = {
        "buyback::buyback": {
            "method": "dr_aipw_hgb_v1",
            "model_family": "hgb",
            "bundle_key": "value_creation::buyback::buyback",
            "residual_std": 0.02,
            "n_train": 10000,
            "n_valid": 1000,
            "treated_rows": 2000,
            "control_rows": 8000,
            "r2": 0.3,
            "oos_r2": 0.2,
            "enabled": True,
        }
    }
    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps(payload))
    bundle = {"value_creation::buyback::buyback": _FailingHGBPredictor()}
    with open(tmp_path / "bundle.pkl", "wb") as fh:
        pickle.dump(bundle, fh, protocol=pickle.HIGHEST_PROTOCOL)

    model = CausalImpactModel.from_path(model_path)
    pred = model.predict(
        action_id="capital_return.open_market_buyback",
        action_type="capital_return",
        action_subtype="open_market_buyback",
        params={"size_pct_market_cap": 0.05, "funding_mix": {"cash": 1.0, "debt": 0.0, "equity": 0.0}},
        features={"market.market_cap": {"value": 1_000_000_000.0}},
        regime={"credit_regime": "neutral", "vol_regime": "normal"},
    )
    assert pred is not None
    assert abs(pred.objectives["value_creation"]["median"] - 1.2) < 1e-6


def test_subtype_keyed_model_fallback_when_action_alias_differs():
    payload = _payload(0.2)
    payload["objectives"]["value_creation"]["dr_models"] = {
        "dividend_increase::dividend_increase": {
            "method": "dr_aipw_hgb_v1",
            "model_family": "linear",
            "intercept": 0.11,
            "coefficients": {
                "base_market_cap": 0.0,
                "action_size": 0.0,
                "funding_mix_cash": 0.0,
            },
            "residual_std": 0.01,
            "n_train": 10000,
            "n_valid": 1000,
            "treated_rows": 2000,
            "control_rows": 8000,
            "r2": 0.3,
            "oos_r2": 0.2,
            "enabled": True,
        }
    }
    model = CausalImpactModel(payload)
    pred = model.predict(
        action_id="capital_return.dividend_increase",
        action_type="capital_return",
        action_subtype="dividend_increase",
        params={"size_pct_market_cap": 0.03, "funding_mix": {"cash": 1.0, "debt": 0.0, "equity": 0.0}},
        features={"market.market_cap": {"value": 2_000_000_000.0}},
        regime={"credit_regime": "neutral", "vol_regime": "normal"},
    )
    assert pred is not None
    assert abs(pred.objectives["value_creation"]["median"] - 0.11) < 1e-6


def test_feature_transform_spec_applies_unit_harmonization_and_signed_log():
    payload = {
        "version": "causal_test_v2",
        "feature_order": ["base_market_cap", "macro_rate_10y", "macro_ig_oas"],
        "feature_transform_spec": {
            "usd_millions_features": ["base_market_cap"],
            "rate_percent_features": ["macro_rate_10y"],
            "oas_percent_features": ["macro_ig_oas"],
            "signed_log1p_features": ["base_market_cap"],
        },
        "feature_stats": {
            # Expect transformed market cap around log1p(2_000_000) ~= 14.508658
            "base_market_cap": {"mean": 13.5, "std": 1.0, "median": 13.5},
            # Expect 0.045 -> 4.5 after rate unit harmonization.
            "macro_rate_10y": {"mean": 3.5, "std": 1.0, "median": 3.5},
            # Expect 120 -> 1.2 after OAS harmonization.
            "macro_ig_oas": {"mean": 1.0, "std": 1.0, "median": 1.0},
        },
        "objectives": {
            "value_creation": {
                "models": {
                    "__global__": {
                        "intercept": 0.0,
                        "coefficients": {
                            "base_market_cap": 1.0,
                            "macro_rate_10y": 1.0,
                            "macro_ig_oas": 1.0,
                        },
                        "residual_std": 1e-6,
                        "n_train": 5000,
                        "n_valid": 600,
                        "r2": 0.45,
                        "oos_r2": 0.25,
                    }
                }
            }
        },
    }
    model = CausalImpactModel(payload)
    pred = model.predict(
        action_id="capital_return.open_market_buyback",
        action_type="capital_return",
        params={"size_pct_market_cap": 0.05, "funding_mix": {"cash": 1.0, "debt": 0.0, "equity": 0.0}},
        features={
            "market.market_cap": {"value": 2_000_000_000_000.0},  # dollars -> 2,000,000 millions
            "macro.rate_10y": {"value": 0.045},  # decimal -> percent
            "market.ig_oas": {"value": 120.0},  # bps -> percent
        },
        regime={"credit_regime": "neutral", "vol_regime": "normal"},
    )
    assert pred is not None
    # Expected standardized sum:
    # z_market_cap ~ (log1p(2,000,000)-13.5) = 1.008658...
    # z_rate = (4.5-3.5)=1
    # z_ig = (1.2-1.0)=0.2
    assert abs(pred.objectives["value_creation"]["median"] - 2.208658) < 1e-5


def test_full_snapshot_input_uses_bundle_canonical_macro_metrics():
    payload = {
        "version": "causal_test_bundle_macro",
        "feature_order": ["macro_rate_10y", "macro_ig_oas", "macro_hy_oas", "macro_vix"],
        "feature_stats": {
            "macro_rate_10y": {"mean": 4.0, "std": 1.0, "median": 4.0},
            "macro_ig_oas": {"mean": 1.0, "std": 1.0, "median": 1.0},
            "macro_hy_oas": {"mean": 3.0, "std": 1.0, "median": 3.0},
            "macro_vix": {"mean": 20.0, "std": 10.0, "median": 20.0},
        },
        "objectives": {
            "value_creation": {
                "models": {
                    "__global__": {
                        "intercept": 0.0,
                        "coefficients": {
                            "macro_rate_10y": 1.0,
                            "macro_ig_oas": 1.0,
                            "macro_hy_oas": 1.0,
                            "macro_vix": 1.0,
                        },
                        "residual_std": 1e-6,
                        "n_train": 5000,
                        "n_valid": 600,
                        "r2": 0.2,
                        "oos_r2": 0.1,
                    }
                }
            }
        },
    }
    model = CausalImpactModel(payload)
    pred = model.predict(
        action_id="capital_return.open_market_buyback",
        action_type="capital_return",
        params={"size_pct_market_cap": 0.05, "funding_mix": {"cash": 1.0, "debt": 0.0, "equity": 0.0}},
        features={
            "company_id": "0000123456",
            "features": {
                "macro.ust_10y_yield": {"value": 4.5, "support_mode": "exact"},
                "macro.ig_oas": {"value": 1.2, "support_mode": "exact"},
                "macro.hy_oas": {"value": 3.4, "support_mode": "exact"},
                "market.vix": {"value": 25.0, "support_mode": "exact"},
            },
        },
        regime={"credit_regime": "neutral", "vol_regime": "normal"},
    )
    assert pred is not None
    # z_10y=(4.5-4.0)=0.5
    # z_ig=(1.2-1.0)=0.2
    # z_hy=(3.4-3.0)=0.4
    # z_vix=(25-20)/10=0.5
    assert abs(pred.objectives["value_creation"]["median"] - 1.6) < 1e-6


def test_legacy_artifact_without_transform_spec_uses_unit_harmonization_defaults():
    payload = {
        "version": "causal_test_legacy",
        "feature_order": ["base_market_cap", "macro_rate_10y", "macro_ig_oas"],
        # No feature_transform_spec on purpose (legacy artifact).
        "feature_stats": {
            # Legacy artifact expects market cap in USD millions (no signed-log).
            "base_market_cap": {"mean": 1_500_000.0, "std": 500_000.0, "median": 1_500_000.0},
            # Legacy artifact expects rates in percent.
            "macro_rate_10y": {"mean": 3.0, "std": 1.0, "median": 3.0},
            # Legacy artifact expects OAS in percent.
            "macro_ig_oas": {"mean": 1.0, "std": 1.0, "median": 1.0},
        },
        "objectives": {
            "value_creation": {
                "models": {
                    "__global__": {
                        "intercept": 0.0,
                        "coefficients": {
                            "base_market_cap": 1.0,
                            "macro_rate_10y": 1.0,
                            "macro_ig_oas": 1.0,
                        },
                        "residual_std": 1e-6,
                        "n_train": 5000,
                        "n_valid": 600,
                        "r2": 0.2,
                        "oos_r2": 0.1,
                    }
                }
            }
        },
    }
    model = CausalImpactModel(payload)
    pred = model.predict(
        action_id="capital_return.open_market_buyback",
        action_type="capital_return",
        params={"size_pct_market_cap": 0.05, "funding_mix": {"cash": 1.0, "debt": 0.0, "equity": 0.0}},
        features={
            "market.market_cap": {"value": 2_000_000_000_000.0},  # dollars -> 2,000,000 millions
            "macro.rate_10y": {"value": 0.04},  # decimal -> percent
            "market.ig_oas": {"value": 120.0},  # bps -> percent
        },
        regime={"credit_regime": "neutral", "vol_regime": "normal"},
    )
    assert pred is not None
    # z_market_cap=(2,000,000-1,500,000)/500,000=1.0
    # z_rate=(4.0-3.0)=1.0
    # z_ig=(1.2-1.0)=0.2
    assert abs(pred.objectives["value_creation"]["median"] - 2.2) < 1e-5


def test_bundle_unpickles_legacy_main_ridge_predictor(tmp_path: Path):
    payload = _payload(0.2)
    payload["model_bundle_path"] = "bundle.pkl"
    payload["objectives"]["value_creation"]["dr_models"] = {
        "buyback::buyback": {
            "method": "dr_aipw_hgb_v1",
            "model_family": "hgb",
            "bundle_key": "value_creation::buyback::buyback",
            "residual_std": 1e-6,
            "n_train": 10000,
            "n_valid": 1000,
            "treated_rows": 2000,
            "control_rows": 8000,
            "r2": 0.3,
            "oos_r2": 0.2,
            "enabled": True,
        }
    }
    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps(payload))

    # Simulate old bundle objects pickled as __main__._RidgePredictor.
    Legacy = type("_RidgePredictor", (), {})
    Legacy.__module__ = "__main__"
    setattr(__main__, "_RidgePredictor", Legacy)
    legacy_obj = Legacy()
    legacy_obj.beta = [0.12, 0.0, 0.0, 0.0]
    with open(tmp_path / "bundle.pkl", "wb") as fh:
        pickle.dump({"value_creation::buyback::buyback": legacy_obj}, fh, protocol=pickle.HIGHEST_PROTOCOL)
    delattr(__main__, "_RidgePredictor")

    model = CausalImpactModel.from_path(model_path)
    pred = model.predict(
        action_id="capital_return.open_market_buyback",
        action_type="capital_return",
        action_subtype="open_market_buyback",
        params={"size_pct_market_cap": 0.05, "funding_mix": {"cash": 1.0, "debt": 0.0, "equity": 0.0}},
        features={"market.market_cap": {"value": 2_000_000_000.0}},
        regime={"credit_regime": "neutral", "vol_regime": "normal"},
    )
    assert pred is not None
    assert abs(pred.objectives["value_creation"]["median"] - 0.12) < 1e-6


def test_action_id_mapping_aligns_runtime_to_outcomes_taxonomy():
    assert action_id_to_outcomes_action_type("capital_structure.new_debt_issuance") == "bond_issuance"
    assert action_id_to_outcomes_action_type("capital_structure.refinancing") == "bond_issuance"
    assert action_id_to_outcomes_action_type("capital_structure.revolver_draw_or_resize") == "loan_issuance"
    assert action_id_to_outcomes_action_type("capital_structure.equity_issuance") == "equity_offering_public_proxy"


def test_min_objective_oos_filter_removes_weak_objective(monkeypatch):
    payload = _payload(0.02)
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
                "n_train": 5000,
                "n_valid": 600,
                "r2": 0.25,
                "oos_r2": 0.20,
            }
        }
    }
    monkeypatch.setenv("CAUSAL_MIN_OBJECTIVE_OOS_R2", "0.05")
    model = CausalImpactModel(payload)
    pred = model.predict(
        action_id="capital_return.open_market_buyback",
        action_type="capital_return",
        params={"size_pct_market_cap": 0.05, "funding_mix": {"cash": 1.0, "debt": 0.0, "equity": 0.0}},
        features={"market.market_cap": {"value": 2_000_000_000.0}},
        regime={"credit_regime": "neutral", "vol_regime": "normal"},
    )
    assert pred is not None
    assert "value_creation" not in pred.objectives
    assert "risk_reduction" in pred.objectives
    assert pred.min_oos_r2 is not None and pred.min_oos_r2 >= 0.20


def test_standardized_feature_vector_uses_global_runtime_aliases_without_action_context(monkeypatch):
    monkeypatch.setenv("AXIOM_ENABLE_RUNTIME_FEATURE_ADAPTER", "1")
    monkeypatch.setenv(
        "AXIOM_RUNTIME_FEATURE_ADAPTER_RULES",
        "normalized_net_debt,normalized_net_leverage,pe_ratio_compatibility_alias,ust_10y_alias,ust_2y_alias,sofr_compatibility_fallback,credit_ig_alias,credit_hy_alias",
    )
    payload = {
        "version": "causal_alias_test_v1",
        "feature_order": [
            "base_leverage",
            "base_pe",
            "macro_rate_10y",
            "macro_rate_2y",
            "macro_sofr",
            "macro_ig_oas",
            "macro_hy_oas",
            "macro_vix",
        ],
        "feature_stats": {
            "base_leverage": {"mean": 0.0, "std": 1.0, "median": 0.0},
            "base_pe": {"mean": 0.0, "std": 1.0, "median": 0.0},
            "macro_rate_10y": {"mean": 0.0, "std": 1.0, "median": 0.0},
            "macro_rate_2y": {"mean": 0.0, "std": 1.0, "median": 0.0},
            "macro_sofr": {"mean": 0.0, "std": 1.0, "median": 0.0},
            "macro_ig_oas": {"mean": 0.0, "std": 1.0, "median": 0.0},
            "macro_hy_oas": {"mean": 0.0, "std": 1.0, "median": 0.0},
            "macro_vix": {"mean": 0.0, "std": 1.0, "median": 0.0},
        },
        "objectives": {},
    }
    model = CausalImpactModel(payload)
    vector = model._standardized_feature_vector(
        params={"funding_mix": {"cash": 1.0}},
        features={
            "capital_structure.net_leverage_normalized": {"value": 2.1, "support_mode": "exact"},
            "market.pe_ratio": {"value": 17.5, "support_mode": "exact"},
            "macro.ust_10y_yield": {"value": 4.58, "support_mode": "exact"},
            "macro.ust_2y_yield": {"value": 4.25, "support_mode": "exact"},
            "macro.sofr": {"value": 4.49, "support_mode": "exact"},
            "macro.ig_oas": {"value": 1.02, "support_mode": "exact"},
            "macro.hy_oas": {"value": 3.44, "support_mode": "exact"},
            "market.vix": {"value": 18.2, "support_mode": "exact"},
        },
        regime={"credit_regime": "neutral", "vol_regime": "normal"},
    )

    assert vector is not None
    assert vector["base_leverage"] == 2.1
    assert vector["base_pe"] == 17.5
    assert vector["macro_rate_10y"] == 4.58
    assert vector["macro_rate_2y"] == 4.25
    assert vector["macro_sofr"] == 4.49
    assert vector["macro_ig_oas"] == 1.02
    assert vector["macro_hy_oas"] == 3.44
    assert vector["macro_vix"] == 18.2


def test_standardized_feature_vector_supports_canonical_contract_feature_order():
    payload = {
        "version": "causal_contract_test_v1",
        "feature_order": [
            "scale.market_cap",
            "capital.net_leverage",
            "macro.ust_10y_yield",
            "action.size_absolute_usd",
        ],
        "feature_stats": {
            "scale.market_cap": {"mean": 0.0, "std": 1.0, "median": 0.0},
            "capital.net_leverage": {"mean": 0.0, "std": 1.0, "median": 0.0},
            "macro.ust_10y_yield": {"mean": 0.0, "std": 1.0, "median": 0.0},
            "action.size_absolute_usd": {"mean": 0.0, "std": 1.0, "median": 0.0},
        },
        "feature_transform_spec": {
            "usd_millions_features": ["scale.market_cap", "action.size_absolute_usd"],
            "rate_percent_features": ["macro.ust_10y_yield"],
            "oas_percent_features": [],
            "signed_log1p_features": [],
        },
        "objectives": {},
    }
    model = CausalImpactModel(payload)
    vector = model._standardized_feature_vector(
        params={"size_pct_market_cap": 0.05},
        features={
            "features": {
                "market.market_cap": {"value": 2_000_000_000.0, "support_mode": "exact"},
                "capital_structure.net_leverage": {"value": 2.4, "support_mode": "exact"},
                "macro.ust_10y_yield": {"value": 4.58, "support_mode": "exact"},
            }
        },
        regime={"credit_regime": "neutral", "vol_regime": "normal"},
    )

    assert vector is not None
    assert vector["scale.market_cap"] > 0.0
    assert vector["capital.net_leverage"] == 2.4
    assert vector["macro.ust_10y_yield"] == 4.58
    assert vector["action.size_absolute_usd"] > 0.0


def test_standardized_feature_vector_supports_retirement_aware_contract_features():
    payload = {
        "version": "causal_retirement_contract_test_v1",
        "feature_order": [
            "capital.net_pension_liability",
            "capital.combined_retirement_liability",
            "capital.net_debt_including_retirement",
            "capital.net_leverage_including_retirement",
            "capital.retirement_regime_combined_retirement_only",
            "capital.retirement_regime_pension_exact",
        ],
        "feature_stats": {
            "capital.net_pension_liability": {"mean": 0.0, "std": 1.0, "median": 0.0},
            "capital.combined_retirement_liability": {"mean": 0.0, "std": 1.0, "median": 0.0},
            "capital.net_debt_including_retirement": {"mean": 0.0, "std": 1.0, "median": 0.0},
            "capital.net_leverage_including_retirement": {"mean": 0.0, "std": 1.0, "median": 0.0},
            "capital.retirement_regime_combined_retirement_only": {"mean": 0.0, "std": 1.0, "median": 0.0},
            "capital.retirement_regime_pension_exact": {"mean": 0.0, "std": 1.0, "median": 0.0},
        },
        "feature_transform_spec": {
            "usd_millions_features": [
                "capital.net_pension_liability",
                "capital.combined_retirement_liability",
                "capital.net_debt_including_retirement",
            ],
            "rate_percent_features": [],
            "oas_percent_features": [],
            "signed_log1p_features": [],
        },
        "objectives": {},
    }
    model = CausalImpactModel(payload)
    vector = model._standardized_feature_vector(
        params={},
        features={
            "features": {
                "capital_structure.net_pension_liability": {"value": 120_000_000.0, "support_mode": "exact"},
                "capital_structure.combined_retirement_liability": {
                    "value": 150_000_000.0,
                    "support_mode": "exact",
                },
                "capital_structure.net_debt_including_retirement": {
                    "value": 900_000_000.0,
                    "support_mode": "proxy_missing_component",
                },
                "capital_structure.net_leverage_including_retirement": {
                    "value": 2.7,
                    "support_mode": "proxy_missing_component",
                },
                "capital_structure.retirement_obligation_regime": {
                    "value": "combined_retirement_only",
                    "support_mode": "exact",
                },
            }
        },
        regime={"credit_regime": "neutral", "vol_regime": "normal"},
    )

    assert vector is not None
    assert vector["capital.net_pension_liability"] == 120.0
    assert vector["capital.combined_retirement_liability"] == 150.0
    assert vector["capital.net_debt_including_retirement"] == 900.0
    assert vector["capital.net_leverage_including_retirement"] == 2.7
    assert vector["capital.retirement_regime_combined_retirement_only"] == 1.0
    assert vector["capital.retirement_regime_pension_exact"] == 0.0
