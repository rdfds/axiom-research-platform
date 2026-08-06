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


def test_strict_causal_gate_honors_action_specific_threshold_overrides():
    brain = _brain()
    ok, reason = brain._passes_strict_causal_gate(
        {
            "model_quality": 0.12,
            "support_score": 0.40,
            "n_train": 2500,
            "out_of_sample_flag": False,
            "min_oos_r2": 0.08,
            "min_treated_rows": 900,
            "min_control_rows": 7000,
            "min_treated_rows_override": 750,
            "quality_floor_override": 0.11,
        }
    )
    assert ok is True
    assert reason == "pass"


def test_causal_action_blocklist_env_supports_exact_and_prefix(monkeypatch):
    monkeypatch.setenv(
        "CAUSAL_ACTION_BLOCKLIST",
        "capital_return.special_dividend,mna.*",
    )
    brain = _brain()
    assert brain._is_causal_action_blocked(
        action_id="capital_return.special_dividend",
        action_type="capital_return",
        action_subtype="special_dividend",
    )
    assert brain._is_causal_action_blocked(
        action_id="mna.platform_acquisition",
        action_type="mna",
        action_subtype="platform_acquisition",
    )
    assert not brain._is_causal_action_blocked(
        action_id="capital_return.dividend_increase",
        action_type="capital_return",
        action_subtype="dividend_increase",
    )


def test_causal_action_blocklist_path_supports_comments(tmp_path, monkeypatch):
    blocklist = tmp_path / "causal_action_blocklist.txt"
    blocklist.write_text(
        "\n".join(
            [
                "# blocked actions",
                "capital_structure.revolver_draw_or_resize",
                "governance.board_refresh # keep on fallback",
            ]
        )
    )
    monkeypatch.delenv("CAUSAL_ACTION_BLOCKLIST", raising=False)
    monkeypatch.setenv("CAUSAL_ACTION_BLOCKLIST_PATH", str(blocklist))
    brain = _brain()
    assert brain._is_causal_action_blocked(
        action_id="capital_structure.revolver_draw_or_resize",
        action_type="capital_structure",
        action_subtype="revolver_draw_or_resize",
    )
    assert brain._is_causal_action_blocked(
        action_id="governance.board_refresh",
        action_type="governance",
        action_subtype="board_refresh",
    )
    assert not brain._is_causal_action_blocked(
        action_id="capital_return.dividend_increase",
        action_type="capital_return",
        action_subtype="dividend_increase",
    )


def test_predict_causal_impact_skips_model_call_for_blocked_action(monkeypatch):
    monkeypatch.setenv("CAUSAL_ACTION_BLOCKLIST", "capital_return.special_dividend")
    stub = _StubCausalModel()
    brain = MechanismBrain(
        action_registry=_DummyRegistry(),
        causal_model=stub,
    )
    out = brain._predict_causal_impact(
        action_id="capital_return.special_dividend",
        action_type="capital_return",
        action_subtype="special_dividend",
        params={},
        features={},
        regime={},
    )
    assert out is None
    assert stub.calls == 0


def test_predict_causal_impact_skips_model_call_for_global_block(monkeypatch):
    monkeypatch.setenv("CAUSAL_ACTION_BLOCKLIST", "*")
    stub = _StubCausalModel()
    brain = MechanismBrain(
        action_registry=_DummyRegistry(),
        causal_model=stub,
    )
    out = brain._predict_causal_impact(
        action_id="capital_structure.equity_issuance",
        action_type="capital_structure",
        action_subtype="equity_issuance",
        params={},
        features={},
        regime={},
    )
    assert out is None
    assert stub.calls == 0


def test_causal_action_blocked_by_routing_config(tmp_path, monkeypatch):
    routing_path = tmp_path / "causal_capital_routing_v1.json"
    routing_path.write_text(
        json.dumps(
            {
                "actions": {
                    "governance.stock_split": {
                        "status": "blocked",
                        "model_action_alias": "cost_program",
                        "model_subtype_alias": "stock_split",
                    }
                }
            }
        )
    )
    monkeypatch.setenv("CAUSAL_ROUTING_CONFIG_PATH", str(routing_path))
    load_causal_routing_config.cache_clear()

    stub = _StubCausalModel()
    brain = MechanismBrain(
        action_registry=_DummyRegistry(),
        causal_model=stub,
    )
    out = brain._predict_causal_impact(
        action_id="governance.stock_split",
        action_type="governance",
        action_subtype="stock_split",
        params={},
        features={},
        regime={},
    )
    assert out is None
    assert stub.calls == 0

    load_causal_routing_config.cache_clear()
