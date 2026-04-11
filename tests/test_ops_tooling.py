from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_causal_rescue_plan as rescue  # noqa: E402
import gate_recommendation_canary as gate  # noqa: E402
import run_manual_replay_benchmark as replay  # noqa: E402
import run_recommendation_prod as prod  # noqa: E402
import train_causal_impact_model as train  # noqa: E402
import train_causal_rescue_model as rescue_train  # noqa: E402


def test_build_causal_rescue_plan_adds_low_row_blocklist_entries():
    audit = {
        "actions": [
            {"action_id": "a.high", "rows": 200, "causal_rate": 0.0, "strict_pass_rate": 0.0},
            {"action_id": "a.low", "rows": 60, "causal_rate": 0.0, "strict_pass_rate": 0.0},
            {"action_id": "a.healthy", "rows": 300, "causal_rate": 1.0, "strict_pass_rate": 1.0},
        ]
    }
    out = rescue.build_causal_rescue_plan(
        audit=audit,
        strict_pass_threshold=0.5,
        min_action_rows=100,
        low_row_blocklist_threshold=50,
    )
    assert [row["action_id"] for row in out["rescue_actions"]] == ["a.high"]
    assert [row["action_id"] for row in out["low_row_blocklist_actions"]] == ["a.low"]
    assert out["suggested_blocklist"] == ["a.high", "a.low"]


def test_train_action_allowlist_helpers():
    allowlist = train._parse_action_id_allowlist("capital_return.*,mna.platform_acquisition", "")
    assert train._matches_action_allowlist("capital_return.special_dividend", allowlist) is True
    assert train._matches_action_allowlist("mna.platform_acquisition", allowlist) is True
    assert train._matches_action_allowlist("capital_structure.refinancing", allowlist) is False


def test_build_targets_tolerates_missing_rating_columns():
    import pandas as pd

    df = pd.DataFrame(
        {
            "outcome_pe_12m": [1.0, 2.0],
            "leverage_delta": [0.1, -0.1],
            "fcf_margin_delta": [0.2, 0.3],
            "credit_spread_change_12m": [0.0, 0.1],
            "revenue_delta": [0.4, 0.5],
            "margin_delta": [0.2, 0.1],
            "eps_delta": [0.3, 0.2],
            "roic_delta": [0.1, 0.2],
        }
    )

    targets = train._build_targets(df)

    assert set(targets.keys()) == {
        "value_creation",
        "risk_reduction",
        "growth",
        "rating_preservation",
        "optionality",
    }
    assert all(len(series) == len(df) for series in targets.values())


def test_run_production_batch_applies_runtime_env_and_writes_run_ids(tmp_path: Path):
    snapshot_root = tmp_path / "snapshots"
    keyed = snapshot_root / "keyed" / "as_of_date=2026-02-28"
    keyed.mkdir(parents=True, exist_ok=True)
    (keyed / "company_id=0000320193.json").write_text(json.dumps({"snapshot_id": "s1"}) + "\n")

    args = argparse.Namespace(
        runs_root=str(tmp_path / "runs"),
        snapshot_root=str(snapshot_root),
        entity_graph_path=str(tmp_path / "entity_graph.parquet"),
        entity_identifier_path=str(tmp_path / "entity_identifier.parquet"),
        outcomes_path=str(tmp_path / "outcomes.parquet"),
        config_path=None,
        as_of="2026-02-28",
        companies=["0000320193"],
        action_ids=None,
        max_candidates=300,
        min_candidates_target=300,
        precedent_top_k=25,
        top_plans=1,
        strict_evidence=False,
        heartbeat_seconds=0.0,
        run_ids_out=str(tmp_path / "run_ids.txt"),
        summary_out="",
        causal_model_path=str(tmp_path / "model.json"),
        causal_routing_config_path=str(tmp_path / "routing.json"),
        causal_action_blocklist_path=str(tmp_path / "blocklist.txt"),
        causal_impact_mode="blend",
        causal_min_objective_oos_r2=0.08,
        causal_strict_quality_floor=0.10,
        causal_strict_support_floor=0.35,
        causal_strict_min_train_rows=1000,
        causal_strict_min_oos_r2=0.0,
        causal_strict_min_treated_rows=1500,
        causal_strict_min_control_rows=20000,
        precedent_workers=2,
    )

    def _stub_create_and_execute(**kwargs):
        return {
            "ok": True,
            "run_id": "run-123",
            "status": "completed",
            "counts": {"candidates": 1, "feasible": 1, "precedent": 1, "plans": 1},
        }

    out = prod.run_production_batch(args, create_and_execute_fn=_stub_create_and_execute)
    assert out["ok"] is True
    assert Path(args.run_ids_out).read_text().strip() == "0000320193 run-123"
    assert out["runtime_env"]["CAUSAL_IMPACT_MODEL_PATH"] == str(tmp_path / "model.json")
    assert out["runtime_env"]["CAUSAL_ROUTING_CONFIG_PATH"] == str(tmp_path / "routing.json")
    assert out["runtime_env"]["RECO_PRECEDENT_WORKERS"] == "2"


