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


def test_resolve_locked_inputs_falls_back_from_stale_localized_bundle_paths(tmp_path: Path, monkeypatch):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"cases": []}))

    canonical_root = tmp_path / "canonical"
    canonical_root.mkdir()
    fallback_paths = {
        "outcomes_path": canonical_root / "action_outcomes_with_credit_ratings.normalized_full.parquet",
        "action_support_manifest": canonical_root / "action_data_support_manifest.json",
        "raw_timeseries_path": canonical_root / "raw_timeseries.parquet",
        "companyfacts_root": canonical_root / "companyfacts",
        "facts_path": canonical_root / "facts",
    }
    fallback_paths["outcomes_path"].write_text("stub")
    fallback_paths["action_support_manifest"].write_text("{}")
    fallback_paths["raw_timeseries_path"].write_text("stub")
    fallback_paths["companyfacts_root"].mkdir()
    fallback_paths["facts_path"].mkdir()

    stale_bundle = tmp_path / "out" / "manual_replay_bundle_20260405_localized" / "inputs"
    config = {
        "defaults": {},
        "artifacts": {
            "outcomes_path": str(tmp_path / "outcomes.parquet"),
            "action_support_manifest": str(stale_bundle / "action_data_support_manifest.json"),
            "corporate_actions_master_path": str(tmp_path / "missing_corporate_actions.parquet"),
            "entity_graph_path": str(tmp_path / "entity_graph.parquet"),
            "entity_identifier_path": str(tmp_path / "entity_identifier.parquet"),
            "entity_table_path": str(tmp_path / "entity.parquet"),
            "raw_timeseries_path": str(stale_bundle / "raw_timeseries.parquet"),
            "event_store_path": str(tmp_path / "event_store.parquet"),
            "ownership_summary_path": str(tmp_path / "ownership_13f_summary.parquet"),
            "issuer_ratings_path": str(tmp_path / "issuer_rating_history.parquet"),
            "companyfacts_root": str(stale_bundle / "companyfacts_buyback_20260407"),
            "facts_path_candidates": [
                str(stale_bundle / "facts_asof_2026"),
            ],
        },
        "benchmarks": {
            "buyback": {
                "manifest": str(manifest_path),
            }
        },
    }

    monkeypatch.setattr(replay, "_CANONICAL_LOCK_ARTIFACT_FALLBACKS", fallback_paths)

    locked = replay._resolve_locked_inputs(config, "buyback")
    resolved_paths = locked["resolved_paths"]

    assert resolved_paths["action_support_manifest"] == fallback_paths["action_support_manifest"]
    assert resolved_paths["outcomes_path"] == fallback_paths["outcomes_path"]
    assert resolved_paths["raw_timeseries_path"] == fallback_paths["raw_timeseries_path"]
    assert resolved_paths["companyfacts_root"] == fallback_paths["companyfacts_root"]
    assert resolved_paths["facts_path"] == fallback_paths["facts_path"]


def test_evaluate_canary_gate_enforces_thresholds():
    audit = {
        "runs_analyzed": 5,
        "status_counts": {"completed": 5, "failed": 0},
        "causal_summary": {
            "causal_rate": {"mean": 0.80},
            "strict_pass_rate_among_all": {"mean": 0.75},
            "strict_pass_rate_among_causal": {"mean": 0.95},
        },
        "precedent_summary": {
            "precedent_confidence_mean": {"mean": 0.40},
            "out_of_sample_rate": {"mean": 0.85},
        },
    }
    args = argparse.Namespace(
        min_causal_rate_mean=0.75,
        min_strict_all_mean=0.70,
        min_strict_causal_mean=0.90,
        min_precedent_conf_mean=0.35,
        max_precedent_oos_mean=0.90,
    )
    result = gate.evaluate_canary_gate(audit, args)
    assert result["gate_pass"] is True
    assert all(bool(c["pass"]) for c in result["checks"])


def test_train_causal_rescue_model_materializes_actions_and_builds_command(tmp_path: Path):
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "actions": [
                    {
                        "action_id": "capital_return.special_dividend",
                        "rows": 200,
                        "causal_rate": 0.0,
                        "strict_pass_rate": 0.0,
                    },
                    {
                        "action_id": "governance.board_refresh",
                        "rows": 300,
                        "causal_rate": 0.0,
                        "strict_pass_rate": 0.0,
                    },
                    {"action_id": "a.healthy", "rows": 500, "causal_rate": 1.0, "strict_pass_rate": 1.0},
                ]
            }
        )
    )
    args = argparse.Namespace(
        audit_json=str(audit_path),
        rescue_actions_file="",
        generated_rescue_actions_out=str(tmp_path / "rescue_actions.txt"),
        strict_pass_threshold=0.5,
        min_action_rows=100,
        low_row_blocklist_threshold=50,
        mapping_path="./config/causal_rescue_action_mapping.json",
        outcomes_path=str(tmp_path / "outcomes.parquet"),
        out_path=str(tmp_path / "model.json"),
        model_card_out=str(tmp_path / "model_card.json"),
        train_end_date="2023-12-31",
        validation_start_date="2024-01-01",
        model_family="hgb",
        cell_level="action_subtype",
        crossfit_folds=3,
        dr_min_treated_rows=1500,
        dr_min_control_rows=20000,
        min_validation_rows=300,
        propensity_clip=0.03,
        gate_min_oos_r2=0.0,
        gate_min_train_rows=8000,
        gate_min_treated_rows=1500,
        gate_min_control_rows=20000,
        progress_every_cells=0,
        quiet=False,
    )

    train_patterns, action_ids_path, recommendation_action_ids, unresolved, coverage = rescue_train.materialize_rescue_action_ids(args)
    assert train_patterns == ["dividend_special.*"]
    assert recommendation_action_ids == ["governance.board_refresh", "capital_return.special_dividend"]
    assert unresolved == ["governance.board_refresh"]
    assert coverage["capital_return.special_dividend"]["status"] == "mapped"
    assert coverage["governance.board_refresh"]["status"] == "unsupported"
    assert action_ids_path.read_text().strip() == "dividend_special.*"

    cmd = rescue_train.build_rescue_train_command(args, action_ids_path)
    assert "--action-id-allowlist-file" in cmd
    assert str(action_ids_path) in cmd
    assert "--subtype-target-normalize" in cmd
