from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.pipeline.types import ImpactDistribution, PrecedentPack
from src.recommendation_run import RecommendationRunStore, create_recommendation_run
from src.recommendation_run_orchestrator import (
    _select_precedent_candidates,
    create_and_execute_recommendation_run,
    execute_recommendation_run,
)


def _write_entity_files(tmp_path: Path) -> tuple[Path, Path]:
    entity_graph = tmp_path / "entity_graph.parquet"
    entity_identifier = tmp_path / "entity_identifier.parquet"

    pd.DataFrame(
        [
            {
                "entity_id": "0000320193",
                "related_id": "001690",
                "valid_from": "2001-01-01T00:00:00Z",
                "effective_at": "2001-01-01T00:00:00Z",
                "published_at": "2001-01-01T00:00:00Z",
                "ingested_at": "2001-01-01T00:00:00Z",
            }
        ]
    ).to_parquet(entity_graph, index=False)

    pd.DataFrame(
        [
            {
                "entity_id": "0000320193",
                "identifier_value": "001690",
            }
        ]
    ).to_parquet(entity_identifier, index=False)

    return entity_graph, entity_identifier


def _write_keyed_snapshot(tmp_path: Path, as_of: str = "2026-02-28") -> Path:
    root = tmp_path / "snapshots"
    keyed = root / "keyed" / f"as_of_date={as_of}"
    keyed.mkdir(parents=True, exist_ok=True)

    row = {
        "snapshot_id": "snap-123",
        "company_id": "0000320193",
        "as_of_time": f"{as_of}T00:00:00+00:00",
        "features": {
            "liquidity.available_for_actions": {"value": 100.0},
            "capital_structure.net_leverage": {"value": 2.0},
            "market.market_cap": {"value": 1000.0},
            "liquidity.cash": {"value": 500.0},
        },
        "regime": {"credit_regime": "neutral"},
        "provenance": {"computation_version": "state_builder_v5"},
    }
    (keyed / "company_id=0000320193.json").write_text(json.dumps(row) + "\n")
    return root


def _stub_precedent_runner(**kwargs):
    action_id = kwargs.get("action_id")
    score_map = {
        "capital_return.open_market_buyback": 0.24,
        "capital_structure.refinancing": 0.12,
        "capital_structure.equity_issuance": -0.08,
    }
    p50 = score_map.get(str(action_id), 0.01)
    dist = ImpactDistribution(
        metric="outcome_pe_12m",
        horizon_months=12,
        p25=p50 - 0.1,
        p50=p50,
        p75=p50 + 0.1,
        n=25,
    )
    return PrecedentPack(
        matches=[{"action_id": str(action_id), "distance": 1.0}],
        distributions=[dist],
        mismatch_diagnostics={},
    )


def test_execute_recommendation_run_lifecycle_and_artifacts(tmp_path: Path):
    entity_graph, entity_identifier = _write_entity_files(tmp_path)
    snapshot_root = _write_keyed_snapshot(tmp_path)

    runs_root = tmp_path / "runs"
    store = RecommendationRunStore(root=runs_root)

    run_id = create_recommendation_run(
        company_id="001690",
        as_of_time="2026-02-28",
        run_store=store,
        snapshot_root=snapshot_root,
        entity_graph_path=entity_graph,
        entity_identifier_path=entity_identifier,
    )

    summary = execute_recommendation_run(
        run_id=run_id,
        runs_root=runs_root,
        snapshot_root=snapshot_root,
        entity_identifier_path=entity_identifier,
        action_ids=["capital_return.open_market_buyback", "capital_structure.refinancing"],
        precedent_runner=_stub_precedent_runner,
        precedent_top_k=2,
    )

    assert summary["ok"] is True
    assert summary["status"] == "completed"
    assert summary["counts"]["candidates"] == 2
    assert summary["counts"]["feasible"] >= 1
    assert summary["counts"]["plans"] >= 1

    run = store.get_run(run_id)
    assert run is not None
    assert run.status == "completed"

    artifact_map = run.metadata.get("artifacts", {})
    for name in ["CandidateSet", "FeasibilityResults", "CausalModelRiskReport", "PrecedentMatches", "PrecedentIndex", "PlanSet", "BoardReadyDossier", "RecommendationPackage"]:
        assert name in artifact_map
        assert Path(artifact_map[name]).exists()

    precedent_payload = json.loads(Path(artifact_map["PrecedentMatches"]).read_text())
    assert precedent_payload.get("results")
    first = precedent_payload["results"][0]
    cand = first.get("candidate", {})
    impact = cand.get("impact_distribution", {})
    assert isinstance(impact, dict)
    assert impact.get("blend_metadata", {}).get("source") == "precedent_distribution"

    dossier_payload = json.loads(Path(artifact_map["BoardReadyDossier"]).read_text())
    assert dossier_payload["executive_summary"]
    assert dossier_payload["recommendation_thesis"]["problem_statement"]
    assert dossier_payload["recommendation_thesis"]["why_now"]
    assert dossier_payload["risk_case"]["kill_criteria"]

    recommendation_payload = json.loads(Path(artifact_map["RecommendationPackage"]).read_text())
    assert recommendation_payload["recommended_posture"] in {"act_now", "conditional_action", "wait"}
    assert recommendation_payload["status_quo_view"]
    assert recommendation_payload["sizing_guidance"]
    assert recommendation_payload["parameter_optimization"]
    assert recommendation_payload["regret_analysis"]
    assert recommendation_payload["rating_cliff_analysis"]
    assert recommendation_payload["signaling_analysis"]
    assert recommendation_payload["top_plan"]
    assert recommendation_payload["ranked_action_views"]
    assert recommendation_payload["plans_preview"]
    assert recommendation_payload["top_plan_summary_explanation"]
    assert recommendation_payload["board_ready_dossier"]["executive_summary"]
    assert isinstance(recommendation_payload["monitoring_triggers"], list)
    assert isinstance(recommendation_payload["contingency_branches"], list)
    assert recommendation_payload["summary"]["top_plan_action_ids"]

    event_types = [e.event_type for e in run.audit_log]
    assert "candidate_generation_started" in event_types
    assert "candidate_generation_completed" in event_types
    assert "feasibility_eval_started" in event_types
    assert "feasibility_eval_completed" in event_types
    assert "precedent_retrieval_started" in event_types
    assert "precedent_retrieval_completed" in event_types
    assert "planning_started" in event_types
    assert "planning_completed" in event_types
    assert "run_completed" in event_types


def test_execute_recommendation_run_hard_constraint_gates_equity(tmp_path: Path):
    entity_graph, entity_identifier = _write_entity_files(tmp_path)
    snapshot_root = _write_keyed_snapshot(tmp_path)

    runs_root = tmp_path / "runs"
    store = RecommendationRunStore(root=runs_root)

    run_id = create_recommendation_run(
        company_id="001690",
        as_of_time="2026-02-28",
        constraints={
            "hard_constraints": [
                {
                    "constraint_type": "no_equity_issuance",
                    "parameters": {},
                    "source": "user_input",
                    "priority": "hard",
                }
            ],
            "soft_constraints": [],
        },
        run_store=store,
        snapshot_root=snapshot_root,
        entity_graph_path=entity_graph,
        entity_identifier_path=entity_identifier,
    )

    execute_recommendation_run(
        run_id=run_id,
        runs_root=runs_root,
        snapshot_root=snapshot_root,
        entity_identifier_path=entity_identifier,
        action_ids=["capital_structure.equity_issuance", "capital_return.open_market_buyback"],
        precedent_runner=_stub_precedent_runner,
    )

    run = store.get_run(run_id)
    assert run is not None
    feas_path = Path(run.metadata["artifacts"]["FeasibilityResults"])
    feas = json.loads(feas_path.read_text())

    by_action = {r["candidate"]["action_id"]: r for r in feas["results"]}
    assert by_action["capital_structure.equity_issuance"]["feasible"] is False
    assert any("no_equity_issuance" in x for x in by_action["capital_structure.equity_issuance"]["hard_constraint_violations"])


def test_execute_recommendation_run_fails_on_snapshot_hash_mismatch(tmp_path: Path):
    entity_graph, entity_identifier = _write_entity_files(tmp_path)
    snapshot_root = _write_keyed_snapshot(tmp_path)

    runs_root = tmp_path / "runs"
    store = RecommendationRunStore(root=runs_root)

    run_id = create_recommendation_run(
        company_id="001690",
        as_of_time="2026-02-28",
        run_store=store,
        snapshot_root=snapshot_root,
        entity_graph_path=entity_graph,
        entity_identifier_path=entity_identifier,
    )

    # mutate snapshot after run creation to force hash mismatch
    p = snapshot_root / "keyed" / "as_of_date=2026-02-28" / "company_id=0000320193.json"
    row = json.loads(p.read_text().strip())
    row["features"]["liquidity.available_for_actions"]["value"] = 999.0
    p.write_text(json.dumps(row) + "\n")

    with pytest.raises(ValueError, match="Frozen snapshot hash mismatch"):
        execute_recommendation_run(
            run_id=run_id,
            runs_root=runs_root,
            snapshot_root=snapshot_root,
            entity_identifier_path=entity_identifier,
            action_ids=["capital_return.open_market_buyback"],
            precedent_runner=_stub_precedent_runner,
        )

    run = store.get_run(run_id)
    assert run is not None
    assert run.status == "failed"
    assert any(e.event_type == "run_failed" for e in run.audit_log)


def test_create_and_execute_recommendation_run_one_shot(tmp_path: Path):
    entity_graph, entity_identifier = _write_entity_files(tmp_path)
    snapshot_root = _write_keyed_snapshot(tmp_path)
    runs_root = tmp_path / "runs"

    summary = create_and_execute_recommendation_run(
        company_id="001690",
        as_of_time="2026-02-28",
        runs_root=runs_root,
        snapshot_root=snapshot_root,
        entity_graph_path=entity_graph,
        entity_identifier_path=entity_identifier,
        action_ids=["capital_return.open_market_buyback"],
        precedent_runner=_stub_precedent_runner,
        top_plans=1,
    )

    assert summary["ok"] is True
    assert summary["status"] == "completed"
    assert summary["counts"]["candidates"] == 1
    assert summary["counts"]["feasible"] == 1
    assert summary["counts"]["plans"] == 1

    run_id = summary["run_id"]
    store = RecommendationRunStore(root=runs_root)
    run = store.get_run(run_id)
    assert run is not None
    assert run.status == "completed"
    assert any(e.event_type == "run_completed" for e in run.audit_log)


def test_execute_recommendation_run_accepts_snapshot_loader(tmp_path: Path):
    entity_graph, entity_identifier = _write_entity_files(tmp_path)
    snapshot_root = _write_keyed_snapshot(tmp_path)
    runs_root = tmp_path / "runs"
    store = RecommendationRunStore(root=runs_root)

    run_id = create_recommendation_run(
        company_id="001690",
        as_of_time="2026-02-28",
        run_store=store,
        snapshot_root=snapshot_root,
        entity_graph_path=entity_graph,
        entity_identifier_path=entity_identifier,
    )

    calls = {"n": 0}

    def loader(company_id: str, as_of_time):
        calls["n"] += 1
        p = snapshot_root / "keyed" / "as_of_date=2026-02-28" / "company_id=0000320193.json"
        return json.loads(p.read_text().strip())

    summary = execute_recommendation_run(
        run_id=run_id,
        runs_root=runs_root,
        snapshot_loader=loader,
        entity_identifier_path=entity_identifier,
        action_ids=["capital_return.open_market_buyback"],
        precedent_runner=_stub_precedent_runner,
        top_plans=1,
    )

    assert summary["ok"] is True
    assert summary["status"] == "completed"
    assert calls["n"] >= 1


def test_execute_recommendation_run_precedent_top_k_limits_calls(tmp_path: Path):
    entity_graph, entity_identifier = _write_entity_files(tmp_path)
    snapshot_root = _write_keyed_snapshot(tmp_path)
    runs_root = tmp_path / "runs"
    store = RecommendationRunStore(root=runs_root)

    run_id = create_recommendation_run(
        company_id="001690",
        as_of_time="2026-02-28",
        run_store=store,
        snapshot_root=snapshot_root,
        entity_graph_path=entity_graph,
        entity_identifier_path=entity_identifier,
    )

    summary = execute_recommendation_run(
        run_id=run_id,
        runs_root=runs_root,
        snapshot_root=snapshot_root,
        entity_identifier_path=entity_identifier,
        action_ids=["capital_return.open_market_buyback", "capital_structure.refinancing"],
        precedent_runner=_stub_precedent_runner,
        precedent_top_k=1,
        top_plans=1,
    )

    assert summary["ok"] is True
    assert summary["counts"]["candidates"] == 2
    assert summary["counts"]["feasible"] >= 1
    assert summary["counts"]["precedent"] == 1


def test_execute_recommendation_run_uses_runtime_feature_adapter_end_to_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("AXIOM_ENABLE_RUNTIME_FEATURE_ADAPTER", "1")
    monkeypatch.setenv(
        "AXIOM_RUNTIME_FEATURE_ADAPTER_RULES",
        "normalized_available_liquidity,normalized_net_debt,normalized_net_leverage,normalized_operating_earnings_fill,pe_ratio_compatibility_alias",
    )
    entity_graph, entity_identifier = _write_entity_files(tmp_path)
    root = tmp_path / "snapshots"
    keyed = root / "keyed" / "as_of_date=2026-02-28"
    keyed.mkdir(parents=True, exist_ok=True)
    row = {
        "snapshot_id": "snap-456",
        "company_id": "0000320193",
        "as_of_time": "2026-02-28T00:00:00+00:00",
        "features": {
            "liquidity.available_liquidity_normalized": {"value": 250.0, "support_mode": "exact"},
            "capital_structure.net_debt_normalized": {"value": 100.0, "support_mode": "exact"},
            "capital_structure.net_leverage_normalized": {"value": 1.5, "support_mode": "exact"},
            "operating.operating_earnings_normalized": {"value": 80.0, "support_mode": "exact"},
            "market.market_cap": {"value": 1000.0},
            "market.pe_ratio": {"value": 12.0, "support_mode": "exact"},
            "liquidity.cash": {"value": 500.0},
        },
        "regime": {"credit_regime": "neutral"},
        "provenance": {"computation_version": "state_builder_v5"},
    }
    (keyed / "company_id=0000320193.json").write_text(json.dumps(row) + "\n")

    runs_root = tmp_path / "runs"
    store = RecommendationRunStore(root=runs_root)
    run_id = create_recommendation_run(
        company_id="001690",
        as_of_time="2026-02-28",
        run_store=store,
        snapshot_root=root,
        entity_graph_path=entity_graph,
        entity_identifier_path=entity_identifier,
    )

    summary = execute_recommendation_run(
        run_id=run_id,
        runs_root=runs_root,
        snapshot_root=root,
        entity_identifier_path=entity_identifier,
        action_ids=["capital_return.open_market_buyback"],
        precedent_runner=_stub_precedent_runner,
    )

    assert summary["ok"] is True
    assert summary["runtime_feature_adapter"]["replacement_count"] >= 3

    run = store.get_run(run_id)
    assert run is not None
    artifact_map = run.metadata.get("artifacts", {})
    adapter_payload = json.loads(Path(artifact_map["RuntimeFeatureAdapterDiagnostics"]).read_text())
    bundle_payload = json.loads(Path(artifact_map["ModelFeatureBundleDiagnostics"]).read_text())
    assert adapter_payload["diagnostics"]["counts_by_target"]["capital_structure.net_leverage"] == 1
    assert "liquidity.available_for_actions" not in adapter_payload["diagnostics"]["counts_by_target"]
    assert bundle_payload["diagnostics"]["canonical_count"] > 0

    dossier_payload = json.loads(Path(artifact_map["BoardReadyDossier"]).read_text())
    if "supporting_evidence" in dossier_payload:
        evidence_by_metric = {row["metric"]: row for row in dossier_payload["supporting_evidence"]}
        assert evidence_by_metric["liquidity.available_for_actions"]["value"] is None
        assert evidence_by_metric["capital_structure.net_leverage"]["value"] == 1.5


def test_select_precedent_candidates_zero_top_k_skips_retrieval():
    feasible = [
        {
            "candidate_id": "c1",
            "action_id": "capital_return.open_market_buyback",
            "generation_confidence": 0.9,
            "evaluation_confidence": 0.7,
        }
    ]
    assert _select_precedent_candidates(feasible, precedent_top_k=0) == []


def test_execute_recommendation_run_persists_execution_config(tmp_path: Path, monkeypatch):
    entity_graph, entity_identifier = _write_entity_files(tmp_path)
    snapshot_root = _write_keyed_snapshot(tmp_path)
    runs_root = tmp_path / "runs"
    store = RecommendationRunStore(root=runs_root)

    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps({"version": "causal_impact_model_v2_hybrid"}))
    blocklist_path = tmp_path / "blocklist.txt"
    blocklist_path.write_text("mna.platform_acquisition\n")

    monkeypatch.setenv("CAUSAL_IMPACT_MODEL_PATH", str(model_path))
    monkeypatch.setenv("CAUSAL_ACTION_BLOCKLIST_PATH", str(blocklist_path))
    monkeypatch.setenv("CAUSAL_STRICT_MIN_CONTROL_ROWS", "20000")

    run_id = create_recommendation_run(
        company_id="001690",
        as_of_time="2026-02-28",
        run_store=store,
        snapshot_root=snapshot_root,
        entity_graph_path=entity_graph,
        entity_identifier_path=entity_identifier,
    )

    execute_recommendation_run(
        run_id=run_id,
        runs_root=runs_root,
        snapshot_root=snapshot_root,
        entity_identifier_path=entity_identifier,
        action_ids=["capital_return.open_market_buyback"],
        max_candidates=50,
        min_candidates_target=50,
        precedent_top_k=5,
        top_plans=1,
        precedent_runner=_stub_precedent_runner,
    )

    run = store.get_run(run_id)
    assert run is not None
    cfg = dict((run.metadata or {}).get("config") or {})
    assert cfg["execution"]["max_candidates"] == 50
    assert cfg["execution"]["precedent_top_k"] == 5
    assert cfg["runtime_env"]["causal"]["model"]["path"] == str(model_path)
    assert "mna.platform_acquisition" in cfg["runtime_env"]["causal"]["blocklist"]["entries"]


def test_select_precedent_candidates_prioritizes_strict_causal_rows():
    feasible = [
        {
            "candidate_id": "c_low",
            "action_id": "capital_return.dividend_increase",
            "generation_confidence": 0.95,
            "evaluation_confidence": 0.60,
            "impact_distribution": {"key_drivers": []},
        },
        {
            "candidate_id": "c_high",
            "action_id": "capital_return.open_market_buyback",
            "generation_confidence": 0.40,
            "evaluation_confidence": 0.70,
            "impact_distribution": {
                "key_drivers": [
                    {"driver_name": "causal_model_mode", "contribution": 1.0},
                    {"driver_name": "causal_model_quality", "contribution": 0.25},
                    {"driver_name": "causal_model_support_score", "contribution": 0.80},
                ]
            },
        },
    ]
    selected = _select_precedent_candidates(feasible, precedent_top_k=1)
    assert len(selected) == 1
    assert selected[0]["candidate_id"] == "c_high"
