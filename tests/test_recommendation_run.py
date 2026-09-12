from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from src.recommendation_run import (
    Constraint,
    ConstraintSet,
    DataCutoffSpec,
    RecommendationRunStore,
    _json_sanitize,
    create_recommendation_run,
    enforce_data_cutoff,
    validate_plan_hard_constraints,
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


def _write_keyed_snapshot(tmp_path: Path, company_id: str = "0000320193", as_of: str = "2026-02-28") -> Path:
    root = tmp_path / "snapshots"
    keyed = root / "keyed" / f"as_of_date={as_of}"
    keyed.mkdir(parents=True, exist_ok=True)

    row = {
        "snapshot_id": "snap-123",
        "company_id": company_id,
        "as_of_time": f"{as_of}T00:00:00+00:00",
        "features": {
            "liquidity.available_for_actions": {"value": 100.0},
            "capital_structure.net_leverage": {"value": 2.0},
            "market.market_cap": {"value": 1000.0},
        },
        "regime": {"credit_regime": "neutral"},
        "provenance": {"computation_version": "state_builder_v5"},
    }

    (keyed / f"company_id={company_id}.json").write_text(json.dumps(row) + "\n")
    return root


def _snapshot_hash(snapshot: dict) -> str:
    txt = json.dumps(snapshot, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(txt.encode("utf-8")).hexdigest()


def test_create_recommendation_run_normalizes_objectives_and_freezes_snapshot(tmp_path: Path):
    entity_graph, entity_identifier = _write_entity_files(tmp_path)
    snapshot_root = _write_keyed_snapshot(tmp_path)
    runs_root = tmp_path / "runs"

    run_id = create_recommendation_run(
        company_id="001690",
        as_of_time="2026-02-28",
        objectives={
            "value_creation_weight": 2.0,
            "risk_reduction_weight": 1.0,
            "growth_weight": 1.0,
            "rating_preservation_weight": 0.0,
            "optionality_weight": 0.0,
        },
        run_store=RecommendationRunStore(root=runs_root),
        snapshot_root=snapshot_root,
        entity_graph_path=entity_graph,
        entity_identifier_path=entity_identifier,
        planner_random_seed=7,
    )

    store = RecommendationRunStore(root=runs_root)
    run = store.get_run(run_id)
    assert run is not None
    assert run.company_id == "001690"
    assert run.status == "initialized"
    assert abs(
        run.objectives.value_creation_weight
        + run.objectives.risk_reduction_weight
        + run.objectives.growth_weight
        + run.objectives.rating_preservation_weight
        + run.objectives.optionality_weight
        - 1.0
    ) < 1e-12
    assert run.data_cutoff.published_at_lte == run.as_of_time
    assert run.data_cutoff.ingested_at_lte == run.as_of_time

    raw_snapshot = json.loads(
        (snapshot_root / "keyed" / "as_of_date=2026-02-28" / "company_id=0000320193.json").read_text().strip()
    )
    assert run.frozen_state.snapshot_hash == _snapshot_hash(raw_snapshot)
    assert run.frozen_state.snapshot_version == "state_builder_v5"

    event_types = [e.event_type for e in run.audit_log]
    assert event_types == ["run_created", "snapshot_frozen"]


def test_create_recommendation_run_persists_runtime_config_metadata(tmp_path: Path, monkeypatch):
    entity_graph, entity_identifier = _write_entity_files(tmp_path)
    snapshot_root = _write_keyed_snapshot(tmp_path)
    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps({"version": "causal_impact_model_v2"}))
    blocklist_path = tmp_path / "blocklist.txt"
    blocklist_path.write_text("capital_return.special_dividend\n")

    monkeypatch.setenv("CAUSAL_IMPACT_MODEL_PATH", str(model_path))
    monkeypatch.setenv("CAUSAL_ACTION_BLOCKLIST_PATH", str(blocklist_path))
    monkeypatch.setenv("CAUSAL_IMPACT_MODE", "None")
    monkeypatch.setenv("CAUSAL_ACTION_BLOCKLIST", "none")
    monkeypatch.setenv("CAUSAL_STRICT_QUALITY_FLOOR", "0.10")

    run_id = create_recommendation_run(
        company_id="001690",
        as_of_time="2026-02-28",
        run_store=RecommendationRunStore(root=tmp_path / "runs"),
        snapshot_root=snapshot_root,
        entity_graph_path=entity_graph,
        entity_identifier_path=entity_identifier,
        planner_random_seed=11,
    )

    run = RecommendationRunStore(root=tmp_path / "runs").get_run(run_id)
    assert run is not None
    cfg = dict((run.metadata or {}).get("config") or {})
    assert cfg["create"]["snapshot_root"] == str(snapshot_root)
    assert cfg["create"]["planner_random_seed"] == 11
    assert cfg["runtime_env"]["causal"]["model"]["path"] == str(model_path)
    assert cfg["runtime_env"]["causal"]["mode"] == "blend"
    assert "none" not in cfg["runtime_env"]["causal"]["blocklist"]["entries"]
    assert "capital_return.special_dividend" in cfg["runtime_env"]["causal"]["blocklist"]["entries"]


def test_snapshot_hash_immutable_after_creation(tmp_path: Path):
    entity_graph, entity_identifier = _write_entity_files(tmp_path)
    snapshot_root = _write_keyed_snapshot(tmp_path)
    store = RecommendationRunStore(root=tmp_path / "runs")

    run_id = create_recommendation_run(
        company_id="001690",
        as_of_time="2026-02-28",
        run_store=store,
        snapshot_root=snapshot_root,
        entity_graph_path=entity_graph,
        entity_identifier_path=entity_identifier,
    )
    run = store.get_run(run_id)
    assert run is not None

    run.frozen_state.snapshot_hash = "tampered"
    with pytest.raises(ValueError, match="snapshot_hash"):
        store.update_run(run)


def test_model_versions_are_immutable(tmp_path: Path):
    entity_graph, entity_identifier = _write_entity_files(tmp_path)
    snapshot_root = _write_keyed_snapshot(tmp_path)
    store = RecommendationRunStore(root=tmp_path / "runs")

    run_id = create_recommendation_run(
        company_id="001690",
        as_of_time="2026-02-28",
        run_store=store,
        snapshot_root=snapshot_root,
        entity_graph_path=entity_graph,
        entity_identifier_path=entity_identifier,
    )
    run = store.get_run(run_id)
    assert run is not None

    run.model_versions.planner_model_version = "planner_model_v99"
    with pytest.raises(ValueError, match="Model versions are immutable"):
        store.update_run(run)


def test_data_cutoff_filter_removes_forward_rows():
    cutoff = DataCutoffSpec(
        published_at_lte="2026-02-28T00:00:00+00:00",
        ingested_at_lte="2026-02-28T00:00:00+00:00",
    )
    df = pd.DataFrame(
        [
            {"x": 1, "published_at": "2026-02-27T00:00:00+00:00", "ingested_at": "2026-02-27T00:00:00+00:00"},
            {"x": 2, "published_at": "2026-03-01T00:00:00+00:00", "ingested_at": "2026-02-27T00:00:00+00:00"},
            {"x": 3, "published_at": "2026-02-27T00:00:00+00:00", "ingested_at": "2026-03-01T00:00:00+00:00"},
        ]
    )

    got = enforce_data_cutoff(df, cutoff)
    assert got["x"].tolist() == [1]


def test_hard_constraints_never_violated_in_plan():
    constraints = ConstraintSet(
        hard_constraints=[
            Constraint(
                constraint_type="no_equity_issuance",
                parameters={},
                source="user_input",
                priority="hard",
            ),
            Constraint(
                constraint_type="forbidden_action_type",
                parameters={"action_id": "capital_structure.equity_issuance"},
                source="user_input",
                priority="hard",
            ),
            Constraint(
                constraint_type="required_action_type",
                parameters={"action_id": "capital_structure.refinancing"},
                source="user_input",
                priority="hard",
            ),
        ]
    )

    violating_plan = [
        {"action_id": "capital_structure.equity_issuance", "action_type": "capital_structure", "params": {}},
        {"action_id": "capital_return.open_market_buyback", "action_type": "capital_return", "params": {}},
    ]
    violations = validate_plan_hard_constraints(violating_plan, constraints)
    assert len(violations) >= 2

    valid_plan = [
        {"action_id": "capital_structure.refinancing", "action_type": "capital_structure", "params": {}},
        {"action_id": "capital_return.open_market_buyback", "action_type": "capital_return", "params": {}},
    ]
    no_violations = validate_plan_hard_constraints(valid_plan, constraints)
    assert no_violations == []


def test_status_lifecycle_appends_audit_events(tmp_path: Path):
    entity_graph, entity_identifier = _write_entity_files(tmp_path)
    snapshot_root = _write_keyed_snapshot(tmp_path)
    store = RecommendationRunStore(root=tmp_path / "runs")

    run_id = create_recommendation_run(
        company_id="001690",
        as_of_time="2026-02-28",
        run_store=store,
        snapshot_root=snapshot_root,
        entity_graph_path=entity_graph,
        entity_identifier_path=entity_identifier,
    )

    store.transition_status(run_id, "candidate_generation")
    store.transition_status(run_id, "feasibility_evaluation")
    store.transition_status(run_id, "precedent_retrieval")
    store.transition_status(run_id, "plan_search")
    store.transition_status(run_id, "completed")

    run = store.get_run(run_id)
    assert run is not None
    assert run.status == "completed"

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


def test_as_of_must_not_be_before_earliest_company_data(tmp_path: Path):
    entity_graph, entity_identifier = _write_entity_files(tmp_path)
    snapshot_root = _write_keyed_snapshot(tmp_path)

    with pytest.raises(ValueError, match="earliest company data"):
        create_recommendation_run(
            company_id="001690",
            as_of_time="1999-01-01",
            run_store=RecommendationRunStore(root=tmp_path / "runs"),
            snapshot_root=snapshot_root,
            entity_graph_path=entity_graph,
            entity_identifier_path=entity_identifier,
        )


def test_create_recommendation_run_accepts_explicit_company_aliases_for_validation_and_snapshot_lookup(tmp_path: Path):
    entity_graph = tmp_path / "entity_graph.parquet"
    entity_identifier = tmp_path / "entity_identifier.parquet"
    pd.DataFrame(
        [
            {
                "entity_id": "resolved-exto",
                "related_id": "issuer-exto",
                "valid_from": "2001-01-01T00:00:00Z",
                "effective_at": "2001-01-01T00:00:00Z",
                "published_at": "2001-01-01T00:00:00Z",
                "ingested_at": "2001-01-01T00:00:00Z",
            }
        ]
    ).to_parquet(entity_graph, index=False)
    pd.DataFrame(
        [
            {"entity_id": "resolved-exto", "identifier_value": "EXTO"},
        ]
    ).to_parquet(entity_identifier, index=False)

    snapshot_root = _write_keyed_snapshot(tmp_path, company_id="EXTO")
    runs_root = tmp_path / "runs"

    run_id = create_recommendation_run(
        company_id="205876",
        company_aliases=["EXTO"],
        as_of_time="2026-02-28",
        run_store=RecommendationRunStore(root=runs_root),
        snapshot_root=snapshot_root,
        entity_graph_path=entity_graph,
        entity_identifier_path=entity_identifier,
    )

    run = RecommendationRunStore(root=runs_root).get_run(run_id)
    assert run is not None
    assert run.company_id == "205876"
    assert run.frozen_state.snapshot_hash == _snapshot_hash(
        json.loads((snapshot_root / "keyed" / "as_of_date=2026-02-28" / "company_id=EXTO.json").read_text().strip())
    )


def test_create_recommendation_run_allows_explicit_alias_snapshot_fallback_when_entity_graph_is_missing(tmp_path: Path):
    entity_graph = tmp_path / "entity_graph.parquet"
    entity_identifier = tmp_path / "entity_identifier.parquet"
    pd.DataFrame(
        columns=[
            "entity_id",
            "related_id",
            "valid_from",
            "effective_at",
            "published_at",
            "ingested_at",
        ]
    ).to_parquet(entity_graph, index=False)
    pd.DataFrame(columns=["entity_id", "identifier_value"]).to_parquet(entity_identifier, index=False)

    runs_root = tmp_path / "runs"

    def snapshot_loader(company_id: str, as_of_time):
        assert company_id == "205876"
        ts = pd.Timestamp(as_of_time)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return {
            "snapshot_id": "snap-exto-fallback",
            "company_id": "EXTO",
            "as_of_time": ts.isoformat(),
            "features": {
                "liquidity.available_for_actions": {"value": 150.0},
                "market.market_cap": {"value": 1000.0},
            },
            "regime": {"credit_regime": "neutral"},
            "provenance": {"computation_version": "state_builder_v5"},
        }

    run_id = create_recommendation_run(
        company_id="205876",
        company_aliases=["EXTO"],
        as_of_time="2026-02-28",
        run_store=RecommendationRunStore(root=runs_root),
        snapshot_loader=snapshot_loader,
        entity_graph_path=entity_graph,
        entity_identifier_path=entity_identifier,
    )

    run = RecommendationRunStore(root=runs_root).get_run(run_id)
    assert run is not None
    assert run.company_id == "205876"
    assert run.frozen_state.snapshot_version == "state_builder_v5"


def test_json_sanitize_handles_naive_pandas_timestamps():
    payload = {"ts": pd.Timestamp("2026-02-28 00:00:00")}
    out = _json_sanitize(payload)
    assert str(out["ts"]).endswith("+00:00")


def test_stage_path_is_unique_per_write(tmp_path: Path):
    store = RecommendationRunStore(root=tmp_path / "runs", temp_dir=tmp_path / "tmp")
    out = store.root / "run_index.parquet"
    staged_a = store._stage_path(out)
    staged_b = store._stage_path(out)

    assert staged_a != staged_b
    assert staged_a.parent.exists()
    assert staged_b.parent.exists()
