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


