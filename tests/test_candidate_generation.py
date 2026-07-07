from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.action_ontology import build_default_action_schema_registry
from src.candidate_generation import CandidateGenerationEngine, _feature_value
from src.recommendation_run import RecommendationRunStore, create_recommendation_run


def _candidate_action_ids(out: dict) -> list[str]:
    return [row["action_id"] for row in out["candidates"]]


def test_feature_value_blocks_unsupported_metric_inputs():
    features = {
        "capital_structure.net_leverage": {
            "value": 6.0,
            "support_mode": "unsupported",
            "applicability_status": "unsupported",
            "quality_flags": ["unsupported_metric"],
        }
    }
    assert _feature_value(features, "capital_structure.net_leverage") is None
    assert _feature_value(features, "capital_structure.net_leverage", default=1.5) == 1.5


def test_feature_value_prefers_fixed_charge_for_lease_heavy_coverage():
    features = {
        "capital_structure.interest_coverage": {
            "value": 5.0,
            "support_mode": "exact",
            "applicability_status": "secondary",
        },
        "capital_structure.fixed_charge_coverage": {
            "value": 2.25,
            "support_mode": "exact",
            "applicability_status": "primary",
        },
    }
    assert _feature_value(features, "capital_structure.interest_coverage") == 2.25


def test_feature_value_only_applies_global_leverage_aliases_without_action_context(monkeypatch):
    monkeypatch.setenv("AXIOM_ENABLE_RUNTIME_FEATURE_ADAPTER", "1")
    monkeypatch.setenv(
        "AXIOM_RUNTIME_FEATURE_ADAPTER_RULES",
        "normalized_net_leverage,normalized_available_liquidity",
    )
    features = {
        "capital_structure.net_leverage_normalized": {"value": 2.1, "support_mode": "exact"},
        "liquidity.available_liquidity_normalized": {"value": 350.0, "support_mode": "exact"},
    }

    assert _feature_value(features, "capital_structure.net_leverage") == 2.1
    assert _feature_value(features, "liquidity.available_for_actions") is None


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


def _write_snapshot(tmp_path: Path, features: dict) -> tuple[Path, dict]:
    root = tmp_path / "snapshots"
    keyed = root / "keyed" / "as_of_date=2026-02-28"
    keyed.mkdir(parents=True, exist_ok=True)
    row = {
        "snapshot_id": "snap-123",
        "company_id": "0000320193",
        "as_of_time": "2026-02-28T00:00:00+00:00",
        "features": features,
        "regime": {"credit_regime": "neutral"},
        "constraint_set": {"hard": [], "soft": []},
        "provenance": {
            "computation_version": "state_builder_v5",
            "inputs_used": {"facts": True, "timeseries": True, "events": True},
        },
    }
    p = keyed / "company_id=0000320193.json"
    p.write_text(json.dumps(row) + "\n")
    return root, row


def _make_run(tmp_path: Path, snapshot_root: Path) -> object:
    entity_graph, entity_identifier = _write_entity_files(tmp_path)
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
    run = store.get_run(run_id)
    assert run is not None
    return run


