from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_recommendation_run_api as api  # noqa: E402

from src.recommendation_run import AuditEvent, RecommendationRunStore, create_recommendation_run


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

    pd.DataFrame([{"entity_id": "0000320193", "identifier_value": "001690"}]).to_parquet(
        entity_identifier, index=False
    )
    return entity_graph, entity_identifier


def _write_keyed_snapshot(tmp_path: Path) -> Path:
    root = tmp_path / "snapshots"
    keyed = root / "keyed" / "as_of_date=2026-02-28"
    keyed.mkdir(parents=True, exist_ok=True)
    row = {
        "snapshot_id": "snap-123",
        "company_id": "0000320193",
        "as_of_time": "2026-02-28T00:00:00+00:00",
        "features": {
            "liquidity.available_for_actions": {"value": 100.0},
            "capital_structure.net_leverage": {"value": 2.0},
            "market.market_cap": {"value": 1000.0},
        },
        "regime": {"credit_regime": "neutral"},
        "provenance": {"computation_version": "state_builder_v5"},
    }
    (keyed / "company_id=0000320193.json").write_text(json.dumps(row) + "\n")
    return root


def test_canonical_request_signature_stable_to_action_order():
    s1 = api._canonical_request_signature(
        company_id="001690",
        as_of="2026-02-28",
        action_ids=["b", "a"],
        action_type=None,
        objectives=None,
        constraints=None,
        scenario=None,
        max_candidates=50,
        min_candidates_target=300,
        precedent_top_k=25,
        strict_evidence=False,
        top_plans=1,
    )
    s2 = api._canonical_request_signature(
        company_id="001690",
        as_of="2026-02-28",
        action_ids=["a", "b"],
        action_type=None,
        objectives=None,
        constraints=None,
        scenario=None,
        max_candidates=50,
        min_candidates_target=300,
        precedent_top_k=25,
        strict_evidence=False,
        top_plans=1,
    )
    assert s1 == s2


