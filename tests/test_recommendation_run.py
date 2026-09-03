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


