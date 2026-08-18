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


