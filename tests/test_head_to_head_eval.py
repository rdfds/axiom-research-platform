from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.head_to_head_eval import build_head_to_head_report, export_blinded_packets, render_head_to_head_markdown
from src.recommendation_run import (
    ConstraintSet,
    DataCutoffSpec,
    FrozenStateReference,
    ModelVersionBundle,
    ObjectiveVector,
    RecommendationRun,
    ScenarioAssumptions,
)


def _run() -> RecommendationRun:
    return RecommendationRun(
        run_id="run-1",
        company_id="0000320193",
        created_at="2026-03-15T00:00:00+00:00",
        as_of_time="2026-02-28T00:00:00+00:00",
        objectives=ObjectiveVector.default(),
        constraints=ConstraintSet(),
        scenario=ScenarioAssumptions(),
        frozen_state=FrozenStateReference(snapshot_id="snap-1", snapshot_hash="hash-1", snapshot_version="v1"),
        model_versions=ModelVersionBundle(
            candidate_generator_version="cg",
            feasibility_model_version="fm",
            mechanism_model_version="mm",
            precedent_retrieval_version="pm",
            planner_model_version="planner",
            regime_model_version="regime",
        ),
        data_cutoff=DataCutoffSpec(
            published_at_lte="2026-02-28T00:00:00+00:00",
            ingested_at_lte="2026-02-28T00:00:00+00:00",
        ),
        status="precedent_retrieval",
        planner_random_seed=7,
    )


def _snapshot_root(tmp_path: Path) -> Path:
    root = tmp_path / "snapshots"
    keyed = root / "keyed" / "as_of_date=2026-02-28"
    keyed.mkdir(parents=True, exist_ok=True)
    (keyed / "company_id=0000320193.json").write_text(
        json.dumps(
            {
                "company_id": "0000320193",
                "as_of_time": "2026-02-28T00:00:00+00:00",
                "features": {
                    "liquidity.available_for_actions": {"value": 600_000_000.0},
                    "market.market_cap": {"value": 6_000_000_000.0},
                    "capital_structure.net_leverage": {"value": 1.7},
                    "capital_structure.maturity_wall_ratio_24m": {"value": 0.19},
                    "operating.fcf_conversion": {"value": 0.79},
                    "operating.revenue_yoy_last_q": {"value": 0.03},
                    "market.credit_window_proxy": {"value": 0.71},
                    "strategic.intent.return_capital_priority": {"value": 0.82},
                },
            }
        )
    )
    return root


def _candidate(action_id: str, *, value_creation: float) -> dict:
    action_type, action_subtype = action_id.split(".", 1)
    return {
        "candidate_id": f"cand-{action_subtype}",
        "run_id": "run-1",
        "action_id": action_id,
        "action_type": action_type,
        "action_subtype": action_subtype,
        "parameters": {},
        "feasibility": {"pass_probability": 0.93},
        "mechanism_activation": {"mechanisms": [{"mechanism_id": "capital_efficiency", "activation_strength": 0.68}]},
        "impact_distribution": {
            "objectives": {
                "value_creation": {"median": value_creation},
                "risk_reduction": {"median": 0.12},
                "growth": {"median": 0.0},
                "rating_preservation": {"median": 0.05},
                "optionality": {"median": 0.08},
            },
            "key_drivers": [{"driver_name": "causal_model_blend_weight", "contribution": 0.2}],
            "uncertainty_score": 0.14,
        },
        "risks": [{"explanation": f"{action_id} requires disciplined execution."}],
        "structural_sanity_flags": [],
        "evaluation_confidence": 0.77,
    }


