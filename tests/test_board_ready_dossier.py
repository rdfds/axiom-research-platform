from __future__ import annotations

from src.action_ontology import build_default_action_schema_registry
from src.board_ready_dossier import (
    _decision_boundaries,
    _humanize_condition,
    _humanize_triggers,
    build_board_ready_dossier,
)
from src.planner_brain import build_plan_set
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
        created_at="2026-03-14T00:00:00+00:00",
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
        status="plan_search",
        planner_random_seed=7,
    )


def _snapshot() -> dict:
    return {
        "company_id": "0000320193",
        "as_of_time": "2026-02-28T00:00:00+00:00",
        "features": {
            "liquidity.available_for_actions": {"value": 850_000_000.0},
            "market.market_cap": {"value": 8_500_000_000.0},
            "capital_structure.net_leverage": {"value": 1.85},
            "capital_structure.maturity_wall_ratio_24m": {"value": 0.22},
            "operating.fcf_conversion": {"value": 0.82},
            "operating.revenue_yoy_last_q": {"value": 0.01},
            "market.credit_window_proxy": {"value": 0.74},
            "market.credit_spread_percentile_2y": {"value": 68.0},
            "market.equity_window_proxy": {"value": 0.57},
            "capital_structure.rating_state": {"value": {"rating": "BBB-", "outlook": "negative", "score": 10.0}},
            "strategic.intent.return_capital_priority": {"value": 0.87},
            "strategic.intent.pursue_mna_priority": {"value": 0.24},
            "strategic.intent.focus_on_core": {"value": 0.39},
            "ownership_governance.activist_signal": {"value": 0.13},
        },
    }


