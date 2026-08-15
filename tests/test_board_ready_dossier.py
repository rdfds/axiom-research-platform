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


def _candidate_row(
    action_id: str,
    *,
    value_creation: float,
    risk_reduction: float = 0.0,
    growth: float = 0.0,
    rating_preservation: float = 0.0,
    optionality: float = 0.0,
    pass_probability: float = 0.92,
    evaluation_confidence: float = 0.74,
    precedent_confidence: float = 0.42,
    params: dict | None = None,
) -> dict:
    action_type, action_subtype = action_id.split(".", 1)
    return {
        "candidate": {
            "candidate_id": f"cand-{action_subtype}",
            "run_id": "run-1",
            "action_id": action_id,
            "action_type": action_type,
            "action_subtype": action_subtype,
            "parameters": params or {},
            "feasibility": {
                "feasibility_status": "feasible",
                "pass_probability": pass_probability,
            },
            "mechanism_activation": {
                "mechanisms": [
                    {
                        "mechanism_id": "capital_efficiency",
                        "activation_strength": 0.65,
                    }
                ],
                "narrative_explanation": f"{action_id} addresses the current strategic setup.",
            },
            "impact_distribution": {
                "objectives": {
                    "value_creation": {"median": value_creation},
                    "risk_reduction": {"median": risk_reduction},
                    "growth": {"median": growth},
                    "rating_preservation": {"median": rating_preservation},
                    "optionality": {"median": optionality},
                },
                "key_drivers": [
                    {
                        "driver_name": "causal_model_blend_weight",
                        "contribution": 0.21,
                        "explanation": f"Causal support is active for {action_id}.",
                    }
                ],
                "uncertainty_score": 0.18,
            },
            "risks": [
                {
                    "risk_type": "execution",
                    "probability": 0.18,
                    "explanation": f"{action_id} requires disciplined execution.",
                }
            ],
            "structural_sanity_flags": [],
            "evaluation_confidence": evaluation_confidence,
        },
        "precedent_pack": {
            "precedent_confidence": precedent_confidence,
            "mismatch_diagnostics": {
                "out_of_sample_flag": False,
                "retrieval_tier": "exact",
            },
            "tail_events": [
                {
                    "metric": "credit_spread_change",
                    "description": "Tail spread widening in the bottom decile.",
                }
            ],
            "second_order_effects": [
                {
                    "follow_on_action_id": "capital_return.open_market_buyback",
                    "frequency": 0.41,
                }
            ],
            "outcome_distributions": {
                "horizon_12m": {
                    "valuation_multiple_change": {"sample_size": 28},
                }
            },
        },
    }


