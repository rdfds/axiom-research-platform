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


def test_build_board_ready_dossier_generates_executive_thesis():
    registry = build_default_action_schema_registry()
    run = _run()
    snapshot = _snapshot()
    rows = [
        _candidate_row(
            "capital_structure.refinancing",
            value_creation=0.16,
            risk_reduction=0.31,
        ),
        _candidate_row(
            "capital_return.open_market_buyback",
            value_creation=0.29,
            optionality=0.11,
            params={"size_pct_market_cap": 0.06},
        ),
        _candidate_row(
            "capital_return.dividend_increase",
            value_creation=0.11,
            rating_preservation=-0.05,
            optionality=-0.03,
        ),
    ]

    plan_set = build_plan_set(
        run=run,
        feasible_candidates=[row["candidate"] for row in rows],
        precedent_matches=rows,
        registry=registry,
        top_plans=3,
    )
    dossier = build_board_ready_dossier(
        run=run,
        snapshot=snapshot,
        plan_set=plan_set,
        feasible_candidates=[row["candidate"] for row in rows],
        precedent_matches=rows,
        registry=registry,
    )

    assert dossier["executive_summary"]
    assert dossier["confidence_posture"] in {"high_conviction", "supported_but_conditional", "conditional"}
    assert "balance-sheet capacity" in dossier["recommendation_thesis"]["problem_statement"].lower()
    assert "sequence matters" in dossier["recommendation_thesis"]["why_this_plan"].lower()
    assert "debt markets are currently" in dossier["recommendation_thesis"]["why_now"].lower()
    assert dossier["status_quo_view"]["recommended_posture"] in {"act_now", "conditional_action", "wait"}
    assert dossier["sizing_guidance"]["recommended_range"]
    assert dossier["sizing_guidance"]["scenario_overrides"]
    assert dossier["parameter_optimization"]["summary"]
    assert dossier["parameter_optimization"]["recommended_parameters"]
    assert any(
        recommendation.get("recommended_value_formatted")
        for recommendation in dossier["parameter_optimization"]["recommended_parameters"].values()
    )
    assert dossier["regret_analysis"]["if_we_act_and_are_wrong"]
    assert dossier["regret_analysis"]["if_we_wait_and_are_wrong"]
    assert dossier["rating_cliff_analysis"]["constraint_posture"]
    assert dossier["signaling_analysis"]["signal_posture"]
    assert dossier["recommendation_thesis"]["sizing_summary"]["recommended_range"]
    assert dossier["recommendation_thesis"]["parameter_summary"]
    assert dossier["recommendation_thesis"]["regret_balance"]
    assert dossier["recommendation_thesis"]["rating_constraint_posture"]
    assert dossier["recommendation_thesis"]["market_signal_posture"]
    assert dossier["ranked_action_views"]
    assert dossier["ranked_action_views"][0]["sizing_guidance"]["recommended_range"]
    assert dossier["ranked_action_views"][0]["parameter_optimization"]["summary"]
    assert dossier["ranked_action_views"][0]["regret_balance"]
    assert dossier["ranked_action_views"][0]["rating_constraint_posture"]
    assert dossier["ranked_action_views"][0]["signal_posture"]
    assert dossier["supporting_evidence"]
    assert dossier["step_theses"][0]["supporting_facts"]
    assert dossier["alternative_analysis"]
    assert any(
        "maturity wall" in item["why_not_preferred"].lower()
        or "stickier payout" in item["why_not_preferred"].lower()
        or "expected utility" in item["why_not_preferred"].lower()
        for item in dossier["alternative_analysis"]
    )
    assert dossier["risk_case"]["kill_criteria"]
    assert dossier["scorecard"]["average_precedent_confidence"] > 0.0


