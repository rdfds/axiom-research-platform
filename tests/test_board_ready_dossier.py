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


def test_build_board_ready_dossier_can_recommend_wait_against_weak_action():
    registry = build_default_action_schema_registry()
    run = _run()
    rows = [
        _candidate_row(
            "capital_return.dividend_initiate",
            value_creation=0.01,
            risk_reduction=0.0,
            optionality=-0.04,
            pass_probability=0.58,
            evaluation_confidence=0.22,
            precedent_confidence=0.12,
        ),
    ]

    plan_set = build_plan_set(
        run=run,
        feasible_candidates=[row["candidate"] for row in rows],
        precedent_matches=rows,
        registry=registry,
        top_plans=1,
    )
    dossier = build_board_ready_dossier(
        run=run,
        snapshot=_snapshot(),
        plan_set=plan_set,
        feasible_candidates=[row["candidate"] for row in rows],
        precedent_matches=rows,
        registry=registry,
    )

    assert dossier["status_quo_view"]["recommended_posture"] == "wait"
    assert dossier["status_quo_view"]["status_quo_preferred"] is True
    assert "wait" in dossier["executive_summary"].lower()
    assert dossier["parameter_optimization"]["summary"]
    assert dossier["parameter_optimization"]["recommended_parameters"]["initial_yield_pct"]["recommended_range"]
    assert dossier["recommendation_thesis"]["recommended_posture"] == "wait"
    assert dossier["ranked_action_views"][0]["recommended_posture"] == "wait"


def test_build_board_ready_dossier_clamps_implausible_debt_tenor():
    registry = build_default_action_schema_registry()
    run = _run()
    snapshot = _snapshot()
    rows = [
        _candidate_row(
            "capital_structure.new_debt_issuance",
            value_creation=0.02,
            risk_reduction=0.05,
            rating_preservation=0.03,
            optionality=0.02,
            params={
                "amount_usd": 2_000_000_000.0,
                "tenor_years": 2_000_000_000.0,
                "secured_flag": False,
                "use_of_proceeds": "general_corporate",
                "fixed_vs_floating": "fixed",
                "instrument_type": "bond",
            },
        ),
    ]

    plan_set = build_plan_set(
        run=run,
        feasible_candidates=[row["candidate"] for row in rows],
        precedent_matches=rows,
        registry=registry,
        top_plans=1,
    )
    dossier = build_board_ready_dossier(
        run=run,
        snapshot=snapshot,
        plan_set=plan_set,
        feasible_candidates=[row["candidate"] for row in rows],
        precedent_matches=rows,
        registry=registry,
    )

    tenor = dossier["parameter_optimization"]["recommended_parameters"]["tenor_years"]
    assert tenor["current_value"] is None
    assert tenor["current_value_formatted"] == "n/a"
    assert tenor["recommended_value"] <= 10.0
    assert tenor["recommended_range"].endswith("years")
    assert "2000000000.0 years" not in dossier["parameter_optimization"]["summary"]
    assert "The edge over waiting" in dossier["executive_summary"]


def test_build_board_ready_dossier_uses_specific_financing_diagnosis_and_role_text():
    registry = build_default_action_schema_registry()
    run = _run()
    snapshot = _snapshot()
    rows = [
        _candidate_row(
            "capital_structure.new_debt_issuance",
            value_creation=0.02,
            risk_reduction=0.05,
            rating_preservation=0.03,
            optionality=0.02,
            params={
                "amount_usd": 2_000_000_000.0,
                "tenor_years": 5.0,
                "secured_flag": False,
                "use_of_proceeds": "refinancing",
                "fixed_vs_floating": "fixed",
                "instrument_type": "bond",
            },
        ),
    ]

    plan_set = build_plan_set(
        run=run,
        feasible_candidates=[row["candidate"] for row in rows],
        precedent_matches=rows,
        registry=registry,
        top_plans=1,
    )
    dossier = build_board_ready_dossier(
        run=run,
        snapshot=snapshot,
        plan_set=plan_set,
        feasible_candidates=[row["candidate"] for row in rows],
        precedent_matches=rows,
        registry=registry,
    )

    problem_statement = dossier["recommendation_thesis"]["problem_statement"].lower()
    why_this_plan = dossier["recommendation_thesis"]["why_this_plan"].lower()
    why_now = dossier["recommendation_thesis"]["why_now"].lower()
    assert "term out an elevated near-term maturity wall" in problem_statement
    assert "raise about $2.0b of new debt" in why_this_plan
    assert "roughly 5.0-year tenor" in why_this_plan
    assert "term out upcoming maturities" in why_this_plan
    assert "24-month maturity wall is already 22.0%" in why_now
    assert "debt markets are currently" in why_now


def test_build_board_ready_dossier_uses_specific_capital_return_diagnosis():
    registry = build_default_action_schema_registry()
    run = _run()
    snapshot = _snapshot()
    rows = [
        _candidate_row(
            "capital_return.open_market_buyback",
            value_creation=0.29,
            optionality=0.11,
            params={"size_pct_market_cap": 0.06},
        ),
    ]

    plan_set = build_plan_set(
        run=run,
        feasible_candidates=[row["candidate"] for row in rows],
        precedent_matches=rows,
        registry=registry,
        top_plans=1,
    )
    dossier = build_board_ready_dossier(
        run=run,
        snapshot=snapshot,
        plan_set=plan_set,
        feasible_candidates=[row["candidate"] for row in rows],
        precedent_matches=rows,
        registry=registry,
    )

    problem_statement = dossier["recommendation_thesis"]["problem_statement"].lower()
    why_this_plan = dossier["recommendation_thesis"]["why_this_plan"].lower()
    assert "excess deployable capital relative to near-term operating demand" in problem_statement
    assert "at about 6.0% of market value" in why_this_plan


def test_build_board_ready_dossier_populates_fallback_alternatives_when_plan_set_is_thin():
    registry = build_default_action_schema_registry()
    run = _run()
    snapshot = _snapshot()
    rows = [
        _candidate_row(
            "capital_structure.new_debt_issuance",
            value_creation=0.02,
            risk_reduction=0.05,
            rating_preservation=0.03,
            optionality=0.02,
            params={"amount_usd": 2_000_000_000.0, "tenor_years": 5.0, "use_of_proceeds": "refinancing"},
        ),
        _candidate_row(
            "capital_structure.refinancing",
            value_creation=0.015,
            risk_reduction=0.04,
            rating_preservation=0.02,
            optionality=0.015,
            params={"amount_refinanced_usd": 1_500_000_000.0, "new_tenor_years": 5.0},
        ),
        _candidate_row(
            "capital_return.open_market_buyback",
            value_creation=0.03,
            optionality=0.01,
            params={"size_pct_market_cap": 0.03},
        ),
    ]

    plan_set = build_plan_set(
        run=run,
        feasible_candidates=[row["candidate"] for row in rows],
        precedent_matches=rows,
        registry=registry,
        top_plans=1,
    )
    dossier = build_board_ready_dossier(
        run=run,
        snapshot=snapshot,
        plan_set=plan_set,
        feasible_candidates=[row["candidate"] for row in rows],
        precedent_matches=rows,
        registry=registry,
    )

    assert dossier["alternative_analysis"]
    assert dossier["alternative_analysis"][0]["action_ids"]
    assert dossier["alternative_analysis"][0]["why_not_preferred"]


def test_build_board_ready_dossier_act_now_case_for_wait_is_not_self_defeating():
    registry = build_default_action_schema_registry()
    run = _run()
    snapshot = _snapshot()
    rows = [
        _candidate_row(
            "capital_return.tender_offer_buyback",
            value_creation=0.06,
            optionality=0.01,
            params={
                "size_absolute_usd": 1_250_000_000.0,
                "funding_mix": {"cash": 1.0, "debt": 0.0, "equity": 0.0},
            },
            evaluation_confidence=0.86,
            precedent_confidence=0.48,
        ),
    ]

    plan_set = build_plan_set(
        run=run,
        feasible_candidates=[row["candidate"] for row in rows],
        precedent_matches=rows,
        registry=registry,
        top_plans=1,
    )
    dossier = build_board_ready_dossier(
        run=run,
        snapshot=snapshot,
        plan_set=plan_set,
        feasible_candidates=[row["candidate"] for row in rows],
        precedent_matches=rows,
        registry=registry,
    )

    assert dossier["recommendation_thesis"]["recommended_posture"] == "act_now"
    assert all("incremental benefit over waiting is still modest" not in item.lower() for item in dossier["recommendation_thesis"]["case_for_wait"])
    assert dossier["risk_case"]["main_failure_modes"]
    assert "adverse tail in" not in dossier["risk_case"]["main_failure_modes"][0].lower()


def test_humanize_condition_rewrites_raw_dependency_rules():
    assert _humanize_condition("use_of_proceeds in ['buyback','general_corporate']") == (
        "Only continue if the use of proceeds remains limited to buyback or general corporate."
    )
    assert _humanize_condition("capital_structure.maturity_wall_ratio_24m drops below 0.2") == (
        "Only continue if the near-term maturity wall ratio drops below 0.2."
    )


def test_humanize_triggers_rewrites_follow_on_explanations():
    triggers = _humanize_triggers(
        [
            {
                "condition": "follow-on capacity remains available after capital_structure.new_debt_issuance",
                "explanation": "Historical follow-on frequency supports capital_return.dividend_increase after capital_structure.new_debt_issuance.",
            }
        ]
    )
    assert triggers[0]["condition"] == "Only continue if capacity still exists after new debt issuance."
    assert triggers[0]["explanation"] == "After new debt issuance, boards often revisit whether a higher recurring payout is supportable."


def test_build_board_ready_dossier_dedupes_monitoring_conditions():
    registry = build_default_action_schema_registry()
    run = _run()
    snapshot = _snapshot()
    rows = [
        _candidate_row(
            "capital_structure.refinancing",
            value_creation=0.16,
            risk_reduction=0.31,
        ),
    ]
    plan_set = {
        "plans": [
            {
                "plan_id": "p1",
                "steps": [
                    {
                        "action_id": "capital_structure.refinancing",
                        "parameters": {"amount_refinanced_usd": 1_000_000_000.0, "new_tenor_years": 5.0},
                    }
                ],
                "actions": [rows[0]["candidate"]],
                "triggers": [
                    {
                        "condition": "follow-on capacity remains available after capital_structure.refinancing",
                        "explanation": "Historical follow-on frequency supports capital_return.dividend_cut after capital_structure.refinancing.",
                        "trigger_probability": 0.7,
                    },
                    {
                        "condition": "follow-on capacity remains available after capital_structure.refinancing",
                        "explanation": "Historical follow-on frequency supports capital_return.dividend_increase after capital_structure.refinancing.",
                        "trigger_probability": 0.7,
                    },
                ],
            }
        ]
    }
    dossier = build_board_ready_dossier(
        run=run,
        snapshot=snapshot,
        plan_set=plan_set,
        feasible_candidates=[row["candidate"] for row in rows],
        precedent_matches=rows,
        registry=registry,
    )
    triggers = dossier["monitoring"]["triggers"]
    assert len(triggers) == 1
    assert triggers[0]["condition"] == "Only continue if capacity still exists after refinancing."


def test_decision_boundaries_avoid_zero_credit_window_baseline():
    boundaries = _decision_boundaries(
        "capital_structure.new_debt_issuance",
        {"features": {"market.credit_window_proxy": {"value": 0.0}}},
        {"triggers": []},
    )
    assert "0.00/1.00" not in " ".join(boundaries)
    assert any("credit conditions deteriorate materially" in item for item in boundaries)


def test_build_board_ready_dossier_handles_missing_plan():
    run = _run()

    dossier = build_board_ready_dossier(
        run=run,
        snapshot=_snapshot(),
        plan_set={"plans": []},
        feasible_candidates=[],
        precedent_matches=[],
        registry=build_default_action_schema_registry(),
    )

    assert dossier["status"] == "no_plan"
    assert dossier["executive_summary"] == "No feasible plan was generated."


def test_risk_case_humanizes_tail_metric_names():
    registry = build_default_action_schema_registry()
    run = _run()
    snapshot = _snapshot()
    rows = [
        _candidate_row(
            "capital_structure.refinancing",
            value_creation=0.16,
            risk_reduction=0.31,
        ),
    ]
    rows[0]["precedent_pack"]["tail_events"] = [
        {
            "metric": "equity_return_vs_sector",
            "description": "Bottom decile historical outcome.",
            "horizon": "12m",
            "value": -85.42,
        }
    ]
    plan_set = build_plan_set(
        run=run,
        feasible_candidates=[row["candidate"] for row in rows],
        precedent_matches=rows,
        registry=registry,
        top_plans=1,
    )
    dossier = build_board_ready_dossier(
        run=run,
        snapshot=snapshot,
        plan_set=plan_set,
        feasible_candidates=[row["candidate"] for row in rows],
        precedent_matches=rows,
        registry=registry,
    )
    assert not any("equity_return_vs_sector" in item for item in dossier["risk_case"]["main_failure_modes"])
    assert any(
        item in {"Adverse tail in 12m relative share performance.", "Bottom decile historical outcome."}
        for item in dossier["risk_case"]["main_failure_modes"]
    )
