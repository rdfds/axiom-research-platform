from __future__ import annotations

from src.action_ontology import build_default_action_schema_registry
from src.planner_brain import (
    _action_specific_penalty,
    _has_causal_support,
    _status_quo_hurdle,
    build_plan_set,
)
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
        status="precedent_retrieval",
        planner_random_seed=7,
    )


def _candidate_row(
    action_id: str,
    *,
    utility: float,
    risk_reduction: float = 0.0,
    growth: float = 0.0,
    rating_preservation: float = 0.0,
    optionality: float = 0.0,
    pass_probability: float = 0.9,
    evaluation_confidence: float = 0.65,
    precedent_confidence: float = 0.4,
    narrative: str = "",
    second_order_effects: list[dict] | None = None,
    tail_events: list[dict] | None = None,
    regime_sensitivity: list[dict] | None = None,
    causal: bool = True,
    causal_blend_weight: float | None = None,
    causal_quality: float | None = None,
    causal_support_score: float | None = None,
    causal_min_oos_r2: float | None = None,
    uncertainty_score: float = 0.2,
    params: dict | None = None,
    feasibility_status: str = "feasible",
    feasibility_blockers: list[dict] | None = None,
    gating_signals: list[dict] | None = None,
) -> dict:
    action_type, action_subtype = action_id.split(".", 1)
    key_drivers = [
        {
            "driver_name": "test_driver",
            "contribution": 0.5,
            "explanation": f"{action_id} directly improves the target condition.",
        }
    ]
    if causal_blend_weight is not None:
        key_drivers.append(
            {
                "driver_name": "causal_model_blend_weight",
                "contribution": causal_blend_weight,
                "explanation": f"Causal support is measured for {action_id}.",
            }
        )
        if causal_quality is not None:
            key_drivers.append(
                {
                    "driver_name": "causal_model_quality",
                    "contribution": causal_quality,
                    "explanation": f"Causal model quality is measured for {action_id}.",
                }
            )
        if causal_support_score is not None:
            key_drivers.append(
                {
                    "driver_name": "causal_model_support_score",
                    "contribution": causal_support_score,
                    "explanation": f"Causal support score is measured for {action_id}.",
                }
            )
        if causal_min_oos_r2 is not None:
            key_drivers.append(
                {
                    "driver_name": "causal_model_min_oos_r2",
                    "contribution": causal_min_oos_r2,
                    "explanation": f"Causal minimum OOS R2 is measured for {action_id}.",
                }
            )
    elif causal:
        key_drivers.append(
            {
                "driver_name": "causal_model_blend_weight",
                "contribution": 0.25,
                "explanation": f"Causal support is active for {action_id}.",
            }
        )
    return {
        "candidate": {
            "candidate_id": f"cand-{action_subtype}",
            "run_id": "run-1",
            "action_id": action_id,
            "action_type": action_type,
            "action_subtype": action_subtype,
            "parameters": params or {},
            "feasibility": {
                "feasibility_status": feasibility_status,
                "pass_probability": pass_probability,
                "blockers": feasibility_blockers or [],
                "gating_signals": gating_signals or [],
            },
            "mechanism_activation": {
                "narrative_explanation": narrative or f"{action_id} addresses the current strategic setup.",
            },
            "impact_distribution": {
                "objectives": {
                    "value_creation": {"median": utility},
                    "risk_reduction": {"median": risk_reduction},
                    "growth": {"median": growth},
                    "rating_preservation": {"median": rating_preservation},
                    "optionality": {"median": optionality},
                },
                "regime_sensitivity": regime_sensitivity or [],
                "key_drivers": key_drivers,
                "uncertainty_score": uncertainty_score,
            },
            "risks": [
                {
                    "risk_type": "execution",
                    "probability": 0.2,
                    "explanation": f"{action_id} requires execution discipline.",
                }
            ],
            "assumptions": [],
            "evaluation_confidence": evaluation_confidence,
        },
        "precedent_pack": {
            "calibration_confidence": precedent_confidence,
            "tail_events": tail_events or [],
            "second_order_effects": second_order_effects or [],
            "mismatch_diagnostics": {"out_of_sample_flag": False},
        },
    }


def test_zero_blend_causal_metadata_does_not_count_as_support():
    inactive = _candidate_row(
        "capital_structure.equity_issuance",
        utility=0.04,
        risk_reduction=0.06,
        rating_preservation=0.04,
        precedent_confidence=0.0,
        causal=False,
        causal_blend_weight=0.0,
    )["candidate"]
    active = _candidate_row(
        "capital_structure.equity_issuance",
        utility=0.04,
        risk_reduction=0.06,
        rating_preservation=0.04,
        precedent_confidence=0.0,
        causal_blend_weight=0.25,
    )["candidate"]

    assert _has_causal_support(inactive) is False
    assert _has_causal_support(active) is True


def test_build_plan_set_constructs_multistep_plan_and_branch():
    registry = build_default_action_schema_registry()
    run = _run()
    rows = [
        _candidate_row(
            "capital_structure.refinancing",
            utility=0.18,
            risk_reduction=0.32,
            second_order_effects=[
                {"follow_on_action_id": "capital_return.open_market_buyback", "frequency": 0.45},
                {"follow_on_action_id": "mna.tuck_in_acquisition", "frequency": 0.25},
            ],
            narrative="Refinancing reduces near-term balance-sheet pressure.",
        ),
        _candidate_row(
            "capital_return.open_market_buyback",
            utility=0.34,
            optionality=0.1,
            narrative="Buybacks deploy excess capital into a discounted share base.",
        ),
        _candidate_row(
            "mna.tuck_in_acquisition",
            utility=-0.05,
            growth=0.12,
            pass_probability=0.72,
            narrative="A tuck-in becomes interesting only after financing capacity improves.",
        ),
    ]

    plan_set = build_plan_set(run=run, precedent_matches=rows, registry=registry, top_plans=3)

    assert plan_set["dependency_graph"]["nodes"] == [
        "capital_return.open_market_buyback",
        "capital_structure.refinancing",
        "mna.tuck_in_acquisition",
    ]
    top_plan = plan_set["plans"][0]
    step_actions = [step["action_id"] for step in top_plan["steps"]]
    assert step_actions[:2] == [
        "capital_structure.refinancing",
        "capital_return.open_market_buyback",
    ]
    assert top_plan["steps"][1]["prerequisites"] == ["capital_structure.refinancing"]
    assert top_plan["timeline"]["step_schedule"][1]["start_time"] == "2026-03-30T00:00:00+00:00"
    assert any(branch["branch_plan_steps"] == ["mna.tuck_in_acquisition"] for branch in top_plan["branches"])
    assert top_plan["triggers"][0]["trigger_type"] == "liquidity_condition"
    assert 0.0 <= top_plan["score"] <= 1.0


def test_build_plan_set_never_combines_conflicting_actions():
    registry = build_default_action_schema_registry()
    run = _run()
    rows = [
        _candidate_row("capital_return.open_market_buyback", utility=0.35),
        _candidate_row("mna.transformational_acquisition", utility=0.45, growth=0.25),
        _candidate_row("capital_structure.refinancing", utility=0.16, risk_reduction=0.22),
    ]

    plan_set = build_plan_set(run=run, precedent_matches=rows, registry=registry, top_plans=5)

    for plan in plan_set["plans"]:
        action_ids = {step["action_id"] for step in plan["steps"]}
        assert not {
            "capital_return.open_market_buyback",
            "mna.transformational_acquisition",
        }.issubset(action_ids)


def test_build_plan_set_ranking_is_deterministic_under_input_reorder():
    registry = build_default_action_schema_registry()
    run = _run()
    rows = [
        _candidate_row("capital_structure.refinancing", utility=0.18, risk_reduction=0.32),
        _candidate_row("capital_return.open_market_buyback", utility=0.34, optionality=0.1),
        _candidate_row("mna.tuck_in_acquisition", utility=0.05, growth=0.12),
    ]

    plan_set_a = build_plan_set(run=run, precedent_matches=rows, registry=registry, top_plans=3)
    plan_set_b = build_plan_set(run=run, precedent_matches=list(reversed(rows)), registry=registry, top_plans=3)

    assert [plan["plan_id"] for plan in plan_set_a["plans"]] == [plan["plan_id"] for plan in plan_set_b["plans"]]
    assert [plan["score"] for plan in plan_set_a["plans"]] == [plan["score"] for plan in plan_set_b["plans"]]


def test_planner_uses_feasible_candidates_beyond_precedent_subset():
    registry = build_default_action_schema_registry()
    run = _run()
    feasible_rows = [
        _candidate_row("capital_structure.refinancing", utility=0.08, risk_reduction=0.24),
        _candidate_row("capital_return.open_market_buyback", utility=0.11, optionality=0.28),
        _candidate_row("capital_return.dividend_increase", utility=0.2, growth=-0.18, rating_preservation=-0.05, optionality=-0.04),
    ]
    precedent_subset = [
        feasible_rows[0],
        feasible_rows[2],
    ]

    plan_set = build_plan_set(
        run=run,
        feasible_candidates=[row["candidate"] for row in feasible_rows],
        precedent_matches=precedent_subset,
        registry=registry,
        top_plans=5,
    )

    assert any(
        "capital_return.open_market_buyback" in [step["action_id"] for step in plan["steps"]]
        for plan in plan_set["plans"]
    )


def test_buyback_refi_fixture_prefers_refi_plus_buyback_over_dividend_policy():
    registry = build_default_action_schema_registry()
    run = _run()
    rows = [
        _candidate_row(
            "capital_structure.refinancing",
            utility=0.12,
            risk_reduction=0.34,
            growth=0.08,
            rating_preservation=0.12,
            precedent_confidence=0.38,
            second_order_effects=[{"follow_on_action_id": "capital_return.open_market_buyback", "frequency": 0.45}],
        ),
        _candidate_row(
            "capital_return.open_market_buyback",
            utility=0.12,
            optionality=0.32,
            precedent_confidence=0.34,
        ),
        _candidate_row(
            "capital_return.dividend_increase",
            utility=0.26,
            growth=-0.22,
            rating_preservation=-0.05,
            optionality=-0.03,
            precedent_confidence=0.41,
        ),
        _candidate_row(
            "capital_return.dividend_cut",
            utility=0.10,
            growth=-0.03,
            rating_preservation=-0.04,
            optionality=-0.03,
            precedent_confidence=0.35,
        ),
    ]

    plan_set = build_plan_set(
        run=run,
        feasible_candidates=[row["candidate"] for row in rows],
        precedent_matches=rows,
        registry=registry,
        top_plans=5,
    )

    top_actions = [step["action_id"] for step in plan_set["plans"][0]["steps"]]
    assert top_actions == [
        "capital_structure.refinancing",
        "capital_return.open_market_buyback",
    ]


def test_acquisition_fixture_prefers_refi_then_tuck_in_when_acquisition_is_supported():
    registry = build_default_action_schema_registry()
    run = _run()
    rows = [
        _candidate_row(
            "capital_structure.refinancing",
            utility=0.10,
            risk_reduction=0.22,
            growth=0.05,
            rating_preservation=0.08,
            precedent_confidence=0.33,
            second_order_effects=[{"follow_on_action_id": "mna.tuck_in_acquisition", "frequency": 0.35}],
        ),
        _candidate_row(
            "mna.tuck_in_acquisition",
            utility=0.32,
            growth=0.42,
            optionality=0.12,
            pass_probability=0.82,
            evaluation_confidence=0.78,
            precedent_confidence=0.39,
            regime_sensitivity=[{"regime_condition": "risk_off", "effect_shift": -0.04}],
        ),
        _candidate_row(
            "capital_return.open_market_buyback",
            utility=0.08,
            optionality=0.16,
            precedent_confidence=0.32,
        ),
        _candidate_row(
            "capital_return.dividend_increase",
            utility=0.22,
            growth=-0.18,
            rating_preservation=-0.04,
            optionality=-0.03,
            precedent_confidence=0.41,
        ),
    ]

    plan_set = build_plan_set(
        run=run,
        feasible_candidates=[row["candidate"] for row in rows],
        precedent_matches=rows,
        registry=registry,
        top_plans=5,
    )

    top_actions = [step["action_id"] for step in plan_set["plans"][0]["steps"]]
    assert top_actions == [
        "capital_structure.refinancing",
        "mna.tuck_in_acquisition",
    ]


def test_divestiture_fixture_prefers_divestiture_over_narrow_payout_moves():
    registry = build_default_action_schema_registry()
    run = _run()
    rows = [
        _candidate_row(
            "portfolio.divestiture_partial",
            utility=0.22,
            risk_reduction=0.28,
            rating_preservation=0.12,
            optionality=0.14,
            pass_probability=0.7,
            evaluation_confidence=0.82,
            precedent_confidence=0.38,
        ),
        _candidate_row(
            "capital_structure.refinancing",
            utility=0.04,
            risk_reduction=0.14,
            growth=0.02,
            precedent_confidence=0.31,
        ),
        _candidate_row(
            "capital_return.dividend_increase",
            utility=0.28,
            growth=-0.16,
            rating_preservation=-0.04,
            optionality=-0.03,
            precedent_confidence=0.40,
        ),
        _candidate_row(
            "capital_return.dividend_cut",
            utility=0.09,
            growth=-0.02,
            rating_preservation=-0.03,
            optionality=-0.03,
            precedent_confidence=0.34,
        ),
    ]

    plan_set = build_plan_set(
        run=run,
        feasible_candidates=[row["candidate"] for row in rows],
        precedent_matches=rows,
        registry=registry,
        top_plans=5,
    )

    assert [step["action_id"] for step in plan_set["plans"][0]["steps"]] == ["portfolio.divestiture_partial"]


def test_dividend_initiation_requires_clear_edge_over_financing_actions():
    registry = build_default_action_schema_registry()
    run = _run()
    rows = [
        _candidate_row(
            "capital_return.dividend_initiate",
            utility=-0.01,
            risk_reduction=-0.01,
            growth=-0.02,
            rating_preservation=-0.02,
            optionality=-0.01,
            pass_probability=0.95,
            evaluation_confidence=0.825,
            precedent_confidence=0.47,
        ),
        _candidate_row(
            "capital_structure.equity_issuance",
            utility=0.025,
            risk_reduction=0.01,
            growth=0.04,
            rating_preservation=0.02,
            optionality=0.01,
            pass_probability=0.95,
            evaluation_confidence=0.833,
            precedent_confidence=0.43,
        ),
        _candidate_row(
            "capital_structure.refinancing",
            utility=0.025,
            risk_reduction=0.01,
            growth=0.03,
            rating_preservation=0.03,
            optionality=0.01,
            pass_probability=0.95,
            evaluation_confidence=0.833,
            precedent_confidence=0.43,
        ),
    ]

    plan_set = build_plan_set(
        run=run,
        feasible_candidates=[row["candidate"] for row in rows],
        precedent_matches=rows,
        registry=registry,
        top_plans=5,
    )

    top_action = plan_set["plans"][0]["steps"][0]["action_id"]
    assert top_action in {"capital_structure.equity_issuance", "capital_structure.refinancing"}
    dividend_plan = next(plan for plan in plan_set["plans"] if plan["steps"][0]["action_id"] == "capital_return.dividend_initiate")
    assert dividend_plan["score_components"]["negative_utility_penalty"] > 0.0
    assert dividend_plan["score_components"]["action_specific_penalty"] >= 0.06


def test_causal_dividend_initiate_relief_is_narrow_and_conservative():
    strong_causal = _candidate_row(
        "capital_return.dividend_initiate",
        utility=0.043,
        risk_reduction=-0.02,
        growth=-0.003,
        rating_preservation=-0.072,
        optionality=-0.026,
        evaluation_confidence=0.66,
        precedent_confidence=0.42,
        causal_blend_weight=0.266,
        causal_quality=0.124,
        causal_support_score=0.944,
        causal_min_oos_r2=0.124,
        uncertainty_score=0.41,
    )["candidate"]
    weak_causal = _candidate_row(
        "capital_return.dividend_initiate",
        utility=0.043,
        risk_reduction=-0.02,
        growth=-0.003,
        rating_preservation=-0.072,
        optionality=-0.026,
        evaluation_confidence=0.66,
        precedent_confidence=0.42,
        causal_blend_weight=0.08,
        causal_quality=0.05,
        causal_support_score=0.7,
        causal_min_oos_r2=0.05,
        uncertainty_score=0.41,
    )["candidate"]

    strong_penalty = _action_specific_penalty(strong_causal, {"calibration_confidence": 0.42})
    weak_penalty = _action_specific_penalty(weak_causal, {"calibration_confidence": 0.42})
    strong_hurdle = _status_quo_hurdle(
        weighted_utility=-0.03,
        weighted_components={
            "value_creation": 0.015,
            "risk_reduction": -0.005,
            "growth": 0.0,
            "rating_preservation": -0.01,
            "optionality": -0.004,
        },
        action_ids=["capital_return.dividend_initiate"],
        support_factor=0.73,
        candidates=[strong_causal],
    )
    weak_hurdle = _status_quo_hurdle(
        weighted_utility=-0.03,
        weighted_components={
            "value_creation": 0.015,
            "risk_reduction": -0.005,
            "growth": 0.0,
            "rating_preservation": -0.01,
            "optionality": -0.004,
        },
        action_ids=["capital_return.dividend_initiate"],
        support_factor=0.73,
        candidates=[weak_causal],
    )

    assert strong_penalty < weak_penalty
    assert strong_penalty >= 0.03
    assert strong_hurdle < weak_hurdle
    assert strong_hurdle >= 0.0


def test_causal_dividend_initiate_support_improves_close_case_scoring():
    registry = build_default_action_schema_registry()
    run = _run()
    strong_rows = [
        _candidate_row(
            "capital_return.dividend_initiate",
            utility=0.043,
            risk_reduction=-0.02,
            growth=-0.003,
            rating_preservation=-0.072,
            optionality=-0.026,
            pass_probability=0.9,
            evaluation_confidence=0.657,
            precedent_confidence=0.42,
            causal_blend_weight=0.266,
            causal_quality=0.124,
            causal_support_score=0.944,
            causal_min_oos_r2=0.124,
            uncertainty_score=0.41,
        ),
        _candidate_row(
            "capital_return.open_market_buyback",
            utility=0.04,
            risk_reduction=-0.016,
            growth=0.002,
            rating_preservation=-0.01,
            optionality=0.06,
            pass_probability=0.9,
            evaluation_confidence=0.636,
            precedent_confidence=0.342,
            causal=False,
            uncertainty_score=0.2,
        ),
    ]
    weak_rows = [
        _candidate_row(
            "capital_return.dividend_initiate",
            utility=0.043,
            risk_reduction=-0.02,
            growth=-0.003,
            rating_preservation=-0.072,
            optionality=-0.026,
            pass_probability=0.9,
            evaluation_confidence=0.657,
            precedent_confidence=0.42,
            causal_blend_weight=0.08,
            causal_quality=0.05,
            causal_support_score=0.7,
            causal_min_oos_r2=0.05,
            uncertainty_score=0.41,
        ),
        _candidate_row(
            "capital_return.open_market_buyback",
            utility=0.04,
            risk_reduction=-0.016,
            growth=0.002,
            rating_preservation=-0.01,
            optionality=0.06,
            pass_probability=0.9,
            evaluation_confidence=0.636,
            precedent_confidence=0.342,
            causal=False,
            uncertainty_score=0.2,
        ),
    ]

    strong_plan_set = build_plan_set(
        run=run,
        feasible_candidates=[row["candidate"] for row in strong_rows],
        precedent_matches=strong_rows,
        registry=registry,
        top_plans=3,
    )
    weak_plan_set = build_plan_set(
        run=run,
        feasible_candidates=[row["candidate"] for row in weak_rows],
        precedent_matches=weak_rows,
        registry=registry,
        top_plans=3,
    )

    strong_dividend_plan = next(plan for plan in strong_plan_set["plans"] if plan["steps"][0]["action_id"] == "capital_return.dividend_initiate")
    weak_dividend_plan = next(plan for plan in weak_plan_set["plans"] if plan["steps"][0]["action_id"] == "capital_return.dividend_initiate")

    assert strong_dividend_plan["score_components"]["action_specific_penalty"] < weak_dividend_plan["score_components"]["action_specific_penalty"]
    assert strong_dividend_plan["score_components"]["raw_total_score"] > weak_dividend_plan["score_components"]["raw_total_score"]


def test_buyback_maturity_wall_relief_is_narrow_and_can_flip_close_case():
    registry = build_default_action_schema_registry()
    run = _run()
    dividend_row = _candidate_row(
        "capital_return.dividend_initiate",
        utility=0.05568,
        risk_reduction=-0.02067,
        growth=-0.00312,
        rating_preservation=-0.083746,
        optionality=-0.02252,
        pass_probability=0.95,
        evaluation_confidence=0.852,
        precedent_confidence=0.0,
        causal_blend_weight=0.266,
        causal_quality=0.124,
        causal_support_score=0.89,
        causal_min_oos_r2=0.124,
        uncertainty_score=0.19,
        params={"initial_yield_pct": 0.01},
    )
    relieved_buyback_row = _candidate_row(
        "capital_return.open_market_buyback",
        utility=0.057987,
        risk_reduction=-0.012833,
        growth=0.001027,
        rating_preservation=-0.008213,
        optionality=-0.009453,
        pass_probability=0.54,
        evaluation_confidence=0.766,
        precedent_confidence=0.0,
        causal_blend_weight=0.0,
        uncertainty_score=0.58,
        params={
            "funding_mix": {"cash": 0.7, "debt": 0.3, "equity": 0.0},
            "size_pct_market_cap": 0.02,
        },
        feasibility_status="conditional",
        feasibility_blockers=[
            {
                "blocker_type": "maturity_wall_conflict",
                "severity": "soft",
                "explanation": "Action consumes liquidity while near-term maturities are elevated.",
            }
        ],
        gating_signals=[
            {"feature_name": "liquidity.runway_months_proforma", "value": 23.23},
            {"feature_name": "capital_structure.maturity_wall_ratio_24m", "value": 0.2836},
            {"feature_name": "capital_structure.proforma_interest_coverage", "value": 34.89},
        ],
    )
    unreleived_buyback_row = _candidate_row(
        "capital_return.open_market_buyback",
        utility=0.057987,
        risk_reduction=-0.012833,
        growth=0.001027,
        rating_preservation=-0.008213,
        optionality=-0.009453,
        pass_probability=0.54,
        evaluation_confidence=0.766,
        precedent_confidence=0.0,
        causal_blend_weight=0.0,
        uncertainty_score=0.58,
        params={
            "funding_mix": {"cash": 0.7, "debt": 0.3, "equity": 0.0},
            "size_pct_market_cap": 0.02,
        },
        feasibility_status="conditional",
        feasibility_blockers=[
            {
                "blocker_type": "maturity_wall_conflict",
                "severity": "soft",
                "explanation": "Action consumes liquidity while near-term maturities are elevated.",
            }
        ],
        gating_signals=[
            {"feature_name": "liquidity.runway_months_proforma", "value": 12.0},
            {"feature_name": "capital_structure.maturity_wall_ratio_24m", "value": 0.2836},
            {"feature_name": "capital_structure.proforma_interest_coverage", "value": 8.0},
        ],
    )

    relieved_penalty = _action_specific_penalty(relieved_buyback_row["candidate"], relieved_buyback_row["precedent_pack"])
    unreleived_penalty = _action_specific_penalty(unreleived_buyback_row["candidate"], unreleived_buyback_row["precedent_pack"])

    relieved_plan_set = build_plan_set(
        run=run,
        feasible_candidates=[dividend_row["candidate"], relieved_buyback_row["candidate"]],
        precedent_matches=[dividend_row, relieved_buyback_row],
        registry=registry,
        top_plans=3,
    )
    unreleived_plan_set = build_plan_set(
        run=run,
        feasible_candidates=[dividend_row["candidate"], unreleived_buyback_row["candidate"]],
        precedent_matches=[dividend_row, unreleived_buyback_row],
        registry=registry,
        top_plans=3,
    )

    relieved_top = relieved_plan_set["plans"][0]["steps"][0]["action_id"]
    unreleived_top = unreleived_plan_set["plans"][0]["steps"][0]["action_id"]

    assert relieved_penalty < unreleived_penalty
    assert relieved_penalty == 0.11
    assert unreleived_penalty == 0.15
    assert relieved_top == "capital_return.open_market_buyback"
    assert unreleived_top == "capital_return.dividend_initiate"


def test_thin_equity_issuance_does_not_dominate_refinancing():
    registry = build_default_action_schema_registry()
    run = _run()
    rows = [
        _candidate_row(
            "capital_structure.equity_issuance",
            utility=0.0144,
            risk_reduction=0.0495,
            growth=0.0027,
            rating_preservation=0.032,
            optionality=0.0261,
            pass_probability=0.95,
            evaluation_confidence=0.8331,
            precedent_confidence=0.43,
        ),
        _candidate_row(
            "capital_structure.refinancing",
            utility=0.0144,
            risk_reduction=0.0495,
            growth=0.0027,
            rating_preservation=0.032,
            optionality=0.0261,
            pass_probability=0.95,
            evaluation_confidence=0.8331,
            precedent_confidence=0.33,
        ),
    ]

    plan_set = build_plan_set(
        run=run,
        feasible_candidates=[row["candidate"] for row in rows],
        precedent_matches=rows,
        registry=registry,
        top_plans=5,
    )

    assert plan_set["plans"][0]["steps"][0]["action_id"] == "capital_structure.refinancing"
    equity_plan = next(plan for plan in plan_set["plans"] if plan["steps"][0]["action_id"] == "capital_structure.equity_issuance")
    assert equity_plan["score_components"]["action_specific_penalty"] >= 0.1
    assert equity_plan["score_components"]["raw_total_score"] < plan_set["plans"][0]["score_components"]["raw_total_score"]


def test_deleveraging_equity_recap_case_can_rank_ahead_of_refinancing():
    registry = build_default_action_schema_registry()
    run = _run()
    rows = [
        _candidate_row(
            "capital_structure.equity_issuance",
            utility=0.016,
            risk_reduction=0.072,
            growth=0.01,
            rating_preservation=0.055,
            optionality=0.04,
            pass_probability=0.9,
            evaluation_confidence=0.83,
            precedent_confidence=0.43,
            params={"amount_usd": 100_000_000.0, "use_of_proceeds": "deleveraging"},
        ),
        _candidate_row(
            "capital_structure.refinancing",
            utility=0.016,
            risk_reduction=0.072,
            growth=0.006,
            rating_preservation=0.05,
            optionality=0.035,
            pass_probability=0.95,
            evaluation_confidence=0.83,
            precedent_confidence=0.33,
        ),
    ]

    plan_set = build_plan_set(
        run=run,
        feasible_candidates=[row["candidate"] for row in rows],
        precedent_matches=rows,
        registry=registry,
        top_plans=5,
    )

    assert plan_set["plans"][0]["steps"][0]["action_id"] == "capital_structure.equity_issuance"
    top_plan = plan_set["plans"][0]
    assert top_plan["score_components"]["structural_bonus"] >= 0.02
    assert top_plan["score_components"]["status_quo_hurdle"] == 0.0


def test_strong_equity_issuance_case_can_still_rank_first():
    registry = build_default_action_schema_registry()
    run = _run()
    rows = [
        _candidate_row(
            "capital_structure.equity_issuance",
            utility=0.09,
            risk_reduction=0.16,
            growth=0.14,
            rating_preservation=0.12,
            optionality=0.08,
            pass_probability=0.9,
            evaluation_confidence=0.84,
            precedent_confidence=0.42,
        ),
        _candidate_row(
            "capital_structure.refinancing",
            utility=0.04,
            risk_reduction=0.09,
            growth=0.03,
            rating_preservation=0.06,
            optionality=0.03,
            pass_probability=0.95,
            evaluation_confidence=0.82,
            precedent_confidence=0.33,
        ),
    ]

    plan_set = build_plan_set(
        run=run,
        feasible_candidates=[row["candidate"] for row in rows],
        precedent_matches=rows,
        registry=registry,
        top_plans=5,
    )

    assert plan_set["plans"][0]["steps"][0]["action_id"] == "capital_structure.equity_issuance"
    top_plan = plan_set["plans"][0]
    assert top_plan["score_components"]["status_quo_hurdle"] == 0.0


def test_weak_governance_default_does_not_beat_financing_action():
    registry = build_default_action_schema_registry()
    run = _run()
    rows = [
        _candidate_row(
            "governance.board_refresh",
            utility=0.0115,
            risk_reduction=0.0,
            growth=0.0,
            rating_preservation=0.0,
            optionality=0.0,
            pass_probability=0.95,
            evaluation_confidence=0.82,
            precedent_confidence=0.43,
        ),
        _candidate_row(
            "capital_structure.refinancing",
            utility=0.0505,
            risk_reduction=0.0,
            growth=0.0,
            rating_preservation=0.0,
            optionality=0.0,
            pass_probability=0.95,
            evaluation_confidence=0.83,
            precedent_confidence=0.32,
        ),
    ]

    plan_set = build_plan_set(
        run=run,
        feasible_candidates=[row["candidate"] for row in rows],
        precedent_matches=rows,
        registry=registry,
        top_plans=5,
    )

    assert plan_set["plans"][0]["steps"][0]["action_id"] == "capital_structure.refinancing"
    governance_plan = next(plan for plan in plan_set["plans"] if plan["steps"][0]["action_id"] == "governance.board_refresh")
    assert governance_plan["score_components"]["action_specific_penalty"] >= 0.07


def test_weak_stock_split_default_does_not_beat_financing_action():
    registry = build_default_action_schema_registry()
    run = _run()
    rows = [
        _candidate_row(
            "governance.stock_split",
            utility=0.012,
            risk_reduction=0.0,
            growth=0.0,
            rating_preservation=0.0,
            optionality=0.0,
            pass_probability=0.95,
            evaluation_confidence=0.82,
            precedent_confidence=0.36,
        ),
        _candidate_row(
            "capital_structure.refinancing",
            utility=0.049,
            risk_reduction=0.01,
            growth=0.0,
            rating_preservation=0.01,
            optionality=0.0,
            pass_probability=0.95,
            evaluation_confidence=0.83,
            precedent_confidence=0.32,
        ),
    ]

    plan_set = build_plan_set(
        run=run,
        feasible_candidates=[row["candidate"] for row in rows],
        precedent_matches=rows,
        registry=registry,
        top_plans=5,
    )

    assert plan_set["plans"][0]["steps"][0]["action_id"] == "capital_structure.refinancing"
    split_plan = next(plan for plan in plan_set["plans"] if plan["steps"][0]["action_id"] == "governance.stock_split")
    assert split_plan["score_components"]["action_specific_penalty"] >= 0.07
    assert split_plan["score_components"]["raw_total_score"] < plan_set["plans"][0]["score_components"]["raw_total_score"]


def test_strong_restructuring_case_can_still_rank_first():
    registry = build_default_action_schema_registry()
    run = _run()
    rows = [
        _candidate_row(
            "restructuring.working_capital_program",
            utility=0.08,
            risk_reduction=0.12,
            growth=0.03,
            rating_preservation=0.09,
            optionality=0.02,
            pass_probability=0.9,
            evaluation_confidence=0.81,
            precedent_confidence=0.42,
        ),
        _candidate_row(
            "capital_structure.refinancing",
            utility=0.03,
            risk_reduction=0.06,
            growth=0.01,
            rating_preservation=0.04,
            optionality=0.01,
            pass_probability=0.95,
            evaluation_confidence=0.83,
            precedent_confidence=0.3,
        ),
    ]

    plan_set = build_plan_set(
        run=run,
        feasible_candidates=[row["candidate"] for row in rows],
        precedent_matches=rows,
        registry=registry,
        top_plans=5,
    )

    assert plan_set["plans"][0]["steps"][0]["action_id"] == "restructuring.working_capital_program"


def test_thin_restructuring_default_does_not_beat_capital_action():
    registry = build_default_action_schema_registry()
    run = _run()
    rows = [
        _candidate_row(
            "restructuring.working_capital_program",
            utility=0.017,
            risk_reduction=0.0,
            growth=0.0,
            rating_preservation=0.0,
            optionality=0.0,
            pass_probability=0.95,
            evaluation_confidence=0.82,
            precedent_confidence=0.41,
        ),
        _candidate_row(
            "capital_structure.new_debt_issuance",
            utility=0.04,
            risk_reduction=0.03,
            growth=0.02,
            rating_preservation=0.03,
            optionality=0.01,
            pass_probability=0.95,
            evaluation_confidence=0.83,
            precedent_confidence=0.31,
        ),
    ]

    plan_set = build_plan_set(
        run=run,
        feasible_candidates=[row["candidate"] for row in rows],
        precedent_matches=rows,
        registry=registry,
        top_plans=5,
    )

    assert plan_set["plans"][0]["steps"][0]["action_id"] == "capital_structure.new_debt_issuance"
    restructuring_plan = next(plan for plan in plan_set["plans"] if plan["steps"][0]["action_id"] == "restructuring.working_capital_program")
    assert restructuring_plan["score_components"]["action_specific_penalty"] >= 0.1
    assert restructuring_plan["score_components"]["raw_total_score"] < plan_set["plans"][0]["score_components"]["raw_total_score"]


def test_unsupported_actions_are_demoted_below_supported_actions():
    registry = build_default_action_schema_registry()
    run = _run()
    supported = _candidate_row(
        "capital_structure.refinancing",
        utility=0.08,
        risk_reduction=0.19,
        growth=0.04,
        precedent_confidence=0.32,
        causal=True,
    )
    unsupported_restructuring = _candidate_row(
        "restructuring.working_capital_program",
        utility=0.13,
        risk_reduction=0.08,
        evaluation_confidence=0.62,
        precedent_confidence=0.0,
        causal=False,
    )
    unsupported_governance = _candidate_row(
        "governance.board_refresh",
        utility=0.12,
        optionality=0.04,
        evaluation_confidence=0.6,
        precedent_confidence=0.0,
        causal=False,
    )

    plan_set = build_plan_set(
        run=run,
        feasible_candidates=[supported["candidate"], unsupported_restructuring["candidate"], unsupported_governance["candidate"]],
        precedent_matches=[supported],
        registry=registry,
        top_plans=5,
    )

    ranked_first_actions = [[step["action_id"] for step in plan["steps"]][0] for plan in plan_set["plans"][:3]]
    assert ranked_first_actions[0] == "capital_structure.refinancing"


def test_real_run_like_buyback_refi_case_prefers_financing_then_buyback():
    registry = build_default_action_schema_registry()
    run = _run()
    rows = [
        _candidate_row(
            "capital_structure.new_debt_issuance",
            utility=0.057,
            risk_reduction=-0.181,
            growth=0.267,
            rating_preservation=0.032,
            optionality=0.026,
            pass_probability=0.95,
            evaluation_confidence=0.70,
            precedent_confidence=0.348,
        ),
        _candidate_row(
            "capital_structure.refinancing",
            utility=0.057,
            risk_reduction=-0.181,
            growth=0.267,
            rating_preservation=0.032,
            optionality=0.026,
            pass_probability=0.95,
            evaluation_confidence=0.70,
            precedent_confidence=0.32,
        ),
        _candidate_row(
            "capital_return.open_market_buyback",
            utility=0.047,
            risk_reduction=-0.016,
            growth=0.002,
            rating_preservation=-0.01,
            optionality=0.462,
            pass_probability=0.90,
            evaluation_confidence=0.636,
            precedent_confidence=0.342,
        ),
        _candidate_row(
            "capital_return.accelerated_share_repurchase",
            utility=0.045,
            risk_reduction=-0.019,
            growth=-0.001,
            rating_preservation=-0.013,
            optionality=0.459,
            pass_probability=0.95,
            evaluation_confidence=0.635,
            precedent_confidence=0.341,
        ),
        _candidate_row(
            "capital_return.dividend_increase",
            utility=0.269,
            risk_reduction=-0.021,
            growth=-0.145,
            rating_preservation=-0.015,
            optionality=-0.027,
            pass_probability=0.95,
            evaluation_confidence=0.712,
            precedent_confidence=0.48,
        ),
        _candidate_row(
            "capital_return.dividend_cut",
            utility=0.141,
            risk_reduction=-0.021,
            growth=-0.003,
            rating_preservation=-0.015,
            optionality=-0.027,
            pass_probability=0.95,
            evaluation_confidence=0.793,
            precedent_confidence=0.473,
        ),
    ]

    plan_set = build_plan_set(
        run=run,
        feasible_candidates=[row["candidate"] for row in rows],
        precedent_matches=rows,
        registry=registry,
        top_plans=5,
    )

    top_actions = [step["action_id"] for step in plan_set["plans"][0]["steps"]]
    assert top_actions[0] in {"capital_structure.new_debt_issuance", "capital_structure.refinancing"}
    assert top_actions[1] in {
        "capital_return.open_market_buyback",
        "capital_return.accelerated_share_repurchase",
    }


def test_real_run_like_divestiture_case_prefers_divestiture():
    registry = build_default_action_schema_registry()
    run = _run()
    rows = [
        _candidate_row(
            "portfolio.divestiture_partial",
            utility=0.104,
            risk_reduction=0.021,
            growth=0.007,
            rating_preservation=0.012,
            optionality=0.026,
            pass_probability=0.59,
            evaluation_confidence=0.80,
            precedent_confidence=0.351,
        ),
        _candidate_row(
            "capital_structure.new_debt_issuance",
            utility=0.019,
            risk_reduction=-0.362,
            growth=0.230,
            rating_preservation=0.032,
            optionality=0.026,
            pass_probability=0.95,
            evaluation_confidence=0.653,
            precedent_confidence=0.301,
        ),
        _candidate_row(
            "capital_structure.refinancing",
            utility=0.019,
            risk_reduction=-0.362,
            growth=0.230,
            rating_preservation=0.032,
            optionality=0.026,
            pass_probability=0.95,
            evaluation_confidence=0.682,
            precedent_confidence=0.306,
        ),
        _candidate_row(
            "capital_return.dividend_increase",
            utility=0.415,
            risk_reduction=-0.021,
            growth=-0.145,
            rating_preservation=-0.015,
            optionality=-0.027,
            pass_probability=0.95,
            evaluation_confidence=0.663,
            precedent_confidence=0.412,
        ),
        _candidate_row(
            "capital_return.dividend_cut",
            utility=0.087,
            risk_reduction=-0.021,
            growth=-0.003,
            rating_preservation=-0.015,
            optionality=-0.027,
            pass_probability=0.95,
            evaluation_confidence=0.811,
            precedent_confidence=0.358,
        ),
    ]

    plan_set = build_plan_set(
        run=run,
        feasible_candidates=[row["candidate"] for row in rows],
        precedent_matches=rows,
        registry=registry,
        top_plans=5,
    )

    assert [step["action_id"] for step in plan_set["plans"][0]["steps"]] == ["portfolio.divestiture_partial"]


def test_real_run_like_acquisition_negative_case_avoids_mna():
    registry = build_default_action_schema_registry()
    run = _run()
    rows = [
        _candidate_row(
            "mna.platform_acquisition",
            utility=-0.296,
            risk_reduction=-0.021,
            growth=-0.191,
            rating_preservation=-0.016,
            optionality=0.002,
            pass_probability=0.51,
            evaluation_confidence=0.728,
            precedent_confidence=0.370,
        ),
        _candidate_row(
            "mna.tuck_in_acquisition",
            utility=-0.295,
            risk_reduction=-0.022,
            growth=-0.189,
            rating_preservation=-0.017,
            optionality=0.002,
            pass_probability=0.59,
            evaluation_confidence=0.738,
            precedent_confidence=0.392,
        ),
        _candidate_row(
            "capital_structure.new_debt_issuance",
            utility=0.029,
            risk_reduction=-0.201,
            growth=0.257,
            rating_preservation=0.026,
            optionality=0.021,
            pass_probability=0.59,
            evaluation_confidence=0.659,
            precedent_confidence=0.307,
        ),
        _candidate_row(
            "capital_structure.refinancing",
            utility=0.029,
            risk_reduction=-0.201,
            growth=0.257,
            rating_preservation=0.026,
            optionality=0.021,
            pass_probability=0.59,
            evaluation_confidence=0.687,
            precedent_confidence=0.306,
        ),
        _candidate_row(
            "capital_return.open_market_buyback",
            utility=0.036,
            risk_reduction=-0.017,
            growth=-0.003,
            rating_preservation=-0.012,
            optionality=0.417,
            pass_probability=0.59,
            evaluation_confidence=0.667,
            precedent_confidence=0.342,
        ),
    ]

    plan_set = build_plan_set(
        run=run,
        feasible_candidates=[row["candidate"] for row in rows],
        precedent_matches=rows,
        registry=registry,
        top_plans=5,
    )

    top_three = plan_set["plans"][:3]
    assert not any(
        any(step["action_id"].startswith("mna.") for step in plan["steps"])
        for plan in top_three
    )
