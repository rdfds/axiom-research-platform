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


