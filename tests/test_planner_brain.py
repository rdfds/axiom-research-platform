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


