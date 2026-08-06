from __future__ import annotations

import json
from pathlib import Path

from src.dossier_eval import build_dossier_eval_report, render_dossier_eval_markdown
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
                    "liquidity.available_for_actions": {"value": 500_000_000.0},
                    "market.market_cap": {"value": 5_000_000_000.0},
                    "capital_structure.net_leverage": {"value": 1.6},
                    "capital_structure.maturity_wall_ratio_24m": {"value": 0.18},
                    "operating.fcf_conversion": {"value": 0.78},
                    "operating.revenue_yoy_last_q": {"value": 0.02},
                    "market.credit_window_proxy": {"value": 0.72},
                    "market.credit_spread_percentile_2y": {"value": 62.0},
                    "market.equity_window_proxy": {"value": 0.61},
                    "capital_structure.rating_state": {"value": {"rating": "BBB", "outlook": "stable", "score": 9.5}},
                    "strategic.intent.return_capital_priority": {"value": 0.83},
                },
            }
        )
    )
    return root


def _candidate(action_id: str) -> dict:
    action_type, action_subtype = action_id.split(".", 1)
    return {
        "candidate_id": f"cand-{action_subtype}",
        "run_id": "run-1",
        "action_id": action_id,
        "action_type": action_type,
        "action_subtype": action_subtype,
        "parameters": {},
        "feasibility": {"pass_probability": 0.94},
        "mechanism_activation": {
            "mechanisms": [{"mechanism_id": "capital_efficiency", "activation_strength": 0.66}],
        },
        "impact_distribution": {
            "objectives": {
                "value_creation": {"median": 0.22},
                "risk_reduction": {"median": 0.11},
                "growth": {"median": 0.0},
                "rating_preservation": {"median": 0.04},
                "optionality": {"median": 0.05},
            },
            "key_drivers": [{"driver_name": "causal_model_blend_weight", "contribution": 0.2}],
            "uncertainty_score": 0.15,
        },
        "risks": [{"explanation": f"{action_id} requires disciplined execution."}],
        "structural_sanity_flags": [],
        "evaluation_confidence": 0.76,
    }


def test_build_dossier_eval_report_and_markdown(tmp_path: Path):
    runs_root = tmp_path / "runs_root"
    (runs_root / "runs").mkdir(parents=True, exist_ok=True)
    artifacts = runs_root / "artifacts" / "run_id=run-1"
    artifacts.mkdir(parents=True, exist_ok=True)

    run = _run()
    (runs_root / "runs" / "run_id=run-1.json").write_text(json.dumps(run.to_dict()))

    buyback = _candidate("capital_return.open_market_buyback")
    refi = _candidate("capital_structure.refinancing")
    precedent_pack = {
        "precedent_confidence": 0.41,
        "mismatch_diagnostics": {"out_of_sample_flag": False, "retrieval_tier": "exact"},
        "tail_events": [
            {
                "metric": "equity_return_vs_sector",
                "horizon": "12m",
                "value": -0.44,
                "description": "Bottom decile historical outcome.",
            }
        ],
        "outcome_distributions": {"horizon_12m": {"valuation_multiple_change": {"sample_size": 24}}},
    }
    (artifacts / "FeasibilityResults.json").write_text(
        json.dumps(
            {
                "results": [
                    {
                        "candidate": {"candidate_id": buyback["candidate_id"], "action_id": buyback["action_id"]},
                        "action_candidate": buyback,
                        "feasible": True,
                        "pass_probability": 0.94,
                    },
                    {
                        "candidate": {"candidate_id": refi["candidate_id"], "action_id": refi["action_id"]},
                        "action_candidate": refi,
                        "feasible": True,
                        "pass_probability": 0.94,
                    },
                ]
            }
        )
    )
    (artifacts / "PrecedentMatches.json").write_text(
        json.dumps(
            {
                "results": [
                    {"candidate": buyback, "precedent_pack": precedent_pack},
                    {"candidate": refi, "precedent_pack": precedent_pack},
                ]
            }
        )
    )

    report = build_dossier_eval_report(
        runs_roots=[runs_root],
        snapshot_root=_snapshot_root(tmp_path),
        review_count=5,
        expected_postures={"0000320193": "act_now"},
    )
    markdown = render_dossier_eval_markdown(report)

    assert report["runs_analyzed"] == 1
    assert report["aggregate"]["heuristic_overall_mean"] > 0.0
    assert report["aggregate"]["status_quo_comparison_rate"] == 1.0
    assert report["aggregate"]["sizing_specificity_rate"] == 1.0
    assert report["aggregate"]["parameter_optimization_rate"] == 1.0
    assert report["aggregate"]["regret_analysis_rate"] == 1.0
    assert report["aggregate"]["scenario_sizing_rate"] == 1.0
    assert report["aggregate"]["rating_analysis_rate"] == 1.0
    assert report["aggregate"]["signaling_analysis_rate"] == 1.0
    assert report["aggregate"]["posture_match_rate"] == 1.0
    assert report["review_queue"]
    assert "Board Dossier Evaluation Report" in markdown
    assert "Why now:" in markdown
    assert "Sizing:" in markdown
    assert "Parameter summary:" in markdown
    assert "Regret balance:" in markdown


def test_build_dossier_eval_report_scores_negative_wait_case(tmp_path: Path):
    runs_root = tmp_path / "runs_root_wait"
    (runs_root / "runs").mkdir(parents=True, exist_ok=True)
    artifacts = runs_root / "artifacts" / "run_id=run-wait"
    artifacts.mkdir(parents=True, exist_ok=True)

    run = _run()
    run.run_id = "run-wait"
    run.company_id = "0000099999"
    (runs_root / "runs" / "run_id=run-wait.json").write_text(json.dumps(run.to_dict()))

    snap_root = tmp_path / "snapshots_wait"
    keyed = snap_root / "keyed" / "as_of_date=2026-02-28"
    keyed.mkdir(parents=True, exist_ok=True)
    (keyed / "company_id=0000099999.json").write_text(
        json.dumps(
            {
                "company_id": "0000099999",
                "as_of_time": "2026-02-28T00:00:00+00:00",
                "features": {
                    "liquidity.available_for_actions": {"value": 850_000_000.0},
                    "market.market_cap": {"value": 8_500_000_000.0},
                    "capital_structure.net_leverage": {"value": 1.85},
                    "capital_structure.maturity_wall_ratio_24m": {"value": 0.22},
                "operating.fcf_conversion": {"value": 0.82},
                "operating.revenue_yoy_last_q": {"value": 0.01},
                "market.credit_window_proxy": {"value": 0.74},
                "market.credit_spread_percentile_2y": {"value": 79.0},
                "market.equity_window_proxy": {"value": 0.57},
                "capital_structure.rating_state": {"value": {"rating": "BBB-", "outlook": "negative", "score": 10.0}},
                "strategic.intent.return_capital_priority": {"value": 0.87},
            },
        }
        )
    )

    weak = _candidate("capital_return.dividend_initiate")
    weak["impact_distribution"]["objectives"]["value_creation"]["median"] = 0.01
    weak["impact_distribution"]["objectives"]["optionality"]["median"] = -0.04
    weak["evaluation_confidence"] = 0.22
    weak["feasibility"]["pass_probability"] = 0.58
    weak_precedent = {
        "precedent_confidence": 0.12,
        "mismatch_diagnostics": {"out_of_sample_flag": False, "retrieval_tier": "exact"},
        "tail_events": [],
        "outcome_distributions": {"horizon_12m": {"valuation_multiple_change": {"sample_size": 8}}},
    }
    (artifacts / "FeasibilityResults.json").write_text(
        json.dumps(
            {
                "results": [
                    {
                        "candidate": {"candidate_id": weak["candidate_id"], "action_id": weak["action_id"]},
                        "action_candidate": weak,
                        "feasible": True,
                        "pass_probability": 0.58,
                    }
                ]
            }
        )
    )
    (artifacts / "PrecedentMatches.json").write_text(
        json.dumps(
            {
                "results": [
                    {"candidate": weak, "precedent_pack": weak_precedent},
                ]
            }
        )
    )

    report = build_dossier_eval_report(
        runs_roots=[runs_root],
        snapshot_root=snap_root,
        review_count=5,
        expected_postures={"0000099999": "wait"},
    )

    assert report["cases"][0]["predicted_posture"] == "wait"
    assert report["cases"][0]["posture_match"] is True
    assert report["aggregate"]["expected_posture_coverage_rate"] == 1.0
    assert report["aggregate"]["negative_case_accuracy"] == 1.0
