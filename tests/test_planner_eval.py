from __future__ import annotations

import json
from pathlib import Path

from src.planner_eval import build_planner_eval_report, render_planner_eval_markdown


def test_build_planner_eval_report_and_markdown(tmp_path: Path):
    runs_root = tmp_path / "runs_root"
    (runs_root / "runs").mkdir(parents=True, exist_ok=True)
    artifacts = runs_root / "artifacts" / "run_id=run-1"
    artifacts.mkdir(parents=True, exist_ok=True)

    run_payload = {
        "run_id": "run-1",
        "company_id": "0000320193",
    }
    (runs_root / "runs" / "run_id=run-1.json").write_text(json.dumps(run_payload))

    feasibility = {
        "results": [
            {
                "feasible": True,
                "pass_probability": 0.92,
                "action_candidate": {
                    "candidate_id": "cand-1",
                    "action_id": "capital_structure.refinancing",
                    "action_type": "capital_structure",
                    "evaluation_confidence": 0.71,
                    "feasibility": {"pass_probability": 0.92},
                    "impact_distribution": {
                        "objectives": {
                            "value_creation": {"median": 0.08},
                            "risk_reduction": {"median": 0.18},
                            "growth": {"median": 0.04},
                            "rating_preservation": {"median": 0.03},
                            "optionality": {"median": 0.01},
                        },
                        "key_drivers": [
                            {"driver_name": "causal_model_blend_weight", "contribution": 0.2},
                        ],
                    },
                },
            }
        ]
    }
    (artifacts / "FeasibilityResults.json").write_text(json.dumps(feasibility))

    precedent = {
        "results": [
            {
                "candidate": {
                    "candidate_id": "cand-1",
                    "action_id": "capital_structure.refinancing",
                },
                "precedent_pack": {
                    "calibration_confidence": 0.33,
                },
            }
        ]
    }
    (artifacts / "PrecedentMatches.json").write_text(json.dumps(precedent))

    plan_set = {
        "plans": [
            {
                "plan_id": "plan_refi",
                "score": 0.41,
                "score_components": {"raw_total_score": 0.41},
                "summary_explanation": "Refinancing reduces funding pressure before other actions.",
                "steps": [
                    {
                        "action_id": "capital_structure.refinancing",
                        "prerequisites": [],
                        "explanation": {
                            "problem_statement": "Funding pressure is elevated.",
                            "why_this_action": "Refinancing extends maturities.",
                            "why_now": "The company can act immediately.",
                        },
                    }
                ],
            }
        ]
    }
    (artifacts / "PlanSet.json").write_text(json.dumps(plan_set))

    report = build_planner_eval_report(runs_roots=[runs_root], review_count=10, rebuild_plan_set=False)
    assert report["runs_analyzed"] == 1
    assert report["aggregate"]["positive_top_plan_rate"] == 1.0
    assert report["aggregate"]["supported_top_plan_rate"] == 1.0
    assert report["cases"][0]["bucket"] == "capital_structure"
    assert report["cases"][0]["top_plan"]["action_path"] == "capital_structure.refinancing"

    markdown = render_planner_eval_markdown(report)
    assert "Planner Evaluation Report" in markdown
    assert "Top-1 plan is strategically sensible" in markdown
    assert "capital_structure.refinancing" in markdown
