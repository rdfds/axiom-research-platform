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


def test_build_head_to_head_report_and_markdown(tmp_path: Path):
    runs_root = tmp_path / "runs_root"
    (runs_root / "runs").mkdir(parents=True, exist_ok=True)
    artifacts = runs_root / "artifacts" / "run_id=run-1"
    artifacts.mkdir(parents=True, exist_ok=True)
    baseline_dir = tmp_path / "baselines"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    run = _run()
    (runs_root / "runs" / "run_id=run-1.json").write_text(json.dumps(run.to_dict()))

    buyback = _candidate("capital_return.open_market_buyback", value_creation=0.24)
    refi = _candidate("capital_structure.refinancing", value_creation=0.11)
    precedent_pack = {
        "precedent_confidence": 0.43,
        "mismatch_diagnostics": {"out_of_sample_flag": False, "retrieval_tier": "exact"},
        "tail_events": [
            {"metric": "equity_return_vs_sector", "horizon": "12m", "value": -0.38, "description": "Bottom decile historical outcome."}
        ],
        "outcome_distributions": {"horizon_12m": {"valuation_multiple_change": {"sample_size": 25}}},
    }
    (artifacts / "FeasibilityResults.json").write_text(
        json.dumps(
            {
                "results": [
                    {"candidate": {"candidate_id": buyback["candidate_id"], "action_id": buyback["action_id"]}, "action_candidate": buyback, "feasible": True},
                    {"candidate": {"candidate_id": refi["candidate_id"], "action_id": refi["action_id"]}, "action_candidate": refi, "feasible": True},
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

    (baseline_dir / "company_id=0000320193.md").write_text(
        "\n".join(
            [
                "Baseline-Type: activist",
                "Task-Match: direct",
                "",
                "# Problem",
                "The company has excess capital and limited near-term better uses.",
                "",
                "# Recommendation",
                "Prefer buybacks because they are more flexible than a special dividend.",
                "",
                "# Why Now",
                "Liquidity is ample, leverage is moderate, and waiting mostly leaves capital idle.",
                "",
                "# Alternatives",
                "- A special dividend is less flexible.",
                "- A tuck-in deal does not clear the hurdle today.",
                "",
                "# Risks",
                "- If conditions weaken, the company may need the cash back.",
                "",
                "# Kill Criteria",
                "- Stop if leverage rises materially.",
                "- Stop if a better strategic use for capital appears.",
                "",
                "# Evidence",
                "- Net leverage is 1.7x.",
                "- Liquidity is about 10% of market value.",
            ]
        )
    )
    realized_path = tmp_path / "realized.parquet"
    pd.DataFrame(
        [
            {
                "company_id": "0000320193",
                "action_date": "2026-04-15",
                "normalized_action_id": "capital_return.open_market_buyback",
                "normalized_action_family": "capital_return",
            },
            {
                "company_id": "0000320193",
                "action_date": "2026-05-20",
                "normalized_action_id": "capital_structure.refinancing",
                "normalized_action_family": "capital_structure",
            },
        ]
    ).to_parquet(realized_path, index=False)

    report = build_head_to_head_report(
        runs_roots=[runs_root],
        snapshot_root=_snapshot_root(tmp_path),
        baseline_dir=baseline_dir,
        realized_outcomes_path=realized_path,
        review_count=5,
    )
    markdown = render_head_to_head_markdown(report)
    export_summary = export_blinded_packets(
        report=report,
        packets_out_dir=tmp_path / "packets",
        answer_key_out=tmp_path / "answer_key.json",
    )

    assert report["runs_analyzed"] == 1
    assert report["cases"][0]["comparison"]["winner"] in {"model", "baseline", "tie"}
    assert report["cases"][0]["baseline_packet"]["baseline_type"] == "activist"
    assert report["cases"][0]["baseline_packet"]["task_match"] == "direct"
    assert report["cases"][0]["blinded_review"]["packet_A"]
    assert report["aggregate"]["sign_test_p_value"] is not None
    assert report["aggregate"]["by_task_match"]["direct"]["case_count"] == 1
    assert report["aggregate"]["ex_post"]["coverage_rate"] == 1.0
    assert report["cases"][0]["ex_post"]["model"]["score"] == 1.0
    assert report["cases"][0]["ex_post"]["baseline"]["score"] == 1.0
    assert export_summary["exported_packets"] == 1
    assert (tmp_path / "packets" / "run_id=run-1.json").exists()
    assert "Head-To-Head Benchmark Report" in markdown
    assert "By Task Match" in markdown
    assert "Winner:" in markdown
    assert "Ex-Post Alignment" in markdown
