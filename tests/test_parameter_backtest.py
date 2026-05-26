from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.parameter_backtest import build_parameter_backtest_report, render_parameter_backtest_markdown
from src.recommendation_run import (
    ConstraintSet,
    DataCutoffSpec,
    FrozenStateReference,
    ModelVersionBundle,
    ObjectiveVector,
    RecommendationRun,
    ScenarioAssumptions,
)


def _run(run_id: str, company_id: str) -> RecommendationRun:
    return RecommendationRun(
        run_id=run_id,
        company_id=company_id,
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
    (keyed / "company_id=0001111111.json").write_text(
        json.dumps(
            {
                "company_id": "0001111111",
                "as_of_time": "2026-02-28T00:00:00+00:00",
                "features": {
                    "liquidity.available_for_actions": {"value": 200_000_000.0},
                    "market.market_cap": {"value": 2_000_000_000.0},
                    "capital_structure.net_leverage": {"value": 1.7},
                    "capital_structure.maturity_wall_ratio_24m": {"value": 0.08},
                    "operating.fcf_conversion": {"value": 0.8},
                    "market.credit_window_proxy": {"value": 0.72},
                    "market.equity_window_proxy": {"value": 0.68},
                },
            }
        )
    )
    (keyed / "company_id=0002222222.json").write_text(
        json.dumps(
            {
                "company_id": "0002222222",
                "as_of_time": "2026-02-28T00:00:00+00:00",
                "features": {
                    "liquidity.available_for_actions": {"value": 150_000_000.0},
                    "market.market_cap": {"value": 1_500_000_000.0},
                    "capital_structure.net_leverage": {"value": 2.4},
                    "capital_structure.maturity_wall_ratio_24m": {"value": 0.22},
                    "operating.fcf_conversion": {"value": 0.65},
                    "market.credit_window_proxy": {"value": 0.49},
                    "market.equity_window_proxy": {"value": 0.55},
                },
            }
        )
    )
    return root


def _candidate(action_id: str, *, run_id: str, value_creation: float, params: dict | None = None) -> dict:
    action_type, action_subtype = action_id.split(".", 1)
    return {
        "candidate_id": f"{run_id}-{action_subtype}",
        "run_id": run_id,
        "action_id": action_id,
        "action_type": action_type,
        "action_subtype": action_subtype,
        "parameters": params or {},
        "feasibility": {"pass_probability": 0.92},
        "mechanism_activation": {
            "mechanisms": [{"mechanism_id": "capital_efficiency", "activation_strength": 0.6}],
        },
        "impact_distribution": {
            "objectives": {
                "value_creation": {"median": value_creation},
                "risk_reduction": {"median": 0.05},
                "growth": {"median": 0.0},
                "rating_preservation": {"median": 0.02},
                "optionality": {"median": 0.03},
            },
            "uncertainty_score": 0.12,
        },
        "risks": [{"explanation": "Execution risk exists."}],
        "structural_sanity_flags": [],
        "evaluation_confidence": 0.72,
    }


def _write_run(
    *,
    runs_root: Path,
    run: RecommendationRun,
    candidates: list[dict],
) -> None:
    (runs_root / "runs").mkdir(parents=True, exist_ok=True)
    (runs_root / "artifacts" / f"run_id={run.run_id}").mkdir(parents=True, exist_ok=True)
    (runs_root / "runs" / f"run_id={run.run_id}.json").write_text(json.dumps(run.to_dict()))
    artifacts = runs_root / "artifacts" / f"run_id={run.run_id}"
    (artifacts / "FeasibilityResults.json").write_text(
        json.dumps(
            {
                "results": [
                    {
                        "candidate": {"candidate_id": cand["candidate_id"], "action_id": cand["action_id"]},
                        "action_candidate": cand,
                        "feasible": True,
                        "pass_probability": cand["feasibility"]["pass_probability"],
                    }
                    for cand in candidates
                ]
            }
        )
    )
    (artifacts / "PrecedentMatches.json").write_text(
        json.dumps(
            {
                "results": [
                    {
                        "candidate": cand,
                        "precedent_pack": {
                            "precedent_confidence": 0.41,
                            "mismatch_diagnostics": {"out_of_sample_flag": False, "retrieval_tier": "exact"},
                            "outcome_distributions": {"horizon_12m": {"valuation_multiple_change": {"sample_size": 24}}},
                        },
                    }
                    for cand in candidates
                ]
            }
        )
    )


def _write_outcomes(path: Path) -> Path:
    rows = []
    for bucket, score_shift in (("small", 0.2), ("medium", 0.6), ("large", 0.1)):
        for i in range(30):
            size_ratio = {"small": 0.03, "medium": 0.09, "large": 0.32}[bucket]
            rows.append(
                {
                    "normalized_action_id": None,
                    "normalized_action_family": "capital_return",
                    "normalized_action_subfamily": "buyback",
                    "family_scale_bucket": bucket,
                    "action_size": size_ratio * 2_000_000_000.0,
                    "base_market_cap": 2_000_000_000.0,
                    "revenue_delta": 0.01 + score_shift,
                    "margin_delta": 0.02 + score_shift,
                    "eps_delta": 0.03 + score_shift,
                    "roic_delta": 0.04 + score_shift,
                    "fcf_margin_delta": 0.01 + score_shift,
                    "outcome_pe_12m": 0.02 + score_shift,
                    "outcome_ev_ebitda_12m": 0.01 + score_shift,
                    "rating_migration_12m": 0.0 + score_shift,
                    "leverage_delta": 0.4 - score_shift,
                    "credit_spread_change_12m": 0.3 - score_shift,
                }
            )
    for i in range(20):
        rows.append(
            {
                "normalized_action_id": "capital_return.dividend_initiate",
                "normalized_action_family": "capital_return",
                "normalized_action_subfamily": "dividend_initiate",
                "family_scale_bucket": None,
                "action_size": None,
                "base_market_cap": 1_500_000_000.0,
                "revenue_delta": 0.03,
                "margin_delta": 0.01,
                "eps_delta": 0.02,
                "roic_delta": 0.01,
                "fcf_margin_delta": 0.0,
                "outcome_pe_12m": 0.01,
                "outcome_ev_ebitda_12m": 0.01,
                "rating_migration_12m": 0.0,
                "leverage_delta": 0.05,
                "credit_spread_change_12m": 0.04,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_parquet(path, index=False)
    return path


