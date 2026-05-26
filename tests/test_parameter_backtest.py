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


