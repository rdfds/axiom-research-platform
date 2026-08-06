from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd

from scripts.paper.build_case_studies import build_case_studies


ROOT = Path(__file__).resolve().parents[1]


def test_smoke_market_expectations_experiment_writes_schema_and_tables(tmp_path: Path) -> None:
    out_dir = tmp_path / "paper_smoke"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "paper" / "run_market_expectations_experiments.py"),
            "--preset",
            "smoke",
            "--out-dir",
            str(out_dir),
        ],
        cwd=ROOT,
        check=True,
    )

    report_path = out_dir / "market_expectations_experiment_report.json"
    report = json.loads(report_path.read_text())
    assert report["schema"] == "market_expectations_paper_report_v1"
    assert report["metadata"]["causal_claim"] is False
    assert report["metadata"]["excluded_sectors"] == ["10", "40", "55", "60"]
    assert report["splits"]

    evaluation_types = {row["evaluation_type"] for row in report["evaluations"]}
    assert {"actual", "baseline", "placebo", "sanity_check"}.issubset(evaluation_types)

    evaluations = pd.DataFrame(report["evaluations"])
    actual_lambdas = sorted(evaluations[evaluations["evaluation_type"].eq("actual")]["lambda"].dropna().unique())
    assert actual_lambdas == [0.0, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0]
    assert evaluations["mae_improvement_vs_fundamentals"].notna().any()

    expected_tables = {
        "table_1_universe",
        "table_2_baselines",
        "table_3_lambda_ablation",
        "table_4_placebos",
        "table_5_sector_robustness",
        "table_6_driver_family_horizon",
    }
    for table_name in expected_tables:
        assert (out_dir / "tables" / f"{table_name}.csv").exists()
        assert (out_dir / "tables" / f"{table_name}.md").exists()


def test_smoke_fixture_has_no_forward_leakage() -> None:
    fixture = pd.read_csv(ROOT / "paper" / "fixtures" / "market_expectations_smoke_panel.csv")
    predictor_dates = pd.to_datetime(fixture["predictor_available_date"])
    outcome_dates = pd.to_datetime(fixture["outcome_available_date"])
    assert (outcome_dates > predictor_dates).all()


def test_publication_preset_can_use_private_panel_env_adapter(tmp_path: Path) -> None:
    out_dir = tmp_path / "paper_publication"
    env = os.environ.copy()
    env["AXIOM_MARKET_EXPECTATIONS_PANEL"] = str(ROOT / "paper" / "fixtures" / "market_expectations_smoke_panel.csv")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "paper" / "run_market_expectations_experiments.py"),
            "--preset",
            "publication",
            "--out-dir",
            str(out_dir),
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )

    report = json.loads((out_dir / "market_expectations_experiment_report.json").read_text())
    assert report["metadata"]["preset"] == "publication"
    assert (out_dir / "tables" / "table_4_placebos.csv").exists()


def test_case_studies_reconcile_gap_decomposition(tmp_path: Path) -> None:
    out = build_case_studies(
        ROOT / "examples" / "hd_market_expectations" / "valuation_driver_data.sample.json",
        tmp_path / "case_studies",
    )
    payload = json.loads(Path(out["json"]).read_text())
    cases = payload["cases"]
    assert {case["ticker"] for case in cases} >= {"HD", "SYN_RESIDUAL", "SYN_MULTI"}

    for case in cases:
        reconciled = (
            case["driver_underwritten_log"]
            + case["residual_outside_measured_drivers_log"]
            + case["reconciliation_error_log"]
        )
        assert abs(case["gap_log"] - reconciled) < 1e-9

    table = pd.read_csv(out["csv"])
    assert {"gap_log", "driver_underwritten_log", "residual_log", "coverage_ratio"}.issubset(table.columns)
