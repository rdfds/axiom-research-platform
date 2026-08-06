#!/usr/bin/env python3
"""Canonical runner for the market-implied valuation-gap paper experiments."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "paper" / "fixtures" / "market_expectations_smoke_panel.csv"
DEFAULT_LAMBDAS = (0.0, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 1.0)
DEFAULT_EXCLUDED_SECTORS = ("10", "40", "55", "60")


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [_json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return value


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> np.ndarray:
    x_train = np.column_stack([np.ones(len(train)), train[cols].to_numpy(dtype=float)])
    y_train = train["delta"].to_numpy(dtype=float)
    coef = np.linalg.pinv(x_train) @ y_train
    x_test = np.column_stack([np.ones(len(test)), test[cols].to_numpy(dtype=float)])
    return x_test @ coef


def _rank_ic(actual: pd.Series, pred: pd.Series) -> float | None:
    if actual.nunique(dropna=True) <= 1 or pred.nunique(dropna=True) <= 1:
        return None
    try:
        out = float(actual.rank().corr(pred.rank()))
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _bootstrap_ci(actual: np.ndarray, baseline: np.ndarray, pred: np.ndarray, seed: int) -> tuple[float | None, float | None]:
    if len(actual) < 4:
        return None, None
    rng = np.random.default_rng(seed)
    vals = []
    idx = np.arange(len(actual))
    for _ in range(400):
        take = rng.choice(idx, size=len(idx), replace=True)
        base_mae = float(np.mean(np.abs(actual[take] - baseline[take])))
        model_mae = float(np.mean(np.abs(actual[take] - pred[take])))
        vals.append(1.0 - model_mae / base_mae if base_mae > 0 else np.nan)
    s = pd.Series(vals).replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return None, None
    return float(s.quantile(0.025)), float(s.quantile(0.975))


def _metric_row(
    *,
    group: tuple[str, str, str],
    model_name: str,
    evaluation_type: str,
    lam: float | None,
    actual: np.ndarray,
    fundamentals: np.ndarray,
    pred: np.ndarray,
    rows: int,
    companies: int,
    sectors: list[str],
    seed: int,
) -> dict[str, Any]:
    base_mae = float(np.mean(np.abs(actual - fundamentals)))
    model_mae = float(np.mean(np.abs(actual - pred)))
    ci_low, ci_high = _bootstrap_ci(actual, fundamentals, pred, seed)
    return {
        "driver_family": group[0],
        "feature": group[1],
        "horizon_label": group[2],
        "model_name": model_name,
        "evaluation_type": evaluation_type,
        "lambda": lam,
        "validation_rows": rows,
        "company_count": companies,
        "sectors": sectors,
        "mae_fundamentals": base_mae,
        "mae_model": model_mae,
        "mae_improvement_vs_fundamentals": 1.0 - model_mae / base_mae if base_mae > 0 else None,
        "mae_improvement_ci_low": ci_low,
        "mae_improvement_ci_high": ci_high,
        "directional_hit_rate": float((np.sign(actual) == np.sign(pred)).mean()),
        "rank_ic": _rank_ic(pd.Series(actual), pd.Series(pred)),
    }


def _prepare_panel(path: Path) -> pd.DataFrame:
    panel = pd.read_csv(path)
    for col in ["predictor_available_date", "outcome_available_date"]:
        panel[col] = pd.to_datetime(panel[col], errors="raise")
    if (panel["outcome_available_date"] <= panel["predictor_available_date"]).any():
        raise ValueError("Leakage guard failed: every outcome date must be after the predictor date.")
    return panel


def _walk_forward_smoke(panel: pd.DataFrame, lambdas: tuple[float, ...], train_ends: tuple[str, ...], test_years: int) -> dict[str, Any]:
    evaluations: list[dict[str, Any]] = []
    split_meta = []
    baseline_cols = ["current", "lag_delta", "cycle_position_score"]
    current_cols = ["current"]
    momentum_cols = ["lag_delta"]

    for group, group_df in panel.groupby(["driver_family", "feature", "horizon_label"], sort=True):
        pred_parts: dict[str, list[np.ndarray]] = {
            "actual": [],
            "fundamentals": [],
            "current_level_only": [],
            "momentum_only": [],
            "peer_median": [],
        }
        lambda_parts = {float(lam): [] for lam in lambdas}
        placebo_parts: dict[str, dict[float, list[np.ndarray]]] = {
            "shuffled_gap": {float(lam): [] for lam in lambdas},
            "wrong_sector_gap": {float(lam): [] for lam in lambdas},
            "random_driver_gap": {float(lam): [] for lam in lambdas},
        }
        row_count = 0
        companies: set[str] = set()
        sectors: set[str] = set()

        for split_idx, train_end_raw in enumerate(train_ends):
            train_end = pd.Timestamp(train_end_raw)
            test_end = train_end + pd.DateOffset(years=int(test_years))
            train = group_df[group_df["outcome_available_date"] <= train_end].copy()
            test = group_df[
                (group_df["predictor_available_date"] > train_end)
                & (group_df["predictor_available_date"] <= test_end)
                & (group_df["outcome_available_date"].notna())
            ].copy()
            if len(train) < 8 or len(test) < 4:
                continue
            p_fund = _fit_predict(train, test, baseline_cols)
            p_current = _fit_predict(train, test, current_cols)
            p_momentum = _fit_predict(train, test, momentum_cols)
            p_median = np.repeat(float(train["delta"].median()), len(test))
            actual = test["delta"].to_numpy(dtype=float)

            pred_parts["actual"].append(actual)
            pred_parts["fundamentals"].append(p_fund)
            pred_parts["current_level_only"].append(p_current)
            pred_parts["momentum_only"].append(p_momentum)
            pred_parts["peer_median"].append(p_median)
            row_count += len(test)
            companies.update(test["company_id"].astype(str))
            sectors.update(test["sector"].astype(str))
            split_meta.append(
                {
                    "driver_family": group[0],
                    "feature": group[1],
                    "horizon_label": group[2],
                    "train_end": train_end.strftime("%Y-%m-%d"),
                    "test_end": test_end.strftime("%Y-%m-%d"),
                    "train_rows": int(len(train)),
                    "test_rows": int(len(test)),
                }
            )

            p_raw = _fit_predict(train, test, baseline_cols + ["gap"])
            for lam in lambdas:
                lam = float(lam)
                lambda_parts[lam].append(p_fund + lam * (p_raw - p_fund))
            for placebo_col, placebo_name in [
                ("wrong_sector_gap", "wrong_sector_gap"),
                ("random_driver_gap", "random_driver_gap"),
            ]:
                p_fake_raw = _fit_predict(train.rename(columns={placebo_col: "gap"}), test.rename(columns={placebo_col: "gap"}), baseline_cols + ["gap"])
                for lam in lambdas:
                    lam = float(lam)
                    placebo_parts[placebo_name][lam].append(p_fund + lam * (p_fake_raw - p_fund))

            shuffled_train = train.copy()
            shuffled_test = test.copy()
            rng = np.random.default_rng(1000 + split_idx)
            shuffled_train["gap"] = rng.permutation(shuffled_train["gap"].to_numpy(dtype=float))
            shuffled_test["gap"] = rng.permutation(shuffled_test["gap"].to_numpy(dtype=float))
            p_shuffle_raw = _fit_predict(shuffled_train, shuffled_test, baseline_cols + ["gap"])
            for lam in lambdas:
                lam = float(lam)
                placebo_parts["shuffled_gap"][lam].append(p_fund + lam * (p_shuffle_raw - p_fund))

        if not pred_parts["actual"]:
            continue
        actual_all = np.concatenate(pred_parts["actual"])
        fundamentals_all = np.concatenate(pred_parts["fundamentals"])
        seed_base = abs(hash(group)) % 10_000
        for name in ["current_level_only", "momentum_only", "peer_median", "fundamentals"]:
            pred = np.concatenate(pred_parts[name])
            evaluations.append(
                _metric_row(
                    group=group,
                    model_name=name,
                    evaluation_type="baseline",
                    lam=None,
                    actual=actual_all,
                    fundamentals=fundamentals_all,
                    pred=pred,
                    rows=row_count,
                    companies=len(companies),
                    sectors=sorted(sectors),
                    seed=seed_base,
                )
            )
        for lam, parts in lambda_parts.items():
            pred = np.concatenate(parts)
            evaluations.append(
                _metric_row(
                    group=group,
                    model_name=f"gap_lambda_{lam:.2f}",
                    evaluation_type="actual",
                    lam=lam,
                    actual=actual_all,
                    fundamentals=fundamentals_all,
                    pred=pred,
                    rows=row_count,
                    companies=len(companies),
                    sectors=sorted(sectors),
                    seed=seed_base + int(lam * 1000),
                )
            )
        for placebo_name, by_lambda in placebo_parts.items():
            for lam, parts in by_lambda.items():
                pred = np.concatenate(parts)
                evaluations.append(
                    _metric_row(
                        group=group,
                        model_name=f"{placebo_name}_lambda_{lam:.2f}",
                        evaluation_type="placebo" if placebo_name == "shuffled_gap" else "sanity_check",
                        lam=lam,
                        actual=actual_all,
                        fundamentals=fundamentals_all,
                        pred=pred,
                        rows=row_count,
                        companies=len(companies),
                        sectors=sorted(sectors),
                        seed=seed_base + int(lam * 1000) + 17,
                    )
                )

    eval_df = pd.DataFrame(evaluations)
    best_actual = eval_df[eval_df["evaluation_type"].eq("actual")].sort_values(
        ["mae_improvement_vs_fundamentals", "directional_hit_rate"], ascending=[False, False]
    )
    summary = {
        "best_lambda": float(best_actual.iloc[0]["lambda"]) if not best_actual.empty else None,
        "best_mean_mae_improvement": float(best_actual.groupby("lambda")["mae_improvement_vs_fundamentals"].mean().max()) if not best_actual.empty else None,
        "actual_evaluation_count": int(eval_df["evaluation_type"].eq("actual").sum()),
        "placebo_evaluation_count": int(eval_df["evaluation_type"].eq("placebo").sum()),
        "sanity_check_evaluation_count": int(eval_df["evaluation_type"].eq("sanity_check").sum()),
    }
    return {"evaluations": evaluations, "splits": split_meta, "summary": summary}


def _write_panel_outputs(args: argparse.Namespace, preset: str) -> Path:
    panel = _prepare_panel(args.fixture)
    lambdas = tuple(float(x) for x in args.lambdas.split(","))
    train_ends = tuple(x.strip() for x in args.walk_forward_train_ends.split(",") if x.strip())
    payload = _walk_forward_smoke(panel, lambdas, train_ends, args.walk_forward_test_years)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "market_expectations_paper_report_v1",
        "metadata": {
            "preset": preset,
            "fixture": str(args.fixture),
            "row_count": int(len(panel)),
            "company_count": int(panel["company_id"].nunique()),
            "sector_count": int(panel["sector"].nunique()),
            "excluded_sectors": list(DEFAULT_EXCLUDED_SECTORS),
            "split_count": len(payload["splits"]),
            "target_claim": "valuation-gap signal improves forward driver-change forecasts beyond fundamentals-only baselines",
            "causal_claim": False,
        },
        "summary": payload["summary"],
        "splits": payload["splits"],
        "evaluations": payload["evaluations"],
    }
    report_path = out_dir / "market_expectations_experiment_report.json"
    report_path.write_text(json.dumps(_json_ready(report), indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame(payload["evaluations"]).to_csv(out_dir / "market_expectations_evaluations.csv", index=False)
    pd.DataFrame(payload["splits"]).to_csv(out_dir / "walk_forward_splits.csv", index=False)
    return report_path


def _write_smoke_outputs(args: argparse.Namespace) -> Path:
    return _write_panel_outputs(args, "smoke")


def _run_publication(args: argparse.Namespace) -> Path:
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "market_expectations_experiment_report.json"
    md_path = out_dir / "market_expectations_experiment_report.md"
    private_panel = os.environ.get("AXIOM_MARKET_EXPECTATIONS_PANEL")
    legacy_runner = ROOT / "scripts" / "validate_forward_gap_lambda_policy.py"
    if private_panel and Path(private_panel).exists():
        args.fixture = Path(private_panel).resolve()
        return _write_panel_outputs(args, "publication")
    if not legacy_runner.exists():
        raise FileNotFoundError(
            "Publication preset requires either AXIOM_MARKET_EXPECTATIONS_PANEL "
            "pointing to a private panel with the smoke fixture schema, or the "
            "private validation runner scripts/validate_forward_gap_lambda_policy.py."
        )
    cmd = [
        sys.executable,
        str(legacy_runner),
        "--validation-mode",
        "walk_forward",
        "--placebo-runs",
        str(args.placebo_runs),
        "--max-companies",
        str(args.max_companies),
        "--per-sector",
        str(args.per_sector),
        "--min-market-cap",
        str(args.min_market_cap),
        "--exclude-sectors",
        ",".join(DEFAULT_EXCLUDED_SECTORS),
        "--out-json",
        str(report_path),
        "--out-md",
        str(md_path),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("smoke", "publication"), default="smoke")
    parser.add_argument("--fixture", type=Path, default=FIXTURE)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--lambdas", default=",".join(str(x) for x in DEFAULT_LAMBDAS))
    parser.add_argument("--walk-forward-train-ends", default="2016-12-31,2018-12-31")
    parser.add_argument("--walk-forward-test-years", type=int, default=2)
    parser.add_argument("--placebo-runs", type=int, default=3)
    parser.add_argument("--max-companies", type=int, default=300)
    parser.add_argument("--per-sector", type=int, default=50)
    parser.add_argument("--min-market-cap", type=float, default=2500.0)
    args = parser.parse_args()
    args.fixture = args.fixture.resolve()
    args.out_dir = (args.out_dir or (ROOT / "paper" / "results" / args.preset)).resolve()

    report_path = _write_smoke_outputs(args) if args.preset == "smoke" else _run_publication(args)
    table_dir = args.out_dir / "tables"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "paper" / "build_paper_tables.py"), "--report", str(report_path), "--out-dir", str(table_dir)],
        cwd=ROOT,
        check=True,
    )
    print(json.dumps({"preset": args.preset, "report": str(report_path), "tables": str(table_dir)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
