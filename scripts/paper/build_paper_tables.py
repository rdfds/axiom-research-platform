#!/usr/bin/env python3
"""Build paper-ready tables from market-expectations experiment reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _num(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if pd.notna(out) else None


def _write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path.with_suffix(".csv"), index=False)
    path.with_suffix(".md").write_text(_to_markdown(df) + "\n", encoding="utf-8")


def _to_markdown(df: pd.DataFrame) -> str:
    """Small dependency-free markdown writer for public smoke environments."""
    if df.empty:
        return ""
    rendered = df.copy()
    for col in rendered.columns:
        rendered[col] = rendered[col].map(lambda v: "" if pd.isna(v) else str(v))
    rows = [list(rendered.columns)] + rendered.values.tolist()
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(rendered.columns))]

    def fmt(row: list[Any]) -> str:
        return "| " + " | ".join(str(value).ljust(widths[i]) for i, value in enumerate(row)) + " |"

    header = fmt(rows[0])
    divider = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [fmt(row) for row in rows[1:]]
    return "\n".join([header, divider, *body])


def _from_paper_schema(report: dict[str, Any]) -> dict[str, pd.DataFrame]:
    evaluations = pd.DataFrame(report.get("evaluations") or [])
    summary = report.get("summary") or {}
    metadata = report.get("metadata") or {}
    if evaluations.empty:
        evaluations = pd.DataFrame()

    universe = pd.DataFrame(
        [
            {
                "preset": metadata.get("preset"),
                "row_count": metadata.get("row_count"),
                "company_count": metadata.get("company_count"),
                "sector_count": metadata.get("sector_count"),
                "excluded_sectors": ", ".join(metadata.get("excluded_sectors") or []),
                "split_count": metadata.get("split_count"),
            }
        ]
    )

    if evaluations.empty:
        return {"table_1_universe": universe}

    actual = evaluations[evaluations["evaluation_type"].eq("actual")].copy()
    baselines = evaluations[evaluations["evaluation_type"].eq("baseline")].copy()
    placebos = evaluations[evaluations["evaluation_type"].isin(["placebo", "sanity_check"])].copy()

    baseline_table = (
        pd.concat([baselines, actual], ignore_index=True)
        .groupby(["model_name"], dropna=False)
        .agg(
            evaluation_count=("mae_improvement_vs_fundamentals", "count"),
            mean_mae=("mae_model", "mean"),
            mean_mae_improvement_vs_fundamentals=("mae_improvement_vs_fundamentals", "mean"),
            mean_directional_hit_rate=("directional_hit_rate", "mean"),
            mean_rank_ic=("rank_ic", "mean"),
        )
        .reset_index()
        .sort_values(["mean_mae_improvement_vs_fundamentals", "mean_directional_hit_rate"], ascending=[False, False])
    )

    lambda_table = (
        actual[actual["lambda"].notna()]
        .groupby(["lambda"], dropna=False)
        .agg(
            evaluation_count=("mae_improvement_vs_fundamentals", "count"),
            mean_mae_improvement=("mae_improvement_vs_fundamentals", "mean"),
            pass_rate=("mae_improvement_vs_fundamentals", lambda s: float((s >= 0).mean())),
            ci_low=("mae_improvement_ci_low", "mean"),
            ci_high=("mae_improvement_ci_high", "mean"),
        )
        .reset_index()
        .sort_values("lambda")
    )

    placebo_table = (
        placebos.groupby(["model_name"], dropna=False)
        .agg(
            evaluation_count=("mae_improvement_vs_fundamentals", "count"),
            mean_mae_improvement=("mae_improvement_vs_fundamentals", "mean"),
            pass_rate=("mae_improvement_vs_fundamentals", lambda s: float((s >= 0).mean())),
        )
        .reset_index()
        .sort_values("mean_mae_improvement", ascending=False)
    )

    sector_source = actual.copy()
    if "sector" not in sector_source.columns and "sectors" in sector_source.columns:
        sector_source = sector_source.explode("sectors").rename(columns={"sectors": "sector"})
    elif "sector" not in sector_source.columns:
        sector_source["sector"] = "unspecified"

    sector_table = (
        sector_source.groupby(["sector"], dropna=False)
        .agg(
            evaluation_count=("mae_improvement_vs_fundamentals", "count"),
            mean_mae_improvement=("mae_improvement_vs_fundamentals", "mean"),
            pass_rate=("mae_improvement_vs_fundamentals", lambda s: float((s >= 0).mean())),
        )
        .reset_index()
        .sort_values("sector")
    )

    family_table = (
        actual.groupby(["driver_family", "horizon_label"], dropna=False)
        .agg(
            evaluation_count=("mae_improvement_vs_fundamentals", "count"),
            mean_mae_improvement=("mae_improvement_vs_fundamentals", "mean"),
            pass_rate=("mae_improvement_vs_fundamentals", lambda s: float((s >= 0).mean())),
            mean_directional_hit_rate=("directional_hit_rate", "mean"),
        )
        .reset_index()
        .sort_values(["driver_family", "horizon_label"])
    )

    return {
        "table_1_universe": universe,
        "table_2_baselines": baseline_table,
        "table_3_lambda_ablation": lambda_table,
        "table_4_placebos": placebo_table,
        "table_5_sector_robustness": sector_table,
        "table_6_driver_family_horizon": family_table,
    }


def _from_legacy_validation_schema(report: dict[str, Any]) -> dict[str, pd.DataFrame]:
    aggregate = report.get("aggregate") or {}
    groups = aggregate.get("groups") or {}
    placebo_groups = aggregate.get("placebo_groups") or {}
    global_group = groups.get("global") or {}

    universe = pd.DataFrame(
        [
            {
                "as_of": report.get("as_of"),
                "candidate_count": report.get("candidate_count"),
                "successful_evaluation_count": aggregate.get("successful_evaluation_count"),
                "validation_mode": report.get("validation_mode"),
                "excluded_sectors": ", ".join(f"{x.get('code')} {x.get('name')}" for x in report.get("excluded_sectors") or []),
            }
        ]
    )

    lambda_rows = []
    for key, row in (global_group.get("lambda_results") or {}).items():
        lambda_rows.append(
            {
                "lambda": row.get("lambda", _num(key)),
                "evaluation_count": row.get("evaluation_count"),
                "mean_mae_improvement": row.get("mean_mae_improvement"),
                "pass_rate": row.get("pass_rate"),
                "validation_rows": row.get("validation_rows"),
            }
        )

    placebo_rows = []
    for key, row in ((placebo_groups.get("global") or {}).get("lambda_results") or {}).items():
        placebo_rows.append(
            {
                "lambda": row.get("lambda", _num(key)),
                "actual_mean_mae_improvement": row.get("actual_mean_mae_improvement"),
                "placebo_mean_mae_improvement": row.get("placebo_mean_mae_improvement"),
                "actual_minus_placebo": row.get("actual_minus_placebo_mean"),
                "actual_beats_placebo": row.get("actual_beats_placebo_rate"),
            }
        )

    sector_rows = []
    for sector, group in (groups.get("by_sector") or {}).items():
        best = group.get("best_lambda")
        metrics = (group.get("lambda_results") or {}).get(str(float(best))) if best is not None else {}
        sector_rows.append(
            {
                "sector": sector,
                "sector_name": group.get("sector_name"),
                "best_lambda": best,
                "evaluation_count": group.get("evaluation_count"),
                "mean_mae_improvement": (metrics or {}).get("mean_mae_improvement"),
                "pass_rate": (metrics or {}).get("pass_rate"),
            }
        )

    family_rows = []
    for group_name, group in (groups.get("by_family_horizon") or {}).items():
        best = group.get("best_lambda")
        metrics = (group.get("lambda_results") or {}).get(str(float(best))) if best is not None else {}
        family, _, horizon = group_name.partition(":")
        family_rows.append(
            {
                "driver_family": family,
                "horizon_label": horizon,
                "best_lambda": best,
                "evaluation_count": group.get("evaluation_count"),
                "mean_mae_improvement": (metrics or {}).get("mean_mae_improvement"),
                "pass_rate": (metrics or {}).get("pass_rate"),
            }
        )

    return {
        "table_1_universe": universe,
        "table_3_lambda_ablation": pd.DataFrame(lambda_rows).sort_values("lambda") if lambda_rows else pd.DataFrame(),
        "table_4_placebos": pd.DataFrame(placebo_rows).sort_values("lambda") if placebo_rows else pd.DataFrame(),
        "table_5_sector_robustness": pd.DataFrame(sector_rows).sort_values("sector") if sector_rows else pd.DataFrame(),
        "table_6_driver_family_horizon": pd.DataFrame(family_rows).sort_values(["driver_family", "horizon_label"]) if family_rows else pd.DataFrame(),
    }


def build_tables(report_path: Path, out_dir: Path) -> dict[str, str]:
    report = json.loads(report_path.read_text())
    schema = report.get("schema")
    tables = _from_paper_schema(report) if schema == "market_expectations_paper_report_v1" else _from_legacy_validation_schema(report)
    written = {}
    for name, df in tables.items():
        if df is None or df.empty:
            continue
        base = out_dir / name
        _write_table(df, base)
        written[name] = str(base.with_suffix(".csv"))
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    written = build_tables(args.report, args.out_dir)
    print(json.dumps({"tables": written}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
