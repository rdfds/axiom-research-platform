#!/usr/bin/env python3
"""Build paper case-study artifacts from the public showcase valuation sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLE = ROOT / "examples" / "hd_market_expectations" / "valuation_driver_data.sample.json"


def _to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    rendered = df.copy()
    for col in rendered.columns:
        rendered[col] = rendered[col].map(lambda v: "" if pd.isna(v) else str(v))
    rows = [list(rendered.columns)] + rendered.values.tolist()
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(rendered.columns))]

    def fmt(row: list[Any]) -> str:
        return "| " + " | ".join(str(value).ljust(widths[i]) for i, value in enumerate(row)) + " |"

    return "\n".join(
        [
            fmt(rows[0]),
            "| " + " | ".join("-" * width for width in widths) + " |",
            *[fmt(row) for row in rows[1:]],
        ]
    )


def _num(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if pd.notna(out) else None


def _case_from_company(company: dict[str, Any]) -> dict[str, Any]:
    forward = company.get("forward_expectations") or {}
    summary = forward.get("summary") or {}
    drivers = forward.get("drivers") or []
    gap_log = _num(summary.get("gap_log")) or 0.0
    driver_log = _num(summary.get("driver_log")) or 0.0
    residual_log = _num(summary.get("outside_log")) or 0.0
    reconciliation_error = gap_log - driver_log - residual_log
    return {
        "ticker": company.get("ticker"),
        "name": company.get("name"),
        "display_multiple": forward.get("display_fair_label"),
        "forward_grade": forward.get("forward_grade"),
        "market_signal_grade": forward.get("market_signal_grade"),
        "forecast_precision_grade": forward.get("forecast_precision_grade"),
        "gap_log": gap_log,
        "driver_underwritten_log": driver_log,
        "residual_outside_measured_drivers_log": residual_log,
        "coverage_ratio": summary.get("coverage_ratio"),
        "reconciliation_error_log": reconciliation_error,
        "case_type": "flagship_driver_underwritten" if abs(driver_log) >= abs(residual_log) * 0.35 else "residual_dominant",
        "top_drivers": [
            {
                "label": d.get("label"),
                "feature": d.get("feature"),
                "horizon_label": d.get("horizon_label"),
                "current_value": d.get("current_value"),
                "implied_value": d.get("implied_value"),
                "market_implied_delta": d.get("market_implied_delta"),
                "gap_share": d.get("gap_share"),
                "contribution_log": d.get("contribution_log"),
                "market_signal_grade": d.get("market_signal_grade"),
                "forecast_precision_grade": d.get("forecast_precision_grade"),
                "economic_materiality": d.get("economic_materiality"),
            }
            for d in drivers[:5]
        ],
    }


def build_case_studies(sample_path: Path, out_dir: Path) -> dict[str, str]:
    payload = json.loads(sample_path.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = [_case_from_company(company) for company in payload.get("companies") or []]

    # Public fixture currently contains HD. Add two explicitly synthetic contrast
    # cases so the paper package always demonstrates residual- and multi-driver
    # interpretation contracts without exposing private company payloads.
    cases.append(
        {
            "ticker": "SYN_RESIDUAL",
            "name": "Synthetic residual-dominant premium case",
            "display_multiple": "P/E",
            "forward_grade": "diagnostic",
            "market_signal_grade": "weak",
            "forecast_precision_grade": "weak",
            "gap_log": 0.22,
            "driver_underwritten_log": 0.025,
            "residual_outside_measured_drivers_log": 0.195,
            "coverage_ratio": 0.1136,
            "reconciliation_error_log": 0.0,
            "case_type": "residual_dominant",
            "top_drivers": [
                {
                    "label": "FCF margin",
                    "feature": "fcf_margin",
                    "horizon_label": "1Y",
                    "gap_share": 0.1136,
                    "contribution_log": 0.025,
                    "market_signal_grade": "weak",
                    "forecast_precision_grade": "weak",
                    "economic_materiality": "immaterial",
                }
            ],
        }
    )
    cases.append(
        {
            "ticker": "SYN_MULTI",
            "name": "Synthetic multi-driver underwritten discount case",
            "display_multiple": "EV/EBITDA",
            "forward_grade": "strong",
            "market_signal_grade": "strong",
            "forecast_precision_grade": "moderate",
            "gap_log": -0.18,
            "driver_underwritten_log": -0.135,
            "residual_outside_measured_drivers_log": -0.045,
            "coverage_ratio": 0.75,
            "reconciliation_error_log": 0.0,
            "case_type": "multi_driver_underwritten",
            "top_drivers": [
                {
                    "label": "EBITDA margin",
                    "feature": "ebitda_margin",
                    "horizon_label": "2Y",
                    "gap_share": 0.45,
                    "contribution_log": -0.081,
                    "market_signal_grade": "strong",
                    "forecast_precision_grade": "moderate",
                    "economic_materiality": "major",
                },
                {
                    "label": "Revenue growth",
                    "feature": "revenue_growth",
                    "horizon_label": "2Y",
                    "gap_share": 0.30,
                    "contribution_log": -0.054,
                    "market_signal_grade": "strong",
                    "forecast_precision_grade": "moderate",
                    "economic_materiality": "material",
                },
            ],
        }
    )

    case_path = out_dir / "market_expectations_case_studies.json"
    case_path.write_text(json.dumps({"cases": cases}, indent=2, allow_nan=False), encoding="utf-8")
    rows = []
    for case in cases:
        rows.append(
            {
                "ticker": case["ticker"],
                "case_type": case["case_type"],
                "display_multiple": case["display_multiple"],
                "gap_log": case["gap_log"],
                "driver_underwritten_log": case["driver_underwritten_log"],
                "residual_log": case["residual_outside_measured_drivers_log"],
                "coverage_ratio": case["coverage_ratio"],
                "reconciliation_error_log": case["reconciliation_error_log"],
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "case_study_decomposition.csv", index=False)
    (out_dir / "case_study_decomposition.md").write_text(_to_markdown(table) + "\n", encoding="utf-8")
    return {"json": str(case_path), "csv": str(out_dir / "case_study_decomposition.csv")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "paper" / "case_studies")
    args = parser.parse_args()
    print(json.dumps(build_case_studies(args.sample, args.out_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
