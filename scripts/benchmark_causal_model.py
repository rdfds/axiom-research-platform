#!/usr/bin/env python
"""Benchmark causal model cards with explicit production gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.causal_benchmark import compare_summaries, evaluate_summary_thresholds, load_model_card, summarize_model_card


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark causal model card quality and production readiness.")
    p.add_argument("--champion-model-card", required=True, help="Path to champion model_card.json")
    p.add_argument("--challenger-model-card", default="", help="Optional challenger model_card.json")
    p.add_argument("--out", default="", help="Optional output path for benchmark JSON")
    p.add_argument("--min-enabled-cells", type=int, default=10)
    p.add_argument("--min-enabled-rate", type=float, default=0.10)
    p.add_argument("--min-enabled-oos-r2-mean", type=float, default=0.05)
    p.add_argument(
        "--required-objective",
        action="append",
        default=[],
        help="Objective name requiring minimum coverage/quality (repeatable).",
    )
    p.add_argument("--required-objective-min-enabled", type=int, default=1)
    p.add_argument("--required-objective-min-oos-r2-mean", type=float, default=0.0)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    champion_card = load_model_card(args.champion_model_card)
    champion_summary = summarize_model_card(champion_card)
    required_objectives: List[str] = list(args.required_objective or [])
    if not required_objectives:
        required_objectives = ["value_creation", "risk_reduction"]

    champion_gates = evaluate_summary_thresholds(
        champion_summary,
        min_enabled_cells=int(args.min_enabled_cells),
        min_enabled_rate=float(args.min_enabled_rate),
        min_enabled_oos_r2_mean=float(args.min_enabled_oos_r2_mean),
        required_objectives=required_objectives,
        required_objective_min_enabled=int(args.required_objective_min_enabled),
        required_objective_min_oos_r2_mean=float(args.required_objective_min_oos_r2_mean),
    )

    report: Dict[str, Any] = {
        "ok": True,
        "champion": {
            "model_card_path": str(Path(args.champion_model_card)),
            "summary": champion_summary,
            "production_gate": champion_gates,
        },
        "thresholds": {
            "min_enabled_cells": int(args.min_enabled_cells),
            "min_enabled_rate": float(args.min_enabled_rate),
            "min_enabled_oos_r2_mean": float(args.min_enabled_oos_r2_mean),
            "required_objectives": required_objectives,
            "required_objective_min_enabled": int(args.required_objective_min_enabled),
            "required_objective_min_oos_r2_mean": float(args.required_objective_min_oos_r2_mean),
        },
    }

    challenger_path = str(args.challenger_model_card or "").strip()
    if challenger_path:
        challenger_card = load_model_card(challenger_path)
        challenger_summary = summarize_model_card(challenger_card)
        challenger_gates = evaluate_summary_thresholds(
            challenger_summary,
            min_enabled_cells=int(args.min_enabled_cells),
            min_enabled_rate=float(args.min_enabled_rate),
            min_enabled_oos_r2_mean=float(args.min_enabled_oos_r2_mean),
            required_objectives=required_objectives,
            required_objective_min_enabled=int(args.required_objective_min_enabled),
            required_objective_min_oos_r2_mean=float(args.required_objective_min_oos_r2_mean),
        )
        report["challenger"] = {
            "model_card_path": str(Path(challenger_path)),
            "summary": challenger_summary,
            "production_gate": challenger_gates,
        }
        report["comparison"] = compare_summaries(champion_summary, challenger_summary)

    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)


if __name__ == "__main__":
    main()

