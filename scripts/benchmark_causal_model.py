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


def _parse_args() :
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


