#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_causal_rescue_plan import build_causal_rescue_plan  # noqa: E402


_REPO_ROOT = Path(__file__).resolve().parents[1]


def _default_path(*parts: str) -> str:
    return str(_REPO_ROOT.joinpath(*parts))


def _parse_args() :
    p = argparse.ArgumentParser(description="Train a targeted causal rescue model for blocked actions.")
    p.add_argument("--audit-json", default="", help="Optional ML audit JSON used to generate rescue actions.")
    p.add_argument("--rescue-actions-file", default="", help="Optional newline-delimited rescue action ids.")
    p.add_argument(
        "--generated-rescue-actions-out",
        default="/tmp/causal_rescue_actions_train.txt",
        help="Where to materialize rescue actions when --audit-json is used.",
    )
    p.add_argument("--strict-pass-threshold", type=float, default=0.50)
    p.add_argument("--min-action-rows", type=int, default=100)
    p.add_argument("--low-row-blocklist-threshold", type=int, default=50)
    p.add_argument(
        "--mapping-path",
        default=_default_path("config", "causal_rescue_action_mapping.json"),
        help="JSON mapping from recommendation actions to trainable dataset patterns.",
    )
    p.add_argument(
        "--outcomes-path",
        default=_default_path("data", "curated", "action_outcomes_with_credit_ratings.parquet"),
    )
    p.add_argument(
        "--out-path",
        default=_default_path("data", "models", "causal_impact_model_rescue_hgb.json"),
    )
    p.add_argument(
        "--model-card-out",
        default=_default_path("data", "models", "causal_impact_model_rescue_hgb.model_card.json"),
    )
    p.add_argument("--train-end-date", default="2023-12-31")
    p.add_argument("--validation-start-date", default="2024-01-01")
    p.add_argument("--model-family", default="hgb")
    p.add_argument("--cell-level", default="action_subtype")
    p.add_argument("--crossfit-folds", type=int, default=3)
    p.add_argument("--dr-min-treated-rows", type=int, default=1500)
    p.add_argument("--dr-min-control-rows", type=int, default=20000)
    p.add_argument("--min-validation-rows", type=int, default=300)
    p.add_argument("--propensity-clip", type=float, default=0.03)
    p.add_argument("--gate-min-oos-r2", type=float, default=0.0)
    p.add_argument("--gate-min-train-rows", type=int, default=8000)
    p.add_argument("--gate-min-treated-rows", type=int, default=1500)
    p.add_argument("--gate-min-control-rows", type=int, default=20000)
    p.add_argument("--progress-every-cells", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def _read_action_ids(path: Path) -> List[str]:
    action_ids: List[str] = []
    for raw in path.read_text().splitlines():
        item = raw.strip()
        if item and not item.startswith("#") and item not in action_ids:
            action_ids.append(item)
    return action_ids


