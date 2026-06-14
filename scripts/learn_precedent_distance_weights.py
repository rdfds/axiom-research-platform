#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.precedent_distance_learning import (  # noqa: E402
    learn_precedent_distance_weights,
    write_precedent_distance_weights,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Learn diagonal precedent-distance weights from historical outcomes.")
    parser.add_argument(
        "--outcomes-path",
        default=str(
            REPO_ROOT / "data/curated/action_outcomes_with_credit_ratings.normalized_full.rich_contract_v3.parquet"
        ),
    )
    parser.add_argument(
        "--out-path",
        default=str(REPO_ROOT / "data/curated/precedent_distance_weights_v1.json"),
    )
    parser.add_argument("--max-pairs", type=int, default=25000)
    parser.add_argument("--min-rows", type=int, default=1500)
    parser.add_argument("--min-state-coverage", type=float, default=0.60)
    parser.add_argument("--min-outcome-coverage", type=float, default=0.50)
    parser.add_argument("--min-outcome-non-null", type=int, default=800)
    parser.add_argument("--ridge-lambda", type=float, default=30.0)
    parser.add_argument("--holdout-frac", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


