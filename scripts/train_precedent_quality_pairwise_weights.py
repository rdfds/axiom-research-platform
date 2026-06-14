#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.precedent_quality_learning import (
    build_pairwise_matrix,
    build_outcome_aware_reranker_matrix,
    build_scope_payload_with_pairwise_weights,
    cross_validate_pairwise_precedent_quality_weights,
    cross_validate_outcome_aware_reranker,
    fit_nonnegative_pairwise_logistic,
    learn_feature_transforms_from_pairwise_supervision,
    load_feature_transform_prior,
    load_feature_weight_prior,
    _outcome_aware_reranker_prior,
    load_penalty_feature_specs,
    load_pairwise_supervision,
    search_latent_regime_models_from_supervision,
    search_pairwise_interactions_from_supervision,
    search_target_regime_mixture_from_supervision,
    write_json,
)
from src.pipeline.run import _default_precedent_outcomes_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train pairwise precedent-quality weights for a scope.")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--scope-key", required=True)
    parser.add_argument("--base-payload-path", required=True)
    parser.add_argument("--out-payload-path", required=True)
    parser.add_argument("--out-summary-path", default="")
    parser.add_argument("--min-feature-coverage-rows", type=int, default=20)
    parser.add_argument("--l2-grid", default="0.25,0.5,1.0,2.0,4.0,8.0")
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-iter", type=int, default=4000)
    parser.add_argument("--learn-feature-transforms", action="store_true")
    parser.add_argument("--feature-transform-mode", choices=("default", "identity"), default="")
    parser.add_argument(
        "--pair-weight-mode",
        choices=("uniform", "teacher_confidence", "target_regime_rarity"),
        default="uniform",
    )
    parser.add_argument("--include-runtime-penalties", action="store_true")
    parser.add_argument("--transform-search-l2-grid", default="0.25,1.0,4.0")
    parser.add_argument("--transform-search-learning-rate", type=float, default=0.05)
    parser.add_argument("--transform-search-max-iter", type=int, default=2000)
    parser.add_argument("--include-interactions", action="store_true")
    parser.add_argument("--search-interactions", action="store_true")
    parser.add_argument("--max-interaction-terms", type=int, default=6)
    parser.add_argument("--interaction-feature-names", default="")
    parser.add_argument("--search-latent-regimes", action="store_true")
    parser.add_argument("--search-target-regime-mixture", action="store_true")
    parser.add_argument("--latent-regime-cluster-grid", default="2,3,4,5,6")
    parser.add_argument("--latent-regime-seed", type=int, default=7)
    parser.add_argument("--latent-regime-max-iter", type=int, default=100)
    parser.add_argument("--train-outcome-aware-reranker", action="store_true")
    parser.add_argument("--outcomes-path", default="")
    parser.add_argument("--outcome-aware-shortlist-size", type=int, default=40)
    return parser.parse_args()


def _parse_grid(value: str) -> Sequence[float]:
    values = [float(item.strip()) for item in str(value or "").split(",") if item.strip()]
    if not values:
        raise ValueError("l2 grid must include at least one value")
    return values


