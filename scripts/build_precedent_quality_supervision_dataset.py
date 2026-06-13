#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import duckdb
import gzip
import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("RECO_DISABLE_PRECEDENT_NARRATIVE", "1")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model_feature_bundle import _STATE_VECTOR_V1_FEATURES, build_model_feature_bundle
from src.pipeline.historical_price_metric_backfill import backfill_historical_price_window_metrics
from src.pipeline.latent_regime_model import fit_latent_regime_kmeans, latent_regime_memberships
from src.pipeline.precedent import _state_vector_baseline_value
from src.pipeline.precedent_brain import (
    _effective_action_subtype,
    _estimate_action_scale,
    _enrich_missing_historical_taxonomy,
    _historical_taxonomy_for_ticker,
    _weighted_state_similarity,
    augment_precedent_state_vector_columns,
)
from src.pipeline.run import _default_precedent_outcomes_path, adapt_snapshot, attach_model_feature_bundle


_PAIRWISE_FEATURE_GAP_SUMMARY_FEATURES = tuple(_STATE_VECTOR_V1_FEATURES)
_REFINITIV_TAXONOMY_REFERENCE_PATH = REPO_ROOT / "data" / "refinitiv" / "fundamentals_all.parquet"
_SEC_TICKER_CIK_PATH = REPO_ROOT / "data" / "mappings" / "sec_ticker_cik.parquet"
_SNAPSHOT_TAXONOMY_LOOKUP_PATH = REPO_ROOT / "data" / "curated" / "snapshot_taxonomy_lookup_2026-02-28.parquet"


def _maybe_backfill_historical_price_window_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    if str(os.environ.get("RECO_DISABLE_HISTORICAL_PRICE_BACKFILL") or "").strip().lower() in {
        "1",
        "true",
        "yes",
    }:
        return frame.copy()
    return backfill_historical_price_window_metrics(frame)


def _prefer_pandas_outcomes_reads() -> bool:
    return str(os.environ.get("RECO_FORCE_PANDAS_OUTCOMES_READ") or "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


@lru_cache(maxsize=4)
def _cached_pandas_outcomes_frame(outcomes_path_str: str) -> pd.DataFrame:
    return pd.read_parquet(outcomes_path_str)


def _resolve_teacher_recipe(
    *,
    teacher_recipe: str,
    positive_source_mode: str,
    include_within_action_hard_negatives: bool,
    include_same_action_positive_ordering: bool,
    actual_anchor_within_action_negative_source: str,
    always_include_actual_anchor_positive: bool,
    same_family_negatives_only_if_available: bool,
    hard_negative_taxonomy_mode: str,
) -> Dict[str, Any]:
    recipe = str(teacher_recipe or "explicit_flags").strip().lower() or "explicit_flags"
    config = {
        "teacher_recipe": recipe,
        "positive_source_mode": str(positive_source_mode or "include_retrieved").strip().lower() or "include_retrieved",
        "include_within_action_hard_negatives": bool(include_within_action_hard_negatives),
        "include_same_action_positive_ordering": bool(include_same_action_positive_ordering),
        "actual_anchor_within_action_negative_source": str(
            actual_anchor_within_action_negative_source or "retrieved_pool"
        ).strip().lower()
        or "retrieved_pool",
        "always_include_actual_anchor_positive": bool(always_include_actual_anchor_positive),
        "same_family_negatives_only_if_available": bool(same_family_negatives_only_if_available),
        "hard_negative_taxonomy_mode": str(hard_negative_taxonomy_mode or "none").strip().lower() or "none",
    }
    if recipe == "explicit_flags":
        return config
    if recipe == "same_action_best_analog":
        config.update(
            {
                "positive_source_mode": "analog_consensus_same_action_universe",
                "include_within_action_hard_negatives": True,
                "include_same_action_positive_ordering": True,
                "actual_anchor_within_action_negative_source": "same_action_universe",
                "always_include_actual_anchor_positive": False,
                "same_family_negatives_only_if_available": False,
            }
        )
        return config
    if recipe == "same_action_regime_best_analog":
        config.update(
            {
                "positive_source_mode": "analog_regime_consensus_same_action_universe",
                "include_within_action_hard_negatives": True,
                "include_same_action_positive_ordering": True,
                "actual_anchor_within_action_negative_source": "same_action_universe",
                "always_include_actual_anchor_positive": False,
                "same_family_negatives_only_if_available": False,
            }
        )
        return config
    if recipe == "same_action_actual_anchor":
        config.update(
            {
                "positive_source_mode": "actual_anchor_preferred",
                "include_within_action_hard_negatives": True,
                "include_same_action_positive_ordering": False,
                "actual_anchor_within_action_negative_source": "same_action_universe",
                "always_include_actual_anchor_positive": True,
                "same_family_negatives_only_if_available": False,
            }
        )
        return config
    raise ValueError(f"Unsupported teacher_recipe: {teacher_recipe}")


