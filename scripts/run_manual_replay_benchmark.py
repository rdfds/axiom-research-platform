#!/usr/bin/env python
from __future__ import annotations

import argparse
import faulthandler
import json
import os
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Tuple

def _bootstrap_runtime_threading_defaults() -> None:
    # Historical replay is strictly offline and occasionally trips Intel/OpenMP
    # runtime issues on this machine; keep the harness single-threaded by default.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    os.environ.setdefault("KMP_AFFINITY", "disabled")
    os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")


_bootstrap_runtime_threading_defaults()

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG_PATH = ROOT / "configs" / "historical_eval_manifests" / "2026-03-17" / "manual_replay_benchmark_lock.json"
_CANONICAL_LOCK_ARTIFACT_FALLBACKS: Dict[str, Path] = {
    "outcomes_path": ROOT / "data" / "curated" / "action_outcomes_with_credit_ratings.normalized_full.parquet",
    "action_support_manifest": ROOT / "configs" / "action_data_support_manifest.json",
    "entity_graph_path": ROOT / "data" / "inputs_layer" / "entity_graph.parquet",
    "entity_identifier_path": ROOT / "data" / "inputs_layer" / "entity_identifier.parquet",
    "entity_table_path": ROOT / "data" / "inputs_layer" / "entity.parquet",
    "raw_timeseries_path": ROOT / "data" / "inputs_layer" / "raw_timeseries.parquet",
    "event_store_path": ROOT / "data" / "inputs_layer" / "event_store.parquet",
    "ownership_summary_path": ROOT / "data" / "inputs_layer" / "ownership_13f_summary.parquet",
    "issuer_ratings_path": ROOT / "data" / "inputs_layer" / "issuer_rating_history.parquet",
    "companyfacts_root": ROOT / "data" / "sec" / "companyfacts",
    "facts_path": ROOT / "data" / "inputs_layer" / "extracted_fact_registry_validity",
}


