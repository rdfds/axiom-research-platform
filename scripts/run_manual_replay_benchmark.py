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


def _bootstrap_env_overrides() -> None:
    config_path = DEFAULT_CONFIG_PATH
    argv = sys.argv[1:]
    for index, token in enumerate(argv):
        if token == "--config" and index + 1 < len(argv):
            config_path = Path(argv[index + 1])
            break
        if token.startswith("--config="):
            config_path = Path(token.split("=", 1)[1])
            break
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    if not config_path.exists():
        return
    try:
        payload = json.loads(config_path.read_text())
    except Exception:
        # Synced placeholder manifests/configs should not block offline replay startup.
        return
    env_override_candidates = dict(payload.get("env_override_candidates", {}) or {})
    if env_override_candidates:
        for name, values in env_override_candidates.items():
            candidates = []
            for value in list(values or []):
                path = Path(value)
                if not path.is_absolute():
                    path = ROOT / path
                candidates.append(path)
            chosen = next((candidate for candidate in candidates if candidate.exists()), candidates[0] if candidates else None)
            if chosen is not None:
                os.environ.setdefault(name, str(chosen))
        return
    env_overrides = dict(payload.get("env_overrides", {}) or {})
    for name, path_value in env_overrides.items():
        path = Path(path_value)
        if not path.is_absolute():
            path = ROOT / path
        os.environ.setdefault(name, str(path))


_bootstrap_env_overrides()

from src.action_data_support import resolve_action_support
from src.backtest_artifacts import (
    build_backtest_artifact_manifest,
    fingerprint_path,
    resolve_backtest_artifact_root,
    resolve_snapshot_cache_dir,
)
from src.backtest_costs import resolve_transaction_cost_model
from src.backtest_protocol import resolve_backtest_protocol
from src.backtest_scorecard import (
    build_portfolio_strategy_scorecard,
    render_portfolio_strategy_scorecard_markdown,
)
from src.company_state_builder import CompanyStateBuilder
from src.company_state_store import SnapshotStore
from src.historical_recommendation_eval import (
    _aggregate_historical_cases,
    _build_historical_alias_overrides,
    _load_action_support_summary,
    _load_fixed_historical_cases,
    _load_realized_outcomes_lookup,
    _score_ex_post_alignment,
    _snapshot_coverage_summary,
    _snapshot_has_meaningful_coverage,
    _top_action_ids,
)
from src.recommendation_run import RecommendationRunStore, create_recommendation_run
from src.recommendation_run_orchestrator import execute_recommendation_run
from src.replay_snapshot_enrichment import enrich_snapshot_with_revenue_growth_inputs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen manual historical replay benchmark.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--benchmark", required=True, help="Benchmark key from the lock config.")
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--snapshot-cache-dir", default="", help="Optional snapshot cache dir. Defaults to a stable path under the artifact root.")
    parser.add_argument("--artifact-root", default="", help="Optional stable artifact root. Defaults to <runs-root>/_backtest_artifacts.")
    parser.add_argument("--out-json", default="", help="Optional report output path. Defaults under the artifact root.")
    parser.add_argument("--scorecard-json", default="", help="Optional standardized scorecard JSON path.")
    parser.add_argument("--scorecard-md", default="", help="Optional standardized scorecard markdown path.")
    parser.add_argument("--manifest-json", default="", help="Optional artifact manifest JSON path.")
    parser.add_argument("--protocol", default="", help="Optional canonical backtest protocol override.")
    parser.add_argument("--cost-model", default="", help="Optional transaction cost model override.")
    parser.add_argument("--precedent-top-k", type=int, help="Optional override for the number of precedent candidates to retrieve per case.")
    parser.add_argument("--case-count", type=int, help="Optional case limit for smoke runs.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-case progress logs.")
    parser.add_argument("--dump-stack-after-seconds", type=int, help="Optional faulthandler timeout for diagnosing stalls.")
    return parser.parse_args()


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


