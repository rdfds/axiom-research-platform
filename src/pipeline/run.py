from __future__ import annotations

import json
import os
import time
import traceback
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..action_ontology import ActionSchemaRegistry, build_default_action_schema_registry
from ..model_feature_bundle import _STATE_VECTOR_V1_FEATURES, attach_model_feature_bundle, feature_view_from_snapshot
from ..runtime_feature_adapter import adapt_snapshot, resolve_feature_value
from .types import ActionCandidate, CompanyStateSnapshot, PrecedentPack


DATA_DIR = Path(__file__).parent.parent.parent / "data"
_DEFAULT_REGISTRY: Optional[ActionSchemaRegistry] = None
_PRECEDENT_DEBUG = os.getenv("RECO_PRECEDENT_DEBUG", "").strip().lower() not in {"", "0", "false", "no"}
_DEFAULT_PRECEDENT_OUTCOMES_CANDIDATES: Tuple[Path, ...] = (
    DATA_DIR / "curated" / "action_outcomes_with_credit_ratings.normalized_full.rich_contract_v3.parquet",
    DATA_DIR / "curated" / "action_outcomes_with_credit_ratings.normalized_full.rich_contract_v2.parquet",
    DATA_DIR / "curated" / "action_outcomes_with_credit_ratings.normalized_full.rich_contract_v1.parquet",
    DATA_DIR / "curated" / "action_outcomes_with_credit_ratings.normalized_full.parquet",
    DATA_DIR / "curated" / "action_outcomes_with_credit_ratings.parquet",
    DATA_DIR / "curated" / "action_outcomes.parquet",
)


def _precedent_debug(stage: str, **details: Any) -> None:
    if not _PRECEDENT_DEBUG:
        return
    payload = {"ok": True, "event": "precedent_wrapper_debug", "stage": stage}
    payload.update(details)
    print(json.dumps(payload, default=str), flush=True)


def _default_precedent_outcomes_path() -> Path:
    for candidate in _DEFAULT_PRECEDENT_OUTCOMES_CANDIDATES:
        if candidate.exists():
            return candidate
    return _DEFAULT_PRECEDENT_OUTCOMES_CANDIDATES[0]


@lru_cache(maxsize=8)
def _load_outcomes_table_cached(path_str: str) -> pd.DataFrame:
    started = time.perf_counter()
    _precedent_debug("load_outcomes_table_cached:start", path=path_str)
    table = pd.read_parquet(path_str)
    _precedent_debug(
        "load_outcomes_table_cached:done",
        path=path_str,
        rows=int(len(table)),
        elapsed_seconds=round(time.perf_counter() - started, 6),
    )
    return table


def _load_outcomes_table(path: Path) -> pd.DataFrame:
    # Keep one in-memory copy per outcomes path for faster repeated precedent calls.
    return _load_outcomes_table_cached(str(path.resolve()))


@lru_cache(maxsize=8)
def _load_precedent_runtime_cached(path_str: str) :
    """Build once per outcomes path: historical stores + retrieval index."""
    from .historical_stores import build_historical_stores_from_outcomes
    from .precedent_brain import augment_precedent_state_vector_columns, build_precedent_retrieval_index

    started = time.perf_counter()
    _precedent_debug("load_precedent_runtime_cached:start", path=path_str)
    full_df = _load_outcomes_table_cached(path_str)
    full_df = augment_precedent_state_vector_columns(full_df)
    stores_started = time.perf_counter()
    _precedent_debug("historical_stores:start", path=path_str, rows=int(len(full_df)))
    stores = build_historical_stores_from_outcomes(full_df, dataset_version=path_str)
    _precedent_debug(
        "historical_stores:done",
        path=path_str,
        elapsed_seconds=round(time.perf_counter() - stores_started, 6),
    )
    index_started = time.perf_counter()
    _precedent_debug("retrieval_index:start", path=path_str, rows=int(len(full_df)))
    retrieval_index = build_precedent_retrieval_index(full_df)
    _precedent_debug(
        "retrieval_index:done",
        path=path_str,
        index_rows=int(getattr(retrieval_index, "n_rows", 0) or 0),
        elapsed_seconds=round(time.perf_counter() - index_started, 6),
    )
    _precedent_debug(
        "load_precedent_runtime_cached:done",
        path=path_str,
        elapsed_seconds=round(time.perf_counter() - started, 6),
    )
    return stores, retrieval_index


