#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Keep OpenMP/BLAS behavior stable before any heavy scientific stack imports.
# Setting these at process start is more reliable than doing it later inside the
# evaluation context because pandas/pyarrow/numexpr may already have imported
# libomp-linked code by then.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from src.historical_recommendation_eval import build_historical_recommendation_report, _load_fixed_historical_cases
from src.pipeline.precedent_distance_v2_learning import (
    build_precedent_distance_v2_payload,
    coordinate_search_scope_configuration,
    default_scope_configuration,
    load_precedent_distance_v2_objective,
    write_precedent_distance_v2_payload,
)

DEFAULT_LOCK_CONFIG = ROOT / "configs" / "historical_eval_manifests" / "2026-03-17" / "manual_replay_benchmark_lock.json"
DEFAULT_OBJECTIVE_CONFIG = ROOT / "configs" / "precedent_distance_v2_objective.json"
FIT_PRECEDENT_CONFIG_PATH = ROOT / "data" / "curated" / "_fit_mode_precedent_config.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit weighted_distance_v2 against frozen historical replay metrics.")
    parser.add_argument("--benchmark", required=True, help="Benchmark key from the lock config.")
    parser.add_argument("--scope", required=True, help="Action family or exact action id to optimize, e.g. capital_structure.")
    parser.add_argument("--runs-root", required=True, help="Base runs root for search evaluations.")
    parser.add_argument("--out-json", required=True, help="Where to write the fitted v2 weight payload.")
    parser.add_argument("--lock-config", default=str(DEFAULT_LOCK_CONFIG))
    parser.add_argument("--objective-config", default=str(DEFAULT_OBJECTIVE_CONFIG))
    parser.add_argument("--fixed-case-path", default="", help="Optional fixed-case report/manifest path.")
    parser.add_argument("--case-count", type=int, default=None)
    parser.add_argument("--max-rounds", type=int, default=1)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def _resolve_candidate_path(values: Iterable[str | Path]) -> Path:
    candidates = [_resolve_path(value) for value in values]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


