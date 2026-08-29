#!/usr/bin/env python
"""HTTP API for RecommendationRun create/execute orchestration.

Endpoints:
  GET  /health
  GET  /precedent_query
  POST /create_run
  POST /execute_run
  POST /create_and_execute_run
"""

from __future__ import annotations

import argparse
from datetime import datetime
from concurrent.futures import Future, ThreadPoolExecutor
import hashlib
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).parent.parent))


def _default_precedent_outcomes_path() -> str:
    repo_root = Path(__file__).resolve().parent.parent
    return str(repo_root / "data" / "curated" / "action_outcomes_with_credit_ratings.normalized_full.parquet")


def _recommendation_run_bindings():
    from src.recommendation_run import (
        RecommendationRunStore,
        _resolve_snapshot,
        _snapshot_company_aliases,
        create_recommendation_run,
    )

    return RecommendationRunStore, _resolve_snapshot, _snapshot_company_aliases, create_recommendation_run


def _build_default_registry():
    from src.action_ontology import build_default_action_schema_registry

    return build_default_action_schema_registry(version="v1.0")


