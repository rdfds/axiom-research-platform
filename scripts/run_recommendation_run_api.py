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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Serve RecommendationRun orchestration over HTTP.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--runs-root", default="data/recommendation_runs")
    p.add_argument("--snapshot-root", default=None)
    p.add_argument("--snapshot-path", default=None)
    p.add_argument("--entity-graph-path", default="data/inputs_layer/entity_graph.parquet")
    p.add_argument("--entity-identifier-path", default="data/inputs_layer/entity_identifier.parquet")
    p.add_argument("--outcomes-path", default=_default_precedent_outcomes_path())
    p.add_argument("--config", default=None)
    p.add_argument("--max-candidates", type=int, default=12)
    p.add_argument("--min-candidates-target", type=int, default=300)
    p.add_argument("--precedent-top-k", type=int, default=25)
    p.add_argument("--top-plans", type=int, default=3)
    p.add_argument("--startup-warmup", dest="startup_warmup", action="store_true", default=True)
    p.add_argument("--no-startup-warmup", dest="startup_warmup", action="store_false")
    p.add_argument("--warmup-company-id", default="0000320193")
    p.add_argument("--warmup-as-of", default="2026-02-28")
    return p.parse_args()


def _json_bytes(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload, default=str).encode("utf-8")


def _read_json_body(handler: BaseHTTPRequestHandler) -> Tuple[Dict[str, Any], Optional[str]]:
    raw_len = handler.headers.get("Content-Length", "0")
    try:
        length = int(raw_len)
    except Exception:
        return {}, "invalid_content_length"
    if length <= 0:
        return {}, "empty_body"
    raw = handler.rfile.read(length)
    try:
        obj = json.loads(raw.decode("utf-8"))
    except Exception:
        return {}, "invalid_json"
    if not isinstance(obj, dict):
        return {}, "body_must_be_object"
    return obj, None


