#!/usr/bin/env python
"""
Minimal on-demand HTTP API for precedent inference.

Endpoints:
  GET  /health
  POST /run_precedent
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Tuple

# Allow running as `python scripts/run_precedent_api.py` from repo root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.run import run_precedent


def _default_precedent_outcomes_path() -> str:
    repo_root = Path(__file__).resolve().parent.parent
    return str(repo_root / "data" / "curated" / "action_outcomes_with_credit_ratings.normalized_full.parquet")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve run_precedent over HTTP.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--config", default=None)
    parser.add_argument("--outcomes-path", default=_default_precedent_outcomes_path())
    parser.add_argument("--state-snapshot-root", default=None)
    parser.add_argument("--state-snapshot-path", default=None)
    return parser.parse_args()


def _json_bytes(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload, default=str).encode("utf-8")


