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


