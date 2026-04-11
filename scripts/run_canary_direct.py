#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import traceback
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run recommendation canary directly via orchestrator (no HTTP API)."
    )
    p.add_argument("--runs-root", required=True)
    p.add_argument("--snapshot-root", required=True)
    p.add_argument("--entity-graph-path", required=True)
    p.add_argument("--entity-identifier-path", required=True)
    p.add_argument("--outcomes-path", required=True)
    p.add_argument("--config-path", default=None)
    p.add_argument("--as-of", default="2026-02-28")
    p.add_argument("--companies", nargs="+", required=True)
    p.add_argument(
        "--action-ids",
        nargs="+",
        default=None,
        help="Optional explicit action_id list to limit runtime for smoke tests.",
    )
    p.add_argument("--max-candidates", type=int, default=300)
    p.add_argument(
        "--min-candidates-target",
        type=int,
        default=300,
        help="Target lower bound on candidate set size (API-equivalent behavior).",
    )
    p.add_argument("--precedent-top-k", type=int, default=25)
    p.add_argument("--top-plans", type=int, default=1)
    p.add_argument(
        "--mock-precedent",
        action="store_true",
        help="Use a local no-op precedent runner for fast smoke testing.",
    )
    p.add_argument(
        "--disable-keyed-loader",
        action="store_true",
        help="Disable direct keyed snapshot loader and fall back to default resolver.",
    )
    p.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=20.0,
        help="Emit a heartbeat log every N seconds while a company run is in progress (0 to disable).",
    )
    p.add_argument("--run-ids-out", required=True)
    p.add_argument("--summary-out", default=None)
    return p.parse_args()


def _safe_run_ids_write(path: Path, pairs: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(pairs) + ("\n" if pairs else ""))


