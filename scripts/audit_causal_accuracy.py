#!/usr/bin/env python
"""Audit causal coverage/strict-gate quality from completed recommendation runs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


_CAUSAL_DRIVER_NAMES = {
    "causal_model_blend_weight",
    "causal_model_quality",
    "causal_model_support_score",
    "causal_model_mode",
}


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        out = float(v)
    except Exception:
        return float(default)
    if out != out:  # NaN
        return float(default)
    return float(out)


def _load_json(path: Path) -> Dict[str, Any]:
    return dict(json.loads(path.read_text()) or {})


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit causal accuracy/coverage from completed run artifacts.")
    p.add_argument(
        "--runs-roots",
        nargs="+",
        default=["/tmp/recommendation_runs_v4_clean", "/tmp/recommendation_runs_fresh"],
        help="Run roots to scan (each must contain runs/ and artifacts/).",
    )
    p.add_argument(
        "--run-ids-file",
        default="",
        help="Optional text file with run IDs to include (one run_id per line, or 'CIK RUN_ID' pairs).",
    )
    p.add_argument("--out", default="/tmp/causal_accuracy_audit.json", help="Output JSON path.")
    p.add_argument("--min-action-rows", type=int, default=50, help="Minimum rows to report action-level stats.")
    return p.parse_args()


def _is_causal_row(driver_names: set[str]) -> bool:
    return bool(driver_names & _CAUSAL_DRIVER_NAMES)


