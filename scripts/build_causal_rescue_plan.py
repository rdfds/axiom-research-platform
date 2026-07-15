#!/usr/bin/env python
"""Create a targeted causal rescue plan from full ML audit output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build causal rescue/blocklist plan from ML audit JSON.")
    p.add_argument("--audit-json", required=True, help="Path to audit_full_ml_status.py output JSON.")
    p.add_argument(
        "--strict-pass-threshold",
        type=float,
        default=0.50,
        help="Actions below this strict pass rate are flagged for rescue/blocklist.",
    )
    p.add_argument(
        "--min-action-rows",
        type=int,
        default=100,
        help="Minimum action rows required for rescue prioritization.",
    )
    p.add_argument(
        "--low-row-blocklist-threshold",
        type=int,
        default=50,
        help="Include low-row actions in blocklist suggestions when they clear this minimum support threshold.",
    )
    p.add_argument(
        "--out-json",
        default="/tmp/causal_rescue_plan.json",
        help="Output plan JSON.",
    )
    p.add_argument(
        "--out-blocklist",
        default="/tmp/causal_action_blocklist_suggested.txt",
        help="Output newline-delimited action blocklist.",
    )
    p.add_argument(
        "--out-rescue-actions",
        default="/tmp/causal_action_rescue_candidates.txt",
        help="Output newline-delimited action rescue candidates.",
    )
    return p.parse_args()


def _f(v: Any, default: float = 0.0) :
    try:
        out = float(v)
    except Exception:
        return float(default)
    if out != out:
        return float(default)
    return float(out)


def _classify_action(action: Dict[str, Any], strict_thr: float) -> Dict[str, Any]:
    row = dict(action)
    strict_rate = _f(row.get("strict_pass_rate"))
    causal_rate = _f(row['causal_rate'])
    rows = int(row.get("rows", 0) or 0)

    if causal_rate <= 0.0:
        row["rescue_reason"] = "no_causal_coverage"
        row["recommended_action"] = "keep_blocked_collect_more_signal"
    elif strict_rate <= 0.0:
        row["rescue_reason"] = "strict_gate_total_failure"
        row["recommended_action"] = "targeted_retrain_and_rebenchmark"
    elif strict_rate < strict_thr:
        row["rescue_reason"] = "strict_gate_under_threshold"
        row["recommended_action"] = "recalibrate_and_rebenchmark"
    else:
        row["rescue_reason"] = "healthy"
        row["recommended_action"] = "no_action"

    if rows <= 0:
        row["rescue_reason"] = "no_support"
        row["recommended_action"] = "keep_blocked_collect_more_signal"
    return row


