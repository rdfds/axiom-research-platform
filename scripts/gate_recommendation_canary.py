#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from audit_full_ml_status import build_ml_status_audit  # noqa: E402


def _parse_args() :
    p = argparse.ArgumentParser(description="Gate recommendation canary runs against regression thresholds.")
    p.add_argument("--runs-roots", nargs="+", required=True)
    p.add_argument("--run-ids-file", required=True)
    p.add_argument("--out", default="/tmp/recommendation_canary_gate.json")
    p.add_argument("--min-action-rows", type=int, default=50)
    p.add_argument("--min-causal-rate-mean", type=float, default=0.75)
    p.add_argument("--min-strict-all-mean", type=float, default=0.70)
    p.add_argument("--min-strict-causal-mean", type=float, default=0.90)
    p.add_argument("--min-precedent-conf-mean", type=float, default=0.35)
    p.add_argument("--max-precedent-oos-mean", type=float, default=0.90)
    return p.parse_args()


def _mean(payload: Dict[str, Any], *keys: str) -> float:
    cur: Any = payload
    for key in keys:
        cur = (cur or {}).get(key) if isinstance(cur, dict) else None
    try:
        out = float(cur)
    except Exception:
        return 0.0
    if out != out:
        return 0.0
    return float(out)


