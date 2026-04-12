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


def _event_ts(event: Dict[str, Any]) -> float | None:
    raw = str(event.get("timestamp", "") or "")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def _run_stage_seconds(runs_root: Path, run_id: str) -> Dict[str, float]:
    p = runs_root / "runs" / f"run_id={run_id}.json"
    if not p.exists():
        return {}
    try:
        run = json.loads(p.read_text())
    except Exception:
        return {}
    events = run.get("audit_log") or []
    if not isinstance(events, list):
        return {}

    by_type: Dict[str, float] = {}
    for e in events:
        if not isinstance(e, dict):
            continue
        t = str(e.get("event_type", "") or "")
        ts = _event_ts(e)
        if t and ts is not None:
            by_type[t] = ts

    out: Dict[str, float] = {}
    def _dur(start: str, end: str, key: str) -> None:
        if start in by_type and end in by_type:
            out[key] = round(max(0.0, by_type[end] - by_type[start]), 3)

    _dur("snapshot_frozen", "candidate_generation_started", "snapshot_load")
    _dur("candidate_generation_started", "candidate_generation_completed", "candidate_generation")
    _dur("feasibility_eval_started", "feasibility_eval_completed", "feasibility")
    _dur("precedent_retrieval_started", "precedent_retrieval_completed", "precedent")
    _dur("planning_started", "planning_completed", "planning")
    return out


def _build_keyed_snapshot_loader(snapshot_root: Path):
    def _loader(company_id: str, as_of_time: datetime) -> Dict[str, Any]:
        as_of_date = as_of_time.strftime("%Y-%m-%d")
        p = (
            snapshot_root
            / "keyed"
            / f"as_of_date={as_of_date}"
            / f"company_id={company_id}.json"
        )
        if not p.exists():
            raise FileNotFoundError(f"Keyed snapshot not found: {p}")
        return json.loads(p.read_text())

    return _loader


def _mock_precedent_runner(**_: Any) -> Dict[str, Any]:
    # Minimal precedent pack shape that downstream planning can consume.
    return {
        "legacy_distributions": [
            {
                "metric": "outcome_pe_12m",
                "p25": 0.0,
                "p50": 0.0,
                "p75": 0.0,
            }
        ],
        "distributions": [],
        "citations": [],
        "match_confidence": 0.0,
        "out_of_sample_rate": 1.0,
    }


def _keyed_snapshot_path(snapshot_root: Path, as_of: str, company_id: str) -> Path:
    return (
        snapshot_root
        / "keyed"
        / f"as_of_date={as_of}"
        / f"company_id={company_id}.json"
    )


