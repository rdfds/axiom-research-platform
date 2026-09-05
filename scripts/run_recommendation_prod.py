#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


_REPO_ROOT = Path(__file__).resolve().parents[1]


def _default_path(*parts: str) -> str:
    return str(_REPO_ROOT.joinpath(*parts))


def _default_precedent_outcomes_path() -> str:
    return _default_path("data", "curated", "action_outcomes_with_credit_ratings.normalized_full.parquet")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="One-command production runner for RecommendationRun.")
    p.add_argument("--runs-root", default="/tmp/recommendation_runs_prod")
    p.add_argument(
        "--snapshot-root",
        default=_default_path("data", "company_state_snapshots", "final_run_2026-02-28"),
    )
    p.add_argument(
        "--entity-graph-path",
        default=_default_path("data", "inputs_layer", "entity_graph.parquet"),
    )
    p.add_argument(
        "--entity-identifier-path",
        default=_default_path("data", "inputs_layer", "entity_identifier.parquet"),
    )
    p.add_argument(
        "--outcomes-path",
        default=_default_precedent_outcomes_path(),
    )
    p.add_argument("--config-path", default=None)
    p.add_argument("--as-of", default="2026-02-28")
    p.add_argument("--companies", nargs="+", required=True)
    p.add_argument("--action-ids", nargs="+", default=None)
    p.add_argument("--max-candidates", type=int, default=300)
    p.add_argument("--min-candidates-target", type=int, default=300)
    p.add_argument("--precedent-top-k", type=int, default=25)
    p.add_argument("--top-plans", type=int, default=1)
    p.add_argument("--strict-evidence", action="store_true")
    p.add_argument("--heartbeat-seconds", type=float, default=20.0)
    p.add_argument("--run-ids-out", default="/tmp/recommendation_prod_run_ids.txt")
    p.add_argument("--summary-out", default="")

    p.add_argument(
        "--causal-model-path",
        default=_default_path("data", "models", "causal_impact_model_v5_5_hybrid.json"),
    )
    p.add_argument(
        "--causal-routing-config-path",
        default=_default_path("configs", "causal_capital_routing_prod_dividend_v2.json"),
    )
    p.add_argument(
        "--causal-action-blocklist-path",
        default=_default_path("config", "causal_action_blocklist_prod_v2.txt"),
    )
    p.add_argument("--causal-impact-mode", default="blend")
    p.add_argument("--causal-min-objective-oos-r2", type=float, default=0.08)
    p.add_argument("--causal-strict-quality-floor", type=float, default=0.08)
    p.add_argument("--causal-strict-support-floor", type=float, default=0.35)
    p.add_argument("--causal-strict-min-train-rows", type=int, default=1000)
    p.add_argument("--causal-strict-min-oos-r2", type=float, default=0.00)
    p.add_argument("--causal-strict-min-treated-rows", type=int, default=1500)
    p.add_argument("--causal-strict-min-control-rows", type=int, default=20000)
    p.add_argument("--precedent-workers", type=int, default=0)
    p.add_argument(
        "--warm-precedent-runtime",
        dest="warm_precedent_runtime",
        action="store_true",
        default=True,
        help="Preload precedent runtime before the first company instead of lazily on demand.",
    )
    p.add_argument(
        "--no-warm-precedent-runtime",
        dest="warm_precedent_runtime",
        action="store_false",
        help="Disable precedent runtime warmup before the first company.",
    )
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
        ts = _event_ts(e)
        if ts is not None:
            by_type[str(e.get("event_type", ""))] = ts

    out: Dict[str, float] = {}

    def _dur(start: str, end: str, key: str) -> None:
        if start in by_type and end in by_type:
            out[key] = round(max(0.0, by_type[end] - by_type[start]), 3)

    _dur("snapshot_frozen", "candidate_generation_started", "snapshot_load")
    _dur("candidate_generation_started", "candidate_generation_completed", "candidate_generation")
    _dur("feasibility_eval_started", "feasibility_eval_completed", "feasibility")
    _dur("precedent_retrieval_started", "precedent_retrieval_completed", "precedent")
    _dur("planning_started", "planning_completed", "planning")
    _dur("run_created", "run_completed", "total_run")
    return out


def _build_keyed_snapshot_loader(snapshot_root: Path):
    def _loader(company_id: str, as_of_time: datetime) -> Dict[str, Any]:
        p = (
            snapshot_root
            / "keyed"
            / f"as_of_date={as_of_time.strftime('%Y-%m-%d')}"
            / f"company_id={company_id}.json"
        )
        if not p.exists():
            raise FileNotFoundError(f"Keyed snapshot not found: {p}")
        return json.loads(p.read_text())

    return _loader


def _keyed_snapshot_path(snapshot_root: Path, as_of: str, company_id: str) -> Path:
    return snapshot_root / "keyed" / f"as_of_date={as_of}" / f"company_id={company_id}.json"


def _start_heartbeat(
    company_id: str,
    start_ts: float,
    every_seconds: float,
) :
    stop = threading.Event()
    if every_seconds <= 0:
        return stop, None

    def _run() -> None:
        while not stop.wait(every_seconds):
            print(
                json.dumps(
                    {
                        "ok": True,
                        "event": "company_heartbeat",
                        "company_id": company_id,
                        "elapsed_seconds": round(time.time() - start_ts, 3),
                    }
                ),
                flush=True,
            )

    th = threading.Thread(target=_run, daemon=True)
    th.start()
    return stop, th


