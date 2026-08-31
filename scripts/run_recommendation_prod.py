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


