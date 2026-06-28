#!/usr/bin/env python
"""Run precedent retrieval only for an existing recommendation run."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))


def _default_precedent_outcomes_path() -> str:
    repo_root = Path(__file__).resolve().parent.parent
    return str(repo_root / "data" / "curated" / "action_outcomes_with_credit_ratings.normalized_full.parquet")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run precedent retrieval only for an existing run")
    p.add_argument("--run-id", required=True)
    p.add_argument("--runs-root", default="data/recommendation_runs")
    p.add_argument("--snapshot-root", default=None)
    p.add_argument("--snapshot-path", default=None)
    p.add_argument("--outcomes-path", default=None)
    p.add_argument("--config-path", default=None)
    p.add_argument("--feasibility-path", default=None)
    p.add_argument("--candidate-set-path", default=None)
    p.add_argument("--precedent-top-k", type=int, default=25)
    p.add_argument("--action-id", action="append", default=[])
    p.add_argument("--artifact-tag", default="precedent_only")
    p.add_argument("--all-candidates", action="store_true", help="Use CandidateSet instead of feasible candidates")
    p.add_argument("--log-candidates", action="store_true", help="Emit per-candidate start/finish logs")
    return p.parse_args()


def _artifact_path(runs_root: Path, run_id: str, name: str) :
    return runs_root / "artifacts" / f"run_id={run_id}" / name


def _metadata_execution_config(run: Any) -> Dict[str, Any]:
    metadata = dict(getattr(run, "metadata", {}) or {})
    config = dict(metadata.get("config", {}) or {})
    return dict(config.get("execution", {}) or {})


def _resolve_path(
    explicit: Optional[str],
    execution_cfg: Dict[str, Any],
    key: str,
) -> Optional[str]:
    if explicit:
        return str(explicit)
    value = execution_cfg.get(key)
    if value:
        return str(value)
    if key == "outcomes_path":
        return _default_precedent_outcomes_path()
    return None


def _infer_snapshot_root(run: Any) -> Optional[str]:
    repo_root = Path(__file__).resolve().parent.parent
    as_of_value = str(getattr(run, "as_of_time", "") or "")
    if not as_of_value:
        return None
    as_of_date = as_of_value[:10]
    candidate = repo_root / "data" / "company_state_snapshots" / f"final_run_{as_of_date}"
    if candidate.exists():
        return str(candidate)
    return None


