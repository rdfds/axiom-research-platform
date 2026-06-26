#!/usr/bin/env python
"""Benchmark targeted precedent families for an existing recommendation run."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))


def _default_precedent_outcomes_path() -> str:
    repo_root = Path(__file__).resolve().parent.parent
    return str(repo_root / "data" / "curated" / "action_outcomes_with_credit_ratings.normalized_full.parquet")


DEFAULT_PRESET: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("debt_issuance", ("capital_structure.new_debt_issuance",)),
    ("refinancing", ("capital_structure.refinancing",)),
    ("platform_acquisition", ("mna.platform_acquisition",)),
    ("tuck_in_acquisition", ("mna.tuck_in_acquisition",)),
    ("go_private_lbo", ("mna.go_private_lbo",)),
    ("divestiture_partial", ("portfolio.divestiture_partial",)),
    ("buyback", ("capital_return.open_market_buyback", "capital_return.accelerated_share_repurchase")),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark targeted precedent families on an existing run")
    p.add_argument("--run-id", required=True)
    p.add_argument("--runs-root", default="data/recommendation_runs")
    p.add_argument("--snapshot-root", default=None)
    p.add_argument("--snapshot-path", default=None)
    p.add_argument("--outcomes-path", default=None)
    p.add_argument("--config-path", default=None)
    p.add_argument("--feasibility-path", default=None)
    p.add_argument("--candidate-set-path", default=None)
    p.add_argument("--precedent-top-k", type=int, default=1)
    p.add_argument("--artifact-prefix", default="precedent_bench")
    p.add_argument("--out", default=None)
    p.add_argument(
        "--slice",
        action="append",
        default=[],
        help="Custom slice as label=action_id[,action_id2,...]. If omitted, uses the built-in targeted preset.",
    )
    return p.parse_args()


def _artifact_path(runs_root: Path, run_id: str, name: str) -> Path:
    return runs_root / "artifacts" / f"run_id={run_id}" / name


def _metadata_execution_config(run: Any) -> Dict[str, Any]:
    metadata = dict(getattr(run, "metadata", {}) or {})
    config = dict(metadata.get("config", {}) or {})
    return dict(config.get("execution", {}) or {})


def _resolve_path(explicit: Optional[str], execution_cfg: Dict[str, Any], key: str) -> Optional[str]:
    if explicit:
        return str(explicit)
    value = execution_cfg.get(key)
    if value:
        return str(value)
    if key == "outcomes_path":
        return _default_precedent_outcomes_path()
    return None


