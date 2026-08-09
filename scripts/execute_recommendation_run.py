#!/usr/bin/env python
"""Execute RecommendationRun lifecycle under run_id."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _default_precedent_outcomes_path() -> str:
    repo_root = Path(__file__).resolve().parent.parent
    return str(repo_root / "data" / "curated" / "action_outcomes_with_credit_ratings.normalized_full.parquet")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Execute recommendation run stages")
    p.add_argument("--run-id", required=True)
    p.add_argument("--runs-root", default="data/recommendation_runs")
    p.add_argument("--snapshot-root", default=None)
    p.add_argument("--snapshot-path", default=None)
    p.add_argument("--entity-identifier-path", default="data/inputs_layer/entity_identifier.parquet")
    p.add_argument("--outcomes-path", default=_default_precedent_outcomes_path())
    p.add_argument("--config", default=None)
    p.add_argument("--action-id", action="append", default=[])
    p.add_argument("--action-type", default=None)
    p.add_argument("--max-candidates", type=int, default=12)
    p.add_argument("--min-candidates-target", type=int, default=0)
    p.add_argument("--precedent-top-k", type=int, default=0, help="Limit precedent retrieval to top-K feasible candidates (0=all)")
    p.add_argument("--top-plans", type=int, default=3)
    p.add_argument("--strict-evidence", action="store_true")
    return p.parse_args()


