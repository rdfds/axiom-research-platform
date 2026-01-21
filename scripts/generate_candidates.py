#!/usr/bin/env python
"""Generate and persist CandidateSet for an existing RecommendationRun."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.action_ontology import build_default_action_schema_registry
from src.candidate_generation import generate_action_candidates
from src.recommendation_run import (
    RecommendationRunStore,
    _apply_scenario_overrides,
    _hash_snapshot,
    _parse_ts,
    _resolve_snapshot,
    _snapshot_company_aliases,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate CandidateSet for run_id")
    p.add_argument("--run-id", required=True)
    p.add_argument("--runs-root", default="data/recommendation_runs")
    p.add_argument("--snapshot-root", default=None)
    p.add_argument("--snapshot-path", default=None)
    p.add_argument("--entity-identifier-path", default="data/inputs_layer/entity_identifier.parquet")
    p.add_argument("--action-id", action="append", default=[])
    p.add_argument("--action-type", default=None)
    p.add_argument("--max-candidates", type=int, default=1500)
    p.add_argument("--min-candidates-target", type=int, default=0)
    p.add_argument("--strict-evidence", action="store_true")
    p.add_argument("--attach-name", default="CandidateSet")
    p.add_argument("--out", default=None, help="Optional explicit JSON output path (in addition to run artifact).")
    return p.parse_args()


