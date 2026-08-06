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


def main() -> None:
    args = parse_args()
    store = RecommendationRunStore(root=args.runs_root)
    run = store.get_run(args.run_id)
    if run is None:
        raise ValueError(f"Run not found: {args.run_id}")

    snapshot_root = Path(args.snapshot_root) if args.snapshot_root else None
    snapshot_path = Path(args.snapshot_path) if args.snapshot_path else None
    entity_identifier_path = Path(args.entity_identifier_path)

    aliases = _snapshot_company_aliases(run.company_id, entity_identifier_path)
    snapshot = _resolve_snapshot(
        company_id=run.company_id,
        as_of_time=_parse_ts(run.as_of_time),
        snapshot_root=snapshot_root,
        snapshot_path=snapshot_path,
        snapshot_builder=None,
        snapshot_loader=None,
        aliases=aliases,
    )
    snapshot = _apply_scenario_overrides(snapshot, run.scenario)
    observed_hash = _hash_snapshot(snapshot)
    if observed_hash != run.frozen_state.snapshot_hash:
        raise ValueError(
            f"Frozen snapshot hash mismatch for run_id={run.run_id} "
            f"expected={run.frozen_state.snapshot_hash} got={observed_hash}"
        )

    registry = build_default_action_schema_registry(version="v1.0")
    candidate_set = generate_action_candidates(
        run=run,
        state_snapshot=snapshot,
        action_registry=registry,
        action_ids=args.action_id or None,
        action_type=args.action_type,
        max_candidates=args.max_candidates,
        min_candidates_target=args.min_candidates_target,
        strict_evidence=args.strict_evidence,
    )

    artifact_path = store.attach_artifact(run.run_id, args.attach_name, candidate_set)

    out_path = None
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(candidate_set, indent=2))

    print(
        json.dumps(
            {
                "ok": True,
                "run_id": run.run_id,
                "artifact": str(artifact_path),
                "out": str(out_path) if out_path else None,
                "count": len(candidate_set.get("candidates", [])),
            }
        )
    )


if __name__ == "__main__":
    main()
