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


def main() -> None:
    t0 = time.time()
    print(json.dumps({"ok": True, "event": "startup", "stage": "import_orchestrator"}), flush=True)
    from src.recommendation_run_orchestrator import execute_recommendation_run

    print(
        json.dumps(
            {
                "ok": True,
                "event": "startup",
                "stage": "import_done",
                "elapsed_seconds": round(time.time() - t0, 3),
            }
        ),
        flush=True,
    )

    args = parse_args()
    summary = execute_recommendation_run(
        run_id=args.run_id,
        runs_root=args.runs_root,
        snapshot_root=args.snapshot_root,
        snapshot_path=args.snapshot_path,
        entity_identifier_path=args.entity_identifier_path,
        action_ids=args.action_id or None,
        action_type=args.action_type,
        max_candidates=args.max_candidates,
        min_candidates_target=args.min_candidates_target,
        strict_evidence=args.strict_evidence,
        precedent_top_k=args.precedent_top_k,
        outcomes_path=args.outcomes_path,
        config_path=args.config,
        top_plans=args.top_plans,
    )
    print(json.dumps(summary, default=str))


if __name__ == "__main__":
    main()
