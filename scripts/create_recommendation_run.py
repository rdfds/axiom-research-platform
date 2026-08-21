#!/usr/bin/env python
"""Create and persist a RecommendationRun."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _load_json_arg(raw: str | None, file_path: str | None) -> dict | None:
    if raw and file_path:
        raise ValueError("Provide only one of inline JSON or file path")
    if raw:
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            raise ValueError("JSON argument must decode to object")
        return obj
    if file_path:
        obj = json.loads(Path(file_path).read_text())
        if not isinstance(obj, dict):
            raise ValueError("JSON file must decode to object")
        return obj
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create RecommendationRun")
    p.add_argument("--company-id", required=True)
    p.add_argument("--as-of", required=True)

    p.add_argument("--objectives-json", default=None, help="Inline JSON object for ObjectiveVector")
    p.add_argument("--objectives-file", default=None, help="Path to JSON file for ObjectiveVector")

    p.add_argument("--constraints-json", default=None, help="Inline JSON object for ConstraintSet")
    p.add_argument("--constraints-file", default=None, help="Path to JSON file for ConstraintSet")

    p.add_argument("--scenario-json", default=None, help="Inline JSON object for ScenarioAssumptions")
    p.add_argument("--scenario-file", default=None, help="Path to JSON file for ScenarioAssumptions")

    p.add_argument("--snapshot-root", default=None)
    p.add_argument("--snapshot-path", default=None)
    p.add_argument("--entity-graph-path", default="data/inputs_layer/entity_graph.parquet")
    p.add_argument("--entity-identifier-path", default="data/inputs_layer/entity_identifier.parquet")
    p.add_argument("--runs-root", default="data/recommendation_runs")
    p.add_argument("--planner-random-seed", type=int, default=None)
    return p.parse_args()


def main() -> None:
    t0 = time.time()
    print(json.dumps({"ok": True, "event": "startup", "stage": "import_recommendation_run"}), flush=True)
    from src.recommendation_run import RecommendationRunStore, create_recommendation_run

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

    objectives = _load_json_arg(args.objectives_json, args.objectives_file)
    constraints = _load_json_arg(args.constraints_json, args.constraints_file)
    scenario = _load_json_arg(args.scenario_json, args.scenario_file)

    store = RecommendationRunStore(root=args.runs_root)
    run_id = create_recommendation_run(
        company_id=args.company_id,
        as_of_time=args.as_of,
        objectives=objectives,
        constraints=constraints,
        scenario=scenario,
        run_store=store,
        snapshot_root=args.snapshot_root,
        snapshot_path=args.snapshot_path,
        entity_graph_path=args.entity_graph_path,
        entity_identifier_path=args.entity_identifier_path,
        planner_random_seed=args.planner_random_seed,
    )

    print(json.dumps({"ok": True, "run_id": run_id, "runs_root": args.runs_root}))


if __name__ == "__main__":
    main()
