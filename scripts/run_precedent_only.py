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


def _artifact_path(runs_root: Path, run_id: str, name: str) -> Path:
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


def _load_candidates_from_feasibility(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text())
    out: List[Dict[str, Any]] = []
    for row in payload.get("results", []):
        if not bool(row.get("feasible")):
            continue
        candidate = dict(row.get("candidate") or row.get("action_candidate") or {})
        if candidate:
            out.append(candidate)
    return out


def _load_candidates_from_candidate_set(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text())
    return [dict(row or {}) for row in payload.get("candidates", []) if isinstance(row, dict)]


def main() -> None:
    t0 = time.time()
    print(json.dumps({"ok": True, "event": "startup", "stage": "import_precedent_only"}), flush=True)
    from src.recommendation_run import RecommendationRunStore
    from src.recommendation_run_orchestrator import (
        _precedent_bindings,
        _precedent_profile,
        _retrieve_precedents,
    )

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
    runs_root = Path(args.runs_root)
    store = RecommendationRunStore(runs_root)
    run = store.get_run(args.run_id)
    if run is None:
        raise SystemExit(f"Run not found: {args.run_id}")

    execution_cfg = _metadata_execution_config(run)
    snapshot_root = _resolve_path(args.snapshot_root, execution_cfg, "snapshot_root")
    snapshot_path = _resolve_path(args.snapshot_path, execution_cfg, "snapshot_path")
    outcomes_path = _resolve_path(args.outcomes_path, execution_cfg, "outcomes_path")
    config_path = _resolve_path(args.config_path, execution_cfg, "config_path")
    if not snapshot_root and not snapshot_path:
        inferred_snapshot_root = _infer_snapshot_root(run)
        if inferred_snapshot_root:
            snapshot_root = inferred_snapshot_root
            print(
                json.dumps(
                    {
                        "ok": True,
                        "event": "precedent_only_snapshot_inferred",
                        "run_id": args.run_id,
                        "snapshot_root": snapshot_root,
                    }
                ),
                flush=True,
            )
        else:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "event": "precedent_only_snapshot_missing",
                        "run_id": args.run_id,
                        "message": "No snapshot_root or snapshot_path available; fallback will rebuild company state.",
                    }
                ),
                flush=True,
            )

    feasibility_path = Path(args.feasibility_path) if args.feasibility_path else _artifact_path(
        runs_root,
        args.run_id,
        "FeasibilityResults.json",
    )
    candidate_set_path = Path(args.candidate_set_path) if args.candidate_set_path else _artifact_path(
        runs_root,
        args.run_id,
        "CandidateSet.json",
    )

    if args.all_candidates:
        candidates = _load_candidates_from_candidate_set(candidate_set_path)
        candidate_source = "candidate_set"
    else:
        candidates = _load_candidates_from_feasibility(feasibility_path)
        candidate_source = "feasibility_results"

    if args.action_id:
        allow = set(args.action_id)
        candidates = [row for row in candidates if str(row.get("action_id", "")) in allow]

    build_precedent_index, run_precedent, _ = _precedent_bindings()
    print(
        json.dumps(
            {
                "ok": True,
                "event": "precedent_only_started",
                "run_id": args.run_id,
                "candidate_source": candidate_source,
                "candidate_count": len(candidates),
                "precedent_top_k": int(args.precedent_top_k),
            }
        ),
        flush=True,
    )

    runner = run_precedent
    if args.log_candidates:
        def _logged_runner(**kwargs: Any) -> Any:
            candidate_id = str(kwargs.get("candidate_id", ""))
            action_id = str(kwargs.get("action_id", ""))
            started = time.time()
            print(
                json.dumps(
                    {
                        "ok": True,
                        "event": "precedent_candidate_started",
                        "run_id": args.run_id,
                        "candidate_id": candidate_id,
                        "action_id": action_id,
                    }
                ),
                flush=True,
            )
            try:
                pack = run_precedent(**kwargs)
            except Exception as exc:
                print(
                    json.dumps(
                        {
                            "ok": False,
                            "event": "precedent_candidate_failed",
                            "run_id": args.run_id,
                            "candidate_id": candidate_id,
                            "action_id": action_id,
                            "elapsed_seconds": round(time.time() - started, 6),
                            "error": str(exc),
                        }
                    ),
                    flush=True,
                )
                raise
            print(
                json.dumps(
                    {
                        "ok": True,
                        "event": "precedent_candidate_completed",
                        "run_id": args.run_id,
                        "candidate_id": candidate_id,
                        "action_id": action_id,
                        "elapsed_seconds": round(time.time() - started, 6),
                        "profiling": dict(getattr(pack, "profiling", {}) or {}),
                    }
                ),
                flush=True,
            )
            return pack

        runner = _logged_runner

    matches = _retrieve_precedents(
        run=run,
        feasible_candidates=candidates,
        precedent_runner=runner,
        precedent_top_k=int(args.precedent_top_k),
        snapshot_root=snapshot_root,
        snapshot_path=snapshot_path,
        outcomes_path=outcomes_path,
        config_path=config_path,
        progress_callback=None,
    )
    profile = _precedent_profile(matches)
    precedent_index = build_precedent_index(run_id=args.run_id, precedent_matches=matches)

    tag = str(args.artifact_tag or "precedent_only").strip().replace(" ", "_")
    matches_key = f"PrecedentMatches_{tag}"
    index_key = f"PrecedentIndex_{tag}"
    matches_path = store.attach_artifact(
        args.run_id,
        matches_key,
        {
            "run_id": args.run_id,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "profile": profile,
            "results": matches,
        },
    )
    index_path = store.attach_artifact(args.run_id, index_key, precedent_index)
    store.merge_metadata(
        args.run_id,
        {
            "precedent_only": {
                tag: {
                    "candidate_source": candidate_source,
                    "candidate_count": len(candidates),
                    "precedent_top_k": int(args.precedent_top_k),
                    "matches_artifact": str(matches_path),
                    "index_artifact": str(index_path),
                }
            }
        },
    )

    print(
        json.dumps(
            {
                "ok": True,
                "run_id": args.run_id,
                "candidate_source": candidate_source,
                "candidate_count": len(candidates),
                "selected_precedent_candidates": len(matches),
                "profile": profile,
                "matches_artifact": str(matches_path),
                "index_artifact": str(index_path),
                "elapsed_seconds": round(time.time() - t0, 3),
            }
        )
    )


if __name__ == "__main__":
    main()
