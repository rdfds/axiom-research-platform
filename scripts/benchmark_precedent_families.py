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


def _parse_slices(values: Sequence[str]) -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
    if not values:
        return DEFAULT_PRESET
    out: List[Tuple[str, Tuple[str, ...]]] = []
    for raw in values:
        label, sep, actions = str(raw).partition("=")
        if not sep or not label.strip() or not actions.strip():
            raise SystemExit(f"Invalid --slice value: {raw!r}")
        action_ids = tuple(a.strip() for a in actions.split(",") if a.strip())
        if not action_ids:
            raise SystemExit(f"Invalid --slice value: {raw!r}")
        out.append((label.strip(), action_ids))
    return tuple(out)


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def _slice_summary(label: str, action_ids: Sequence[str], matches: Sequence[Dict[str, Any]], elapsed_seconds: float) -> Dict[str, Any]:
    confs: List[float] = []
    oos_flags: List[float] = []
    action_scores: List[float] = []
    sims: List[float] = []
    tiers: Dict[str, int] = {}
    family_scale_keys: Dict[str, int] = {}
    family_keys: Dict[str, int] = {}
    pool_sizes: List[float] = []

    for row in matches:
        pack = dict(row.get("precedent_pack", {}) or {})
        md = dict(pack.get("mismatch_diagnostics", {}) or {})
        prof = dict(pack.get("profiling", {}) or {})
        confs.append(float(pack.get("precedent_confidence", pack.get("calibration_confidence", 0.0)) or 0.0))
        oos_flags.append(1.0 if bool(md.get("out_of_sample_flag")) else 0.0)
        action_scores.append(float(md.get("top_action_match_score") or 0.0))
        sims.append(float(md.get("top_similarity_mean") or 0.0))
        tier = str(md.get("retrieval_tier", "") or "")
        tiers[tier] = tiers.get(tier, 0) + 1
        for key in prof.get("selected_family_scale_keys") or []:
            s = str(key)
            family_scale_keys[s] = family_scale_keys.get(s, 0) + 1
        for key in prof.get("selected_family_keys") or []:
            s = str(key)
            family_keys[s] = family_keys.get(s, 0) + 1
        if prof.get("candidate_pool_size_after_prefilter") is not None:
            pool_sizes.append(float(prof.get("candidate_pool_size_after_prefilter") or 0.0))

    return {
        "label": label,
        "action_ids": list(action_ids),
        "selected_precedent_candidates": int(len(matches)),
        "precedent_conf_mean": round(_mean(confs), 6),
        "oos_rate": round(_mean(oos_flags), 6),
        "top_action_match_mean": round(_mean(action_scores), 6),
        "top_similarity_mean": round(_mean(sims), 6),
        "candidate_pool_size_after_prefilter_mean": round(_mean(pool_sizes), 3) if pool_sizes else 0.0,
        "tiers": dict(sorted(tiers.items())),
        "selected_family_scale_keys": dict(sorted(family_scale_keys.items())),
        "selected_family_keys": dict(sorted(family_keys.items())),
        "elapsed_seconds": round(float(elapsed_seconds), 6),
    }


def _print_table(rows: Sequence[Dict[str, Any]]) -> None:
    headers = (
        ("label", 22),
        ("selected_precedent_candidates", 6),
        ("precedent_conf_mean", 8),
        ("oos_rate", 8),
        ("top_action_match_mean", 8),
        ("top_similarity_mean", 8),
        ("candidate_pool_size_after_prefilter_mean", 8),
        ("elapsed_seconds", 8),
    )
    header_line = " ".join(f"{name[:width]:<{width}}" for name, width in headers)
    print(header_line)
    print("-" * len(header_line))
    for row in rows:
        print(
            " ".join(
                f"{str(row.get(name, ''))[:width]:<{width}}"
                for name, width in headers
            )
        )


def main() -> None:
    t0 = time.time()
    args = parse_args()
    slices = _parse_slices(args.slice)

    print(json.dumps({"ok": True, "event": "startup", "stage": "import_precedent_benchmark"}), flush=True)
    from src.recommendation_run import RecommendationRunStore
    from src.recommendation_run_orchestrator import _precedent_bindings, _precedent_profile, _retrieve_precedents
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
        snapshot_root = _infer_snapshot_root(run)

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

    feasible_candidates = _load_candidates_from_feasibility(feasibility_path)
    all_candidates = _load_candidates_from_candidate_set(candidate_set_path)
    build_precedent_index, run_precedent, _ = _precedent_bindings()

    summaries: List[Dict[str, Any]] = []
    artifacts: Dict[str, Dict[str, str]] = {}
    for label, action_ids in slices:
        candidate_source = "feasibility_results"
        selected = [row for row in feasible_candidates if str(row.get("action_id", "")) in set(action_ids)]
        if not selected:
            candidate_source = "candidate_set"
            selected = [row for row in all_candidates if str(row.get("action_id", "")) in set(action_ids)]

        print(
            json.dumps(
                {
                    "ok": True,
                    "event": "benchmark_slice_started",
                    "label": label,
                    "action_ids": list(action_ids),
                    "candidate_source": candidate_source,
                    "candidate_count": len(selected),
                }
            ),
            flush=True,
        )
        started = time.time()
        matches = _retrieve_precedents(
            run=run,
            feasible_candidates=selected,
            precedent_runner=run_precedent,
            precedent_top_k=int(args.precedent_top_k),
            snapshot_root=snapshot_root,
            snapshot_path=snapshot_path,
            outcomes_path=outcomes_path,
            config_path=config_path,
            progress_callback=None,
        )
        elapsed = time.time() - started
        profile = _precedent_profile(matches)
        index = build_precedent_index(run_id=args.run_id, precedent_matches=matches)
        tag = f"{args.artifact_prefix}_{label}".strip().replace(" ", "_")
        matches_key = f"PrecedentMatches_{tag}"
        index_key = f"PrecedentIndex_{tag}"
        matches_path = store.attach_artifact(
            args.run_id,
            matches_key,
            {
                "run_id": args.run_id,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "label": label,
                "action_ids": list(action_ids),
                "candidate_source": candidate_source,
                "candidate_count": len(selected),
                "profile": profile,
                "results": matches,
            },
        )
        index_path = store.attach_artifact(args.run_id, index_key, index)
        artifacts[label] = {
            "matches_artifact": str(matches_path),
            "index_artifact": str(index_path),
        }
        summary = _slice_summary(label, action_ids, matches, elapsed)
        summary["candidate_count"] = len(selected)
        summary["candidate_source"] = candidate_source
        summaries.append(summary)
        print(
            json.dumps(
                {
                    "ok": True,
                    "event": "benchmark_slice_completed",
                    "label": label,
                    "summary": summary,
                    "artifacts": artifacts[label],
                }
            ),
            flush=True,
        )

    payload = {
        "ok": True,
        "run_id": args.run_id,
        "runs_root": str(runs_root),
        "precedent_top_k": int(args.precedent_top_k),
        "summaries": summaries,
        "artifacts": artifacts,
        "elapsed_seconds": round(time.time() - t0, 3),
    }
    if args.out:
        out_path = Path(args.out)
        out_path.write_text(json.dumps(payload, indent=2))
        payload["out"] = str(out_path)

    print()
    _print_table(summaries)
    print()
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
