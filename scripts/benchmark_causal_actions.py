#!/usr/bin/env python
"""Benchmark targeted causal routing on an existing recommendation run."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))


def _default_model_path() -> str:
    repo_root = Path(__file__).resolve().parent.parent
    return str(repo_root / "data" / "models" / "causal_impact_model_v5_5_hybrid.json")


DEFAULT_PRESET: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("platform_acquisition", ("mna.platform_acquisition",)),
    ("tuck_in_acquisition", ("mna.tuck_in_acquisition",)),
    ("special_dividend", ("capital_return.special_dividend",)),
    ("dividend_initiate", ("capital_return.dividend_initiate",)),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark targeted causal routing on an existing run")
    p.add_argument("--run-id", required=True)
    p.add_argument("--runs-root", default="data/recommendation_runs")
    p.add_argument("--snapshot-root", default=None)
    p.add_argument("--snapshot-path", default=None)
    p.add_argument("--model-path", default=None)
    p.add_argument("--feasibility-path", default=None)
    p.add_argument("--candidate-set-path", default=None)
    p.add_argument("--artifact-prefix", default="causal_bench")
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


def _metadata_config(run: Any) -> Dict[str, Any]:
    metadata = dict(getattr(run, "metadata", {}) or {})
    return dict(metadata.get("config", {}) or {})


def _resolve_path(explicit: Optional[str], cfg: Dict[str, Any], key: str) -> Optional[str]:
    if explicit:
        return str(explicit)
    create_cfg = dict(cfg.get("create", {}) or {})
    value = create_cfg.get(key)
    if value:
        return str(value)
    if key == "model_path":
        return _default_model_path()
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


def _print_table(rows: Sequence[Dict[str, Any]]) -> None:
    headers = (
        ("label", 22),
        ("selected_causal_candidates", 6),
        ("coverage_score_mean", 8),
        ("model_quality_mean", 8),
        ("support_score_mean", 8),
        ("blend_weight_mean", 8),
        ("oos_rate", 8),
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


def _as_of_datetime(raw: str) -> datetime:
    s = str(raw).strip()
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _load_snapshot(
    run: Any,
    snapshot_root: Optional[str],
    snapshot_path: Optional[str],
    entity_identifier_path: str,
) -> Dict[str, Any]:
    from src.recommendation_run import _resolve_snapshot, _snapshot_company_aliases

    aliases = _snapshot_company_aliases(
        str(run.company_id),
        Path(entity_identifier_path),
    )
    return _resolve_snapshot(
        company_id=str(run.company_id),
        as_of_time=_as_of_datetime(str(run.as_of_time)),
        snapshot_root=Path(snapshot_root) if snapshot_root else None,
        snapshot_path=Path(snapshot_path) if snapshot_path else None,
        snapshot_builder=None,
        snapshot_loader=None,
        aliases=aliases,
    )


def _slice_summary(
    label: str,
    action_ids: Sequence[str],
    diagnostics_rows: Sequence[Dict[str, Any]],
    elapsed_seconds: float,
) -> Dict[str, Any]:
    coverages: List[float] = []
    qualities: List[float] = []
    supports: List[float] = []
    blends: List[float] = []
    oos_flags: List[float] = []
    min_oos_values: List[float] = []
    selected_key_counts: Dict[str, int] = {}
    selected_objective_counts: List[float] = []

    for row in diagnostics_rows:
        diag = dict(row.get("causal_diagnostics", {}) or {})
        if not diag:
            continue
        coverages.append(float(diag.get("coverage_score") or 0.0))
        qualities.append(float(diag.get("model_quality") or 0.0))
        supports.append(float(diag.get("support_score") or 0.0))
        blends.append(float(diag.get("blend_weight") or 0.0))
        oos_flags.append(1.0 if bool(diag.get("out_of_sample_flag")) else 0.0)
        if diag.get("min_oos_r2") is not None:
            min_oos_values.append(float(diag.get("min_oos_r2") or 0.0))
        objectives = dict(diag.get("selected_models_by_objective") or {})
        selected_objective_counts.append(float(len(objectives)))
        for payload in objectives.values():
            key = str((payload or {}).get("selected_key", "") or "")
            if key:
                selected_key_counts[key] = selected_key_counts.get(key, 0) + 1

    return {
        "label": label,
        "action_ids": list(action_ids),
        "selected_causal_candidates": int(len(diagnostics_rows)),
        "coverage_score_mean": round(_mean(coverages), 6),
        "model_quality_mean": round(_mean(qualities), 6),
        "support_score_mean": round(_mean(supports), 6),
        "blend_weight_mean": round(_mean(blends), 6),
        "min_oos_r2_mean": round(_mean(min_oos_values), 6) if min_oos_values else None,
        "selected_objectives_mean": round(_mean(selected_objective_counts), 6),
        "oos_rate": round(_mean(oos_flags), 6),
        "selected_model_keys": dict(sorted(selected_key_counts.items())),
        "elapsed_seconds": round(float(elapsed_seconds), 6),
    }


def main() -> None:
    t0 = time.time()
    args = parse_args()
    slices = _parse_slices(args.slice)

    print(json.dumps({"ok": True, "event": "startup", "stage": "import_causal_benchmark"}), flush=True)
    from src.causal_impact_model import CausalImpactModel
    from src.recommendation_run import RecommendationRunStore
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

    cfg = _metadata_config(run)
    snapshot_root = _resolve_path(args.snapshot_root, cfg, "snapshot_root")
    snapshot_path = _resolve_path(args.snapshot_path, cfg, "snapshot_path")
    model_path = _resolve_path(args.model_path, cfg, "model_path")
    entity_identifier_path = _resolve_path(None, cfg, "entity_identifier_path") or "data/inputs_layer/entity_identifier.parquet"
    if not snapshot_root and not snapshot_path:
        snapshot_root = _infer_snapshot_root(run)
    if not model_path:
        model_path = _default_model_path()

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
    snapshot = _load_snapshot(run, snapshot_root, snapshot_path, entity_identifier_path)
    features = dict(snapshot.get("features", {}) or {})
    regime = dict(snapshot.get("regime", {}) or {})
    model = CausalImpactModel.from_path(Path(model_path))

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
        diagnostics_rows: List[Dict[str, Any]] = []
        for cand in selected:
            action_id = str(cand.get("action_id", "") or "")
            if not action_id:
                continue
            action_type = action_id.split(".", 1)[0] if "." in action_id else str(cand.get("action_type", "") or "")
            action_subtype = action_id.split(".", 1)[1] if "." in action_id else str(cand.get("action_subtype", "") or "")
            params = dict(cand.get("params") or cand.get("parameters") or {})
            diag = model.diagnose(
                action_id=action_id,
                action_type=action_type,
                action_subtype=action_subtype,
                params=params,
                features=features,
                regime=regime,
            )
            if diag is None:
                continue
            diagnostics_rows.append(
                {
                    "candidate": cand,
                    "causal_diagnostics": {
                        "action_alias": diag.action_alias,
                        "subtype_alias": diag.subtype_alias,
                        "blend_weight": diag.blend_weight,
                        "coverage_score": diag.coverage_score,
                        "n_train": diag.n_train,
                        "model_version": diag.model_version,
                        "model_quality": diag.model_quality,
                        "support_score": diag.support_score,
                        "out_of_sample_flag": diag.out_of_sample_flag,
                        "min_oos_r2": diag.min_oos_r2,
                        "min_treated_rows": diag.min_treated_rows,
                        "min_control_rows": diag.min_control_rows,
                        "selected_model_keys": list(diag.selected_model_keys),
                        "selected_models_by_objective": dict(diag.selected_models_by_objective),
                        "gate_reason": diag.gate_reason,
                    },
                }
            )
        elapsed = time.time() - started
        summary = _slice_summary(label, action_ids, diagnostics_rows, elapsed)
        summary["candidate_count"] = len(selected)
        summary["candidate_source"] = candidate_source
        summaries.append(summary)

        tag = f"{args.artifact_prefix}_{label}".strip().replace(" ", "_")
        bench_key = f"CausalBenchmark_{tag}"
        bench_path = store.attach_artifact(
            args.run_id,
            bench_key,
            {
                "run_id": args.run_id,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "label": label,
                "action_ids": list(action_ids),
                "candidate_source": candidate_source,
                "candidate_count": len(selected),
                "results": diagnostics_rows,
                "summary": summary,
            },
        )
        artifacts[label] = {"benchmark_artifact": str(bench_path)}
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
        "model_path": str(model_path),
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
