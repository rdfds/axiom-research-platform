#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


_REPO_ROOT = Path(__file__).resolve().parents[1]


def _default_path(*parts: str) -> str:
    return str(_REPO_ROOT.joinpath(*parts))


def _default_precedent_outcomes_path() -> str:
    return _default_path("data", "curated", "action_outcomes_with_credit_ratings.normalized_full.parquet")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="One-command production runner for RecommendationRun.")
    p.add_argument("--runs-root", default="/tmp/recommendation_runs_prod")
    p.add_argument(
        "--snapshot-root",
        default=_default_path("data", "company_state_snapshots", "final_run_2026-02-28"),
    )
    p.add_argument(
        "--entity-graph-path",
        default=_default_path("data", "inputs_layer", "entity_graph.parquet"),
    )
    p.add_argument(
        "--entity-identifier-path",
        default=_default_path("data", "inputs_layer", "entity_identifier.parquet"),
    )
    p.add_argument(
        "--outcomes-path",
        default=_default_precedent_outcomes_path(),
    )
    p.add_argument("--config-path", default=None)
    p.add_argument("--as-of", default="2026-02-28")
    p.add_argument("--companies", nargs="+", required=True)
    p.add_argument("--action-ids", nargs="+", default=None)
    p.add_argument("--max-candidates", type=int, default=300)
    p.add_argument("--min-candidates-target", type=int, default=300)
    p.add_argument("--precedent-top-k", type=int, default=25)
    p.add_argument("--top-plans", type=int, default=1)
    p.add_argument("--strict-evidence", action="store_true")
    p.add_argument("--heartbeat-seconds", type=float, default=20.0)
    p.add_argument("--run-ids-out", default="/tmp/recommendation_prod_run_ids.txt")
    p.add_argument("--summary-out", default="")

    p.add_argument(
        "--causal-model-path",
        default=_default_path("data", "models", "causal_impact_model_v5_5_hybrid.json"),
    )
    p.add_argument(
        "--causal-routing-config-path",
        default=_default_path("configs", "causal_capital_routing_prod_dividend_v2.json"),
    )
    p.add_argument(
        "--causal-action-blocklist-path",
        default=_default_path("config", "causal_action_blocklist_prod_v2.txt"),
    )
    p.add_argument("--causal-impact-mode", default="blend")
    p.add_argument("--causal-min-objective-oos-r2", type=float, default=0.08)
    p.add_argument("--causal-strict-quality-floor", type=float, default=0.08)
    p.add_argument("--causal-strict-support-floor", type=float, default=0.35)
    p.add_argument("--causal-strict-min-train-rows", type=int, default=1000)
    p.add_argument("--causal-strict-min-oos-r2", type=float, default=0.00)
    p.add_argument("--causal-strict-min-treated-rows", type=int, default=1500)
    p.add_argument("--causal-strict-min-control-rows", type=int, default=20000)
    p.add_argument("--precedent-workers", type=int, default=0)
    p.add_argument(
        "--warm-precedent-runtime",
        dest="warm_precedent_runtime",
        action="store_true",
        default=True,
        help="Preload precedent runtime before the first company instead of lazily on demand.",
    )
    p.add_argument(
        "--no-warm-precedent-runtime",
        dest="warm_precedent_runtime",
        action="store_false",
        help="Disable precedent runtime warmup before the first company.",
    )
    return p.parse_args()


def _safe_run_ids_write(path: Path, pairs: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(pairs) + ("\n" if pairs else ""))


def _event_ts(event: Dict[str, Any]) -> float | None:
    raw = str(event.get("timestamp", "") or "")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def _run_stage_seconds(runs_root: Path, run_id: str) -> Dict[str, float]:
    p = runs_root / "runs" / f"run_id={run_id}.json"
    if not p.exists():
        return {}
    try:
        run = json.loads(p.read_text())
    except Exception:
        return {}
    events = run.get("audit_log") or []
    if not isinstance(events, list):
        return {}

    by_type: Dict[str, float] = {}
    for e in events:
        if not isinstance(e, dict):
            continue
        ts = _event_ts(e)
        if ts is not None:
            by_type[str(e.get("event_type", ""))] = ts

    out: Dict[str, float] = {}

    def _dur(start: str, end: str, key: str) -> None:
        if start in by_type and end in by_type:
            out[key] = round(max(0.0, by_type[end] - by_type[start]), 3)

    _dur("snapshot_frozen", "candidate_generation_started", "snapshot_load")
    _dur("candidate_generation_started", "candidate_generation_completed", "candidate_generation")
    _dur("feasibility_eval_started", "feasibility_eval_completed", "feasibility")
    _dur("precedent_retrieval_started", "precedent_retrieval_completed", "precedent")
    _dur("planning_started", "planning_completed", "planning")
    _dur("run_created", "run_completed", "total_run")
    return out


def _build_keyed_snapshot_loader(snapshot_root: Path):
    def _loader(company_id: str, as_of_time: datetime) -> Dict[str, Any]:
        p = (
            snapshot_root
            / "keyed"
            / f"as_of_date={as_of_time.strftime('%Y-%m-%d')}"
            / f"company_id={company_id}.json"
        )
        if not p.exists():
            raise FileNotFoundError(f"Keyed snapshot not found: {p}")
        return json.loads(p.read_text())

    return _loader


def _keyed_snapshot_path(snapshot_root: Path, as_of: str, company_id: str) -> Path:
    return snapshot_root / "keyed" / f"as_of_date={as_of}" / f"company_id={company_id}.json"


def _start_heartbeat(
    company_id: str,
    start_ts: float,
    every_seconds: float,
) -> tuple[threading.Event, Optional[threading.Thread]]:
    stop = threading.Event()
    if every_seconds <= 0:
        return stop, None

    def _run() -> None:
        while not stop.wait(every_seconds):
            print(
                json.dumps(
                    {
                        "ok": True,
                        "event": "company_heartbeat",
                        "company_id": company_id,
                        "elapsed_seconds": round(time.time() - start_ts, 3),
                    }
                ),
                flush=True,
            )

    th = threading.Thread(target=_run, daemon=True)
    th.start()
    return stop, th


def _apply_runtime_env(args: argparse.Namespace) -> Dict[str, str]:
    env_map = {
        "RECOMMENDATION_RUN_TMP_DIR": str(Path(args.runs_root) / "tmp"),
        "CAUSAL_IMPACT_MODEL_PATH": str(args.causal_model_path),
        "CAUSAL_ROUTING_CONFIG_PATH": str(args.causal_routing_config_path),
        "CAUSAL_IMPACT_MODE": str(args.causal_impact_mode),
        "CAUSAL_MIN_OBJECTIVE_OOS_R2": str(args.causal_min_objective_oos_r2),
        "CAUSAL_STRICT_QUALITY_FLOOR": str(args.causal_strict_quality_floor),
        "CAUSAL_STRICT_SUPPORT_FLOOR": str(args.causal_strict_support_floor),
        "CAUSAL_STRICT_MIN_TRAIN_ROWS": str(args.causal_strict_min_train_rows),
        "CAUSAL_STRICT_MIN_OOS_R2": str(args.causal_strict_min_oos_r2),
        "CAUSAL_STRICT_MIN_TREATED_ROWS": str(args.causal_strict_min_treated_rows),
        "CAUSAL_STRICT_MIN_CONTROL_ROWS": str(args.causal_strict_min_control_rows),
    }
    if str(args.causal_action_blocklist_path or "").strip():
        env_map["CAUSAL_ACTION_BLOCKLIST_PATH"] = str(args.causal_action_blocklist_path)
    if int(args.precedent_workers or 0) > 0:
        env_map["RECO_PRECEDENT_WORKERS"] = str(int(args.precedent_workers))
    for key, value in env_map.items():
        os.environ[str(key)] = str(value)
    Path(env_map["RECOMMENDATION_RUN_TMP_DIR"]).mkdir(parents=True, exist_ok=True)
    return env_map


def run_production_batch(
    args: argparse.Namespace,
    create_and_execute_fn: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    applied_env = _apply_runtime_env(args)
    import_t0 = time.time()
    print(json.dumps({"ok": True, "event": "startup", "stage": "import_orchestrator"}), flush=True)
    registry = None
    precedent_runner = None
    if create_and_execute_fn is None:
        print(
            json.dumps({"ok": True, "event": "startup", "stage": "import_recommendation_run_orchestrator"}),
            flush=True,
        )
        from src.recommendation_run_orchestrator import create_and_execute_recommendation_run

        print(
            json.dumps({"ok": True, "event": "startup", "stage": "import_action_ontology"}),
            flush=True,
        )
        from src.action_ontology import build_default_action_schema_registry

        print(
            json.dumps({"ok": True, "event": "startup", "stage": "import_causal_model_risk"}),
            flush=True,
        )
        from src.causal_model_risk import build_causal_model_risk_report as _warm_causal_model_risk  # noqa: F401

        print(
            json.dumps({"ok": True, "event": "startup", "stage": "import_precedent_runtime"}),
            flush=True,
        )
        from src.pipeline.run import run_precedent, warm_precedent_runtime

        print(
            json.dumps({"ok": True, "event": "startup", "stage": "build_action_schema_registry:start"}),
            flush=True,
        )
        registry = build_default_action_schema_registry(version="v1.0")
        print(
            json.dumps({"ok": True, "event": "startup", "stage": "build_action_schema_registry:done"}),
            flush=True,
        )
        precedent_runner = run_precedent
        if bool(args.warm_precedent_runtime):
            print(
                json.dumps(
                    {
                        "ok": True,
                        "event": "startup",
                        "stage": "warm_precedent_runtime:start",
                        "outcomes_path": str(args.outcomes_path),
                    }
                ),
                flush=True,
            )
            try:
                warm_precedent_runtime(args.outcomes_path)
                print(
                    json.dumps({"ok": True, "event": "startup", "stage": "warm_precedent_runtime:done"}),
                    flush=True,
                )
            except Exception as exc:
                print(
                    json.dumps(
                        {
                            "ok": False,
                            "event": "startup",
                            "stage": "warm_precedent_runtime:failed",
                            "error": str(exc),
                        }
                    ),
                    flush=True,
                )

        create_and_execute_fn = create_and_execute_recommendation_run
    print(
        json.dumps(
            {
                "ok": True,
                "event": "startup",
                "stage": "import_done",
                "elapsed_seconds": round(time.time() - import_t0, 3),
            }
        ),
        flush=True,
    )

    runs_root = Path(args.runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)
    snapshot_root = Path(args.snapshot_root)
    keyed_loader = _build_keyed_snapshot_loader(snapshot_root)

    run_pairs: List[str] = []
    summaries: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for company_id in args.companies:
        t0 = time.time()
        hb_stop, hb_thread = _start_heartbeat(str(company_id), t0, float(args.heartbeat_seconds))
        keyed_snapshot = _keyed_snapshot_path(snapshot_root, str(args.as_of), str(company_id))
        snapshot_path_arg = str(keyed_snapshot) if keyed_snapshot.exists() else None
        print(
            json.dumps(
                {
                    "ok": True,
                    "event": "company_started",
                    "company_id": str(company_id),
                    "snapshot_mode": "keyed_file" if snapshot_path_arg else "snapshot_root_lookup",
                }
            ),
            flush=True,
        )
        try:
            summary = create_and_execute_fn(
                company_id=str(company_id),
                as_of_time=str(args.as_of),
                runs_root=str(runs_root),
                snapshot_root=str(snapshot_root),
                snapshot_path=snapshot_path_arg,
                snapshot_loader=keyed_loader,
                entity_graph_path=str(args.entity_graph_path),
                entity_identifier_path=str(args.entity_identifier_path),
                action_ids=[str(x) for x in (args.action_ids or [])] or None,
                max_candidates=int(args.max_candidates),
                min_candidates_target=int(args.min_candidates_target),
                strict_evidence=bool(args.strict_evidence),
                precedent_top_k=int(args.precedent_top_k),
                outcomes_path=str(args.outcomes_path),
                config_path=str(args.config_path) if args.config_path else None,
                top_plans=int(args.top_plans),
                metadata={"runner": {"script": str(Path(__file__).resolve()), "mode": "production"}},
                registry=registry,
                precedent_runner=precedent_runner,
            )
            run_id = str(summary.get("run_id", "") or "")
            if run_id:
                run_pairs.append(f"{company_id} {run_id}")
            stage_seconds = _run_stage_seconds(runs_root, run_id) if run_id else {}
            company_summary = {
                "ok": True,
                "event": "company_completed",
                "company_id": str(company_id),
                "run_id": run_id,
                "status": summary.get("status"),
                "counts": summary.get("counts", {}),
                "stage_seconds": stage_seconds,
                "elapsed_seconds": round(time.time() - t0, 3),
            }
            summaries.append(company_summary)
            print(json.dumps(company_summary), flush=True)
        except Exception as exc:
            failure = {
                "ok": False,
                "company_id": str(company_id),
                "error": str(exc),
                "traceback": traceback.format_exc(limit=3),
                "elapsed_seconds": round(time.time() - t0, 3),
            }
            failures.append(failure)
            print(json.dumps(failure), flush=True)
        finally:
            hb_stop.set()
            if hb_thread is not None:
                hb_thread.join(timeout=1.0)

    run_ids_out = Path(str(args.run_ids_out))
    _safe_run_ids_write(run_ids_out, run_pairs)
    final = {
        "ok": len(failures) == 0,
        "runs_root": str(runs_root),
        "run_ids_out": str(run_ids_out),
        "requested_companies": len(args.companies),
        "completed_runs": len(run_pairs),
        "failed_runs": len(failures),
        "runtime_env": applied_env,
        "failures": failures,
        "summaries": summaries,
    }
    print(json.dumps(final), flush=True)
    if str(args.summary_out or "").strip():
        out_path = Path(str(args.summary_out))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(final, indent=2))
    return final


def main() -> None:
    args = _parse_args()
    run_production_batch(args)


if __name__ == "__main__":
    main()
