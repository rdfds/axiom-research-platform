#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import traceback
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run recommendation canary directly via orchestrator (no HTTP API)."
    )
    p.add_argument("--runs-root", required=True)
    p.add_argument("--snapshot-root", required=True)
    p.add_argument("--entity-graph-path", required=True)
    p.add_argument("--entity-identifier-path", required=True)
    p.add_argument("--outcomes-path", required=True)
    p.add_argument("--config-path", default=None)
    p.add_argument("--as-of", default="2026-02-28")
    p.add_argument("--companies", nargs="+", required=True)
    p.add_argument(
        "--action-ids",
        nargs="+",
        default=None,
        help="Optional explicit action_id list to limit runtime for smoke tests.",
    )
    p.add_argument("--max-candidates", type=int, default=300)
    p.add_argument(
        "--min-candidates-target",
        type=int,
        default=300,
        help="Target lower bound on candidate set size (API-equivalent behavior).",
    )
    p.add_argument("--precedent-top-k", type=int, default=25)
    p.add_argument("--top-plans", type=int, default=1)
    p.add_argument(
        "--mock-precedent",
        action="store_true",
        help="Use a local no-op precedent runner for fast smoke testing.",
    )
    p.add_argument(
        "--disable-keyed-loader",
        action="store_true",
        help="Disable direct keyed snapshot loader and fall back to default resolver.",
    )
    p.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=20.0,
        help="Emit a heartbeat log every N seconds while a company run is in progress (0 to disable).",
    )
    p.add_argument("--run-ids-out", required=True)
    p.add_argument("--summary-out", default=None)
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
        t = str(e.get("event_type", "") or "")
        ts = _event_ts(e)
        if t and ts is not None:
            by_type[t] = ts

    out: Dict[str, float] = {}
    def _dur(start: str, end: str, key: str) -> None:
        if start in by_type and end in by_type:
            out[key] = round(max(0.0, by_type[end] - by_type[start]), 3)

    _dur("snapshot_frozen", "candidate_generation_started", "snapshot_load")
    _dur("candidate_generation_started", "candidate_generation_completed", "candidate_generation")
    _dur("feasibility_eval_started", "feasibility_eval_completed", "feasibility")
    _dur("precedent_retrieval_started", "precedent_retrieval_completed", "precedent")
    _dur("planning_started", "planning_completed", "planning")
    return out


def _build_keyed_snapshot_loader(snapshot_root: Path):
    def _loader(company_id: str, as_of_time: datetime) -> Dict[str, Any]:
        as_of_date = as_of_time.strftime("%Y-%m-%d")
        p = (
            snapshot_root
            / "keyed"
            / f"as_of_date={as_of_date}"
            / f"company_id={company_id}.json"
        )
        if not p.exists():
            raise FileNotFoundError(f"Keyed snapshot not found: {p}")
        return json.loads(p.read_text())

    return _loader


def _mock_precedent_runner(**_: Any) -> Dict[str, Any]:
    # Minimal precedent pack shape that downstream planning can consume.
    return {
        "legacy_distributions": [
            {
                "metric": "outcome_pe_12m",
                "p25": 0.0,
                "p50": 0.0,
                "p75": 0.0,
            }
        ],
        "distributions": [],
        "citations": [],
        "match_confidence": 0.0,
        "out_of_sample_rate": 1.0,
    }


def _keyed_snapshot_path(snapshot_root: Path, as_of: str, company_id: str) -> Path:
    return (
        snapshot_root
        / "keyed"
        / f"as_of_date={as_of}"
        / f"company_id={company_id}.json"
    )


def _start_heartbeat(
    company_id: str, start_ts: float, every_seconds: float
) -> tuple[threading.Event, threading.Thread | None]:
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


def main() -> None:
    import_t0 = time.time()
    print(
        json.dumps(
            {
                "ok": True,
                "event": "startup",
                "stage": "import_orchestrator",
            }
        ),
        flush=True,
    )
    from src.recommendation_run_orchestrator import create_and_execute_recommendation_run

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

    args = _parse_args()
    runs_root = Path(args.runs_root)
    snapshot_root = Path(args.snapshot_root)
    keyed_loader = None if args.disable_keyed_loader else _build_keyed_snapshot_loader(snapshot_root)
    precedent_runner = _mock_precedent_runner if args.mock_precedent else None
    runs_root.mkdir(parents=True, exist_ok=True)

    run_pairs: List[str] = []
    summaries: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for company_id in args.companies:
        t0 = time.time()
        keyed_snapshot = _keyed_snapshot_path(
            snapshot_root=snapshot_root,
            as_of=str(args.as_of),
            company_id=str(company_id),
        )
        snapshot_path_arg = str(keyed_snapshot) if keyed_snapshot.exists() else None
        hb_stop, hb_thread = _start_heartbeat(
            company_id=str(company_id),
            start_ts=t0,
            every_seconds=float(args.heartbeat_seconds),
        )
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
            summary = create_and_execute_recommendation_run(
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
                precedent_top_k=int(args.precedent_top_k),
                outcomes_path=str(args.outcomes_path),
                config_path=str(args.config_path) if args.config_path else None,
                top_plans=int(args.top_plans),
                precedent_runner=precedent_runner,
            )
            rid = str(summary.get("run_id", ""))
            if rid:
                run_pairs.append(f"{company_id} {rid}")
            summaries.append(summary)
            stage_seconds = _run_stage_seconds(runs_root=runs_root, run_id=rid) if rid else {}
            print(
                json.dumps(
                    {
                        "ok": True,
                        "event": "company_completed",
                        "company_id": str(company_id),
                        "run_id": rid,
                        "status": summary.get("status"),
                        "counts": summary.get("counts", {}),
                        "stage_seconds": stage_seconds,
                        "elapsed_seconds": round(time.time() - t0, 3),
                    }
                ),
                flush=True,
            )
        except Exception as exc:  # pragma: no cover - operational guard
            failure = {
                "company_id": str(company_id),
                "error": str(exc),
                "traceback": traceback.format_exc(limit=3),
                "elapsed_seconds": round(time.time() - t0, 3),
            }
            failures.append(failure)
            print(json.dumps({"ok": False, **failure}), flush=True)
        finally:
            hb_stop.set()
            if hb_thread is not None:
                hb_thread.join(timeout=1.0)

    run_ids_out = Path(args.run_ids_out)
    _safe_run_ids_write(run_ids_out, run_pairs)

    final = {
        "ok": len(failures) == 0,
        "runs_root": str(runs_root),
        "run_ids_out": str(run_ids_out),
        "requested_companies": len(args.companies),
        "completed_runs": len(run_pairs),
        "failed_runs": len(failures),
        "failures": failures,
    }
    print(json.dumps(final), flush=True)

    if args.summary_out:
        out_path = Path(args.summary_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "final": final,
                    "summaries": summaries,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
