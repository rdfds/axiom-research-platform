#!/usr/bin/env python
"""HTTP API for RecommendationRun create/execute orchestration.

Endpoints:
  GET  /health
  GET  /precedent_query
  POST /create_run
  POST /execute_run
  POST /create_and_execute_run
"""

from __future__ import annotations

import argparse
from datetime import datetime
from concurrent.futures import Future, ThreadPoolExecutor
import hashlib
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).parent.parent))


def _default_precedent_outcomes_path() -> str:
    repo_root = Path(__file__).resolve().parent.parent
    return str(repo_root / "data" / "curated" / "action_outcomes_with_credit_ratings.normalized_full.parquet")


def _recommendation_run_bindings():
    from src.recommendation_run import (
        RecommendationRunStore,
        _resolve_snapshot,
        _snapshot_company_aliases,
        create_recommendation_run,
    )

    return RecommendationRunStore, _resolve_snapshot, _snapshot_company_aliases, create_recommendation_run


def _build_default_registry():
    from src.action_ontology import build_default_action_schema_registry

    return build_default_action_schema_registry(version="v1.0")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Serve RecommendationRun orchestration over HTTP.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--runs-root", default="data/recommendation_runs")
    p.add_argument("--snapshot-root", default=None)
    p.add_argument("--snapshot-path", default=None)
    p.add_argument("--entity-graph-path", default="data/inputs_layer/entity_graph.parquet")
    p.add_argument("--entity-identifier-path", default="data/inputs_layer/entity_identifier.parquet")
    p.add_argument("--outcomes-path", default=_default_precedent_outcomes_path())
    p.add_argument("--config", default=None)
    p.add_argument("--max-candidates", type=int, default=12)
    p.add_argument("--min-candidates-target", type=int, default=300)
    p.add_argument("--precedent-top-k", type=int, default=25)
    p.add_argument("--top-plans", type=int, default=3)
    p.add_argument("--startup-warmup", dest="startup_warmup", action="store_true", default=True)
    p.add_argument("--no-startup-warmup", dest="startup_warmup", action="store_false")
    p.add_argument("--warmup-company-id", default="0000320193")
    p.add_argument("--warmup-as-of", default="2026-02-28")
    return p.parse_args()


def _json_bytes(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload, default=str).encode("utf-8")


def _read_json_body(handler: BaseHTTPRequestHandler) -> Tuple[Dict[str, Any], Optional[str]]:
    raw_len = handler.headers.get("Content-Length", "0")
    try:
        length = int(raw_len)
    except Exception:
        return {}, "invalid_content_length"
    if length <= 0:
        return {}, "empty_body"
    raw = handler.rfile.read(length)
    try:
        obj = json.loads(raw.decode("utf-8"))
    except Exception:
        return {}, "invalid_json"
    if not isinstance(obj, dict):
        return {}, "body_must_be_object"
    return obj, None


def _pick(body: Dict[str, Any], key: str, default: Any) -> Any:
    return body[key] if key in body else default


def _parse_bool_flag(raw: str, default: bool = False) -> bool:
    v = str(raw or "").strip().lower()
    if not v:
        return bool(default)
    if v in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "f", "no", "n", "off"}:
        return False
    return bool(default)


def _coerce_action_ids(raw: Any) -> Optional[List[str]]:
    if raw is None:
        return None
    if isinstance(raw, str):
        val = raw.strip()
        return [val] if val else None
    if isinstance(raw, list):
        out = [str(x).strip() for x in raw if str(x).strip()]
        return out or None
    raise ValueError("action_ids must be string or list")


def _canonical_request_signature(
    company_id: str,
    as_of: str,
    action_ids: Optional[List[str]],
    action_type: Optional[str],
    objectives: Optional[Dict[str, Any]],
    constraints: Optional[Dict[str, Any]],
    scenario: Optional[Dict[str, Any]],
    max_candidates: int,
    min_candidates_target: int,
    precedent_top_k: int,
    strict_evidence: bool,
    top_plans: int,
) -> str:
    payload = {
        "company_id": str(company_id),
        "as_of": str(as_of),
        "action_ids": sorted(action_ids or []),
        "action_type": action_type,
        "objectives": objectives or {},
        "constraints": constraints or {},
        "scenario": scenario or {},
        "max_candidates": int(max_candidates),
        "min_candidates_target": int(min_candidates_target),
        "precedent_top_k": int(precedent_top_k),
        "strict_evidence": bool(strict_evidence),
        "top_plans": int(top_plans),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _read_json_if_exists(path: str | Path) -> Optional[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        obj = json.loads(p.read_text())
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _audit_event_to_dict(event: Any) -> Dict[str, Any]:
    if isinstance(event, dict):
        return dict(event)
    return {
        "event_id": str(getattr(event, "event_id", "")),
        "timestamp": str(getattr(event, "timestamp", "")),
        "event_type": str(getattr(event, "event_type", "")),
        "details": dict(getattr(event, "details", {}) or {}),
    }


def _find_cached_completed_run(
    store: RecommendationRunStore,
    company_id: str,
    as_of: str,
    request_signature: str,
) -> Optional[Dict[str, Any]]:
    runs = store.list_runs(company_id=company_id, as_of_time=as_of, status="completed")
    for run in reversed(runs):
        md = run.metadata if isinstance(run.metadata, dict) else {}
        if str(md.get("request_signature", "")) != str(request_signature):
            continue
        artifacts = md.get("artifacts", {}) if isinstance(md.get("artifacts", {}), dict) else {}
        rec_path = artifacts.get("RecommendationPackage")
        if not rec_path:
            continue
        rec = _read_json_if_exists(rec_path)
        if rec is None:
            continue
        return {
            "run_id": run.run_id,
            "status": run.status,
            "artifacts": artifacts,
            "recommendation_package": rec,
            "created_at": run.created_at,
        }
    return None


class _WarmRuntime:
    def __init__(self, snapshot_cache_size: int = 512, max_workers: int = 2) -> None:
        self._registry = None
        self.snapshot_cache_size = int(max(8, snapshot_cache_size))
        self._snapshot_cache: Dict[Tuple[str, str, str, str, str], Dict[str, Any]] = {}
        self._aliases_cache: Dict[Tuple[str, str, str], List[str]] = {}
        self._executor = ThreadPoolExecutor(max_workers=max(1, int(max_workers)))
        self._jobs: Dict[str, Future] = {}
        self._jobs_meta: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()
        self._execute_run_fn: Optional[Callable[..., Dict[str, Any]]] = None
        self._create_and_execute_run_fn: Optional[Callable[..., Dict[str, Any]]] = None
        self._warmup_meta: Dict[str, Any] = {"state": "idle"}
        self._warmup_thread: Optional[Thread] = None

    @property
    def registry(self):
        if self._registry is None:
            self._registry = _build_default_registry()
        return self._registry

    @staticmethod
    def _entity_version(path: str | Path) -> str:
        p = Path(path)
        if not p.exists():
            return "missing"
        st = p.stat()
        return f"{st.st_mtime_ns}:{st.st_size}"

    def _cache_put_snapshot(self, key: Tuple[str, str, str, str, str], value: Dict[str, Any]) -> None:
        self._snapshot_cache[key] = value
        if len(self._snapshot_cache) <= self.snapshot_cache_size:
            return
        oldest_key = next(iter(self._snapshot_cache.keys()))
        self._snapshot_cache.pop(oldest_key, None)

    def _aliases_for(self, company_id: str, entity_identifier_path: str | Path) -> List[str]:
        _, _, snapshot_company_aliases, _ = _recommendation_run_bindings()
        path = str(entity_identifier_path)
        version = self._entity_version(path)
        key = (path, version, str(company_id))
        if key in self._aliases_cache:
            return list(self._aliases_cache[key])
        aliases = snapshot_company_aliases(str(company_id), Path(path))
        self._aliases_cache[key] = list(aliases)
        return list(aliases)

    def build_snapshot_loader(
        self,
        snapshot_root: Optional[str | Path],
        snapshot_path: Optional[str | Path],
        entity_identifier_path: str | Path,
    ) -> Callable[[str, datetime], Dict[str, Any]]:
        snap_root = Path(snapshot_root) if snapshot_root else None
        snap_path = Path(snapshot_path) if snapshot_path else None
        snap_root_s = str(snap_root) if snap_root else ""
        snap_path_s = str(snap_path) if snap_path else ""
        ent_path_s = str(entity_identifier_path)

        def _loader(company_id: str, as_of_time: datetime) -> Dict[str, Any]:
            _, resolve_snapshot, _, _ = _recommendation_run_bindings()
            key = (
                str(company_id),
                as_of_time.strftime("%Y-%m-%d"),
                snap_root_s,
                snap_path_s,
                self._entity_version(ent_path_s),
            )
            cached = self._snapshot_cache.get(key)
            if cached is not None:
                return dict(cached)

            aliases = self._aliases_for(company_id, ent_path_s)
            snap = resolve_snapshot(
                company_id=str(company_id),
                as_of_time=as_of_time,
                snapshot_root=snap_root,
                snapshot_path=snap_path,
                snapshot_builder=None,
                snapshot_loader=None,
                aliases=aliases,
            )
            self._cache_put_snapshot(key, dict(snap))
            return dict(snap)

        return _loader

    def _ensure_orchestrator(self) -> None:
        if self._execute_run_fn is not None and self._create_and_execute_run_fn is not None:
            return
        from src.recommendation_run_orchestrator import (  # lazy import for faster API boot
            create_and_execute_recommendation_run,
            execute_recommendation_run,
        )

        self._execute_run_fn = execute_recommendation_run
        self._create_and_execute_run_fn = create_and_execute_recommendation_run

    def execute_run_fn(self) -> Callable[..., Dict[str, Any]]:
        self._ensure_orchestrator()
        assert self._execute_run_fn is not None
        return self._execute_run_fn

    def create_and_execute_run_fn(self) -> Callable[..., Dict[str, Any]]:
        self._ensure_orchestrator()
        assert self._create_and_execute_run_fn is not None
        return self._create_and_execute_run_fn

    def submit_execution(
        self,
        run_id: str,
        exec_fn: Callable[..., Dict[str, Any]],
        exec_kwargs: Dict[str, Any],
    ) -> None:
        with self._lock:
            if run_id in self._jobs and not self._jobs[run_id].done():
                return

            self._jobs_meta[run_id] = {"state": "queued", "submitted_at": _now_iso()}

            def _runner() -> Dict[str, Any]:
                with self._lock:
                    self._jobs_meta[run_id] = {"state": "running", "started_at": _now_iso()}
                try:
                    summary = exec_fn(**exec_kwargs)
                    with self._lock:
                        self._jobs_meta[run_id] = {
                            "state": "completed",
                            "completed_at": _now_iso(),
                            "summary": summary,
                        }
                    return summary
                except Exception as exc:
                    with self._lock:
                        self._jobs_meta[run_id] = {
                            "state": "failed",
                            "failed_at": _now_iso(),
                            "error": str(exc),
                        }
                    raise

            fut = self._executor.submit(_runner)
            self._jobs[run_id] = fut

    def job_state(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            meta = self._jobs_meta.get(run_id)
            return dict(meta) if isinstance(meta, dict) else None

    def warmup_state(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._warmup_meta)

    def start_background_warmup(
        self,
        *,
        outcomes_path: Optional[str | Path],
        snapshot_root: Optional[str | Path],
        snapshot_path: Optional[str | Path],
        entity_identifier_path: str | Path,
        warmup_company_id: str,
        warmup_as_of: str,
    ) -> None:
        if not outcomes_path:
            with self._lock:
                self._warmup_meta = {"state": "skipped", "reason": "outcomes_path_not_set", "updated_at": _now_iso()}
            return
        with self._lock:
            state = str(self._warmup_meta.get("state", ""))
            if state in {"queued", "running", "completed"}:
                return
            self._warmup_meta = {"state": "queued", "queued_at": _now_iso(), "outcomes_path": str(outcomes_path)}

        def _runner() -> None:
            with self._lock:
                self._warmup_meta = {
                    **self._warmup_meta,
                    "state": "running",
                    "started_at": _now_iso(),
                }
            try:
                from src.pipeline.run import warm_precedent_runtime

                report = warm_precedent_runtime(str(outcomes_path))

                # Optional snapshot-loader warmup for common request path.
                snapshot_warm = {"attempted": False, "ok": False}
                if warmup_company_id and warmup_as_of:
                    snapshot_warm["attempted"] = True
                    try:
                        loader = self.build_snapshot_loader(
                            snapshot_root=snapshot_root,
                            snapshot_path=snapshot_path,
                            entity_identifier_path=entity_identifier_path,
                        )
                        asof = datetime.fromisoformat(str(warmup_as_of).replace("Z", "+00:00"))
                        _ = loader(str(warmup_company_id), asof)
                        snapshot_warm["ok"] = True
                    except Exception as exc:
                        snapshot_warm["error"] = str(exc)

                with self._lock:
                    self._warmup_meta = {
                        "state": "completed",
                        "completed_at": _now_iso(),
                        "report": report,
                        "snapshot_warmup": snapshot_warm,
                    }
            except Exception as exc:
                with self._lock:
                    self._warmup_meta = {
                        "state": "failed",
                        "failed_at": _now_iso(),
                        "error": str(exc),
                    }

        t = Thread(target=_runner, daemon=True, name="api-startup-warmup")
        self._warmup_thread = t
        t.start()


def _execute_recommendation_run_entrypoint(**kwargs: Any) -> Dict[str, Any]:
    from src.recommendation_run_orchestrator import execute_recommendation_run

    return execute_recommendation_run(**kwargs)


def _create_and_execute_recommendation_run_entrypoint(**kwargs: Any) -> Dict[str, Any]:
    from src.recommendation_run_orchestrator import create_and_execute_recommendation_run

    return create_and_execute_recommendation_run(**kwargs)


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def build_handler(defaults: argparse.Namespace):
    runtime = _WarmRuntime(max_workers=2)
    if bool(getattr(defaults, "startup_warmup", True)):
        runtime.start_background_warmup(
            outcomes_path=_pick(vars(defaults), "outcomes_path", None),
            snapshot_root=_pick(vars(defaults), "snapshot_root", None),
            snapshot_path=_pick(vars(defaults), "snapshot_path", None),
            entity_identifier_path=_pick(vars(defaults), "entity_identifier_path", "data/inputs_layer/entity_identifier.parquet"),
            warmup_company_id=str(_pick(vars(defaults), "warmup_company_id", "0000320193")),
            warmup_as_of=str(_pick(vars(defaults), "warmup_as_of", "2026-02-28")),
        )

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: Dict[str, Any]) -> None:
            body = _json_bytes(payload)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args) -> None:
            return

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send(200, {"ok": True, "warmup": runtime.warmup_state()})
                return
            parsed = urlparse(self.path)
            if parsed.path == "/precedent_query":
                self._handle_precedent_query(parsed)
                return
            if parsed.path == "/run_status":
                self._handle_run_status(parsed)
                return
            if parsed.path == "/run_result":
                self._handle_run_result(parsed)
                return
            self._send(404, {"ok": False, "error": "not_found"})

        def do_POST(self) -> None:
            if self.path not in {
                "/create_run",
                "/execute_run",
                "/create_and_execute_run",
                "/recommend",
                "/execute_run_async",
            }:
                self._send(404, {"ok": False, "error": "not_found"})
                return

            body, err = _read_json_body(self)
            if err:
                self._send(400, {"ok": False, "error": err})
                return

            try:
                if self.path == "/create_run":
                    self._handle_create_run(body)
                    return
                if self.path == "/execute_run":
                    self._handle_execute_run(body)
                    return
                if self.path == "/execute_run_async":
                    self._handle_execute_run_async(body)
                    return
                if self.path == "/recommend":
                    self._handle_recommend(body)
                    return
                self._handle_create_and_execute_run(body)
            except ValueError as exc:
                self._send(400, {"ok": False, "error": str(exc)})
            except Exception as exc:
                self._send(500, {"ok": False, "error": str(exc)})

        def _load_store(self, body: Dict[str, Any]) -> RecommendationRunStore:
            RecommendationRunStore, _, _, _ = _recommendation_run_bindings()
            return RecommendationRunStore(root=_pick(body, "runs_root", defaults.runs_root))

        def _handle_precedent_query(self, parsed) -> None:
            q = parse_qs(parsed.query)
            run_id = str((q.get("run_id") or [""])[0]).strip()
            if not run_id:
                self._send(400, {"ok": False, "error": "run_id_required"})
                return
            runs_root = str((q.get("runs_root") or [defaults.runs_root])[0])
            RecommendationRunStore, _, _, _ = _recommendation_run_bindings()
            store = RecommendationRunStore(root=runs_root)
            run = store.get_run(run_id)
            if run is None:
                self._send(404, {"ok": False, "error": "run_not_found"})
                return
            md = run.metadata if isinstance(run.metadata, dict) else {}
            artifacts = md.get("artifacts", {}) if isinstance(md.get("artifacts"), dict) else {}
            idx_path = str(artifacts.get("PrecedentIndex") or "")
            idx = _read_json_if_exists(idx_path) if idx_path else None
            from src.pipeline.precedent_index import INDEX_VERSION as PRECEDENT_INDEX_VERSION

            stale_index = not (isinstance(idx, dict) and str(idx.get("index_version", "")) == PRECEDENT_INDEX_VERSION)
            if idx is None or stale_index:
                pm_path = str(artifacts.get("PrecedentMatches") or "")
                pm_obj = _read_json_if_exists(pm_path) if pm_path else None
                if not isinstance(pm_obj, dict):
                    self._send(409, {"ok": False, "error": "precedent_artifact_not_ready"})
                    return
                from src.pipeline.precedent_index import build_precedent_index

                idx = build_precedent_index(run_id=run_id, precedent_matches=pm_obj.get("results", []))
                try:
                    idx_saved = store.attach_artifact(run_id, "PrecedentIndex", idx)
                    idx_path = str(idx_saved)
                except Exception:
                    idx_path = ""
            from src.pipeline.precedent_index import query_precedent_index

            try:
                limit = int((q.get("limit") or ["200"])[0])
            except Exception:
                limit = 200
            try:
                min_sample_size = int((q.get("min_sample_size") or ["0"])[0])
            except Exception:
                min_sample_size = 0
            try:
                min_precedent_confidence = float((q.get("min_precedent_confidence") or ["0"])[0])
            except Exception:
                min_precedent_confidence = 0.0
            exclude_out_of_sample = _parse_bool_flag((q.get("exclude_out_of_sample") or ["false"])[0], default=False)
            result = query_precedent_index(
                idx if isinstance(idx, dict) else {},
                action_type=str((q.get("action_type") or [""])[0]).strip() or None,
                action_id=str((q.get("action_id") or [""])[0]).strip() or None,
                regime=str((q.get("regime") or [""])[0]).strip() or None,
                sector=str((q.get("sector") or [""])[0]).strip() or None,
                time_horizon=str((q.get("time_horizon") or [""])[0]).strip() or None,
                min_sample_size=max(0, min_sample_size),
                min_precedent_confidence=max(0.0, min_precedent_confidence),
                exclude_out_of_sample=exclude_out_of_sample,
                limit=max(1, min(2000, limit)),
            )
            self._send(
                200,
                {
                    "ok": True,
                    "run_id": run_id,
                    "status": run.status,
                    "index_artifact": idx_path,
                    **result,
                },
            )

        def _handle_run_status(self, parsed) -> None:
            q = parse_qs(parsed.query)
            run_id = str((q.get("run_id") or [""])[0]).strip()
            if not run_id:
                self._send(400, {"ok": False, "error": "run_id_required"})
                return
            runs_root = str((q.get("runs_root") or [defaults.runs_root])[0])
            RecommendationRunStore, _, _, _ = _recommendation_run_bindings()
            store = RecommendationRunStore(root=runs_root)
            run = store.get_run(run_id)
            if run is None:
                self._send(404, {"ok": False, "error": "run_not_found"})
                return
            events = [_audit_event_to_dict(e) for e in run.audit_log[-10:]]
            metadata = run.metadata if isinstance(run.metadata, dict) else {}
            self._send(
                200,
                {
                    "ok": True,
                    "run_id": run_id,
                    "status": run.status,
                    "job_state": runtime.job_state(run_id),
                    "last_events": events,
                    "artifacts": metadata.get("artifacts", {}),
                    "config": metadata.get("config"),
                    "progress": metadata.get("progress"),
                    "progress_by_stage": metadata.get("progress_by_stage", {}),
                },
            )

        def _handle_run_result(self, parsed) -> None:
            q = parse_qs(parsed.query)
            run_id = str((q.get("run_id") or [""])[0]).strip()
            if not run_id:
                self._send(400, {"ok": False, "error": "run_id_required"})
                return
            runs_root = str((q.get("runs_root") or [defaults.runs_root])[0])
            RecommendationRunStore, _, _, _ = _recommendation_run_bindings()
            store = RecommendationRunStore(root=runs_root)
            run = store.get_run(run_id)
            if run is None:
                self._send(404, {"ok": False, "error": "run_not_found"})
                return
            metadata = run.metadata if isinstance(run.metadata, dict) else {}
            artifacts = metadata.get("artifacts", {})
            rec = None
            if isinstance(artifacts, dict) and artifacts.get("RecommendationPackage"):
                rec = _read_json_if_exists(artifacts["RecommendationPackage"])
            self._send(
                200,
                {
                    "ok": True,
                    "run_id": run_id,
                    "status": run.status,
                    "artifacts": artifacts,
                    "config": metadata.get("config"),
                    "progress": metadata.get("progress"),
                    "recommendation_package": rec,
                },
            )

        def _handle_create_run(self, body: Dict[str, Any]) -> None:
            company_id = str(body.get("company_id") or "").strip()
            as_of = str(body.get("as_of") or body.get("as_of_time") or "").strip()
            if not company_id or not as_of:
                raise ValueError("company_id and as_of are required")

            snapshot_root = _pick(body, "snapshot_root", defaults.snapshot_root)
            snapshot_path = _pick(body, "snapshot_path", defaults.snapshot_path)
            entity_identifier_path = _pick(body, "entity_identifier_path", defaults.entity_identifier_path)
            raw_action_ids = body.get("action_ids")
            if raw_action_ids is None and body.get("action_id") is not None:
                raw_action_ids = [body.get("action_id")]
            action_ids = _coerce_action_ids(raw_action_ids)
            signature = _canonical_request_signature(
                company_id=company_id,
                as_of=as_of,
                action_ids=action_ids,
                action_type=_pick(body, "action_type", None),
                objectives=_pick(body, "objectives", None),
                constraints=_pick(body, "constraints", None),
                scenario=_pick(body, "scenario", None),
                max_candidates=int(_pick(body, "max_candidates", defaults.max_candidates)),
                min_candidates_target=int(
                    _pick(body, "min_candidates_target", defaults.min_candidates_target)
                ),
                precedent_top_k=int(_pick(body, "precedent_top_k", defaults.precedent_top_k)),
                strict_evidence=bool(_pick(body, "strict_evidence", False)),
                top_plans=int(_pick(body, "top_plans", defaults.top_plans)),
            )
            metadata = dict(_pick(body, "metadata", None) or {})
            metadata["request_signature"] = signature

            RecommendationRunStore, _, _, create_recommendation_run = _recommendation_run_bindings()
            run_id = create_recommendation_run(
                company_id=company_id,
                as_of_time=as_of,
                objectives=_pick(body, "objectives", None),
                constraints=_pick(body, "constraints", None),
                scenario=_pick(body, "scenario", None),
                run_store=RecommendationRunStore(root=_pick(body, "runs_root", defaults.runs_root)),
                snapshot_root=snapshot_root,
                snapshot_path=snapshot_path,
                snapshot_loader=runtime.build_snapshot_loader(
                    snapshot_root=snapshot_root,
                    snapshot_path=snapshot_path,
                    entity_identifier_path=entity_identifier_path,
                ),
                entity_graph_path=_pick(body, "entity_graph_path", defaults.entity_graph_path),
                entity_identifier_path=entity_identifier_path,
                planner_random_seed=_pick(body, "planner_random_seed", None),
                metadata=metadata,
            )
            self._send(
                200,
                {
                    "ok": True,
                    "run_id": run_id,
                    "runs_root": _pick(body, "runs_root", defaults.runs_root),
                },
            )

        def _handle_execute_run(self, body: Dict[str, Any]) -> None:
            run_id = str(body.get("run_id") or "").strip()
            if not run_id:
                raise ValueError("run_id is required")

            snapshot_root = _pick(body, "snapshot_root", defaults.snapshot_root)
            snapshot_path = _pick(body, "snapshot_path", defaults.snapshot_path)
            entity_identifier_path = _pick(body, "entity_identifier_path", defaults.entity_identifier_path)

            raw_action_ids = body.get("action_ids")
            if raw_action_ids is None and body.get("action_id") is not None:
                raw_action_ids = [body.get("action_id")]

            summary = _execute_recommendation_run_entrypoint(
                run_id=run_id,
                runs_root=_pick(body, "runs_root", defaults.runs_root),
                snapshot_root=snapshot_root,
                snapshot_path=snapshot_path,
                snapshot_loader=runtime.build_snapshot_loader(
                    snapshot_root=snapshot_root,
                    snapshot_path=snapshot_path,
                    entity_identifier_path=entity_identifier_path,
                ),
                entity_identifier_path=entity_identifier_path,
                action_ids=_coerce_action_ids(raw_action_ids),
                action_type=_pick(body, "action_type", None),
                max_candidates=int(_pick(body, "max_candidates", defaults.max_candidates)),
                min_candidates_target=int(
                    _pick(body, "min_candidates_target", defaults.min_candidates_target)
                ),
                precedent_top_k=int(_pick(body, "precedent_top_k", defaults.precedent_top_k)),
                strict_evidence=bool(_pick(body, "strict_evidence", False)),
                outcomes_path=_pick(body, "outcomes_path", defaults.outcomes_path),
                config_path=_pick(body, "config", defaults.config),
                top_plans=int(_pick(body, "top_plans", defaults.top_plans)),
                registry=runtime.registry,
            )
            self._send(200, summary)

        def _exec_kwargs(self, body: Dict[str, Any], run_id: str) -> Dict[str, Any]:
            snapshot_root = _pick(body, "snapshot_root", defaults.snapshot_root)
            snapshot_path = _pick(body, "snapshot_path", defaults.snapshot_path)
            entity_identifier_path = _pick(body, "entity_identifier_path", defaults.entity_identifier_path)
            raw_action_ids = body.get("action_ids")
            if raw_action_ids is None and body.get("action_id") is not None:
                raw_action_ids = [body.get("action_id")]
            return {
                "run_id": run_id,
                "runs_root": _pick(body, "runs_root", defaults.runs_root),
                "snapshot_root": snapshot_root,
                "snapshot_path": snapshot_path,
                "snapshot_loader": runtime.build_snapshot_loader(
                    snapshot_root=snapshot_root,
                    snapshot_path=snapshot_path,
                    entity_identifier_path=entity_identifier_path,
                ),
                "entity_identifier_path": entity_identifier_path,
                "action_ids": _coerce_action_ids(raw_action_ids),
                "action_type": _pick(body, "action_type", None),
                "max_candidates": int(_pick(body, "max_candidates", defaults.max_candidates)),
                "min_candidates_target": int(
                    _pick(body, "min_candidates_target", defaults.min_candidates_target)
                ),
                "precedent_top_k": int(_pick(body, "precedent_top_k", defaults.precedent_top_k)),
                "strict_evidence": bool(_pick(body, "strict_evidence", False)),
                "outcomes_path": _pick(body, "outcomes_path", defaults.outcomes_path),
                "config_path": _pick(body, "config", defaults.config),
                "top_plans": int(_pick(body, "top_plans", defaults.top_plans)),
                "registry": runtime.registry,
            }

        def _handle_execute_run_async(self, body: Dict[str, Any]) -> None:
            run_id = str(body.get("run_id") or "").strip()
            if not run_id:
                raise ValueError("run_id is required")
            exec_kwargs = self._exec_kwargs(body, run_id)
            runtime.submit_execution(run_id, _execute_recommendation_run_entrypoint, exec_kwargs)
            self._send(
                202,
                {
                    "ok": True,
                    "run_id": run_id,
                    "status": "accepted",
                    "job_state": runtime.job_state(run_id),
                },
            )

        def _create_run_for_request(self, body: Dict[str, Any]) -> Tuple[str, str]:
            company_id = str(body.get("company_id") or "").strip()
            as_of = str(body.get("as_of") or body.get("as_of_time") or "").strip()
            if not company_id or not as_of:
                raise ValueError("company_id and as_of are required")
            raw_action_ids = body.get("action_ids")
            if raw_action_ids is None and body.get("action_id") is not None:
                raw_action_ids = [body.get("action_id")]
            action_ids = _coerce_action_ids(raw_action_ids)

            signature = _canonical_request_signature(
                company_id=company_id,
                as_of=as_of,
                action_ids=action_ids,
                action_type=_pick(body, "action_type", None),
                objectives=_pick(body, "objectives", None),
                constraints=_pick(body, "constraints", None),
                scenario=_pick(body, "scenario", None),
                max_candidates=int(_pick(body, "max_candidates", defaults.max_candidates)),
                min_candidates_target=int(
                    _pick(body, "min_candidates_target", defaults.min_candidates_target)
                ),
                precedent_top_k=int(_pick(body, "precedent_top_k", defaults.precedent_top_k)),
                strict_evidence=bool(_pick(body, "strict_evidence", False)),
                top_plans=int(_pick(body, "top_plans", defaults.top_plans)),
            )
            snapshot_root = _pick(body, "snapshot_root", defaults.snapshot_root)
            snapshot_path = _pick(body, "snapshot_path", defaults.snapshot_path)
            entity_identifier_path = _pick(body, "entity_identifier_path", defaults.entity_identifier_path)
            metadata = dict(_pick(body, "metadata", None) or {})
            metadata["request_signature"] = signature
            metadata["request_action_ids"] = action_ids or []
            metadata["request_action_type"] = _pick(body, "action_type", None)

            RecommendationRunStore, _, _, create_recommendation_run = _recommendation_run_bindings()
            run_id = create_recommendation_run(
                company_id=company_id,
                as_of_time=as_of,
                objectives=_pick(body, "objectives", None),
                constraints=_pick(body, "constraints", None),
                scenario=_pick(body, "scenario", None),
                run_store=self._load_store(body),
                snapshot_root=snapshot_root,
                snapshot_path=snapshot_path,
                snapshot_loader=runtime.build_snapshot_loader(
                    snapshot_root=snapshot_root,
                    snapshot_path=snapshot_path,
                    entity_identifier_path=entity_identifier_path,
                ),
                entity_graph_path=_pick(body, "entity_graph_path", defaults.entity_graph_path),
                entity_identifier_path=entity_identifier_path,
                planner_random_seed=_pick(body, "planner_random_seed", None),
                metadata=metadata,
            )
            return run_id, signature

        def _handle_recommend(self, body: Dict[str, Any]) -> None:
            company_id = str(body.get("company_id") or "").strip()
            as_of = str(body.get("as_of") or body.get("as_of_time") or "").strip()
            if not company_id or not as_of:
                raise ValueError("company_id and as_of are required")
            raw_action_ids = body.get("action_ids")
            if raw_action_ids is None and body.get("action_id") is not None:
                raw_action_ids = [body.get("action_id")]
            action_ids = _coerce_action_ids(raw_action_ids)
            signature = _canonical_request_signature(
                company_id=company_id,
                as_of=as_of,
                action_ids=action_ids,
                action_type=_pick(body, "action_type", None),
                objectives=_pick(body, "objectives", None),
                constraints=_pick(body, "constraints", None),
                scenario=_pick(body, "scenario", None),
                max_candidates=int(_pick(body, "max_candidates", defaults.max_candidates)),
                min_candidates_target=int(
                    _pick(body, "min_candidates_target", defaults.min_candidates_target)
                ),
                precedent_top_k=int(_pick(body, "precedent_top_k", defaults.precedent_top_k)),
                strict_evidence=bool(_pick(body, "strict_evidence", False)),
                top_plans=int(_pick(body, "top_plans", defaults.top_plans)),
            )
            store = self._load_store(body)
            force_refresh = bool(_pick(body, "force_refresh", False))
            if not force_refresh:
                cached = _find_cached_completed_run(store, company_id, as_of, signature)
                if cached is not None:
                    self._send(
                        200,
                        {
                            "ok": True,
                            "mode": "cache_hit",
                            "cached": True,
                            **cached,
                        },
                    )
                    return

            run_id, _ = self._create_run_for_request(body)
            exec_kwargs = self._exec_kwargs(body, run_id)
            runtime.submit_execution(run_id, _execute_recommendation_run_entrypoint, exec_kwargs)
            self._send(
                202,
                {
                    "ok": True,
                    "mode": "started",
                    "cached": False,
                    "run_id": run_id,
                    "status": "accepted",
                    "job_state": runtime.job_state(run_id),
                },
            )

        def _handle_create_and_execute_run(self, body: Dict[str, Any]) -> None:
            company_id = str(body.get("company_id") or "").strip()
            as_of = str(body.get("as_of") or body.get("as_of_time") or "").strip()
            if not company_id or not as_of:
                raise ValueError("company_id and as_of are required")

            snapshot_root = _pick(body, "snapshot_root", defaults.snapshot_root)
            snapshot_path = _pick(body, "snapshot_path", defaults.snapshot_path)
            entity_identifier_path = _pick(body, "entity_identifier_path", defaults.entity_identifier_path)
            raw_action_ids = body.get("action_ids")
            if raw_action_ids is None and body.get("action_id") is not None:
                raw_action_ids = [body.get("action_id")]
            action_ids = _coerce_action_ids(raw_action_ids)
            signature = _canonical_request_signature(
                company_id=company_id,
                as_of=as_of,
                action_ids=action_ids,
                action_type=_pick(body, "action_type", None),
                objectives=_pick(body, "objectives", None),
                constraints=_pick(body, "constraints", None),
                scenario=_pick(body, "scenario", None),
                max_candidates=int(_pick(body, "max_candidates", defaults.max_candidates)),
                min_candidates_target=int(
                    _pick(body, "min_candidates_target", defaults.min_candidates_target)
                ),
                precedent_top_k=int(_pick(body, "precedent_top_k", defaults.precedent_top_k)),
                strict_evidence=bool(_pick(body, "strict_evidence", False)),
                top_plans=int(_pick(body, "top_plans", defaults.top_plans)),
            )
            metadata = dict(_pick(body, "metadata", None) or {})
            metadata["request_signature"] = signature
            metadata["request_action_ids"] = action_ids or []
            metadata["request_action_type"] = _pick(body, "action_type", None)

            summary = _create_and_execute_recommendation_run_entrypoint(
                company_id=company_id,
                as_of_time=as_of,
                objectives=_pick(body, "objectives", None),
                constraints=_pick(body, "constraints", None),
                scenario=_pick(body, "scenario", None),
                runs_root=_pick(body, "runs_root", defaults.runs_root),
                snapshot_root=snapshot_root,
                snapshot_path=snapshot_path,
                snapshot_loader=runtime.build_snapshot_loader(
                    snapshot_root=snapshot_root,
                    snapshot_path=snapshot_path,
                    entity_identifier_path=entity_identifier_path,
                ),
                entity_graph_path=_pick(body, "entity_graph_path", defaults.entity_graph_path),
                entity_identifier_path=entity_identifier_path,
                planner_random_seed=_pick(body, "planner_random_seed", None),
                metadata=metadata,
                action_ids=action_ids,
                action_type=_pick(body, "action_type", None),
                max_candidates=int(_pick(body, "max_candidates", defaults.max_candidates)),
                min_candidates_target=int(
                    _pick(body, "min_candidates_target", defaults.min_candidates_target)
                ),
                precedent_top_k=int(_pick(body, "precedent_top_k", defaults.precedent_top_k)),
                strict_evidence=bool(_pick(body, "strict_evidence", False)),
                outcomes_path=_pick(body, "outcomes_path", defaults.outcomes_path),
                config_path=_pick(body, "config", defaults.config),
                top_plans=int(_pick(body, "top_plans", defaults.top_plans)),
                registry=runtime.registry,
            )
            self._send(200, summary)

    return Handler


def main() -> None:
    args = _parse_args()
    handler = build_handler(args)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(
        json.dumps(
            {
                "ok": True,
                "message": "recommendation_run_api_started",
                "host": args.host,
                "port": args.port,
                "runs_root": args.runs_root,
            }
        ),
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
