#!/usr/bin/env python
from __future__ import annotations

import argparse
import faulthandler
import json
import os
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Tuple

def _bootstrap_runtime_threading_defaults() -> None:
    # Historical replay is strictly offline and occasionally trips Intel/OpenMP
    # runtime issues on this machine; keep the harness single-threaded by default.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    os.environ.setdefault("KMP_AFFINITY", "disabled")
    os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")


_bootstrap_runtime_threading_defaults()

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG_PATH = ROOT / "configs" / "historical_eval_manifests" / "2026-03-17" / "manual_replay_benchmark_lock.json"
_CANONICAL_LOCK_ARTIFACT_FALLBACKS: Dict[str, Path] = {
    "outcomes_path": ROOT / "data" / "curated" / "action_outcomes_with_credit_ratings.normalized_full.parquet",
    "action_support_manifest": ROOT / "configs" / "action_data_support_manifest.json",
    "entity_graph_path": ROOT / "data" / "inputs_layer" / "entity_graph.parquet",
    "entity_identifier_path": ROOT / "data" / "inputs_layer" / "entity_identifier.parquet",
    "entity_table_path": ROOT / "data" / "inputs_layer" / "entity.parquet",
    "raw_timeseries_path": ROOT / "data" / "inputs_layer" / "raw_timeseries.parquet",
    "event_store_path": ROOT / "data" / "inputs_layer" / "event_store.parquet",
    "ownership_summary_path": ROOT / "data" / "inputs_layer" / "ownership_13f_summary.parquet",
    "issuer_ratings_path": ROOT / "data" / "inputs_layer" / "issuer_rating_history.parquet",
    "companyfacts_root": ROOT / "data" / "sec" / "companyfacts",
    "facts_path": ROOT / "data" / "inputs_layer" / "extracted_fact_registry_validity",
}


def _bootstrap_env_overrides() -> None:
    config_path = DEFAULT_CONFIG_PATH
    argv = sys.argv[1:]
    for index, token in enumerate(argv):
        if token == "--config" and index + 1 < len(argv):
            config_path = Path(argv[index + 1])
            break
        if token.startswith("--config="):
            config_path = Path(token.split("=", 1)[1])
            break
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    if not config_path.exists():
        return
    try:
        payload = json.loads(config_path.read_text())
    except Exception:
        # Synced placeholder manifests/configs should not block offline replay startup.
        return
    env_override_candidates = dict(payload.get("env_override_candidates", {}) or {})
    if env_override_candidates:
        for name, values in env_override_candidates.items():
            candidates = []
            for value in list(values or []):
                path = Path(value)
                if not path.is_absolute():
                    path = ROOT / path
                candidates.append(path)
            chosen = next((candidate for candidate in candidates if candidate.exists()), candidates[0] if candidates else None)
            if chosen is not None:
                os.environ.setdefault(name, str(chosen))
        return
    env_overrides = dict(payload.get("env_overrides", {}) or {})
    for name, path_value in env_overrides.items():
        path = Path(path_value)
        if not path.is_absolute():
            path = ROOT / path
        os.environ.setdefault(name, str(path))


_bootstrap_env_overrides()

from src.action_data_support import resolve_action_support
from src.backtest_artifacts import (
    build_backtest_artifact_manifest,
    fingerprint_path,
    resolve_backtest_artifact_root,
    resolve_snapshot_cache_dir,
)
from src.backtest_costs import resolve_transaction_cost_model
from src.backtest_protocol import resolve_backtest_protocol
from src.backtest_scorecard import (
    build_portfolio_strategy_scorecard,
    render_portfolio_strategy_scorecard_markdown,
)
from src.company_state_builder import CompanyStateBuilder
from src.company_state_store import SnapshotStore
from src.historical_recommendation_eval import (
    _aggregate_historical_cases,
    _build_historical_alias_overrides,
    _load_action_support_summary,
    _load_fixed_historical_cases,
    _load_realized_outcomes_lookup,
    _score_ex_post_alignment,
    _snapshot_coverage_summary,
    _snapshot_has_meaningful_coverage,
    _top_action_ids,
)
from src.recommendation_run import RecommendationRunStore, create_recommendation_run
from src.recommendation_run_orchestrator import execute_recommendation_run
from src.replay_snapshot_enrichment import enrich_snapshot_with_revenue_growth_inputs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen manual historical replay benchmark.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--benchmark", required=True, help="Benchmark key from the lock config.")
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--snapshot-cache-dir", default="", help="Optional snapshot cache dir. Defaults to a stable path under the artifact root.")
    parser.add_argument("--artifact-root", default="", help="Optional stable artifact root. Defaults to <runs-root>/_backtest_artifacts.")
    parser.add_argument("--out-json", default="", help="Optional report output path. Defaults under the artifact root.")
    parser.add_argument("--scorecard-json", default="", help="Optional standardized scorecard JSON path.")
    parser.add_argument("--scorecard-md", default="", help="Optional standardized scorecard markdown path.")
    parser.add_argument("--manifest-json", default="", help="Optional artifact manifest JSON path.")
    parser.add_argument("--protocol", default="", help="Optional canonical backtest protocol override.")
    parser.add_argument("--cost-model", default="", help="Optional transaction cost model override.")
    parser.add_argument("--precedent-top-k", type=int, help="Optional override for the number of precedent candidates to retrieve per case.")
    parser.add_argument("--case-count", type=int, help="Optional case limit for smoke runs.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-case progress logs.")
    parser.add_argument("--dump-stack-after-seconds", type=int, help="Optional faulthandler timeout for diagnosing stalls.")
    return parser.parse_args()


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def _looks_like_stale_manual_replay_bundle_path(path: Path) -> bool:
    text = str(path)
    return "/out/manual_replay_bundle_" in text or text.endswith("/companyfacts_buyback_20260407")


def _resolve_locked_artifact_path(artifact_key: str, value: str | Path) -> Path:
    path = _resolve_path(value)
    if path.exists():
        return path
    if _looks_like_stale_manual_replay_bundle_path(path):
        fallback = _CANONICAL_LOCK_ARTIFACT_FALLBACKS.get(artifact_key)
        if fallback is not None and fallback.exists():
            return fallback
    return path


def _resolve_candidate_path(values: Iterable[str | Path], artifact_key: str = "") -> Path:
    candidates = [_resolve_path(value) for value in values]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if candidates and any(_looks_like_stale_manual_replay_bundle_path(candidate) for candidate in candidates):
        fallback = _CANONICAL_LOCK_ARTIFACT_FALLBACKS.get(artifact_key)
        if fallback is not None and fallback.exists():
            return fallback
    return candidates[0]


def _path_metadata(path: Path) -> Dict[str, Any]:
    exists = path.exists()
    info: Dict[str, Any] = {
        "path": str(path),
        "exists": exists,
    }
    if exists:
        stat = path.stat()
        info["size_bytes"] = stat.st_size
        info["modified_at"] = pd.Timestamp(stat.st_mtime, unit="s", tz="UTC").isoformat()
    return info


def _resolve_output_path(value: str, default_path: Path) -> Path:
    raw = str(value or "").strip()
    return Path(raw) if raw else default_path


def _load_lock_config(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _resolve_locked_inputs(config: Dict[str, Any], benchmark_key: str) -> Dict[str, Any]:
    benchmarks = dict(config.get("benchmarks", {}) or {})
    if benchmark_key not in benchmarks:
        raise KeyError(f"Unknown benchmark '{benchmark_key}'. Available: {sorted(benchmarks)}")
    defaults = dict(config.get("defaults", {}) or {})
    artifacts = dict(config.get("artifacts", {}) or {})
    benchmark = dict(benchmarks[benchmark_key] or {})

    resolved_paths = {
        "manifest": _resolve_path(benchmark["manifest"]),
        "outcomes_path": _resolve_locked_artifact_path("outcomes_path", artifacts["outcomes_path"]),
        "action_support_manifest": _resolve_locked_artifact_path("action_support_manifest", artifacts["action_support_manifest"]),
        "corporate_actions_master_path": _resolve_locked_artifact_path("corporate_actions_master_path", artifacts["corporate_actions_master_path"]),
        "entity_graph_path": _resolve_locked_artifact_path("entity_graph_path", artifacts["entity_graph_path"]),
        "entity_identifier_path": _resolve_locked_artifact_path("entity_identifier_path", artifacts["entity_identifier_path"]),
        "entity_table_path": _resolve_locked_artifact_path("entity_table_path", artifacts["entity_table_path"]),
        "raw_timeseries_path": _resolve_locked_artifact_path("raw_timeseries_path", artifacts["raw_timeseries_path"]),
        "event_store_path": _resolve_locked_artifact_path("event_store_path", artifacts["event_store_path"]),
        "ownership_summary_path": _resolve_locked_artifact_path("ownership_summary_path", artifacts["ownership_summary_path"]),
        "issuer_ratings_path": _resolve_locked_artifact_path("issuer_ratings_path", artifacts["issuer_ratings_path"]),
        "companyfacts_root": _resolve_locked_artifact_path("companyfacts_root", artifacts["companyfacts_root"]),
        "facts_path": _resolve_candidate_path(artifacts["facts_path_candidates"], artifact_key="facts_path"),
    }
    env_override_candidates = dict(config.get("env_override_candidates", {}) or {})
    if env_override_candidates:
        resolved_env = {
            name: str(_resolve_candidate_path(values))
            for name, values in env_override_candidates.items()
        }
    else:
        env_overrides = dict(config.get("env_overrides", {}) or {})
        resolved_env = {name: str(_resolve_path(path)) for name, path in env_overrides.items()}
    return {
        "defaults": defaults,
        "benchmark": benchmark,
        "resolved_paths": resolved_paths,
        "resolved_env": resolved_env,
    }


def _progress(payload: Dict[str, Any], quiet: bool) -> None:
    if quiet:
        return
    print(json.dumps(payload, sort_keys=True), flush=True)


def _derive_facts_years(selected_cases: List[Dict[str, Any]], lookback_years: int) -> List[int] | None:
    years: set[int] = set()
    for spec in selected_cases:
        raw_as_of = spec.get("as_of_time")
        if not raw_as_of:
            continue
        try:
            ts = pd.Timestamp(raw_as_of)
        except Exception:
            continue
        year = int(ts.year)
        for candidate_year in range(year - max(0, int(lookback_years)), year + 1):
            years.add(candidate_year)
    return sorted(years) if years else None


def main() -> None:
    args = _parse_args()
    if args.dump_stack_after_seconds:
        faulthandler.dump_traceback_later(int(args.dump_stack_after_seconds), repeat=False)
    lock_path = Path(args.config)
    config = _load_lock_config(lock_path)
    locked = _resolve_locked_inputs(config, args.benchmark)

    for env_name, env_value in locked["resolved_env"].items():
        os.environ[env_name] = env_value
    runs_root_path = Path(args.runs_root)
    artifact_root = resolve_backtest_artifact_root(
        runs_root=runs_root_path,
        artifact_root=Path(args.artifact_root) if args.artifact_root else None,
    )
    snapshot_cache_dir = resolve_snapshot_cache_dir(
        runs_root=runs_root_path,
        artifact_root=artifact_root,
        snapshot_cache_dir=Path(args.snapshot_cache_dir) if args.snapshot_cache_dir else None,
    )
    artifact_root.mkdir(parents=True, exist_ok=True)
    snapshot_cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = _resolve_output_path(
        args.out_json,
        artifact_root / f"{args.benchmark}.historical_replay_report.json",
    )
    scorecard_json_path = _resolve_output_path(
        args.scorecard_json,
        artifact_root / f"{args.benchmark}.canonical_scorecard.json",
    )
    scorecard_md_path = _resolve_output_path(
        args.scorecard_md,
        artifact_root / f"{args.benchmark}.canonical_scorecard.md",
    )
    manifest_json_path = _resolve_output_path(
        args.manifest_json,
        artifact_root / f"{args.benchmark}.artifact_manifest.json",
    )

    run_tmp_dir = runs_root_path / "_tmp"
    os.environ["RECOMMENDATION_RUN_TMP_DIR"] = str(run_tmp_dir)
    locked["resolved_env"]["RECOMMENDATION_RUN_TMP_DIR"] = str(run_tmp_dir)

    defaults = locked["defaults"]
    benchmark = locked["benchmark"]
    paths = locked["resolved_paths"]
    precedent_top_k = int(args.precedent_top_k) if args.precedent_top_k is not None else int(defaults.get("precedent_top_k", 0) or 0)

    manifest = paths["manifest"]
    outcomes_path = paths["outcomes_path"]
    selected_cases = _load_fixed_historical_cases([manifest], case_count=args.case_count)
    facts_years = _derive_facts_years(
        selected_cases,
        lookback_years=int(defaults.get("facts_years_lookback", 2)),
    )
    alias_overrides = _build_historical_alias_overrides(selected_cases)
    outcomes_lookup = _load_realized_outcomes_lookup(outcomes_path)
    action_support_summary = _load_action_support_summary(
        outcomes_path=outcomes_path,
        manifest_path=paths["action_support_manifest"],
    )

    builder = CompanyStateBuilder(
        raw_timeseries_path=paths["raw_timeseries_path"],
        event_store_path=paths["event_store_path"],
        corporate_actions_master_path=paths["corporate_actions_master_path"],
        facts_path=paths["facts_path"],
        ownership_summary_path=paths["ownership_summary_path"],
        issuer_ratings_path=paths["issuer_ratings_path"],
        entity_graph_path=paths["entity_graph_path"],
        entity_identifier_path=paths["entity_identifier_path"],
        entity_table_path=paths["entity_table_path"],
        historical_backfill_mode=bool(defaults.get("historical_backfill_mode", True)),
        facts_years=facts_years,
        companyfacts_root=paths["companyfacts_root"],
        enable_market_relevant_smart_normalized_inputs=bool(
            defaults.get("enable_market_relevant_smart_normalized_inputs", True)
        ),
    )

    snapshot_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
    protocol = resolve_backtest_protocol(
        protocol_key=args.protocol or None,
        benchmark_key=args.benchmark,
    )
    cost_model = resolve_transaction_cost_model(args.cost_model or protocol.cost_model_key)
    if precedent_top_k > 0:
        from src.pipeline.run import warm_precedent_runtime

        _progress(
            {
                "event": "precedent_runtime_warm_start",
                "outcomes_path": str(outcomes_path),
            },
            args.quiet,
        )
        warm_summary = warm_precedent_runtime(outcomes_path)
        _progress(
            {
                "event": "precedent_runtime_warm_complete",
                **warm_summary,
            },
            args.quiet,
        )

    snapshot_store = None
    legacy_snapshot_cache_root = snapshot_cache_dir
    try:
        snapshot_store = SnapshotStore(root=snapshot_cache_dir)
    except Exception:
        snapshot_store = None

    def _load_persisted_snapshot(company_id: str, ts: pd.Timestamp) -> Dict[str, Any] | None:
        if snapshot_store is not None:
            try:
                cached = snapshot_store.load_keyed_snapshot(str(company_id), ts.strftime("%Y-%m-%d"))
            except Exception:
                cached = None
            if isinstance(cached, dict) and cached:
                return cached
        legacy_path = legacy_snapshot_cache_root / f"company_id={company_id}" / f"snapshot_as_of={ts.strftime('%Y%m%dT%H%M%SZ')}.json"
        if legacy_path.exists():
            try:
                return json.loads(legacy_path.read_text())
            except Exception:
                return None
        return None

    def snapshot_loader(company_id: str, as_of_dt) -> Dict[str, Any]:
        ts = pd.Timestamp(as_of_dt)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        key = (str(company_id), ts.isoformat())
        cache_path = snapshot_cache_dir / f"company_id={company_id}" / f"snapshot_as_of={ts.strftime('%Y%m%dT%H%M%SZ')}.json"
        keyed_as_of = ts.strftime("%Y-%m-%d")

        def _persist_snapshot_payload(payload: Dict[str, Any]) -> None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(payload, default=str))
            if snapshot_store is not None:
                snapshot_store.upsert_keyed_snapshot(payload, as_of=keyed_as_of)

        def _enrich_snapshot_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
            enriched, changed, enrichment_summary = enrich_snapshot_with_revenue_growth_inputs(
                payload,
                companyfacts_root=paths["companyfacts_root"],
                company_id=str(company_id),
                as_of_time=ts.isoformat(),
            )
            if changed:
                _persist_snapshot_payload(enriched)
                _progress(
                    {
                        "event": "snapshot_matching_enrichment_complete",
                        "company_id": str(company_id),
                        "as_of_time": ts.isoformat(),
                        "metrics": enrichment_summary.get("metrics") or {},
                    },
                    args.quiet,
                )
            return enriched

        if key not in snapshot_cache:
            cached_snapshot = _load_persisted_snapshot(str(company_id), ts)
            if isinstance(cached_snapshot, dict) and cached_snapshot:
                snapshot_cache[key] = _enrich_snapshot_payload(cached_snapshot)
                _progress(
                    {
                        "event": "snapshot_cache_hit",
                        "company_id": str(company_id),
                        "as_of_time": ts.isoformat(),
                    },
                    args.quiet,
                )
                return snapshot_cache[key]
            extra_aliases = list(alias_overrides.get(key, []) or [])
            _progress(
                {
                    "event": "snapshot_build_start",
                    "company_id": str(company_id),
                    "as_of_time": ts.isoformat(),
                    "alias_count": len(extra_aliases),
                },
                args.quiet,
            )
            snapshot = builder.build(
                company_id=str(company_id),
                as_of_time=ts.isoformat(),
                extra_aliases=extra_aliases,
            )
            payload = _enrich_snapshot_payload(asdict(snapshot))
            snapshot_cache[key] = payload
            _progress(
                {
                    "event": "snapshot_build_complete",
                    "company_id": str(company_id),
                    "as_of_time": ts.isoformat(),
                },
                args.quiet,
            )
            _persist_snapshot_payload(payload)
            _progress(
                {
                    "event": "snapshot_cache_write_complete",
                    "company_id": str(company_id),
                    "cache_path": str(cache_path),
                },
                args.quiet,
            )
        return snapshot_cache[key]

    store = RecommendationRunStore(root=runs_root_path)
    cases: List[Dict[str, Any]] = []

    for index, spec in enumerate(selected_cases, start=1):
        company_id = str(spec["company_id"])
        source_company_id = str(spec.get("source_company_id") or company_id)
        as_of_time = str(spec["as_of_time"])
        company_aliases = list(alias_overrides.get((company_id, as_of_time), []) or [])
        anchor_action_id = str(spec["anchor_action_id"])
        anchor_action_family = str(spec["anchor_action_family"])
        anchor_action_date = str(spec["anchor_action_date"])
        anchor_action_support = resolve_action_support(
            action_id=anchor_action_id,
            action_family=anchor_action_family,
            support_report=action_support_summary,
        )
        try:
            _progress({"event": "case_start", "index": index, "total": len(selected_cases), "company_id": company_id}, args.quiet)
            prebuilt_snapshot = snapshot_loader(company_id, pd.Timestamp(as_of_time).to_pydatetime())
            snapshot_coverage = _snapshot_coverage_summary(prebuilt_snapshot)
            if not _snapshot_has_meaningful_coverage(
                snapshot_coverage,
                min_non_missing_core_features=int(defaults.get("min_non_missing_core_features", 3)),
            ):
                cases.append(
                    {
                        "company_id": company_id,
                        "source_company_id": source_company_id,
                        "as_of_time": as_of_time,
                        "anchor_action_id": anchor_action_id,
                        "anchor_action_family": anchor_action_family,
                        "anchor_action_date": anchor_action_date,
                        "anchor_action_support": anchor_action_support,
                        "unsupported_reason": "insufficient_snapshot_coverage",
                        "snapshot_coverage": snapshot_coverage,
                    }
                )
                _progress({"event": "case_skip", "index": index, "company_id": company_id, "reason": "insufficient_snapshot_coverage"}, args.quiet)
                continue
            run_id = create_recommendation_run(
                company_id=company_id,
                as_of_time=as_of_time,
                run_store=store,
                snapshot_loader=snapshot_loader,
                entity_graph_path=paths["entity_graph_path"],
                entity_identifier_path=paths["entity_identifier_path"],
                company_aliases=company_aliases,
                skip_as_of_lower_bound_validation=True,
                metadata={
                    "historical_eval": {
                        "anchor_action_id": anchor_action_id,
                        "anchor_action_family": anchor_action_family,
                        "anchor_action_date": anchor_action_date,
                        "lookback_days": int(defaults.get("lookback_days", 365)),
                        "alignment_horizon_days": int(defaults.get("alignment_horizon_days", 120)),
                        "precedent_top_k": precedent_top_k,
                    }
                },
            )
            summary = execute_recommendation_run(
                run_id=run_id,
                runs_root=Path(args.runs_root),
                snapshot_loader=snapshot_loader,
                snapshot_root=snapshot_cache_dir,
                entity_identifier_path=paths["entity_identifier_path"],
                outcomes_path=outcomes_path,
                precedent_top_k=precedent_top_k,
                top_plans=int(defaults.get("top_plans", 3)),
            )
            artifacts = dict(summary.get("artifacts", {}) or {})
            package_path = artifacts.get("RecommendationPackage")
            package = json.loads(Path(package_path).read_text()) if package_path else {}
            top_action_ids = _top_action_ids(package)
            recommended_action_support = [
                resolve_action_support(
                    action_id=action_id,
                    action_family=action_id.split(".", 1)[0] if "." in action_id else "",
                    support_report=action_support_summary,
                )
                for action_id in top_action_ids
            ]
            alignment = _score_ex_post_alignment(
                company_id=source_company_id,
                as_of_time=as_of_time,
                recommended_action_ids=top_action_ids,
                outcomes_lookup=outcomes_lookup,
                alignment_horizon_days=int(defaults.get("alignment_horizon_days", 120)),
                anchor_action_id=anchor_action_id,
                anchor_action_family=anchor_action_family,
                anchor_action_support=anchor_action_support,
                recommended_action_support=recommended_action_support,
            )
            case = {
                "company_id": company_id,
                "source_company_id": source_company_id,
                "as_of_time": as_of_time,
                "run_id": summary.get("run_id"),
                "anchor_action_id": anchor_action_id,
                "anchor_action_family": anchor_action_family,
                "anchor_action_date": anchor_action_date,
                "anchor_action_support": anchor_action_support,
                "recommended_posture": package.get("recommended_posture"),
                "top_action_ids": top_action_ids,
                "recommended_action_support": recommended_action_support,
                "historical_alignment": alignment,
                "snapshot_coverage": snapshot_coverage,
                "artifacts": artifacts,
            }
            if not top_action_ids:
                case["unsupported_reason"] = "no_feasible_plan_generated"
            cases.append(case)
            _progress(
                {
                    "event": "case_complete",
                    "index": index,
                    "total": len(selected_cases),
                    "company_id": company_id,
                    "run_id": summary.get("run_id"),
                    "alignment_score": alignment.get("score"),
                    "alignment_reason": alignment.get("reason"),
                },
                args.quiet,
            )
        except Exception as exc:
            cases.append(
                {
                    "company_id": company_id,
                    "source_company_id": source_company_id,
                    "as_of_time": as_of_time,
                    "anchor_action_id": anchor_action_id,
                    "anchor_action_family": anchor_action_family,
                    "anchor_action_date": anchor_action_date,
                    "anchor_action_support": anchor_action_support,
                    "error": str(exc),
                }
            )
            _progress({"event": "case_error", "index": index, "company_id": company_id, "error": str(exc)}, args.quiet)

    report = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "benchmark_key": args.benchmark,
        "benchmark_label": benchmark.get("label"),
        "benchmark_lock_path": str(lock_path),
        "lock_version": config.get("version"),
        "manifest": str(manifest),
        "runs_root": str(runs_root_path),
        "artifact_root": str(artifact_root),
        "snapshot_cache_dir": str(snapshot_cache_dir),
        "case_count_requested": len(selected_cases),
        "runs_analyzed": len(cases),
        "reference_metrics": benchmark.get("reference_metrics", {}),
        "resolved_paths": {name: _path_metadata(path) for name, path in paths.items()},
        "resolved_env_overrides": {
            name: _path_metadata(Path(value))
            for name, value in locked["resolved_env"].items()
        },
        "resolved_path_fingerprints": {
            name: fingerprint_path(path)
            for name, path in paths.items()
        },
        "resolved_env_override_fingerprints": {
            name: fingerprint_path(value)
            for name, value in locked["resolved_env"].items()
        },
        "defaults": defaults,
        "facts_years": facts_years,
        "cases": cases,
        "aggregate": _aggregate_historical_cases(cases),
    }
    scorecard = build_portfolio_strategy_scorecard(
        report,
        protocol=protocol,
        cost_model=cost_model,
    )
    report["protocol"] = protocol.to_dict()
    report["cost_model"] = cost_model.to_dict()
    report["scorecard"] = scorecard

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str))

    scorecard_json_path.parent.mkdir(parents=True, exist_ok=True)
    scorecard_json_path.write_text(json.dumps(scorecard, indent=2, default=str))

    scorecard_md_path.parent.mkdir(parents=True, exist_ok=True)
    scorecard_md_path.write_text(render_portfolio_strategy_scorecard_markdown(scorecard))

    manifest = build_backtest_artifact_manifest(
        suite=str(config.get("suite") or "manual_replay_historical_benchmark"),
        benchmark_key=args.benchmark,
        protocol=protocol,
        cost_model=cost_model,
        lock_path=lock_path,
        runs_root=runs_root_path,
        artifact_root=artifact_root,
        snapshot_cache_dir=snapshot_cache_dir,
        resolved_paths=paths,
        resolved_env=locked["resolved_env"],
        outputs={
            "historical_report": out_path,
            "scorecard_json": scorecard_json_path,
            "scorecard_markdown": scorecard_md_path,
        },
    )
    manifest_json_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_json_path.write_text(json.dumps(manifest, indent=2, default=str))

    report["artifact_manifest_path"] = str(manifest_json_path)
    report["artifact_manifest"] = manifest
    out_path.write_text(json.dumps(report, indent=2, default=str))

    aggregate = dict(report.get("aggregate", {}) or {})
    portfolio_proxy = dict(scorecard.get("portfolio_proxy", {}) or {})
    print(
        json.dumps(
            {
                "ok": True,
                "benchmark_key": args.benchmark,
                "out_json": str(out_path),
                "scorecard_json": str(scorecard_json_path),
                "manifest_json": str(manifest_json_path),
                "mean_alignment_score": aggregate.get("mean_alignment_score"),
                "net_mean_alignment_score": portfolio_proxy.get("net_mean_alignment_score"),
                "anchor_primary_exact_rate": aggregate.get("anchor_primary_exact_rate"),
                "anchor_primary_family_rate": aggregate.get("anchor_primary_family_rate"),
                "unsupported_case_count": aggregate.get("unsupported_case_count"),
            }
        )
    )
    if args.dump_stack_after_seconds:
        faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
