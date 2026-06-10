#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Keep OpenMP/BLAS behavior stable before any heavy scientific stack imports.
# Setting these at process start is more reliable than doing it later inside the
# evaluation context because pandas/pyarrow/numexpr may already have imported
# libomp-linked code by then.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from src.historical_recommendation_eval import build_historical_recommendation_report, _load_fixed_historical_cases
from src.pipeline.precedent_distance_v2_learning import (
    build_precedent_distance_v2_payload,
    coordinate_search_scope_configuration,
    default_scope_configuration,
    load_precedent_distance_v2_objective,
    write_precedent_distance_v2_payload,
)

DEFAULT_LOCK_CONFIG = ROOT / "configs" / "historical_eval_manifests" / "2026-03-17" / "manual_replay_benchmark_lock.json"
DEFAULT_OBJECTIVE_CONFIG = ROOT / "configs" / "precedent_distance_v2_objective.json"
FIT_PRECEDENT_CONFIG_PATH = ROOT / "data" / "curated" / "_fit_mode_precedent_config.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit weighted_distance_v2 against frozen historical replay metrics.")
    parser.add_argument("--benchmark", required=True, help="Benchmark key from the lock config.")
    parser.add_argument("--scope", required=True, help="Action family or exact action id to optimize, e.g. capital_structure.")
    parser.add_argument("--runs-root", required=True, help="Base runs root for search evaluations.")
    parser.add_argument("--out-json", required=True, help="Where to write the fitted v2 weight payload.")
    parser.add_argument("--lock-config", default=str(DEFAULT_LOCK_CONFIG))
    parser.add_argument("--objective-config", default=str(DEFAULT_OBJECTIVE_CONFIG))
    parser.add_argument("--fixed-case-path", default="", help="Optional fixed-case report/manifest path.")
    parser.add_argument("--case-count", type=int, default=None)
    parser.add_argument("--max-rounds", type=int, default=1)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def _resolve_candidate_path(values: Iterable[str | Path]) -> Path:
    candidates = [_resolve_path(value) for value in values]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _load_lock_config(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _ensure_fit_precedent_config(path: Path = FIT_PRECEDENT_CONFIG_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "action_effects": {},
        "macro_series": {},
        "outcome": {
            "horizons_months": [3, 6, 12],
            "primary_metric": "pe",
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _resolve_locked_inputs(config: Dict[str, Any], benchmark_key: str) -> Dict[str, Any]:
    benchmarks = dict(config.get("benchmarks", {}) or {})
    defaults = dict(config.get("defaults", {}) or {})
    artifacts = dict(config.get("artifacts", {}) or {})
    benchmark = dict(benchmarks.get(benchmark_key, {}) or {})
    if not benchmark:
        raise KeyError(f"Unknown benchmark '{benchmark_key}'. Available: {sorted(benchmarks)}")
    resolved_paths = {
        "manifest": _resolve_path(benchmark["manifest"]),
        "outcomes_path": _resolve_path(artifacts["outcomes_path"]),
        "action_support_manifest": _resolve_path(artifacts["action_support_manifest"]),
        "entity_graph_path": _resolve_path(artifacts["entity_graph_path"]),
        "entity_identifier_path": _resolve_path(artifacts["entity_identifier_path"]),
        "entity_table_path": _resolve_path(artifacts["entity_table_path"]),
        "raw_timeseries_path": _resolve_path(artifacts["raw_timeseries_path"]),
        "event_store_path": _resolve_path(artifacts["event_store_path"]),
        "ownership_summary_path": _resolve_path(artifacts["ownership_summary_path"]),
        "issuer_ratings_path": _resolve_path(artifacts["issuer_ratings_path"]),
        "companyfacts_root": _resolve_path(artifacts["companyfacts_root"]),
        "facts_path": _resolve_candidate_path(artifacts["facts_path_candidates"]),
    }
    return {
        "defaults": defaults,
        "benchmark": benchmark,
        "resolved_paths": resolved_paths,
    }


def _filter_cases_by_scope(cases: List[Dict[str, Any]], scope_key: str, case_count: int | None) -> List[Dict[str, Any]]:
    scope = str(scope_key or "").strip().lower()
    filtered: List[Dict[str, Any]] = []
    for case in cases:
        anchor_action_id = str(case.get("anchor_action_id") or "").strip().lower()
        anchor_family = str(case.get("anchor_action_family") or "").strip().lower()
        if "." in scope:
            keep = anchor_action_id == scope
        else:
            keep = anchor_family == scope
        if keep:
            filtered.append(case)
        if case_count is not None and len(filtered) >= int(case_count):
            break
    return filtered


@contextmanager
def _temporary_env(overrides: Dict[str, str]):
    old: Dict[str, Any] = {}
    try:
        for key, value in overrides.items():
            old[key] = os.environ.get(key)
            os.environ[key] = str(value)
        yield
    finally:
        for key, previous in old.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous


def main() -> None:
    args = _parse_args()
    objective_config = load_precedent_distance_v2_objective(args.objective_config)
    lock_config = _load_lock_config(Path(args.lock_config))
    fit_precedent_config_path = _ensure_fit_precedent_config()
    locked = _resolve_locked_inputs(lock_config, args.benchmark)
    defaults = dict(locked["defaults"] or {})
    paths = dict(locked["resolved_paths"] or {})
    config_dir = Path(paths["action_support_manifest"]).parent
    metric_policy_path = config_dir / "market_metric_policy_v1.json"
    methodology_registry_path = config_dir / "consumer_industrials_canonical_registry_v1.json"
    input_source_registry_path = config_dir / "company_state_input_source_registry_v1.json"
    smart_metric_registry_path = config_dir / "smart_metric_registry_v1.json"
    market_availability_overrides_path = config_dir / "liquidity_market_availability_overrides.json"

    fixed_case_source = Path(args.fixed_case_path) if args.fixed_case_path else Path(paths["manifest"])
    selected_cases = _load_fixed_historical_cases([fixed_case_source], case_count=args.case_count)
    filtered_cases = _filter_cases_by_scope(selected_cases, args.scope, args.case_count)
    if not filtered_cases:
        raise SystemExit(f"No fixed cases found for scope '{args.scope}' in {fixed_case_source}")

    runs_root = Path(args.runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)
    out_json_path = Path(args.out_json)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="precedent_distance_v2_search_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        fixed_case_path = tmpdir_path / f"{str(args.scope).replace('.', '_')}_fixed_cases.json"
        fixed_case_path.write_text(json.dumps({"cases": filtered_cases}, indent=2))
        evaluation_counter = {"value": 0}
        shared_snapshot_cache = runs_root / "_shared_snapshot_cache"
        shared_snapshot_cache.mkdir(parents=True, exist_ok=True)

        def evaluate_scope_config(scope_config: Dict[str, Any]) -> Dict[str, Any]:
            evaluation_counter["value"] += 1
            eval_index = int(evaluation_counter["value"])
            candidate_payload = build_precedent_distance_v2_payload(
                scopes={str(scope_config.get("scope_key") or args.scope): dict(scope_config)},
                objective_config=objective_config,
                benchmark_key=args.benchmark,
                notes={"evaluation_index": eval_index},
            )
            candidate_payload_path = tmpdir_path / f"candidate_{eval_index:03d}.json"
            write_precedent_distance_v2_payload(candidate_payload, candidate_payload_path)
            eval_runs_root = runs_root / f"{str(args.scope).replace('.', '_')}_eval_{eval_index:03d}"
            env = {
                "PRECEDENT_DISTANCE_PROFILE_VERSION": "weighted_distance_v2",
                "PRECEDENT_DISTANCE_V2_WEIGHTS_PATH": str(candidate_payload_path),
                "PRECEDENT_DISABLE_LEARNED_DISTANCE_WEIGHTS": "1",
                "AXIOM_SKIP_CAUSAL_MODEL_RISK_REPORT": "1",
                "AXIOM_SKIP_DOSSIER_PACKAGE_BUILD": "1",
                "AXIOM_SKIP_ESTIMATES": "1",
                "AXIOM_SKIP_DEALSCAN": "1",
                "AXIOM_SKIP_RUN_COMPANY_VALIDATION": "1",
                "CAUSAL_IMPACT_MODEL_PATH": str(ROOT / "data" / "curated" / "_fit_mode_no_causal_model.json"),
                "MECHANISM_MODEL_VERSION": "mechanism_model_fit_stub",
                # Weight fitting is about precedent quality, not causal-model scoring.
                # Block causal impact inference so the search stays deterministic and
                # avoids loading the large bundle.pkl on every fit process.
                "CAUSAL_ACTION_BLOCKLIST": "*",
                "AXIOM_WAREHOUSE_FINANCIAL_YEARS_BACK": "4",
                "RECO_PRECEDENT_WORKERS": "1",
                # The precedent runtime pulls in libraries that may initialize
                # libomp again inside the replay search process. These settings
                # keep the search stable instead of dying in OpenMP startup.
                "KMP_DUPLICATE_LIB_OK": "TRUE",
                "KMP_INIT_AT_FORK": "FALSE",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            }
            if metric_policy_path.exists():
                env["AXIOM_METRIC_POLICY_PATH"] = str(metric_policy_path)
            if methodology_registry_path.exists():
                env["AXIOM_METHODOLOGY_REGISTRY_PATH"] = str(methodology_registry_path)
            if input_source_registry_path.exists():
                env["AXIOM_INPUT_SOURCE_REGISTRY_PATH"] = str(input_source_registry_path)
            if smart_metric_registry_path.exists():
                env["AXIOM_SMART_METRIC_REGISTRY_PATH"] = str(smart_metric_registry_path)
            if market_availability_overrides_path.exists():
                env["AXIOM_MARKET_AVAILABILITY_OVERRIDES_PATH"] = str(market_availability_overrides_path)
            if not args.quiet:
                print(
                    json.dumps(
                        {
                            "event": "evaluate_candidate",
                            "evaluation_index": eval_index,
                            "scope": str(args.scope),
                            "runs_root": str(eval_runs_root),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            def progress_logger(payload: Dict[str, Any]) -> None:
                if args.quiet:
                    return
                event_payload = {
                    "event": "evaluation_progress",
                    "evaluation_index": eval_index,
                    **dict(payload or {}),
                }
                print(json.dumps(event_payload, sort_keys=True), flush=True)
            with _temporary_env(env):
                report = build_historical_recommendation_report(
                    runs_root=eval_runs_root,
                    outcomes_path=paths["outcomes_path"],
                    config_path=fit_precedent_config_path,
                    entity_graph_path=paths["entity_graph_path"],
                    entity_identifier_path=paths["entity_identifier_path"],
                    companyfacts_root=paths["companyfacts_root"],
                    entity_table_path=paths["entity_table_path"],
                    case_count=len(filtered_cases),
                    lookback_days=int(defaults.get("lookback_days", 365)),
                    alignment_horizon_days=int(defaults.get("alignment_horizon_days", 120)),
                    families=[str(args.scope).split(".", 1)[0]],
                    top_plans=int(defaults.get("top_plans", 3)),
                    precedent_top_k=int(defaults.get("precedent_top_k", 0) or 0),
                    raw_timeseries_path=paths["raw_timeseries_path"],
                    event_store_path=paths["event_store_path"],
                    facts_path=paths["facts_path"],
                    ownership_summary_path=paths["ownership_summary_path"],
                    issuer_ratings_path=paths["issuer_ratings_path"],
                    snapshot_cache_dir=shared_snapshot_cache,
                    historical_backfill_mode=bool(defaults.get("historical_backfill_mode", True)),
                    min_non_missing_core_features=int(defaults.get("min_non_missing_core_features", 3)),
                    cache_facts=False,
                    cache_events=False,
                    cache_timeseries=False,
                    cache_ownership=False,
                    cache_ratings=False,
                    fixed_case_paths=[fixed_case_path],
                    action_support_manifest_path=paths["action_support_manifest"],
                    progress_logger=progress_logger,
                )
            if not args.quiet:
                print(
                    json.dumps(
                        {
                            "event": "candidate_complete",
                            "evaluation_index": eval_index,
                            "aggregate": dict(report.get("aggregate", {}) or {}),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            return report

        search = coordinate_search_scope_configuration(
            scope_key=str(args.scope),
            objective_config=objective_config,
            evaluate_scope_config=evaluate_scope_config,
            max_rounds=int(args.max_rounds),
        )

        final_scope_key = str(search["scope_key"])
        final_scope_config = dict(search["best_config"])
        final_scope_config["metrics"] = dict(search["best_aggregate"])
        final_payload = build_precedent_distance_v2_payload(
            scopes={final_scope_key: final_scope_config},
            objective_config=objective_config,
            benchmark_key=args.benchmark,
            notes={
                "fixed_case_source": str(fixed_case_source),
                "filtered_case_count": len(filtered_cases),
                "evaluation_count": int(evaluation_counter["value"]),
                "history": search["history"],
            },
        )
        write_precedent_distance_v2_payload(final_payload, out_json_path)
        if not args.quiet:
            print(
                json.dumps(
                    {
                        "event": "search_complete",
                        "scope": final_scope_key,
                        "out_json": str(out_json_path),
                        "best_aggregate": search["best_aggregate"],
                        "evaluation_count": int(evaluation_counter["value"]),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
