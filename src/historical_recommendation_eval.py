from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import duckdb
import pandas as pd

from .action_data_support import build_action_support_report, load_action_support_report, resolve_action_support
from .company_state_builder import CompanyStateBuilder
from .model_feature_bundle import attach_model_feature_bundle
from .recommendation_run_orchestrator import _run_store_bindings, execute_recommendation_run


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPANYFACTS_ROOT = ROOT / "data" / "sec" / "companyfacts"
DEFAULT_FAMILIES: Tuple[str, ...] = (
    "capital_return",
    "capital_structure",
    "mna",
    "portfolio",
)


def build_historical_recommendation_report(
    *,
    runs_root: str | Path,
    outcomes_path: str | Path,
    entity_graph_path: str | Path,
    entity_identifier_path: str | Path,
    companyfacts_root: Optional[str | Path] = None,
    case_count: Optional[int] = 12,
    lookback_days: int = 120,
    alignment_horizon_days: int = 365,
    families: Optional[Sequence[str]] = None,
    max_cases_per_company: int = 1,
    top_plans: int = 3,
    max_candidates: int = 12,
    strict_evidence: bool = False,
    precedent_top_k: int = 0,
    planner_random_seed: int = 7,
    limit: Optional[int] = None,
    raw_timeseries_path: str | Path = "data/inputs_layer/raw_timeseries.parquet",
    macro_timeseries_path: Optional[str | Path] = None,
    event_store_path: str | Path = "data/inputs_layer/event_store.parquet",
    facts_path: str | Path = "data/inputs_layer/extracted_fact_registry_validity",
    ownership_summary_path: str | Path = "data/inputs_layer/ownership_13f_summary.parquet",
    issuer_ratings_path: str | Path = "data/inputs_layer/issuer_rating_history.parquet",
    entity_table_path: str | Path = "data/inputs_layer/entity.parquet",
    skip_timeseries: bool = False,
    skip_macro: bool = False,
    skip_events: bool = False,
    skip_peers: bool = False,
    debug: bool = False,
    cache_facts: bool = True,
    cache_events: bool = True,
    cache_timeseries: bool = True,
    cache_ownership: bool = True,
    cache_ratings: bool = True,
    snapshot_cache_dir: Optional[str | Path] = None,
    progress_logger: Optional[Callable[[Dict[str, Any]], None]] = None,
    min_non_missing_core_features: int = 3,
    selection_multiplier: int = 10,
    max_candidate_cases: Optional[int] = None,
    historical_backfill_mode: bool = True,
    exclude_report_paths: Optional[Sequence[str | Path]] = None,
    fixed_case_paths: Optional[Sequence[str | Path]] = None,
    action_support_manifest_path: Optional[str | Path] = None,
    config_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    outcomes_path = Path(outcomes_path)
    runs_root = Path(runs_root)
    snapshot_cache_path = Path(snapshot_cache_dir) if snapshot_cache_dir else (runs_root / "_historical_snapshot_cache")
    snapshot_cache_path.mkdir(parents=True, exist_ok=True)
    selection_diagnostics = _summarize_historical_selection_pool(
        outcomes_path=outcomes_path,
        families=families,
        alignment_horizon_days=alignment_horizon_days,
    )
    action_support_summary = _load_action_support_summary(
        outcomes_path=outcomes_path,
        manifest_path=Path(action_support_manifest_path) if action_support_manifest_path else None,
    )
    requested_case_count = int(case_count) if case_count is not None else None
    selection_mode = "dynamic_selection"
    if fixed_case_paths:
        selected_cases = _load_fixed_historical_cases(
            fixed_case_paths,
            case_count=requested_case_count,
        )
        target_supported_cases = len(selected_cases) if requested_case_count is None else max(1, requested_case_count)
        candidate_case_count = len(selected_cases)
        selection_mode = "fixed_cases"
        case_support_prefilter: Dict[str, Dict[str, Any]] = {}
        family_prefilter_summary: Dict[str, Dict[str, Any]] = {}
    else:
        target_supported_cases = max(1, requested_case_count or 12)
        candidate_case_count = max(
            target_supported_cases,
            int(max_candidate_cases or (target_supported_cases * max(1, int(selection_multiplier)))),
        )
        exclude_case_keys = _load_excluded_historical_case_keys(exclude_report_paths)
        selected_cases = _select_historical_cases(
            outcomes_path=outcomes_path,
            entity_identifier_path=Path(entity_identifier_path),
            entity_table_path=Path(entity_table_path),
            case_count=candidate_case_count,
            lookback_days=lookback_days,
            alignment_horizon_days=alignment_horizon_days,
            families=families,
            max_cases_per_company=max_cases_per_company,
            limit=limit,
            exclude_case_keys=exclude_case_keys,
        )
        case_support_prefilter = _prefilter_case_support(
            selected_cases,
            facts_path=Path(facts_path),
            raw_timeseries_path=Path(raw_timeseries_path),
            event_store_path=Path(event_store_path),
            ownership_summary_path=Path(ownership_summary_path),
            issuer_ratings_path=Path(issuer_ratings_path),
            historical_backfill_mode=historical_backfill_mode,
        )
        family_prefilter_summary = _summarize_case_support_by_family(selected_cases, case_support_prefilter)
        selected_cases = _prioritize_historical_cases(
            selected_cases,
            case_support_prefilter=case_support_prefilter,
            family_prefilter_summary=family_prefilter_summary,
        )
    _emit_progress(
        progress_logger,
        {
            "event": "selected_cases",
            "selected_case_count": len(selected_cases),
            "target_supported_case_count": target_supported_cases,
            "candidate_case_count": candidate_case_count,
            "snapshot_cache_dir": str(snapshot_cache_path),
            "family_prefilter_summary": family_prefilter_summary,
            "selection_mode": selection_mode,
            "selection_diagnostics": selection_diagnostics,
        },
    )
    outcomes_lookup = _load_realized_outcomes_lookup(outcomes_path)
    resolved_companyfacts_root = (
        Path(companyfacts_root)
        if companyfacts_root is not None
        else (DEFAULT_COMPANYFACTS_ROOT if DEFAULT_COMPANYFACTS_ROOT.exists() else None)
    )
    builder = CompanyStateBuilder(
        raw_timeseries_path=raw_timeseries_path,
        macro_timeseries_path=macro_timeseries_path,
        event_store_path=event_store_path,
        facts_path=facts_path,
        ownership_summary_path=ownership_summary_path,
        issuer_ratings_path=issuer_ratings_path,
        entity_graph_path=entity_graph_path,
        entity_identifier_path=entity_identifier_path,
        entity_table_path=entity_table_path,
        skip_timeseries=skip_timeseries,
        skip_macro=skip_macro,
        skip_events=skip_events,
        skip_peer_context=skip_peers,
        debug=debug,
        cache_facts=cache_facts,
        cache_events=cache_events,
        cache_timeseries=cache_timeseries,
        cache_ownership=cache_ownership,
        cache_ratings=cache_ratings,
        historical_backfill_mode=historical_backfill_mode,
        companyfacts_root=resolved_companyfacts_root,
        enable_market_relevant_smart_normalized_inputs=True,
    )
    alias_overrides = _build_historical_alias_overrides(selected_cases)
    snapshot_loader = _cached_snapshot_loader(
        builder,
        cache_dir=snapshot_cache_path,
        progress_logger=progress_logger,
        alias_overrides=alias_overrides,
    )
    RecommendationRunStore, _, _, _, _, _, create_recommendation_run, _ = _run_store_bindings()
    run_store = RecommendationRunStore(root=runs_root)

    cases: List[Dict[str, Any]] = []
    supported_case_count = 0
    total_cases = len(selected_cases)
    for index, spec in enumerate(selected_cases, start=1):
        if supported_case_count >= target_supported_cases:
            _emit_progress(
                progress_logger,
                {
                    "event": "target_supported_case_count_reached",
                    "attempted_case_count": len(cases),
                    "supported_case_count": supported_case_count,
                    "target_supported_case_count": target_supported_cases,
                },
            )
            break
        company_id = str(spec["company_id"])
        source_company_id = str(spec.get("source_company_id") or company_id)
        as_of_time = str(spec["as_of_time"])
        company_aliases = list(alias_overrides.get((company_id, as_of_time), []) or [])
        anchor_action_id = str(spec["anchor_action_id"])
        anchor_family = str(spec["anchor_action_family"])
        anchor_date = str(spec["anchor_action_date"])
        anchor_action_support = resolve_action_support(
            action_id=anchor_action_id,
            action_family=anchor_family,
            support_report=action_support_summary,
        )
        case_started_at = time.perf_counter()
        _emit_progress(
            progress_logger,
            {
                "event": "case_start",
                "index": index,
                "total": total_cases,
                "company_id": company_id,
                "as_of_time": as_of_time,
                "anchor_action_id": anchor_action_id,
                "anchor_action_family": anchor_family,
            },
        )
        metadata = {
            "historical_eval": {
                "anchor_action_id": anchor_action_id,
                "anchor_action_family": anchor_family,
                "anchor_action_date": anchor_date,
                "lookback_days": int(lookback_days),
                "alignment_horizon_days": int(alignment_horizon_days),
            }
        }
        prefilter_support = dict(case_support_prefilter.get(_historical_case_key(spec), {}) or {})
        try:
            if not _prefilter_support_is_eligible(prefilter_support):
                cases.append(
                    {
                        "company_id": company_id,
                        "source_company_id": source_company_id,
                        "as_of_time": as_of_time,
                        "anchor_action_id": anchor_action_id,
                        "anchor_action_family": anchor_family,
                        "anchor_action_date": anchor_date,
                        "anchor_action_support": anchor_action_support,
                        "unsupported_reason": "prefilter_low_source_support",
                        "prefilter_support": prefilter_support,
                    }
                )
                _emit_progress(
                    progress_logger,
                    {
                        "event": "case_skipped",
                        "index": index,
                        "total": total_cases,
                        "company_id": company_id,
                        "unsupported_reason": "prefilter_low_source_support",
                        "prefilter_support": prefilter_support,
                    },
                )
                continue
            prebuilt_snapshot = snapshot_loader(company_id, pd.Timestamp(as_of_time).to_pydatetime())
            snapshot_coverage = _snapshot_coverage_summary(prebuilt_snapshot)
            if not _snapshot_has_meaningful_coverage(
                snapshot_coverage,
                min_non_missing_core_features=min_non_missing_core_features,
            ):
                cases.append(
                    {
                        "company_id": company_id,
                        "source_company_id": source_company_id,
                        "as_of_time": as_of_time,
                        "anchor_action_id": anchor_action_id,
                        "anchor_action_family": anchor_family,
                        "anchor_action_date": anchor_date,
                        "anchor_action_support": anchor_action_support,
                        "unsupported_reason": "insufficient_snapshot_coverage",
                        "prefilter_support": prefilter_support,
                        "snapshot_coverage": snapshot_coverage,
                    }
                )
                _emit_progress(
                    progress_logger,
                    {
                        "event": "case_skipped",
                        "index": index,
                        "total": total_cases,
                        "company_id": company_id,
                        "unsupported_reason": "insufficient_snapshot_coverage",
                        "snapshot_coverage": snapshot_coverage,
                    },
                )
                continue
            _emit_progress(
                progress_logger,
                {
                    "event": "run_create_start",
                    "index": index,
                    "total": total_cases,
                    "company_id": company_id,
                    "as_of_time": as_of_time,
                },
            )
            run_id = create_recommendation_run(
                company_id=company_id,
                as_of_time=as_of_time,
                run_store=run_store,
                snapshot_loader=snapshot_loader,
                entity_graph_path=entity_graph_path,
                entity_identifier_path=entity_identifier_path,
                company_aliases=company_aliases,
                skip_as_of_lower_bound_validation=historical_backfill_mode,
                planner_random_seed=planner_random_seed,
                metadata=metadata,
            )
            _emit_progress(
                progress_logger,
                {
                    "event": "run_create_complete",
                    "index": index,
                    "total": total_cases,
                    "company_id": company_id,
                    "run_id": run_id,
                    "elapsed_seconds": round(time.perf_counter() - case_started_at, 3),
                },
            )
            _emit_progress(
                progress_logger,
                {
                    "event": "run_execute_start",
                    "index": index,
                    "total": total_cases,
                    "company_id": company_id,
                    "run_id": run_id,
                },
            )
            summary = execute_recommendation_run(
                run_id=run_id,
                runs_root=runs_root,
                snapshot_loader=snapshot_loader,
                entity_identifier_path=entity_identifier_path,
                max_candidates=max_candidates,
                strict_evidence=strict_evidence,
                precedent_top_k=precedent_top_k,
                outcomes_path=outcomes_path,
                config_path=config_path,
                top_plans=top_plans,
            )
            _emit_progress(
                progress_logger,
                {
                    "event": "run_execute_complete",
                    "index": index,
                    "total": total_cases,
                    "company_id": company_id,
                    "run_id": summary.get("run_id"),
                    "elapsed_seconds": round(time.perf_counter() - case_started_at, 3),
                },
            )
            artifacts = dict(summary.get("artifacts", {}) or {})
            package_path = artifacts.get("RecommendationPackage")
            dossier_path = artifacts.get("BoardReadyDossier")
            precedent_index_path = artifacts.get("PrecedentIndex")
            package = json.loads(Path(package_path).read_text()) if package_path else {}
            dossier = json.loads(Path(dossier_path).read_text()) if dossier_path else {}
            precedent_index = json.loads(Path(precedent_index_path).read_text()) if precedent_index_path else {}
            top_action_ids = _top_action_ids(package)
            recommended_action_support = [
                resolve_action_support(
                    action_id=action_id,
                    action_family=_action_family(action_id),
                    support_report=action_support_summary,
                )
                for action_id in top_action_ids
            ]
            alignment = _score_ex_post_alignment(
                company_id=source_company_id or company_id,
                as_of_time=as_of_time,
                recommended_action_ids=top_action_ids,
                outcomes_lookup=outcomes_lookup,
                alignment_horizon_days=alignment_horizon_days,
                anchor_action_id=anchor_action_id,
                anchor_action_family=anchor_family,
                anchor_action_support=anchor_action_support,
                recommended_action_support=recommended_action_support,
            )
            precedent_ranking = _score_precedent_ranking(
                precedent_index=precedent_index,
                anchor_action_id=anchor_action_id,
                anchor_action_family=anchor_family,
                anchor_action_support=anchor_action_support,
            )
            cases.append(
                {
                    "company_id": company_id,
                    "source_company_id": source_company_id,
                    "as_of_time": as_of_time,
                    "run_id": summary.get("run_id"),
                    "anchor_action_id": anchor_action_id,
                    "anchor_action_family": anchor_family,
                    "anchor_action_date": anchor_date,
                    "anchor_action_support": anchor_action_support,
                    "recommended_posture": package.get("recommended_posture"),
                    "top_action_ids": top_action_ids,
                    "recommended_action_support": recommended_action_support,
                    "executive_summary": dossier.get("executive_summary"),
                    "problem_statement": ((dossier.get("recommendation_thesis", {}) or {}).get("problem_statement")),
                    "why_this_plan": ((dossier.get("recommendation_thesis", {}) or {}).get("why_this_plan")),
                    "prefilter_support": prefilter_support,
                    "historical_alignment": alignment,
                    "precedent_ranking": precedent_ranking,
                    "snapshot_coverage": snapshot_coverage,
                    "artifacts": artifacts,
                }
            )
            if not top_action_ids:
                cases[-1]["unsupported_reason"] = "no_feasible_plan_generated"
                _emit_progress(
                    progress_logger,
                    {
                        "event": "case_unsupported",
                        "index": index,
                        "total": total_cases,
                        "company_id": company_id,
                        "run_id": summary.get("run_id"),
                        "unsupported_reason": "no_feasible_plan_generated",
                    },
                )
                continue
            supported_case_count += 1
            _emit_progress(
                progress_logger,
                {
                    "event": "case_complete",
                    "index": index,
                    "total": total_cases,
                    "company_id": company_id,
                    "run_id": summary.get("run_id"),
                    "elapsed_seconds": round(time.perf_counter() - case_started_at, 3),
                    "top_action_ids": top_action_ids,
                    "alignment_reason": alignment.get("reason"),
                    "alignment_score": alignment.get("score"),
                },
            )
        except Exception as exc:
            cases.append(
                {
                    "company_id": company_id,
                    "source_company_id": source_company_id,
                    "as_of_time": as_of_time,
                    "anchor_action_id": anchor_action_id,
                    "anchor_action_family": anchor_family,
                    "anchor_action_date": anchor_date,
                    "anchor_action_support": anchor_action_support,
                    "prefilter_support": prefilter_support,
                    "error": str(exc),
                }
            )
            _emit_progress(
                progress_logger,
                {
                    "event": "case_error",
                    "index": index,
                    "total": total_cases,
                    "company_id": company_id,
                    "elapsed_seconds": round(time.perf_counter() - case_started_at, 3),
                    "error": str(exc),
                },
            )

    aggregate = _aggregate_historical_cases(cases)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runs_root": str(runs_root),
        "outcomes_path": str(outcomes_path),
        "case_count_requested": int(target_supported_cases),
        "candidate_case_count": int(candidate_case_count),
        "runs_analyzed": len(cases),
        "supported_case_count": supported_case_count,
        "family_prefilter_summary": family_prefilter_summary,
        "selection_mode": selection_mode,
        "selection_diagnostics": selection_diagnostics,
        "action_support_summary": action_support_summary,
        "aggregate": aggregate,
        "cases": cases,
    }


def _historical_case_key(spec: Dict[str, Any]) -> str:
    return "|".join(
        [
            str(spec.get("company_id") or ""),
            str(spec.get("as_of_time") or ""),
            str(spec.get("anchor_action_id") or ""),
        ]
    )


def _prefilter_support_is_eligible(profile: Dict[str, Any]) -> bool:
    if not profile:
        return True
    return bool(profile.get("estimated_supported", False))


