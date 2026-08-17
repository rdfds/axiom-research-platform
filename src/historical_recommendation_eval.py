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


def _prefilter_case_support(
    cases: Sequence[Dict[str, Any]],
    *,
    facts_path: Path,
    raw_timeseries_path: Path,
    event_store_path: Path,
    ownership_summary_path: Path,
    issuer_ratings_path: Path,
    historical_backfill_mode: bool,
) -> Dict[str, Dict[str, Any]]:
    profiles: Dict[str, Dict[str, Any]] = {}
    if not cases:
        return profiles
    rows: List[Dict[str, Any]] = []
    for spec in cases:
        case_key = _historical_case_key(spec)
        rows.append(
            {
                "case_key": case_key,
                "company_id": str(spec.get("company_id") or ""),
                "source_company_id": str(spec.get("source_company_id") or ""),
                "anchor_action_family": str(spec.get("anchor_action_family") or ""),
                "as_of_time": pd.Timestamp(spec.get("as_of_time")).tz_convert("UTC").to_pydatetime(),
            }
        )
        profiles[case_key] = {
            "facts_hits": 0,
            "timeseries_hits": 0,
            "events_hits": 0,
            "ownership_hits": 0,
            "ratings_hits": 0,
            "strong_source_count": 0,
            "total_hits": 0,
            "estimated_supported": False,
            "score": 0.0,
            "support_bucket": "none",
        }
    case_df = pd.DataFrame(rows)
    con = duckdb.connect()
    con.register("hist_cases", case_df)

    _update_prefilter_hits(
        profiles,
        "facts_hits",
        _query_source_hits(
            con,
            source_path=facts_path,
            source_name="facts",
            query=_build_facts_prefilter_query(
                source_path=facts_path,
                historical_backfill_mode=historical_backfill_mode,
            ),
        ),
    )
    _update_prefilter_hits(
        profiles,
        "timeseries_hits",
        _query_source_hits(
            con,
            source_path=raw_timeseries_path,
            source_name="timeseries",
            query=_build_timeseries_prefilter_query(
                source_path=raw_timeseries_path,
                historical_backfill_mode=historical_backfill_mode,
            ),
        ),
    )
    _update_prefilter_hits(
        profiles,
        "events_hits",
        _query_source_hits(
            con,
            source_path=event_store_path,
            source_name="events",
            query=_build_events_prefilter_query(
                source_path=event_store_path,
                historical_backfill_mode=historical_backfill_mode,
            ),
        ),
    )
    _update_prefilter_hits(
        profiles,
        "ownership_hits",
        _query_source_hits(
            con,
            source_path=ownership_summary_path,
            source_name="ownership",
            query=_build_ownership_prefilter_query(
                source_path=ownership_summary_path,
                historical_backfill_mode=historical_backfill_mode,
            ),
        ),
    )
    _update_prefilter_hits(
        profiles,
        "ratings_hits",
        _query_source_hits(
            con,
            source_path=issuer_ratings_path,
            source_name="ratings",
            query=_build_ratings_prefilter_query(
                source_path=issuer_ratings_path,
                historical_backfill_mode=historical_backfill_mode,
            ),
        ),
    )
    con.close()

    for profile in profiles.values():
        strong_source_count = sum(
            1
            for key in ("facts_hits", "timeseries_hits", "events_hits", "ratings_hits")
            if int(profile.get(key, 0) or 0) > 0
        )
        total_hits = sum(int(profile.get(key, 0) or 0) for key in ("facts_hits", "timeseries_hits", "events_hits", "ownership_hits", "ratings_hits"))
        score = (
            min(int(profile.get("facts_hits", 0) or 0), 50) * 0.08
            + min(int(profile.get("timeseries_hits", 0) or 0), 200) * 0.02
            + min(int(profile.get("events_hits", 0) or 0), 50) * 0.05
            + min(int(profile.get("ownership_hits", 0) or 0), 10) * 0.05
            + min(int(profile.get("ratings_hits", 0) or 0), 10) * 0.1
            + (1.5 if int(profile.get("facts_hits", 0) or 0) > 0 else 0.0)
            + (1.0 if int(profile.get("timeseries_hits", 0) or 0) > 0 else 0.0)
            + (1.0 if int(profile.get("events_hits", 0) or 0) > 0 else 0.0)
            + (0.5 if int(profile.get("ratings_hits", 0) or 0) > 0 else 0.0)
        )
        estimated_supported = bool(
            (int(profile.get("facts_hits", 0) or 0) > 0 or int(profile.get("events_hits", 0) or 0) > 0 or int(profile.get("ratings_hits", 0) or 0) > 0)
            and (int(profile.get("facts_hits", 0) or 0) > 0 or int(profile.get("timeseries_hits", 0) or 0) > 0)
        )
        if estimated_supported and strong_source_count >= 3:
            support_bucket = "strong"
        elif estimated_supported:
            support_bucket = "moderate"
        elif total_hits > 0:
            support_bucket = "weak"
        else:
            support_bucket = "none"
        profile.update(
            {
                "strong_source_count": strong_source_count,
                "total_hits": total_hits,
                "estimated_supported": estimated_supported,
                "score": round(float(score), 3),
                "support_bucket": support_bucket,
            }
        )
    return profiles


def _summarize_case_support_by_family(
    cases: Sequence[Dict[str, Any]],
    case_support_prefilter: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for spec in cases:
        family = str(spec.get("anchor_action_family") or "")
        profile = dict(case_support_prefilter.get(_historical_case_key(spec), {}) or {})
        buckets.setdefault(family, []).append(profile)
    summary: Dict[str, Dict[str, Any]] = {}
    for family, profiles in buckets.items():
        candidate_count = len(profiles)
        estimated_supported = sum(1 for profile in profiles if bool(profile.get("estimated_supported")))
        mean_score = round(sum(float(profile.get("score", 0.0) or 0.0) for profile in profiles) / candidate_count, 3) if candidate_count else 0.0
        summary[family] = {
            "candidate_count": candidate_count,
            "estimated_supported_count": estimated_supported,
            "estimated_supported_rate": round(estimated_supported / candidate_count, 3) if candidate_count else 0.0,
            "mean_prefilter_score": mean_score,
        }
    return dict(
        sorted(
            summary.items(),
            key=lambda item: (
                -float(item[1].get("estimated_supported_rate", 0.0) or 0.0),
                -float(item[1].get("mean_prefilter_score", 0.0) or 0.0),
                item[0],
            ),
        )
    )


def _prioritize_historical_cases(
    cases: Sequence[Dict[str, Any]],
    *,
    case_support_prefilter: Dict[str, Dict[str, Any]],
    family_prefilter_summary: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    def _sort_key(spec: Dict[str, Any]) -> Tuple[Any, ...]:
        profile = dict(case_support_prefilter.get(_historical_case_key(spec), {}) or {})
        family = str(spec.get("anchor_action_family") or "")
        family_summary = dict(family_prefilter_summary.get(family, {}) or {})
        as_of_ts = pd.Timestamp(spec.get("as_of_time") or "1970-01-01T00:00:00Z").timestamp()
        return (
            -int(bool(profile.get("estimated_supported"))),
            -float(family_summary.get("estimated_supported_rate", 0.0) or 0.0),
            -float(profile.get("score", 0.0) or 0.0),
            -int(profile.get("strong_source_count", 0) or 0),
            -float(as_of_ts),
            str(spec.get("company_id") or ""),
        )

    return sorted(list(cases), key=_sort_key, reverse=False)


def _query_source_hits(
    con: duckdb.DuckDBPyConnection,
    *,
    source_path: Path,
    source_name: str,
    query: Optional[str],
) -> Dict[str, int]:
    if query is None or not source_path.exists():
        return {}
    try:
        frame = con.execute(query).df()
    except Exception:
        return {}
    if frame.empty:
        return {}
    case_key_col = "case_key"
    hits_col = f"{source_name}_hits"
    out: Dict[str, int] = {}
    for row in frame.itertuples(index=False):
        key = str(getattr(row, case_key_col))
        out[key] = int(getattr(row, hits_col) or 0)
    return out


def _update_prefilter_hits(
    profiles: Dict[str, Dict[str, Any]],
    field_name: str,
    source_hits: Dict[str, int],
) -> None:
    for case_key, hits in source_hits.items():
        profile = profiles.get(case_key)
        if profile is None:
            continue
        profile[field_name] = int(hits or 0)


def _build_facts_prefilter_query(*, source_path: Path, historical_backfill_mode: bool) -> Optional[str]:
    source_sql = _parquet_source_sql(source_path)
    if not source_sql:
        return None
    ingested_clause = "" if historical_backfill_mode else "AND (f.ingested_at IS NULL OR try_cast(f.ingested_at AS TIMESTAMP) <= c.as_of_time)"
    return f"""
        SELECT c.case_key, COUNT(*) AS facts_hits
        FROM hist_cases c
        JOIN read_parquet({source_sql}, union_by_name=True) f
          ON CAST(f.entity_id AS VARCHAR) = c.company_id
        WHERE (f.published_at IS NULL OR try_cast(f.published_at AS TIMESTAMP) <= c.as_of_time)
          AND (f.effective_at IS NULL OR try_cast(f.effective_at AS TIMESTAMP) <= c.as_of_time)
          AND (f.valid_from IS NULL OR try_cast(f.valid_from AS TIMESTAMP) <= c.as_of_time)
          AND (f.valid_to IS NULL OR try_cast(f.valid_to AS TIMESTAMP) > c.as_of_time)
          {ingested_clause}
        GROUP BY 1
    """


def _build_timeseries_prefilter_query(*, source_path: Path, historical_backfill_mode: bool) -> Optional[str]:
    source_sql = _parquet_source_sql(source_path)
    if not source_sql:
        return None
    ingested_clause = (
        ""
        if historical_backfill_mode
        else """
          AND (
                (ts.ingested_at IS NULL OR try_cast(ts.ingested_at AS TIMESTAMP) <= c.as_of_time)
                AND (ts.ingestion_time IS NULL OR try_cast(ts.ingestion_time AS TIMESTAMP) <= c.as_of_time)
              )
        """
    )
    return f"""
        SELECT c.case_key, COUNT(*) AS timeseries_hits
        FROM hist_cases c
        JOIN read_parquet({source_sql}, union_by_name=True) ts
          ON (
               CAST(ts.entity_id AS VARCHAR) = c.company_id
               OR CAST(ts.company_id AS VARCHAR) = c.company_id
             )
        WHERE (
                coalesce(
                    try_cast(ts.published_at AS TIMESTAMP),
                    try_cast(ts.available_time AS TIMESTAMP),
                    try_cast(ts.trade_date AS TIMESTAMP),
                    try_cast(ts.event_time AS TIMESTAMP)
                ) IS NULL
                OR coalesce(
                    try_cast(ts.published_at AS TIMESTAMP),
                    try_cast(ts.available_time AS TIMESTAMP),
                    try_cast(ts.trade_date AS TIMESTAMP),
                    try_cast(ts.event_time AS TIMESTAMP)
                ) <= c.as_of_time
              )
          {ingested_clause}
        GROUP BY 1
    """


def _build_events_prefilter_query(*, source_path: Path, historical_backfill_mode: bool) -> Optional[str]:
    source_sql = _parquet_source_sql(source_path)
    if not source_sql:
        return None
    ingested_clause = (
        ""
        if historical_backfill_mode
        else """
          AND (
                (e.ingested_at IS NULL OR try_cast(e.ingested_at AS TIMESTAMP) <= c.as_of_time)
                AND (e.created_at IS NULL OR try_cast(e.created_at AS TIMESTAMP) <= c.as_of_time)
              )
        """
    )
    return f"""
        SELECT c.case_key, COUNT(*) AS events_hits
        FROM hist_cases c
        JOIN read_parquet({source_sql}, union_by_name=True) e
          ON (
               CAST(e.company_id AS VARCHAR) = c.company_id
               OR CAST(e.company_id AS VARCHAR) = c.source_company_id
             )
        WHERE (
                coalesce(
                    try_cast(e.published_at AS TIMESTAMP),
                    try_cast(e.announced_at AS TIMESTAMP),
                    try_cast(e.effective_at AS TIMESTAMP)
                ) IS NULL
                OR coalesce(
                    try_cast(e.published_at AS TIMESTAMP),
                    try_cast(e.announced_at AS TIMESTAMP),
                    try_cast(e.effective_at AS TIMESTAMP)
                ) <= c.as_of_time
              )
          AND (e.effective_at IS NULL OR try_cast(e.effective_at AS TIMESTAMP) <= c.as_of_time)
          {ingested_clause}
        GROUP BY 1
    """


def _build_ownership_prefilter_query(*, source_path: Path, historical_backfill_mode: bool) -> Optional[str]:
    source_sql = _parquet_source_sql(source_path)
    if not source_sql:
        return None
    ingested_clause = "" if historical_backfill_mode else "AND (o.ingested_at IS NULL OR try_cast(o.ingested_at AS TIMESTAMP) <= c.as_of_time)"
    return f"""
        SELECT c.case_key, COUNT(*) AS ownership_hits
        FROM hist_cases c
        JOIN read_parquet({source_sql}, union_by_name=True) o
          ON CAST(o.company_id AS VARCHAR) = c.company_id
        WHERE (o.published_at IS NULL OR try_cast(o.published_at AS TIMESTAMP) <= c.as_of_time)
          AND (o.effective_at IS NULL OR try_cast(o.effective_at AS TIMESTAMP) <= c.as_of_time)
          {ingested_clause}
        GROUP BY 1
    """


def _build_ratings_prefilter_query(*, source_path: Path, historical_backfill_mode: bool) -> Optional[str]:
    source_sql = _parquet_source_sql(source_path)
    if not source_sql:
        return None
    ingested_clause = "" if historical_backfill_mode else "AND (r.ingested_at IS NULL OR try_cast(r.ingested_at AS TIMESTAMP) <= c.as_of_time)"
    return f"""
        SELECT c.case_key, COUNT(*) AS ratings_hits
        FROM hist_cases c
        JOIN read_parquet({source_sql}, union_by_name=True) r
          ON CAST(r.company_id AS VARCHAR) = c.company_id
        WHERE (r.published_at IS NULL OR try_cast(r.published_at AS TIMESTAMP) <= c.as_of_time)
          AND (r.effective_at IS NULL OR try_cast(r.effective_at AS TIMESTAMP) <= c.as_of_time)
          {ingested_clause}
        GROUP BY 1
    """


def _parquet_source_sql(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    if path.is_file():
        return _sql_quote(path.as_posix())
    files = sorted(path.glob("year=*/part.parquet"))
    if not files:
        files = sorted(path.rglob("*.parquet"))
    if not files:
        return None
    return "[" + ", ".join(_sql_quote(file.as_posix()) for file in files) + "]"


def _sql_quote(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def render_historical_recommendation_markdown(report: Dict[str, Any]) -> str:
    aggregate = dict(report.get("aggregate", {}) or {})
    family_prefilter = dict(report.get("family_prefilter_summary", {}) or {})
    action_support = dict(report.get("action_support_summary", {}) or {})
    lines = [
        "# Historical Recommendation Validation",
        "",
        f"- Supported case target: `{int(report.get('case_count_requested', 0) or 0)}`",
        f"- Candidate cases scanned: `{int(report.get('candidate_case_count', 0) or 0)}`",
        f"- Runs analyzed: `{int(report.get('runs_analyzed', 0) or 0)}`",
        f"- Supported runs: `{int(report.get('supported_case_count', 0) or 0)}`",
        f"- Completed runs: `{int(aggregate.get('completed_case_count', 0) or 0)}`",
        f"- Scored runs: `{int(aggregate.get('scored_case_count', 0) or 0)}`",
        f"- Unsupported runs: `{int(aggregate.get('unsupported_case_count', 0) or 0)}`",
        f"- Mean alignment score: `{aggregate.get('mean_alignment_score', 0.0):.3f}`",
        f"- Strong alignment rate: `{aggregate.get('strong_alignment_rate', 0.0):.3f}`",
        f"- Primary exact-match rate: `{aggregate.get('anchor_primary_exact_rate', 0.0):.3f}`",
        f"- Primary family-match rate: `{aggregate.get('anchor_primary_family_rate', 0.0):.3f}`",
        f"- Primary support-adjusted rate: `{aggregate.get('anchor_primary_support_adjusted_rate', 0.0):.3f}`",
        f"- Any exact-match rate: `{aggregate.get('future_any_exact_rate', 0.0):.3f}`",
        f"- Any family-match rate: `{aggregate.get('future_any_family_rate', 0.0):.3f}`",
        f"- Any support-adjusted rate: `{aggregate.get('future_any_support_adjusted_rate', 0.0):.3f}`",
        "",
        "## Candidate Support",
        "",
    ]
    if family_prefilter:
        for family, summary in family_prefilter.items():
            lines.append(
                f"- `{family}` candidates=`{int(summary.get('candidate_count', 0) or 0)}` "
                f"estimated_supported=`{int(summary.get('estimated_supported_count', 0) or 0)}` "
                f"rate=`{float(summary.get('estimated_supported_rate', 0.0) or 0.0):.3f}` "
                f"mean_score=`{float(summary.get('mean_prefilter_score', 0.0) or 0.0):.3f}`"
            )
    else:
        lines.append("- No candidate support profile available.")
    lines.extend([
        "",
        "## Action Data Support",
        "",
    ])
    if action_support:
        exact_status_counts = dict(action_support.get("exact_status_counts", {}) or {})
        support_mode_counts = dict(action_support.get("support_mode_counts", {}) or {})
        lines.append(f"- Exact support statuses: `{json.dumps(exact_status_counts, sort_keys=True)}`")
        lines.append(f"- Support modes: `{json.dumps(support_mode_counts, sort_keys=True)}`")
    else:
        lines.append("- No action support summary available.")
    lines.extend([
        "",
        "## Review Queue",
        "",
    ])
    for case in report.get("cases", [])[:20]:
        if case.get("error"):
            lines.append(f"- `{case.get('company_id')}` `{case.get('as_of_time')}`: error `{case.get('error')}`")
            continue
        if case.get("unsupported_reason"):
            prefilter = dict(case.get("prefilter_support", {}) or {})
            suffix = ""
            if prefilter:
                suffix = (
                    f" prefilter=`{prefilter.get('support_bucket', '')}`"
                    f" score=`{float(prefilter.get('score', 0.0) or 0.0):.3f}`"
                )
            lines.append(
                f"- `{case.get('company_id')}` `{case.get('as_of_time')}`: unsupported "
                f"`{case.get('unsupported_reason')}`{suffix}"
            )
            continue
        hist = dict(case.get("historical_alignment", {}) or {})
        score_value = hist.get("score")
        score_text = "n/a" if score_value is None else f"{float(score_value):.3f}"
        anchor_support = dict(case.get("anchor_action_support", {}) or {})
        anchor_support_text = str(anchor_support.get("support_mode") or "")
        lines.append(
            f"- `{case['company_id']}` `{case.get('as_of_time')}` "
            f"`{','.join(case.get('top_action_ids', []) or [])}` "
            f"vs `{case.get('anchor_action_id')}` "
            f"score=`{score_text}` reason=`{hist.get('reason', '')}` support=`{anchor_support_text}`"
        )
    return "\n".join(lines).strip() + "\n"


def _select_historical_cases(
    *,
    outcomes_path: Path,
    entity_identifier_path: Path,
    entity_table_path: Path,
    case_count: int,
    lookback_days: int,
    alignment_horizon_days: int,
    families: Optional[Sequence[str]],
    max_cases_per_company: int,
    limit: Optional[int],
    exclude_case_keys: Optional[Set[Tuple[str, pd.Timestamp, str]]] = None,
) -> List[Dict[str, Any]]:
    family_values = list(families or DEFAULT_FAMILIES)
    max_event_date = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=max(1, int(alignment_horizon_days)))
    max_event_date_naive = max_event_date.tz_convert(None).to_pydatetime()
    query = """
        SELECT
            CAST(company_id AS VARCHAR) AS company_id,
            CAST(ticker AS VARCHAR) AS ticker,
            CAST(action_date AS TIMESTAMP) AS action_date,
            CAST(normalized_action_id AS VARCHAR) AS normalized_action_id,
            CAST(normalized_action_family AS VARCHAR) AS normalized_action_family
        FROM read_parquet(?)
        WHERE action_date IS NOT NULL
          AND normalized_action_id IS NOT NULL
          AND normalized_action_family IS NOT NULL
          AND action_date <= ?
    """
    params: List[Any] = [str(outcomes_path), max_event_date_naive]
    if family_values:
        query += " AND normalized_action_family IN (" + ",".join(["?"] * len(family_values)) + ")"
        params.extend(family_values)
    query += " ORDER BY action_date DESC"
    frame = duckdb.execute(query, params).df()
    if frame.empty:
        return []
    frame["action_date"] = pd.to_datetime(frame["action_date"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["company_id", "action_date", "normalized_action_id", "normalized_action_family"])
    frame = _filter_excluded_historical_cases(frame, exclude_case_keys)
    if frame.empty:
        return []
    frame = _resolve_supported_historical_entities(
        frame=frame,
        entity_identifier_path=entity_identifier_path,
        entity_table_path=entity_table_path,
        lookback_days=lookback_days,
    )
    if frame.empty:
        return []
    frame = frame.sort_values("action_date", ascending=False).reset_index(drop=True)
    if limit:
        frame = frame.head(int(limit)).reset_index(drop=True)
    return _select_historical_cases_from_frame(
        frame=frame,
        case_count=case_count,
        lookback_days=lookback_days,
        max_cases_per_company=max_cases_per_company,
    )


