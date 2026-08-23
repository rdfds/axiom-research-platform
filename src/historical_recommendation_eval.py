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
            f"- `{case.get('company_id')}` `{case.get('as_of_time')}` "
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


def _summarize_historical_selection_pool(
    *,
    outcomes_path: Path,
    families: Optional[Sequence[str]],
    alignment_horizon_days: int,
) -> Dict[str, Any]:
    family_values = list(families or DEFAULT_FAMILIES)
    max_event_date = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=max(1, int(alignment_horizon_days)))
    max_event_date_naive = max_event_date.tz_convert(None).to_pydatetime()
    query = """
        SELECT
            CAST(normalized_action_family AS VARCHAR) AS normalized_action_family,
            COUNT(*) AS row_count,
            SUM(CASE WHEN normalized_action_id IS NOT NULL THEN 1 ELSE 0 END) AS with_action_id_count,
            SUM(CASE WHEN normalized_action_id IS NULL THEN 1 ELSE 0 END) AS missing_action_id_count
        FROM read_parquet(?)
        WHERE action_date IS NOT NULL
          AND normalized_action_family IS NOT NULL
          AND action_date <= ?
    """
    params: List[Any] = [str(outcomes_path), max_event_date_naive]
    if family_values:
        query += " AND normalized_action_family IN (" + ",".join(["?"] * len(family_values)) + ")"
        params.extend(family_values)
    query += " GROUP BY 1 ORDER BY 1"
    frame = duckdb.execute(query, params).df()
    if frame.empty:
        return {
            "families": family_values,
            "total_rows": 0,
            "with_action_id_count": 0,
            "missing_action_id_count": 0,
            "family_counts": {},
        }

    family_counts: Dict[str, Dict[str, int]] = {}
    total_rows = 0
    with_action_id_count = 0
    missing_action_id_count = 0
    for row in frame.itertuples(index=False):
        family = str(row.normalized_action_family or "")
        row_count = int(row.row_count or 0)
        present = int(row.with_action_id_count or 0)
        missing = int(row.missing_action_id_count or 0)
        family_counts[family] = {
            "row_count": row_count,
            "with_action_id_count": present,
            "missing_action_id_count": missing,
        }
        total_rows += row_count
        with_action_id_count += present
        missing_action_id_count += missing

    return {
        "families": family_values,
        "total_rows": total_rows,
        "with_action_id_count": with_action_id_count,
        "missing_action_id_count": missing_action_id_count,
        "family_counts": family_counts,
    }


def _load_action_support_summary(
    *,
    outcomes_path: Path,
    manifest_path: Optional[Path],
) -> Dict[str, Any]:
    if manifest_path and manifest_path.exists():
        try:
            manifest = load_action_support_report(manifest_path)
            if str(manifest.get("outcomes_path") or "") == str(outcomes_path):
                return manifest
        except Exception:
            pass
    return build_action_support_report(outcomes_path=outcomes_path)


def _select_historical_cases_from_frame(
    *,
    frame: pd.DataFrame,
    case_count: int,
    lookback_days: int,
    max_cases_per_company: int,
) -> List[Dict[str, Any]]:
    if frame.empty or case_count <= 0:
        return []
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    company_counts: Dict[str, int] = {}
    for row in frame.itertuples(index=False):
        company_id = str(row.company_id)
        if company_counts.get(company_id, 0) >= max(1, int(max_cases_per_company)):
            continue
        family = str(row.normalized_action_family or "")
        resolved_company_id = str(getattr(row, "resolved_company_id", "") or company_id)
        spec = {
            "company_id": resolved_company_id,
            "source_company_id": company_id,
            "ticker": str(getattr(row, "ticker", "") or ""),
            "mapping_method": str(getattr(row, "mapping_method", "") or ""),
            "anchor_action_date": pd.Timestamp(row.action_date).tz_convert("UTC").isoformat(),
            "anchor_action_id": str(row.normalized_action_id or ""),
            "anchor_action_family": family,
            "as_of_time": (pd.Timestamp(row.action_date).tz_convert("UTC") - pd.Timedelta(days=max(1, int(lookback_days)))).isoformat(),
        }
        buckets.setdefault(family, []).append(spec)
        company_counts[company_id] = company_counts.get(company_id, 0) + 1
    ordered_families = sorted(buckets.keys(), key=lambda key: (-len(buckets[key]), key))
    selected: List[Dict[str, Any]] = []
    while len(selected) < case_count:
        made_progress = False
        for family in ordered_families:
            bucket = buckets.get(family, [])
            if not bucket:
                continue
            selected.append(bucket.pop(0))
            made_progress = True
            if len(selected) >= case_count:
                break
        if not made_progress:
            break
    return selected


def _load_fixed_historical_cases(
    report_paths: Sequence[str | Path],
    *,
    case_count: Optional[int] = None,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    max_cases = max(1, int(case_count)) if case_count is not None else None
    for path_like in report_paths:
        path = Path(path_like)
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        for raw_case in payload.get("cases", []) or []:
            spec = _normalize_fixed_historical_case(raw_case)
            if spec is None:
                continue
            case_key = _historical_case_key(spec)
            if case_key in seen:
                continue
            seen.add(case_key)
            selected.append(spec)
            if max_cases is not None and len(selected) >= max_cases:
                return selected
    return selected


def _normalize_fixed_historical_case(raw_case: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    company_id = str(raw_case.get("company_id") or "").strip()
    if not company_id:
        return None
    anchor_action_id = str(raw_case.get("anchor_action_id") or "").strip()
    anchor_action_family = str(raw_case.get("anchor_action_family") or "").strip()
    anchor_action_date_raw = raw_case.get("anchor_action_date")
    as_of_time_raw = raw_case.get("as_of_time")
    if not anchor_action_id or not anchor_action_family or not anchor_action_date_raw or not as_of_time_raw:
        return None
    anchor_action_date = pd.to_datetime(anchor_action_date_raw, utc=True, errors="coerce")
    as_of_time = pd.to_datetime(as_of_time_raw, utc=True, errors="coerce")
    if pd.isna(anchor_action_date) or pd.isna(as_of_time):
        return None
    return {
        "company_id": company_id,
        "source_company_id": str(raw_case.get("source_company_id") or company_id),
        "ticker": str(raw_case.get("ticker") or ""),
        "mapping_method": str(raw_case.get("mapping_method") or ""),
        "anchor_action_date": pd.Timestamp(anchor_action_date).isoformat(),
        "anchor_action_id": anchor_action_id,
        "anchor_action_family": anchor_action_family,
        "as_of_time": pd.Timestamp(as_of_time).isoformat(),
    }


def _load_excluded_historical_case_keys(
    report_paths: Optional[Sequence[str | Path]],
) -> Set[Tuple[str, pd.Timestamp, str]]:
    excluded: Set[Tuple[str, pd.Timestamp, str]] = set()
    for path_like in report_paths or []:
        path = Path(path_like)
        if not path.exists():
            continue
        try:
            report = json.loads(path.read_text())
        except Exception:
            continue
        for case in report.get("cases", []) or []:
            company_id = str(case.get("source_company_id") or case.get("company_id") or "").strip()
            action_id = str(case.get("anchor_action_id") or "").strip()
            action_date_raw = case.get("anchor_action_date")
            if not company_id or not action_id or not action_date_raw:
                continue
            action_date = pd.to_datetime(action_date_raw, utc=True, errors="coerce")
            if pd.isna(action_date):
                continue
            excluded.add((company_id, pd.Timestamp(action_date), action_id))
    return excluded


def _filter_excluded_historical_cases(
    frame: pd.DataFrame,
    exclude_case_keys: Optional[Set[Tuple[str, pd.Timestamp, str]]],
) -> pd.DataFrame:
    if frame.empty or not exclude_case_keys:
        return frame
    keep_mask = []
    for row in frame.itertuples(index=False):
        key = (
            str(row.company_id),
            pd.Timestamp(row.action_date).tz_convert("UTC"),
            str(row.normalized_action_id),
        )
        keep_mask.append(key not in exclude_case_keys)
    return frame.loc[keep_mask].reset_index(drop=True)


def _resolve_supported_historical_entities(
    *,
    frame: pd.DataFrame,
    entity_identifier_path: Path,
    entity_table_path: Path,
    lookback_days: int,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    try:
        entity_df = duckdb.execute(
            "SELECT CAST(entity_id AS VARCHAR) AS entity_id FROM read_parquet(?)",
            [str(entity_table_path)],
        ).df()
    except Exception:
        return frame.iloc[0:0].copy()
    valid_entity_ids = set(entity_df["entity_id"].astype(str))
    if not valid_entity_ids:
        return frame.iloc[0:0].copy()
    try:
        ticker_df = duckdb.execute(
            """
            SELECT
                CAST(entity_id AS VARCHAR) AS entity_id,
                upper(CAST(identifier_value AS VARCHAR)) AS ticker,
                CAST(valid_from AS TIMESTAMP) AS valid_from,
                CAST(valid_to AS TIMESTAMP) AS valid_to
            FROM read_parquet(?)
            WHERE identifier_type = 'ticker'
              AND identifier_value IS NOT NULL
            """,
            [str(entity_identifier_path)],
        ).df()
    except Exception:
        ticker_df = pd.DataFrame(columns=["entity_id", "ticker", "valid_from", "valid_to"])
    if not ticker_df.empty:
        ticker_df["valid_from"] = pd.to_datetime(ticker_df["valid_from"], utc=True, errors="coerce")
        ticker_df["valid_to"] = pd.to_datetime(ticker_df["valid_to"], utc=True, errors="coerce")
        ticker_df = ticker_df[ticker_df["entity_id"].astype(str).isin(valid_entity_ids)].reset_index(drop=True)
    ticker_groups: Dict[str, pd.DataFrame] = {
        str(ticker): group.sort_values(["valid_from", "valid_to"], ascending=[False, False], na_position="last").reset_index(drop=True)
        for ticker, group in ticker_df.groupby("ticker", sort=False)
    }

    resolved_rows: List[Dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        source_company_id = str(row.company_id or "")
        ticker = str(getattr(row, "ticker", "") or "").upper()
        as_of_time = pd.Timestamp(row.action_date).tz_convert("UTC") - pd.Timedelta(days=max(1, int(lookback_days)))
        resolved_company_id = None
        mapping_method = None
        if source_company_id in valid_entity_ids:
            resolved_company_id = source_company_id
            mapping_method = "company_id_direct"
        elif ticker:
            candidates = ticker_groups.get(ticker)
            if candidates is not None and not candidates.empty:
                valid = candidates[
                    ((candidates["valid_from"].isna()) | (candidates["valid_from"] <= as_of_time))
                    & ((candidates["valid_to"].isna()) | (candidates["valid_to"] > as_of_time))
                ]
                chosen = valid.iloc[0] if not valid.empty else candidates.iloc[0]
                resolved_company_id = str(chosen["entity_id"])
                mapping_method = "ticker_identifier"
        if not resolved_company_id:
            continue
        payload = row._asdict()
        payload["resolved_company_id"] = resolved_company_id
        payload["mapping_method"] = mapping_method or ""
        resolved_rows.append(payload)
    return pd.DataFrame(resolved_rows)


def _cached_snapshot_loader(
    builder: CompanyStateBuilder,
    *,
    cache_dir: Optional[Path] = None,
    progress_logger: Optional[Callable[[Dict[str, Any]], None]] = None,
    alias_overrides: Optional[Dict[Tuple[str, str], List[str]]] = None,
):
    cache: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def _loader(company_id: str, as_of_time: datetime) -> Dict[str, Any]:
        as_of_ts = pd.Timestamp(as_of_time).tz_convert("UTC")
        key = (str(company_id), as_of_ts.isoformat())
        if key in cache:
            return cache[key]
        extra_aliases = list((alias_overrides or {}).get(key, []) or [])
        cache_path = _snapshot_cache_path(
            cache_dir,
            company_id=str(company_id),
            as_of_ts=as_of_ts,
            extra_aliases=extra_aliases,
            historical_backfill_mode=bool(getattr(builder, "historical_backfill_mode", False)),
        )
        if cache_path is not None and cache_path.exists():
            try:
                payload = json.loads(cache_path.read_text())
                payload = attach_model_feature_bundle(payload)
                cache_path.write_text(json.dumps(payload))
                cache[key] = payload
                _emit_progress(
                    progress_logger,
                    {
                        "event": "snapshot_cache_hit",
                        "company_id": str(company_id),
                        "as_of_time": as_of_ts.isoformat(),
                        "cache_path": str(cache_path),
                    },
                )
                return payload
            except Exception:
                _emit_progress(
                    progress_logger,
                    {
                        "event": "snapshot_cache_rebuild",
                        "company_id": str(company_id),
                        "as_of_time": as_of_ts.isoformat(),
                        "cache_path": str(cache_path),
                    },
                )
        build_started_at = time.perf_counter()
        _emit_progress(
            progress_logger,
            {
                "event": "snapshot_build_start",
                "company_id": str(company_id),
                "as_of_time": as_of_ts.isoformat(),
                "cache_path": str(cache_path) if cache_path is not None else None,
                "extra_aliases": extra_aliases[:10],
            },
        )
        snap = builder.build(
            company_id=str(company_id),
            as_of_time=as_of_ts.isoformat(),
            extra_aliases=extra_aliases,
        )
        payload = attach_model_feature_bundle(asdict(snap))
        cache[key] = payload
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(payload))
        _emit_progress(
            progress_logger,
            {
                "event": "snapshot_build_complete",
                "company_id": str(company_id),
                "as_of_time": as_of_ts.isoformat(),
                "cache_path": str(cache_path) if cache_path is not None else None,
                "elapsed_seconds": round(time.perf_counter() - build_started_at, 3),
            },
        )
        return payload

    return _loader


def _snapshot_cache_path(
    cache_dir: Optional[Path],
    *,
    company_id: str,
    as_of_ts: pd.Timestamp,
    extra_aliases: Optional[Sequence[str]] = None,
    historical_backfill_mode: bool = False,
) -> Optional[Path]:
    if cache_dir is None:
        return None
    stamp = as_of_ts.strftime("%Y%m%dT%H%M%SZ")
    alias_part = "|".join(sorted({str(x) for x in list(extra_aliases or []) if str(x)}))
    digest = hashlib.sha1(
        f"{company_id}|{as_of_ts.isoformat()}|backfill={int(historical_backfill_mode)}|aliases={alias_part}".encode("utf-8")
    ).hexdigest()[:12]
    return cache_dir / f"company_id={company_id}" / f"snapshot_as_of={stamp}_{digest}.json"


def _emit_progress(
    progress_logger: Optional[Callable[[Dict[str, Any]], None]],
    payload: Dict[str, Any],
) -> None:
    if progress_logger is None:
        return
    progress_logger(dict(payload))


def _build_historical_alias_overrides(selected_cases: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, str], List[str]]:
    overrides: Dict[Tuple[str, str], List[str]] = {}
    for case in selected_cases:
        resolved_company_id = str(case.get("company_id") or "")
        as_of_time = str(case.get("as_of_time") or "")
        source_company_id = str(case.get("source_company_id") or "")
        ticker = str(case.get("ticker") or "").upper()
        aliases: List[str] = []
        if source_company_id and source_company_id != resolved_company_id:
            aliases.append(source_company_id)
        if ticker:
            aliases.append(ticker)
        if aliases:
            overrides[(resolved_company_id, as_of_time)] = aliases
    return overrides


def _load_realized_outcomes_lookup(path: Path) -> Dict[str, List[Tuple[pd.Timestamp, str, str]]]:
    frame = duckdb.execute(
        """
        SELECT
            CAST(company_id AS VARCHAR) AS company_id,
            CAST(action_date AS TIMESTAMP) AS action_date,
            CAST(normalized_action_id AS VARCHAR) AS normalized_action_id,
            CAST(normalized_action_family AS VARCHAR) AS normalized_action_family
        FROM read_parquet(?)
        WHERE company_id IS NOT NULL AND action_date IS NOT NULL
        ORDER BY company_id, action_date
        """,
        [str(path)],
    ).df()
    frame["action_date"] = pd.to_datetime(frame["action_date"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["company_id", "action_date"])
    lookup: Dict[str, List[Tuple[pd.Timestamp, str, str]]] = {}
    for company_id, group in frame.groupby("company_id", sort=False):
        lookup[str(company_id)] = [
            (
                pd.Timestamp(row.action_date).tz_convert("UTC"),
                str(row.normalized_action_id or ""),
                str(row.normalized_action_family or ""),
            )
            for row in group.itertuples(index=False)
        ]
    return lookup


def _top_action_ids(package: Dict[str, Any]) -> List[str]:
    ranked = list(package.get("ranked_action_views", []) or [])
    if ranked:
        top = dict(ranked[0] or {})
        ids = [str(x) for x in list(top.get("action_ids", []) or []) if str(x)]
        if ids:
            return ids
    primary = str(package.get("primary_recommendation", "") or "")
    return [primary] if primary else []


def _score_precedent_ranking(
    *,
    precedent_index: Dict[str, Any],
    anchor_action_id: str,
    anchor_action_family: str,
    anchor_action_support: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    candidate_rows = list(precedent_index.get("candidate_rows", []) or [])
    if not candidate_rows:
        return {"reason": "no_precedent_candidate_rows"}

    action_scores: Dict[str, float] = {}
    family_scores: Dict[str, float] = {}
    for row in candidate_rows:
        action_id = str(row.get("action_id") or "").strip()
        if not action_id:
            continue
        confidence = row.get("precedent_confidence")
        if confidence is None:
            continue
        score = float(confidence)
        family = _action_family(action_id)
        if action_id not in action_scores or score > action_scores[action_id]:
            action_scores[action_id] = score
        if family and (family not in family_scores or score > family_scores[family]):
            family_scores[family] = score

    if not action_scores and not family_scores:
        return {"reason": "no_precedent_scores"}

    sorted_actions = sorted(action_scores.items(), key=lambda item: (-item[1], item[0]))
    sorted_families = sorted(family_scores.items(), key=lambda item: (-item[1], item[0]))
    action_rank_lookup = {action_id: index + 1 for index, (action_id, _) in enumerate(sorted_actions)}
    family_rank_lookup = {family: index + 1 for index, (family, _) in enumerate(sorted_families)}

    anchor_action_score = action_scores.get(anchor_action_id)
    anchor_family_score = family_scores.get(anchor_action_family)
    anchor_action_rank = action_rank_lookup.get(anchor_action_id)
    anchor_family_rank = family_rank_lookup.get(anchor_action_family)

    best_other_action_score = max((score for action_id, score in sorted_actions if action_id != anchor_action_id), default=None)
    best_other_family_score = max((score for family, score in sorted_families if family != anchor_action_family), default=None)

    anchor_support_mode = str((anchor_action_support or {}).get("support_mode") or "exact_supported")
    if anchor_support_mode == "exact_supported":
        support_adjusted_rank = anchor_action_rank
        support_adjusted_score = anchor_action_score
        support_adjusted_margin = (
            None
            if anchor_action_score is None or best_other_action_score is None
            else float(anchor_action_score) - float(best_other_action_score)
        )
    else:
        support_adjusted_rank = anchor_family_rank
        support_adjusted_score = anchor_family_score
        support_adjusted_margin = (
            None
            if anchor_family_score is None or best_other_family_score is None
            else float(anchor_family_score) - float(best_other_family_score)
        )

    return {
        "reason": "ok",
        "candidate_row_count": len(candidate_rows),
        "anchor_action_precedent_score": anchor_action_score,
        "anchor_action_precedent_rank": anchor_action_rank,
        "anchor_action_precedent_mrr": (1.0 / float(anchor_action_rank)) if anchor_action_rank else 0.0,
        "anchor_action_precedent_top1": bool(anchor_action_rank == 1),
        "anchor_action_precedent_margin": (
            None
            if anchor_action_score is None or best_other_action_score is None
            else float(anchor_action_score) - float(best_other_action_score)
        ),
        "anchor_family_precedent_score": anchor_family_score,
        "anchor_family_precedent_rank": anchor_family_rank,
        "anchor_family_precedent_mrr": (1.0 / float(anchor_family_rank)) if anchor_family_rank else 0.0,
        "anchor_family_precedent_top1": bool(anchor_family_rank == 1),
        "anchor_family_precedent_margin": (
            None
            if anchor_family_score is None or best_other_family_score is None
            else float(anchor_family_score) - float(best_other_family_score)
        ),
        "anchor_support_adjusted_precedent_score": support_adjusted_score,
        "anchor_support_adjusted_precedent_rank": support_adjusted_rank,
        "anchor_support_adjusted_precedent_mrr": (1.0 / float(support_adjusted_rank)) if support_adjusted_rank else 0.0,
        "anchor_support_adjusted_precedent_top1": bool(support_adjusted_rank == 1),
        "anchor_support_adjusted_precedent_margin": support_adjusted_margin,
    }


def _score_ex_post_alignment(
    *,
    company_id: str,
    as_of_time: str,
    recommended_action_ids: Sequence[str],
    outcomes_lookup: Dict[str, List[Tuple[pd.Timestamp, str, str]]],
    alignment_horizon_days: int,
    anchor_action_id: str,
    anchor_action_family: str,
    anchor_action_support: Optional[Dict[str, Any]] = None,
    recommended_action_support: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    future_events = list(outcomes_lookup.get(company_id, []) or [])
    if not future_events:
        return {"score": None, "reason": "no_company_events"}
    as_of_ts = pd.Timestamp(as_of_time)
    if as_of_ts.tzinfo is None:
        as_of_ts = as_of_ts.tz_localize("UTC")
    else:
        as_of_ts = as_of_ts.tz_convert("UTC")
    end_ts = as_of_ts + pd.Timedelta(days=max(1, int(alignment_horizon_days)))
    window_events = [(ts, action_id, family) for ts, action_id, family in future_events if ts > as_of_ts and ts <= end_ts]
    if not window_events:
        return {"score": None, "reason": "no_future_events_within_horizon"}
    primary = str((recommended_action_ids[0] if recommended_action_ids else "") or "")
    target_ids = [str(x) for x in recommended_action_ids if str(x)]
    target_families = sorted({_action_family(x) for x in target_ids if _action_family(x)})
    future_ids = {action_id for _, action_id, _ in window_events if action_id}
    future_families = {family for _, _, family in window_events if family}
    primary_family = _action_family(primary)
    anchor_support_mode = str((anchor_action_support or {}).get("support_mode") or "exact_supported")
    recommended_support_modes = [
        str((item or {}).get("support_mode") or "exact_supported")
        for item in list(recommended_action_support or [])
    ]
    primary_support_mode = recommended_support_modes[0] if recommended_support_modes else "exact_supported"

    primary_exact = bool(primary and primary == anchor_action_id)
    any_exact = bool(set(target_ids) & future_ids)
    primary_family_match = bool(primary_family and primary_family == anchor_action_family)
    any_family_match = bool(set(target_families) & future_families)
    primary_benchmark_mode = "exact" if anchor_support_mode == "exact_supported" and primary_support_mode == "exact_supported" else "family_only"
    any_benchmark_mode = "exact" if anchor_support_mode == "exact_supported" and all(mode == "exact_supported" for mode in recommended_support_modes or ["exact_supported"]) else "family_only"
    primary_support_adjusted_match = primary_exact if primary_benchmark_mode == "exact" else primary_family_match
    any_support_adjusted_match = any_exact if any_benchmark_mode == "exact" else any_family_match
    if primary_support_adjusted_match:
        score = 1.0
        reason = "anchor_primary_exact" if primary_benchmark_mode == "exact" else "anchor_primary_family_support_adjusted"
    elif any_support_adjusted_match:
        score = 0.85
        reason = "future_exact_match" if any_benchmark_mode == "exact" else "future_family_support_adjusted"
    elif primary_family_match:
        score = 0.6
        reason = "anchor_primary_family_match"
    elif any_family_match:
        score = 0.4
        reason = "future_family_match"
    else:
        score = 0.0
        reason = "no_alignment"
    return {
        "score": round(score, 6),
        "reason": reason,
        "primary_exact_match": primary_exact,
        "any_exact_match": any_exact,
        "primary_family_match": primary_family_match,
        "any_family_match": any_family_match,
        "primary_support_adjusted_match": primary_support_adjusted_match,
        "any_support_adjusted_match": any_support_adjusted_match,
        "primary_benchmark_mode": primary_benchmark_mode,
        "any_benchmark_mode": any_benchmark_mode,
        "anchor_action_id": anchor_action_id,
        "anchor_action_family": anchor_action_family,
        "future_action_ids_sample": sorted(future_ids)[:5],
        "future_action_families_sample": sorted(future_families)[:5],
    }


def _aggregate_historical_cases(cases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    unsupported = [case for case in cases if case.get("unsupported_reason")]
    completed = [case for case in cases if not case.get("error") and not case.get("unsupported_reason")]
    scored = [
        case for case in completed
        if ((case.get("historical_alignment", {}) or {}).get("score")) is not None
    ]
    if not cases:
        return {
            "completed_case_count": 0,
            "scored_case_count": 0,
            "mean_alignment_score": 0.0,
            "strong_alignment_rate": 0.0,
            "anchor_primary_exact_rate": 0.0,
            "anchor_primary_family_rate": 0.0,
            "future_any_exact_rate": 0.0,
            "future_any_family_rate": 0.0,
            "error_rate": 0.0,
            "unsupported_case_count": 0,
            "coverage_skip_rate": 0.0,
        }
    alignment_scores = [float(((case.get("historical_alignment", {}) or {}).get("score", 0.0) or 0.0)) for case in scored]
    precedent_ranking_cases = [
        case for case in completed
        if str(((case.get("precedent_ranking", {}) or {}).get("reason") or "")).strip().lower() == "ok"
    ]
    exact_precedent_mrr = [
        float(((case.get("precedent_ranking", {}) or {}).get("anchor_action_precedent_mrr", 0.0) or 0.0))
        for case in precedent_ranking_cases
    ]
    family_precedent_mrr = [
        float(((case.get("precedent_ranking", {}) or {}).get("anchor_family_precedent_mrr", 0.0) or 0.0))
        for case in precedent_ranking_cases
    ]
    support_adjusted_precedent_mrr = [
        float(((case.get("precedent_ranking", {}) or {}).get("anchor_support_adjusted_precedent_mrr", 0.0) or 0.0))
        for case in precedent_ranking_cases
    ]
    exact_precedent_margin = [
        float(value)
        for value in [((case.get("precedent_ranking", {}) or {}).get("anchor_action_precedent_margin")) for case in precedent_ranking_cases]
        if value is not None
    ]
    family_precedent_margin = [
        float(value)
        for value in [((case.get("precedent_ranking", {}) or {}).get("anchor_family_precedent_margin")) for case in precedent_ranking_cases]
        if value is not None
    ]
    support_adjusted_precedent_margin = [
        float(value)
        for value in [((case.get("precedent_ranking", {}) or {}).get("anchor_support_adjusted_precedent_margin")) for case in precedent_ranking_cases]
        if value is not None
    ]
    exact_negative_margin_case_count = sum(1 for value in exact_precedent_margin if float(value) < 0.0)
    family_negative_margin_case_count = sum(1 for value in family_precedent_margin if float(value) < 0.0)
    support_adjusted_negative_margin_case_count = sum(1 for value in support_adjusted_precedent_margin if float(value) < 0.0)
    return {
        "completed_case_count": len(completed),
        "scored_case_count": len(scored),
        "unsupported_case_count": len(unsupported),
        "mean_alignment_score": round(sum(alignment_scores) / len(alignment_scores), 6) if alignment_scores else 0.0,
        "strong_alignment_rate": round(sum(1 for case in scored if float(((case.get("historical_alignment", {}) or {}).get("score", 0.0) or 0.0)) >= 0.6) / len(scored), 6) if scored else 0.0,
        "anchor_primary_exact_rate": round(sum(1 for case in scored if bool(((case.get("historical_alignment", {}) or {}).get("primary_exact_match")))) / len(scored), 6) if scored else 0.0,
        "anchor_primary_family_rate": round(sum(1 for case in scored if bool(((case.get("historical_alignment", {}) or {}).get("primary_family_match")))) / len(scored), 6) if scored else 0.0,
        "anchor_primary_support_adjusted_rate": round(sum(1 for case in scored if bool(((case.get("historical_alignment", {}) or {}).get("primary_support_adjusted_match")))) / len(scored), 6) if scored else 0.0,
        "future_any_exact_rate": round(sum(1 for case in scored if bool(((case.get("historical_alignment", {}) or {}).get("any_exact_match")))) / len(scored), 6) if scored else 0.0,
        "future_any_family_rate": round(sum(1 for case in scored if bool(((case.get("historical_alignment", {}) or {}).get("any_family_match")))) / len(scored), 6) if scored else 0.0,
        "future_any_support_adjusted_rate": round(sum(1 for case in scored if bool(((case.get("historical_alignment", {}) or {}).get("any_support_adjusted_match")))) / len(scored), 6) if scored else 0.0,
        "precedent_ranking_case_count": len(precedent_ranking_cases),
        "anchor_action_precedent_top1_rate": round(sum(1 for case in precedent_ranking_cases if bool(((case.get("precedent_ranking", {}) or {}).get("anchor_action_precedent_top1")))) / len(precedent_ranking_cases), 6) if precedent_ranking_cases else 0.0,
        "anchor_action_precedent_mrr_mean": round(sum(exact_precedent_mrr) / len(exact_precedent_mrr), 6) if exact_precedent_mrr else 0.0,
        "anchor_action_precedent_margin_mean": round(sum(exact_precedent_margin) / len(exact_precedent_margin), 6) if exact_precedent_margin else 0.0,
        "anchor_action_precedent_margin_min": round(min(exact_precedent_margin), 6) if exact_precedent_margin else 0.0,
        "anchor_action_precedent_negative_margin_case_count": exact_negative_margin_case_count,
        "anchor_family_precedent_top1_rate": round(sum(1 for case in precedent_ranking_cases if bool(((case.get("precedent_ranking", {}) or {}).get("anchor_family_precedent_top1")))) / len(precedent_ranking_cases), 6) if precedent_ranking_cases else 0.0,
        "anchor_family_precedent_mrr_mean": round(sum(family_precedent_mrr) / len(family_precedent_mrr), 6) if family_precedent_mrr else 0.0,
        "anchor_family_precedent_margin_mean": round(sum(family_precedent_margin) / len(family_precedent_margin), 6) if family_precedent_margin else 0.0,
        "anchor_family_precedent_margin_min": round(min(family_precedent_margin), 6) if family_precedent_margin else 0.0,
        "anchor_family_precedent_negative_margin_case_count": family_negative_margin_case_count,
        "anchor_support_adjusted_precedent_top1_rate": round(sum(1 for case in precedent_ranking_cases if bool(((case.get("precedent_ranking", {}) or {}).get("anchor_support_adjusted_precedent_top1")))) / len(precedent_ranking_cases), 6) if precedent_ranking_cases else 0.0,
        "anchor_support_adjusted_precedent_mrr_mean": round(sum(support_adjusted_precedent_mrr) / len(support_adjusted_precedent_mrr), 6) if support_adjusted_precedent_mrr else 0.0,
        "anchor_support_adjusted_precedent_margin_mean": round(sum(support_adjusted_precedent_margin) / len(support_adjusted_precedent_margin), 6) if support_adjusted_precedent_margin else 0.0,
        "anchor_support_adjusted_precedent_margin_min": round(min(support_adjusted_precedent_margin), 6) if support_adjusted_precedent_margin else 0.0,
        "anchor_support_adjusted_precedent_negative_margin_case_count": support_adjusted_negative_margin_case_count,
        "error_rate": round(sum(1 for case in cases if case.get("error")) / len(cases), 6),
        "coverage_skip_rate": round(len(unsupported) / len(cases), 6),
        "recommended_posture_counts": _count_values(case.get("recommended_posture") for case in completed),
        "recommended_family_counts": _count_values(_action_family((case.get("top_action_ids") or [""])[0]) for case in completed),
        "anchor_family_counts": _count_values(case.get("anchor_action_family") for case in completed),
        "anchor_support_mode_counts": _count_values(
            ((case.get("anchor_action_support", {}) or {}).get("support_mode")) for case in completed
        ),
        "recommended_support_mode_counts": _count_values(
            (
                ((case.get("recommended_action_support") or [{}])[0] or {}).get("support_mode")
                if case.get("recommended_action_support")
                else None
            )
            for case in completed
        ),
    }


def _count_values(values: Iterable[Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        out[text] = out.get(text, 0) + 1
    return dict(sorted(out.items()))


def _action_family(action_id: str) -> str:
    text = str(action_id or "")
    if "." not in text:
        return ""
    return text.split(".", 1)[0]


def _snapshot_coverage_summary(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    features = dict(snapshot.get("features", {}) or {})
    category_counts: Dict[str, int] = {}
    non_missing_core_feature_count = 0
    non_missing_feature_names: List[str] = []
    for name, payload in features.items():
        if not isinstance(payload, dict):
            continue
        if str(name).startswith("strategic.intent."):
            continue
        missing_reason = payload.get("missing_reason")
        value = payload.get("value")
        if missing_reason is None and value is not None:
            non_missing_core_feature_count += 1
            non_missing_feature_names.append(str(name))
            category = str(name).split(".", 1)[0]
            category_counts[category] = category_counts.get(category, 0) + 1
    return {
        "feature_count": len(features),
        "non_missing_core_feature_count": non_missing_core_feature_count,
        "non_missing_categories": sorted(category_counts.keys()),
        "non_missing_category_counts": dict(sorted(category_counts.items())),
        "sample_non_missing_features": non_missing_feature_names[:10],
    }


def _snapshot_has_meaningful_coverage(
    snapshot_coverage: Dict[str, Any],
    *,
    min_non_missing_core_features: int,
) -> bool:
    core_categories = {"liquidity", "capital_structure", "market", "operating", "ownership_governance", "strategic"}
    non_missing_count = int(snapshot_coverage.get("non_missing_core_feature_count", 0) or 0)
    categories = set(snapshot_coverage.get("non_missing_categories", []) or [])
    return non_missing_count >= max(1, int(min_non_missing_core_features)) and bool(categories & core_categories)


__all__ = [
    "build_historical_recommendation_report",
    "render_historical_recommendation_markdown",
    "_historical_case_key",
    "_prefilter_support_is_eligible",
    "_prioritize_historical_cases",
    "_select_historical_cases_from_frame",
    "_score_ex_post_alignment",
    "_snapshot_coverage_summary",
    "_snapshot_has_meaningful_coverage",
    "_summarize_case_support_by_family",
]
