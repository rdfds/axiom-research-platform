"""RecommendationRun execution orchestration.

Executes staged lifecycle under a frozen run_id and persists artifacts:
- CandidateSet
- FeasibilityResults
- PrecedentMatches
- PrecedentIndex
- PlanSet
- BoardReadyDossier
- RecommendationPackage
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from typing import TYPE_CHECKING

from .recommendation_runtime_config import (
    build_execution_config,
    capture_runtime_env_config,
)
from .model_feature_bundle import attach_model_feature_bundle, feature_view_from_snapshot
from .runtime_feature_adapter import adapt_snapshot, resolve_feature_value

if TYPE_CHECKING:
    from .action_ontology import ActionSchemaRegistry
    from .pipeline.types import PrecedentPack
    from .recommendation_run import RecommendationRun, RecommendationRunStore


def _candidate_bindings():
    from .action_ontology import build_default_action_schema_registry
    from .candidate_generation import generate_action_candidates
    from .mechanism_brain import MechanismBrain

    return build_default_action_schema_registry, generate_action_candidates, MechanismBrain


def _causal_bindings():
    from .causal_model_risk import build_causal_model_risk_report

    return build_causal_model_risk_report


def _precedent_bindings():
    from .pipeline.precedent_index import build_precedent_index
    from .pipeline.run import run_precedent
    from .pipeline.types import PrecedentPack

    return build_precedent_index, run_precedent, PrecedentPack


def _dossier_bindings():
    from .board_ready_dossier import build_board_ready_dossier

    return build_board_ready_dossier


def _truthy_env(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def _run_store_bindings():
    from .recommendation_run import (
        RecommendationRunStore,
        _apply_scenario_overrides,
        _hash_snapshot,
        _parse_ts,
        _resolve_snapshot,
        _snapshot_company_aliases,
        create_recommendation_run,
        validate_plan_hard_constraints,
    )

    return (
        RecommendationRunStore,
        _apply_scenario_overrides,
        _hash_snapshot,
        _parse_ts,
        _resolve_snapshot,
        _snapshot_company_aliases,
        create_recommendation_run,
        validate_plan_hard_constraints,
    )


def execute_recommendation_run(
    run_id: str,
    runs_root: str | Path = "data/recommendation_runs",
    snapshot_root: Optional[str | Path] = None,
    snapshot_path: Optional[str | Path] = None,
    snapshot_loader: Optional[Callable[[str, datetime], Dict[str, Any]]] = None,
    entity_identifier_path: str | Path = "data/inputs_layer/entity_identifier.parquet",
    action_ids: Optional[Sequence[str]] = None,
    action_type: Optional[str] = None,
    max_candidates: int = 12,
    min_candidates_target: int = 0,
    strict_evidence: bool = False,
    precedent_top_k: int = 0,
    outcomes_path: Optional[str | Path] = None,
    config_path: Optional[str | Path] = None,
    top_plans: int = 3,
    registry: Optional[ActionSchemaRegistry] = None,
    precedent_runner: Optional[Callable[..., PrecedentPack]] = None,
) -> Dict[str, Any]:
    """Execute RecommendationRun lifecycle and attach all stage artifacts."""
    import json as _json
    import time as _time

    t0 = _time.time()

    def _debug(stage: str, **extra: Any) -> None:
        payload = {
            "ok": True,
            "event": "execute_debug",
            "stage": stage,
            "elapsed_seconds": round(_time.time() - t0, 3),
        }
        if extra:
            payload.update(extra)
        print(_json.dumps(payload, default=str), flush=True)

    _debug("bind_run_store:start")
    RecommendationRunStore, _, _, _, _, _, _, _ = _run_store_bindings()
    _debug("bind_run_store:done")
    _debug("bind_candidate:start")
    build_default_action_schema_registry, _, _ = _candidate_bindings()
    _debug("bind_candidate:done")
    _debug("bind_precedent:start")
    build_precedent_index, run_precedent, _ = _precedent_bindings()
    _debug("bind_precedent:done")
    _debug("store_init:start", runs_root=str(runs_root))
    store = RecommendationRunStore(root=runs_root)
    _debug("store_init:done")
    _debug("load_run:start", run_id=str(run_id))
    run = store.get_run(run_id)
    _debug("load_run:done", found=bool(run))
    if run is None:
        raise ValueError(f"Run not found: {run_id}")
    if run.status in {"completed", "failed"}:
        raise ValueError(f"Run is terminal ({run.status}); create a new run_id")

    _debug("persist_execution_config:start")
    _persist_execution_config(
        store=store,
        run_id=run_id,
        runs_root=runs_root,
        snapshot_root=snapshot_root,
        snapshot_path=snapshot_path,
        entity_identifier_path=entity_identifier_path,
        action_ids=action_ids,
        action_type=action_type,
        max_candidates=max_candidates,
        min_candidates_target=min_candidates_target,
        strict_evidence=strict_evidence,
        precedent_top_k=precedent_top_k,
        outcomes_path=outcomes_path,
        config_path=config_path,
        top_plans=top_plans,
    )
    _debug("persist_execution_config:done")

    _debug("build_registry:start")
    registry = registry or build_default_action_schema_registry(version="v1.0")
    _debug("build_registry:done")
    precedent_runner = precedent_runner or run_precedent
    adapter_path = None

    try:
        _debug("load_snapshot:start")
        snapshot = _load_and_verify_frozen_snapshot(
            run=run,
            snapshot_root=Path(snapshot_root) if snapshot_root else None,
            snapshot_path=Path(snapshot_path) if snapshot_path else None,
            snapshot_loader=snapshot_loader,
            entity_identifier_path=Path(entity_identifier_path),
        )
        snapshot, adapter_diagnostics = adapt_snapshot(snapshot)
        snapshot = attach_model_feature_bundle(snapshot)
        adapter_path = store.attach_artifact(
            run_id,
            "RuntimeFeatureAdapterDiagnostics",
            {
                "run_id": run_id,
                "generated_at": _now_iso(),
                "diagnostics": adapter_diagnostics,
            },
        )
        bundle_path = store.attach_artifact(
            run_id,
            "ModelFeatureBundleDiagnostics",
            {
                "run_id": run_id,
                "generated_at": _now_iso(),
                "diagnostics": dict((snapshot.get("_model_feature_bundle", {}) or {}).get("diagnostics", {}) or {}),
            },
        )
        store.merge_metadata(
            run_id,
            {
                "runtime": {
                    "feature_adapter": adapter_diagnostics,
                    "model_feature_bundle": dict((snapshot.get("_model_feature_bundle", {}) or {}).get("diagnostics", {}) or {}),
                },
                "artifacts": {
                    "RuntimeFeatureAdapterDiagnostics": str(adapter_path),
                    "ModelFeatureBundleDiagnostics": str(bundle_path),
                },
            },
        )
        _debug("load_snapshot:done")

        # Stage 1: candidate generation
        store.transition_status(run_id, "candidate_generation", {"max_candidates": max_candidates})
        candidate_set = _generate_candidates(
            run=run,
            snapshot=snapshot,
            registry=registry,
            action_ids=action_ids,
            action_type=action_type,
            max_candidates=max_candidates,
            min_candidates_target=min_candidates_target,
            strict_evidence=strict_evidence,
        )
        candidates = list(candidate_set.get("candidates", []))
        cand_artifact = dict(candidate_set)
        cand_artifact["count"] = len(candidates)
        candidate_path = store.attach_artifact(run_id, "CandidateSet", cand_artifact)

        # Stage 2: feasibility
        store.transition_status(run_id, "feasibility_evaluation", {"candidate_count": len(candidates)})
        feasibility = _evaluate_feasibility(
            run=run,
            registry=registry,
            candidates=candidates,
            snapshot=snapshot,
            strict_evidence=strict_evidence,
        )
        feasibility_profile = _feasibility_profile(feasibility)
        _emit_stage_profile("feasibility_evaluation", feasibility_profile)
        feasibility_path = store.attach_artifact(
            run_id,
            "FeasibilityResults",
            {
                "run_id": run_id,
                "generated_at": _now_iso(),
                "candidate_count": len(candidates),
                "feasible_count": sum(1 for x in feasibility if x.get("feasible")),
                "profile": feasibility_profile,
                "results": feasibility,
            },
        )
        causal_risk_path = None
        if str(os.environ.get("AXIOM_SKIP_CAUSAL_MODEL_RISK_REPORT", "")).strip().lower() not in {
            "1",
            "true",
            "yes",
            "on",
        }:
            _debug("bind_causal:start")
            build_causal_model_risk_report = _causal_bindings()
            _debug("bind_causal:done")
            prev_causal_report = _latest_previous_causal_report(store=store, run=run)
            causal_risk_report = build_causal_model_risk_report(
                run=run,
                snapshot=snapshot,
                feasibility_results=feasibility,
                previous_report=prev_causal_report,
            )
            causal_risk_path = store.attach_artifact(run_id, "CausalModelRiskReport", causal_risk_report)

        feasible_candidates = [
            x.get("action_candidate", x.get("candidate"))
            for x in feasibility
            if x.get("feasible")
        ]

        # Stage 3: precedent retrieval
        store.transition_status(
            run_id,
            "precedent_retrieval",
            {
                "feasible_count": len(feasible_candidates),
                "precedent_top_k": int(precedent_top_k),
            },
        )
        _update_stage_progress(
            store=store,
            run_id=run_id,
            stage="precedent_retrieval",
            completed=0,
            total=min(len(feasible_candidates), int(precedent_top_k or 0) or len(feasible_candidates)),
        )
        precedent_matches = _retrieve_precedents(
            run=run,
            feasible_candidates=feasible_candidates,
            precedent_runner=precedent_runner,
            precedent_top_k=precedent_top_k,
            snapshot=snapshot,
            snapshot_root=snapshot_root,
            snapshot_path=snapshot_path,
            outcomes_path=outcomes_path,
            config_path=config_path,
            progress_callback=lambda completed, total: _update_stage_progress(
                store=store,
                run_id=run_id,
                stage="precedent_retrieval",
                completed=completed,
                total=total,
            ),
        )
        precedent_profile = _precedent_profile(precedent_matches)
        _emit_stage_profile("precedent_retrieval", precedent_profile)
        precedent_path = store.attach_artifact(
            run_id,
            "PrecedentMatches",
            {
                "run_id": run_id,
                "generated_at": _now_iso(),
                "candidate_count": len(feasible_candidates),
                "profile": precedent_profile,
                "results": precedent_matches,
            },
        )
        precedent_index = build_precedent_index(run_id=run_id, precedent_matches=precedent_matches)
        precedent_index_path = store.attach_artifact(run_id, "PrecedentIndex", precedent_index)

        # Stage 4: plan search
        store.transition_status(run_id, "plan_search", {"precedent_candidates": len(precedent_matches)})
        plan_set = _build_plan_set(
            run=run,
            feasible_candidates=feasible_candidates,
            precedent_matches=precedent_matches,
            registry=registry,
            top_plans=top_plans,
        )
        plan_path = store.attach_artifact(run_id, "PlanSet", plan_set)
        plans = list(plan_set.get("plans", []) or [])
        top_plan = plans[0] if plans else None
        skip_dossier_package = _truthy_env("AXIOM_SKIP_DOSSIER_PACKAGE_BUILD")
        if skip_dossier_package:
            board_ready_dossier = {
                "run_id": run_id,
                "generated_at": _now_iso(),
                "executive_summary": "Fit-mode dossier skipped to reduce heavy downstream warehouse access.",
                "confidence_posture": "fit_mode_lightweight",
                "status_quo_view": {"recommended_posture": None},
                "ranked_action_views": [],
                "monitoring": {"triggers": [], "branches": []},
                "recommendation_thesis": {},
                "fit_mode_skipped": True,
            }
        else:
            build_board_ready_dossier = _dossier_bindings()
            board_ready_dossier = build_board_ready_dossier(
                run=run,
                snapshot=snapshot,
                plan_set=plan_set,
                feasible_candidates=feasible_candidates,
                precedent_matches=precedent_matches,
                registry=registry,
            )
        dossier_path = store.attach_artifact(run_id, "BoardReadyDossier", board_ready_dossier)

        def _plan_preview(plan: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "plan_id": plan.get("plan_id"),
                "score": plan.get("score"),
                "action_ids": [step.get("action_id") for step in list(plan.get("steps", []) or [])],
                "summary_explanation": plan.get("summary_explanation"),
                "main_failure_modes": list(((plan.get("risks", {}) or {}).get("main_failure_modes", []) or []))[:3],
                "confidence_posture": board_ready_dossier.get("confidence_posture"),
            }

        top_plan_action_ids = [step.get("action_id") for step in list((top_plan or {}).get("steps", []) or []) if step.get("action_id")]
        ranked_action_views = list(board_ready_dossier.get("ranked_action_views", []) or [])
        if skip_dossier_package and top_plan_action_ids:
            ranked_action_views = [
                {
                    "action_ids": top_plan_action_ids,
                    "plan_id": (top_plan or {}).get("plan_id"),
                    "summary_explanation": (top_plan or {}).get("summary_explanation"),
                }
            ]

        recommendation_package = {
            "run_id": run_id,
            "company_id": run.company_id,
            "as_of_time": run.as_of_time,
            "generated_at": _now_iso(),
            "planner_random_seed": int(run.planner_random_seed if run.planner_random_seed is not None else 0),
            "recommended_posture": ((board_ready_dossier.get("status_quo_view", {}) or {}).get("recommended_posture")),
            "status_quo_view": board_ready_dossier.get("status_quo_view"),
            "sizing_guidance": board_ready_dossier.get("sizing_guidance"),
            "parameter_optimization": board_ready_dossier.get("parameter_optimization"),
            "regret_analysis": board_ready_dossier.get("regret_analysis"),
            "rating_cliff_analysis": board_ready_dossier.get("rating_cliff_analysis"),
            "signaling_analysis": board_ready_dossier.get("signaling_analysis"),
            "top_plan": top_plan,
            "ranked_action_views": ranked_action_views,
            "primary_recommendation": top_plan_action_ids[0] if top_plan_action_ids else None,
            "plans_preview": [_plan_preview(plan) for plan in plans[:3]],
            "monitoring_triggers": list(((board_ready_dossier.get("monitoring", {}) or {}).get("triggers", []) or [])),
            "contingency_branches": list(((board_ready_dossier.get("monitoring", {}) or {}).get("branches", []) or [])),
            "top_plan_summary_explanation": board_ready_dossier.get("executive_summary") or (top_plan or {}).get("summary_explanation"),
            "board_ready_dossier": board_ready_dossier,
            "summary": {
                "candidate_count": len(candidates),
                "feasible_count": len(feasible_candidates),
                "precedent_candidate_count": len(precedent_matches),
                "plan_count": len(plans),
                "top_plan_action_ids": [step.get("action_id") for step in list((top_plan or {}).get("steps", []) or [])],
            },
        }
        recommendation_path = store.attach_artifact(run_id, "RecommendationPackage", recommendation_package)

        store.transition_status(run_id, "completed", {"plan_count": len(plan_set.get("plans", []))})

        return {
            "ok": True,
            "run_id": run_id,
            "status": "completed",
            "artifacts": {
                "CandidateSet": str(candidate_path),
                "FeasibilityResults": str(feasibility_path),
                "CausalModelRiskReport": str(causal_risk_path),
                "PrecedentMatches": str(precedent_path),
                "PrecedentIndex": str(precedent_index_path),
                "PlanSet": str(plan_path),
                "BoardReadyDossier": str(dossier_path),
                "RecommendationPackage": str(recommendation_path),
                "RuntimeFeatureAdapterDiagnostics": str(adapter_path) if adapter_path is not None else None,
            },
            "counts": {
                "candidates": len(candidates),
                "feasible": len(feasible_candidates),
                "precedent": len(precedent_matches),
                "plans": len(plan_set.get("plans", [])),
            },
            "runtime_feature_adapter": adapter_diagnostics,
        }

    except Exception as exc:
        # Best effort failure transition for traceability.
        try:
            current = store.get_run(run_id)
            if current is not None and current.status not in {"completed", "failed"}:
                store.transition_status(run_id, "failed", {"error": str(exc)})
        except Exception:
            pass
        raise


def create_and_execute_recommendation_run(
    company_id: str,
    as_of_time: str | datetime,
    objectives: Optional[Dict[str, Any]] = None,
    constraints: Optional[Dict[str, Any]] = None,
    scenario: Optional[Dict[str, Any]] = None,
    runs_root: str | Path = "data/recommendation_runs",
    snapshot_root: Optional[str | Path] = None,
    snapshot_path: Optional[str | Path] = None,
    snapshot_loader: Optional[Callable[[str, datetime], Dict[str, Any]]] = None,
    entity_graph_path: str | Path = "data/inputs_layer/entity_graph.parquet",
    entity_identifier_path: str | Path = "data/inputs_layer/entity_identifier.parquet",
    planner_random_seed: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
    action_ids: Optional[Sequence[str]] = None,
    action_type: Optional[str] = None,
    max_candidates: int = 12,
    min_candidates_target: int = 0,
    strict_evidence: bool = False,
    precedent_top_k: int = 0,
    outcomes_path: Optional[str | Path] = None,
    config_path: Optional[str | Path] = None,
    top_plans: int = 3,
    registry: Optional[ActionSchemaRegistry] = None,
    precedent_runner: Optional[Callable[..., PrecedentPack]] = None,
) -> Dict[str, Any]:
    """One-shot helper: create a run, then execute all stages."""
    RecommendationRunStore, _, _, _, _, _, create_recommendation_run, _ = _run_store_bindings()
    store = RecommendationRunStore(root=runs_root)
    run_id = create_recommendation_run(
        company_id=company_id,
        as_of_time=as_of_time,
        objectives=objectives,
        constraints=constraints,
        scenario=scenario,
        run_store=store,
        snapshot_root=snapshot_root,
        snapshot_path=snapshot_path,
        snapshot_loader=snapshot_loader,
        entity_graph_path=entity_graph_path,
        entity_identifier_path=entity_identifier_path,
        planner_random_seed=planner_random_seed,
        metadata=metadata,
    )
    summary = execute_recommendation_run(
        run_id=run_id,
        runs_root=runs_root,
        snapshot_root=snapshot_root,
        snapshot_path=snapshot_path,
        snapshot_loader=snapshot_loader,
        entity_identifier_path=entity_identifier_path,
        action_ids=action_ids,
        action_type=action_type,
        max_candidates=max_candidates,
        min_candidates_target=min_candidates_target,
        strict_evidence=strict_evidence,
        precedent_top_k=precedent_top_k,
        outcomes_path=outcomes_path,
        config_path=config_path,
        top_plans=top_plans,
        registry=registry,
        precedent_runner=precedent_runner,
    )
    return summary


def _load_and_verify_frozen_snapshot(
    run: RecommendationRun,
    snapshot_root: Optional[Path],
    snapshot_path: Optional[Path],
    snapshot_loader: Optional[Callable[[str, datetime], Dict[str, Any]]],
    entity_identifier_path: Path,
) -> Dict[str, Any]:
    _, _apply_scenario_overrides, _hash_snapshot, _parse_ts, _resolve_snapshot, _snapshot_company_aliases, _, _ = _run_store_bindings()
    as_of_dt = _parse_ts(run.as_of_time)
    if snapshot_path is not None or _truthy_env("AXIOM_SKIP_RUN_COMPANY_VALIDATION"):
        aliases = [str(run.company_id)]
    else:
        aliases = _snapshot_company_aliases(run.company_id, entity_identifier_path)
    snapshot = _resolve_snapshot(
        company_id=run.company_id,
        as_of_time=as_of_dt,
        snapshot_root=snapshot_root,
        snapshot_path=snapshot_path,
        snapshot_builder=None,
        snapshot_loader=snapshot_loader,
        aliases=aliases,
    )
    snapshot = _apply_scenario_overrides(snapshot, run.scenario)
    observed_hash = _hash_snapshot(snapshot)
    if observed_hash != run.frozen_state.snapshot_hash:
        raise ValueError(
            "Frozen snapshot hash mismatch for run_id={} expected={} got={}".format(
                run.run_id,
                run.frozen_state.snapshot_hash,
                observed_hash,
            )
        )
    return snapshot


def _generate_candidates(
    run: RecommendationRun,
    snapshot: Dict[str, Any],
    registry: ActionSchemaRegistry,
    action_ids: Optional[Sequence[str]],
    action_type: Optional[str],
    max_candidates: int,
    min_candidates_target: int,
    strict_evidence: bool,
) -> Dict[str, Any]:
    _, generate_action_candidates, _ = _candidate_bindings()
    if action_ids:
        missing = [aid for aid in action_ids if registry.get_action(str(aid)) is None]
        if missing:
            raise ValueError(f"Unknown action_id in candidate request: {missing}")
    if action_type and not registry.get_actions_by_type(action_type):
        raise ValueError(f"No actions under action_type={action_type}")

    return generate_action_candidates(
        run=run,
        state_snapshot=snapshot,
        action_registry=registry,
        action_ids=action_ids,
        action_type=action_type,
        max_candidates=max_candidates,
        min_candidates_target=min_candidates_target,
        strict_evidence=strict_evidence,
    )


def _snapshot_features(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    feats = feature_view_from_snapshot(snapshot, view_name="candidate_generation")
    return feats if isinstance(feats, dict) else {}


def _snapshot_feature_value(v: Any) -> Any:
    if isinstance(v, dict):
        return v.get("value")
    return v


def _flatten_projected_state(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    dossier_features = feature_view_from_snapshot(snapshot, view_name="dossier")
    for k, v in dossier_features.items():
        out[k] = _snapshot_feature_value(v)
    for key in (
        "capital_structure.net_debt",
        "capital_structure.net_leverage",
        "capital_structure.gross_leverage",
        "liquidity.available_for_actions",
        "operating.ebitda_ttm",
        "macro.rate_10y",
        "macro.rate_2y",
        "macro.sofr",
        "market.ig_oas",
        "market.hy_oas",
        "market.pe",
    ):
        value = resolve_feature_value(dossier_features, key)
        if value is not None:
            out[key] = value

    # Optional helper signal for rating-preservation constraints.
    rating_state = out.get("capital_structure.rating_state")
    if isinstance(rating_state, dict):
        rating = str(rating_state.get("rating", "") or "")
        is_ig = False
        if rating:
            upper = rating.upper()
            # Broad heuristic: treat any BB+/below as non-IG.
            is_ig = not upper.startswith("BB") and not upper.startswith("B") and not upper.startswith("CCC")
        out["capital_structure.rating_state.is_investment_grade"] = is_ig
    return out


def _infer_evidence_classes(snapshot: Dict[str, Any]) -> List[str]:
    classes = {"financial_disclosure"}
    prov = snapshot.get("provenance", {}) if isinstance(snapshot.get("provenance"), dict) else {}
    inputs = prov.get("inputs_used", {}) if isinstance(prov.get("inputs_used"), dict) else {}
    if inputs.get("facts"):
        classes.update({"management_statement", "capital_policy_statement", "liquidity_disclosure"})
    if inputs.get("timeseries") or inputs.get("macro"):
        classes.add("market_signal")
    if inputs.get("events"):
        classes.update({"recent_action_history", "peer_context_signal"})
    if inputs.get("issuer_ratings"):
        classes.add("rating_disclosure")
    return sorted(classes)


def _constraint_tokens(run: RecommendationRun) -> List[str]:
    out: List[str] = []
    for c in run.constraints.hard_constraints + run.constraints.soft_constraints:
        out.append(c.constraint_type)
        out.append(c.constraint_id)
    return list(dict.fromkeys(out))


