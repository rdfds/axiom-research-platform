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


def _evaluate_feasibility(
    run: RecommendationRun,
    registry: ActionSchemaRegistry,
    candidates: List[Dict[str, Any]],
    snapshot: Dict[str, Any],
    strict_evidence: bool,
) -> List[Dict[str, Any]]:
    _, _, MechanismBrain = _candidate_bindings()
    _, _, _, _, _, _, _, validate_plan_hard_constraints = _run_store_bindings()
    features = _snapshot_features(snapshot)
    feature_names = sorted(features.keys())
    evidence_classes = _infer_evidence_classes(snapshot)
    projected_state = _flatten_projected_state(snapshot)
    constraint_tokens = _constraint_tokens(run)
    mechanism_brain = MechanismBrain(action_registry=registry)
    mechanism_eval_t0 = time.perf_counter()
    evaluated = mechanism_brain.evaluate_candidate_set(
        run=run,
        state_snapshot=snapshot,
        candidates=candidates,
    )
    mechanism_eval_seconds = time.perf_counter() - mechanism_eval_t0
    mechanism_batch_profile = dict(getattr(mechanism_brain, "last_evaluation_set_profile", {}) or {})

    out: List[Dict[str, Any]] = []
    for cand, action_eval in zip(candidates, evaluated):
        candidate_t0 = time.perf_counter()
        action_eval_to_dict_t0 = time.perf_counter()
        action_eval_dict = action_eval.to_dict()
        action_eval_to_dict_seconds = time.perf_counter() - action_eval_to_dict_t0

        validation_t0 = time.perf_counter()
        validation = registry.validate_candidate(
            {
                "action_id": cand["action_id"],
                "parameters": cand.get("params", {}),
                "available_features": feature_names,
                "available_evidence_classes": evidence_classes,
                "constraints": constraint_tokens,
            },
            strict_evidence=bool(strict_evidence),
        )
        validation_seconds = time.perf_counter() - validation_t0

        eval_profile = dict(action_eval_dict.get("evaluation_profile", {}) or {})
        hard_constraint_t0 = time.perf_counter()
        hard_violations = validate_plan_hard_constraints([action_eval_dict], run.constraints, projected_state=projected_state)
        hard_constraint_seconds = time.perf_counter() - hard_constraint_t0
        mech_feasible = str(action_eval.feasibility.feasibility_status) != "infeasible"
        feasible = bool(validation.valid) and len(hard_violations) == 0 and mech_feasible

        out.append(
            {
                "candidate": cand,
                "feasible": feasible,
                "feasibility_status": action_eval.feasibility.feasibility_status,
                "pass_probability": action_eval.feasibility.pass_probability,
                "action_candidate": action_eval_dict,
                "validation": validation.to_dict(),
                "hard_constraint_violations": hard_violations,
                "profiling": {
                    "action_eval_to_dict_seconds": round(action_eval_to_dict_seconds, 6),
                    "validation_seconds": round(validation_seconds, 6),
                    "hard_constraint_seconds": round(hard_constraint_seconds, 6),
                    "post_eval_seconds": round(time.perf_counter() - candidate_t0, 6),
                    "schema_lookup_seconds": round(float(eval_profile.get("schema_lookup_seconds", 0.0) or 0.0), 6),
                    "mechanism_feasibility_seconds": round(float(eval_profile.get("feasibility_seconds", 0.0) or 0.0), 6),
                    "mechanism_activation_seconds": round(float(eval_profile.get("mechanism_activation_seconds", 0.0) or 0.0), 6),
                    "impact_distribution_seconds": round(float(eval_profile.get("impact_distribution_seconds", 0.0) or 0.0), 6),
                    "structural_checks_seconds": round(float(eval_profile.get("structural_checks_seconds", 0.0) or 0.0), 6),
                    "risk_identification_seconds": round(float(eval_profile.get("risk_identification_seconds", 0.0) or 0.0), 6),
                    "assumptions_seconds": round(float(eval_profile.get("assumptions_seconds", 0.0) or 0.0), 6),
                    "evaluation_confidence_seconds": round(float(eval_profile.get("evaluation_confidence_seconds", 0.0) or 0.0), 6),
                    "candidate_id_seconds": round(float(eval_profile.get("candidate_id_seconds", 0.0) or 0.0), 6),
                    "mechanism_total_seconds": round(float(eval_profile.get("total_seconds", 0.0) or 0.0), 6),
                },
            }
        )
    if out:
        bulk_share = mechanism_eval_seconds / float(len(out))
        for row in out:
            profiling = dict(row.get("profiling", {}) or {})
            profiling["bulk_mechanism_eval_seconds_share"] = round(bulk_share, 6)
            if mechanism_batch_profile:
                profiling["bulk_mechanism_unattributed_seconds_share"] = round(
                    float(mechanism_batch_profile.get("unattributed_seconds", 0.0) or 0.0) / float(len(out)),
                    6,
                )
            profiling["estimated_total_seconds"] = round(
                float(profiling.get("post_eval_seconds", 0.0)) + bulk_share,
                6,
            )
            row["profiling"] = profiling
    if mechanism_batch_profile and out:
        mechanism_batch_profile["per_candidate_share_seconds"] = round(mechanism_eval_seconds / float(len(out)), 6)
        mechanism_batch_profile["per_candidate_unattributed_share_seconds"] = round(
            float(mechanism_batch_profile.get("unattributed_seconds", 0.0) or 0.0) / float(len(out)),
            6,
        )
    return out


def _pack_to_dict(pack: Any) -> Dict[str, Any]:
    _, _, PrecedentPack = _precedent_bindings()
    if isinstance(pack, dict):
        return pack
    if isinstance(pack, PrecedentPack):
        return pack.to_dict()
    if hasattr(pack, "to_dict"):
        return pack.to_dict()
    if is_dataclass(pack):
        return asdict(pack)
    raise ValueError(f"Unsupported precedent pack type: {type(pack)}")


def _latest_previous_causal_report(store: RecommendationRunStore, run: RecommendationRun) -> Optional[Dict[str, Any]]:
    try:
        runs = store.list_runs(company_id=run.company_id, status="completed")
    except Exception:
        return None

    # Most recent first, excluding current run_id.
    for prev in reversed(runs):
        if str(prev.run_id) == str(run.run_id):
            continue
        artifacts = dict(prev.metadata.get("artifacts", {}) or {})
        path = artifacts.get("CausalModelRiskReport")
        if not path:
            continue
        try:
            p = Path(str(path))
            if not p.exists():
                continue
            import json as _json

            obj = _json.loads(p.read_text())
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


def _persist_execution_config(
    *,
    store: RecommendationRunStore,
    run_id: str,
    runs_root: str | Path,
    snapshot_root: Optional[str | Path],
    snapshot_path: Optional[str | Path],
    entity_identifier_path: str | Path,
    action_ids: Optional[Sequence[str]],
    action_type: Optional[str],
    max_candidates: int,
    min_candidates_target: int,
    strict_evidence: bool,
    precedent_top_k: int,
    outcomes_path: Optional[str | Path],
    config_path: Optional[str | Path],
    top_plans: int,
) -> None:
    try:
        store.merge_metadata(
            run_id,
            {
                "config": {
                    "execution": build_execution_config(
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
                    ),
                    "runtime_env": capture_runtime_env_config(),
                }
            },
        )
    except Exception:
        return


def _retrieve_precedents(
    run: RecommendationRun,
    feasible_candidates: List[Dict[str, Any]],
    precedent_runner: Callable[..., Any],
    precedent_top_k: int,
    snapshot: Dict[str, Any],
    snapshot_root: Optional[str | Path],
    snapshot_path: Optional[str | Path],
    outcomes_path: Optional[str | Path],
    config_path: Optional[str | Path],
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[Dict[str, Any]]:
    _, _, _, _parse_ts, _, _, _, _ = _run_store_bindings()
    _, _, MechanismBrain = _candidate_bindings()
    as_of_date = _parse_ts(run.as_of_time).strftime("%Y-%m-%d")
    selected_candidates = _select_precedent_candidates(feasible_candidates, precedent_top_k)
    total = len(selected_candidates)
    if progress_callback:
        progress_callback(0, total)
    if total == 0:
        return []

    def _build_match(cand: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        pack = precedent_runner(
            company_id=run.company_id,
            as_of_date=as_of_date,
            action_id=cand["action_id"],
            action_type=cand.get("action_type"),
            action_subtype=cand.get("action_subtype"),
            action_params=dict(cand.get("parameters", cand.get("params", {})) or {}),
            config_path=str(config_path) if config_path else None,
            outcomes_path=str(outcomes_path) if outcomes_path else None,
            state_snapshot_root=str(snapshot_root) if snapshot_root else None,
            state_snapshot_path=str(snapshot_path) if snapshot_path else None,
            state_snapshot=snapshot,
            run_id=str(getattr(run, "run_id", "")),
            candidate_id=str(cand.get("candidate_id", "")),
        )
        pack_dict = _pack_to_dict(pack)
        blended_candidate = MechanismBrain.blend_precedent_into_action_candidate(
            action_candidate=dict(cand),
            precedent_pack=pack_dict,
        )
        return {
            "candidate": blended_candidate,
            "precedent_pack": pack_dict,
            "profiling": {
                "precedent_seconds": round(time.perf_counter() - t0, 6),
            },
        }

    workers = _precedent_worker_count(total)
    if workers <= 1:
        out: List[Dict[str, Any]] = []
        for idx, cand in enumerate(selected_candidates, start=1):
            out.append(_build_match(cand))
            if progress_callback and (idx == total or idx % 5 == 0):
                progress_callback(idx, total)
        return out

    ordered: List[Optional[Dict[str, Any]]] = [None] * total
    completed = 0

    # Warm the cold path once before fan-out. Without this, the first wave of
    # threadpool tasks all pay the same one-time import/cache setup cost, which
    # shows up as pathological first-company precedent latency.
    ordered[0] = _build_match(selected_candidates[0])
    completed = 1
    if progress_callback:
        progress_callback(completed, total)

    if total == 1:
        return [row for row in ordered if row is not None]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_index = {
            pool.submit(_build_match, cand): i for i, cand in enumerate(selected_candidates[1:], start=1)
        }
        for fut in as_completed(future_to_index):
            idx = future_to_index[fut]
            ordered[idx] = fut.result()
            completed += 1
            if progress_callback and (completed == total or completed % 5 == 0):
                progress_callback(completed, total)
    return [row for row in ordered if row is not None]


def _precedent_worker_count(total_candidates: int) -> int:
    if total_candidates <= 1:
        return 1
    if total_candidates < 10:
        return 1
    env_value = str(os.environ.get("RECO_PRECEDENT_WORKERS", "")).strip()
    if env_value:
        try:
            return max(1, min(total_candidates, int(env_value)))
        except Exception:
            pass
    return max(1, min(total_candidates, 6))


def _update_stage_progress(
    store: RecommendationRunStore,
    run_id: str,
    stage: str,
    completed: int,
    total: int,
) -> None:
    try:
        run = store.get_run(run_id)
        if run is None:
            return
        metadata = dict(run.metadata or {})
        by_stage = dict(metadata.get("progress_by_stage", {}) or {})
        safe_total = max(0, int(total))
        safe_completed = max(0, int(completed))
        ratio = 1.0 if safe_total == 0 else min(1.0, safe_completed / float(safe_total))
        by_stage[stage] = {
            "stage": stage,
            "completed": safe_completed,
            "total": safe_total,
            "ratio": ratio,
            "updated_at": _now_iso(),
        }
        metadata["progress_by_stage"] = by_stage
        metadata["progress"] = by_stage.get(stage)
        run.metadata = metadata
        store.update_run(run, sync_index=False)
    except Exception:
        return


def _percentile_seconds(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    arr = sorted(float(v) for v in values)
    idx = min(len(arr) - 1, max(0, int(round(q * (len(arr) - 1)))))
    return float(arr[idx])


def _slowest_candidates(
    rows: Sequence[Dict[str, Any]],
    latency_key: str,
    count: int = 10,
) -> List[Dict[str, Any]]:
    ranked: List[Dict[str, Any]] = []
    for row in rows:
        profiling = dict(row.get("profiling", {}) or {})
        seconds = float(profiling.get(latency_key, 0.0) or 0.0)
        candidate = dict(row.get("candidate", {}) or row.get("action_candidate", {}) or {})
        item: Dict[str, Any] = {
            "candidate_id": str(candidate.get("candidate_id", "")),
            "action_id": str(candidate.get("action_id", "")),
            "seconds": round(seconds, 6),
        }
        for extra_key in (
            "bulk_mechanism_eval_seconds_share",
            "post_eval_seconds",
            "validation_seconds",
            "action_eval_to_dict_seconds",
            "hard_constraint_seconds",
        ):
            if extra_key in profiling:
                item[extra_key] = round(float(profiling.get(extra_key, 0.0) or 0.0), 6)
        ranked.append(item)
    ranked.sort(key=lambda x: (-float(x["seconds"]), x["action_id"], x["candidate_id"]))
    return ranked[:count]


def _feasibility_profile(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    vals = [
        float(dict(row.get("profiling", {}) or {}).get("estimated_total_seconds", 0.0) or 0.0)
        for row in rows
    ]
    post_eval_vals = [
        float(dict(row.get("profiling", {}) or {}).get("post_eval_seconds", 0.0) or 0.0)
        for row in rows
    ]
    validation_vals = [
        float(dict(row.get("profiling", {}) or {}).get("validation_seconds", 0.0) or 0.0)
        for row in rows
    ]
    to_dict_vals = [
        float(dict(row.get("profiling", {}) or {}).get("action_eval_to_dict_seconds", 0.0) or 0.0)
        for row in rows
    ]
    hard_constraint_vals = [
        float(dict(row.get("profiling", {}) or {}).get("hard_constraint_seconds", 0.0) or 0.0)
        for row in rows
    ]
    bulk_share_vals = [
        float(dict(row.get("profiling", {}) or {}).get("bulk_mechanism_eval_seconds_share", 0.0) or 0.0)
        for row in rows
    ]
    bulk_unattributed_vals = [
        float(dict(row.get("profiling", {}) or {}).get("bulk_mechanism_unattributed_seconds_share", 0.0) or 0.0)
        for row in rows
    ]
    schema_lookup_vals = [
        float(dict(row.get("profiling", {}) or {}).get("schema_lookup_seconds", 0.0) or 0.0)
        for row in rows
    ]
    mech_feas_vals = [
        float(dict(row.get("profiling", {}) or {}).get("mechanism_feasibility_seconds", 0.0) or 0.0)
        for row in rows
    ]
    mech_activation_vals = [
        float(dict(row.get("profiling", {}) or {}).get("mechanism_activation_seconds", 0.0) or 0.0)
        for row in rows
    ]
    impact_vals = [
        float(dict(row.get("profiling", {}) or {}).get("impact_distribution_seconds", 0.0) or 0.0)
        for row in rows
    ]
    structural_vals = [
        float(dict(row.get("profiling", {}) or {}).get("structural_checks_seconds", 0.0) or 0.0)
        for row in rows
    ]
    risk_vals = [
        float(dict(row.get("profiling", {}) or {}).get("risk_identification_seconds", 0.0) or 0.0)
        for row in rows
    ]
    assumptions_vals = [
        float(dict(row.get("profiling", {}) or {}).get("assumptions_seconds", 0.0) or 0.0)
        for row in rows
    ]
    eval_conf_vals = [
        float(dict(row.get("profiling", {}) or {}).get("evaluation_confidence_seconds", 0.0) or 0.0)
        for row in rows
    ]
    candidate_id_vals = [
        float(dict(row.get("profiling", {}) or {}).get("candidate_id_seconds", 0.0) or 0.0)
        for row in rows
    ]
    mechanism_total_vals = [
        float(dict(row.get("profiling", {}) or {}).get("mechanism_total_seconds", 0.0) or 0.0)
        for row in rows
    ]
    return {
        "count": int(len(rows)),
        "p50_seconds": round(_percentile_seconds(vals, 0.50), 6),
        "p95_seconds": round(_percentile_seconds(vals, 0.95), 6),
        "max_seconds": round(max(vals) if vals else 0.0, 6),
        "post_eval_p50_seconds": round(_percentile_seconds(post_eval_vals, 0.50), 6),
        "post_eval_p95_seconds": round(_percentile_seconds(post_eval_vals, 0.95), 6),
        "validation_p50_seconds": round(_percentile_seconds(validation_vals, 0.50), 6),
        "validation_p95_seconds": round(_percentile_seconds(validation_vals, 0.95), 6),
        "action_eval_to_dict_p50_seconds": round(_percentile_seconds(to_dict_vals, 0.50), 6),
        "action_eval_to_dict_p95_seconds": round(_percentile_seconds(to_dict_vals, 0.95), 6),
        "hard_constraint_p50_seconds": round(_percentile_seconds(hard_constraint_vals, 0.50), 6),
        "hard_constraint_p95_seconds": round(_percentile_seconds(hard_constraint_vals, 0.95), 6),
        "schema_lookup_p50_seconds": round(_percentile_seconds(schema_lookup_vals, 0.50), 6),
        "mechanism_feasibility_p50_seconds": round(_percentile_seconds(mech_feas_vals, 0.50), 6),
        "mechanism_activation_p50_seconds": round(_percentile_seconds(mech_activation_vals, 0.50), 6),
        "impact_distribution_p50_seconds": round(_percentile_seconds(impact_vals, 0.50), 6),
        "structural_checks_p50_seconds": round(_percentile_seconds(structural_vals, 0.50), 6),
        "risk_identification_p50_seconds": round(_percentile_seconds(risk_vals, 0.50), 6),
        "assumptions_p50_seconds": round(_percentile_seconds(assumptions_vals, 0.50), 6),
        "evaluation_confidence_p50_seconds": round(_percentile_seconds(eval_conf_vals, 0.50), 6),
        "candidate_id_p50_seconds": round(_percentile_seconds(candidate_id_vals, 0.50), 6),
        "mechanism_total_p50_seconds": round(_percentile_seconds(mechanism_total_vals, 0.50), 6),
        "mechanism_total_p95_seconds": round(_percentile_seconds(mechanism_total_vals, 0.95), 6),
        "bulk_mechanism_eval_seconds_share": round(_percentile_seconds(bulk_share_vals, 0.50), 6),
        "bulk_mechanism_eval_seconds_total": round(sum(bulk_share_vals), 6),
        "bulk_mechanism_unattributed_seconds_share": round(_percentile_seconds(bulk_unattributed_vals, 0.50), 6),
        "bulk_mechanism_unattributed_seconds_total": round(sum(bulk_unattributed_vals), 6),
        "slowest": _slowest_candidates(rows, "estimated_total_seconds"),
    }


def _precedent_profile(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    vals = [
        float(dict(row.get("profiling", {}) or {}).get("precedent_seconds", 0.0) or 0.0)
        for row in rows
    ]
    action_counts: Dict[str, int] = {}
    for row in rows:
        candidate = dict(row.get("candidate", {}) or {})
        action_id = str(candidate.get("action_id", "") or "")
        if not action_id:
            continue
        action_counts[action_id] = action_counts.get(action_id, 0) + 1
    return {
        "count": int(len(rows)),
        "p50_seconds": round(_percentile_seconds(vals, 0.50), 6),
        "p95_seconds": round(_percentile_seconds(vals, 0.95), 6),
        "max_seconds": round(max(vals) if vals else 0.0, 6),
        "slowest": _slowest_candidates(rows, "precedent_seconds"),
        "selected_action_counts": dict(sorted(action_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:10]),
    }


def _emit_stage_profile(stage: str, profile: Dict[str, Any]) -> None:
    try:
        print(
            json.dumps(
                {
                    "ok": True,
                    "event": "stage_profile",
                    "stage": str(stage),
                    **dict(profile or {}),
                }
            ),
            flush=True,
        )
    except Exception:
        return


def _select_precedent_candidates(
    feasible_candidates: List[Dict[str, Any]],
    precedent_top_k: int,
) -> List[Dict[str, Any]]:
    debt_action_ids = {
        "capital_structure.new_debt_issuance",
        "capital_structure.refinancing",
    }
    k = int(precedent_top_k or 0)
    if k <= 0:
        return []
    if len(feasible_candidates) <= k:
        return feasible_candidates

    def _driver_contribution(c: Dict[str, Any], driver_name: str) -> float:
        impact = dict(c.get("impact_distribution", {}) or {})
        drivers = impact.get("key_drivers") or []
        for row in drivers:
            if not isinstance(row, dict):
                continue
            if str(row.get("driver_name", "")) != str(driver_name):
                continue
            try:
                return float(row.get("contribution", 0.0) or 0.0)
            except Exception:
                return 0.0
        return 0.0

    def _strict_causal_priority(c: Dict[str, Any]) -> float:
        mode = _driver_contribution(c, "causal_model_mode")
        blend = _driver_contribution(c, "causal_model_blend_weight")
        if mode >= 0.5 or blend > 0.0:
            quality = max(0.0, _driver_contribution(c, "causal_model_quality"))
            support = max(0.0, _driver_contribution(c, "causal_model_support_score"))
            return 1.0 + (quality * support)
        return 0.0

    def _key(c: Dict[str, Any]) -> Any:
        strict_priority = _strict_causal_priority(c)
        eval_conf = float(c.get("evaluation_confidence", 0.0) or 0.0)
        score = float(c.get("generation_confidence", 0.0) or 0.0)
        action_id = str(c.get("action_id", "") or "")
        signature = str(c.get("candidate_signature", "") or "")
        cid = str(c.get("candidate_id", "") or "")
        return (-strict_priority, -eval_conf, -score, action_id, signature, cid)

    ranked = sorted(feasible_candidates, key=_key)

    env_cap = str(os.environ.get("RECO_PRECEDENT_MAX_PER_ACTION", "")).strip()
    try:
        per_action_cap = max(1, int(env_cap)) if env_cap else 3
    except Exception:
        per_action_cap = 3
    env_debt_cap = str(os.environ.get("RECO_PRECEDENT_MAX_PER_DEBT_ACTION", "")).strip()
    try:
        debt_action_cap = max(1, int(env_debt_cap)) if env_debt_cap else per_action_cap
    except Exception:
        debt_action_cap = per_action_cap

    selected: List[Dict[str, Any]] = []
    overflow: List[Dict[str, Any]] = []
    action_counts: Dict[str, int] = {}

    for cand in ranked:
        action_id = str(cand.get("action_id", "") or "")
        max_for_action = debt_action_cap if action_id in debt_action_ids else per_action_cap
        current = int(action_counts.get(action_id, 0))
        if current < max_for_action:
            selected.append(cand)
            action_counts[action_id] = current + 1
            if len(selected) >= k:
                return selected[:k]
        else:
            overflow.append(cand)

    if len(selected) < k:
        selected.extend(overflow[: max(0, k - len(selected))])
    return selected[:k]


def _select_distribution(pack: Dict[str, Any]) -> Dict[str, Any]:
    dists = pack.get("legacy_distributions", []) if isinstance(pack.get("legacy_distributions"), list) else []
    if not dists and isinstance(pack.get("distributions"), list):
        dists = pack.get("distributions", [])
    if not dists:
        return {}
    # Prefer 12m PE distribution, then any 12m, then first.
    for metric in ("outcome_pe_12m", "outcome_ev_ebitda_12m", "outcome_pe_6m"):
        for d in dists:
            if str(d.get("metric", "")) == metric:
                return d
    for d in dists:
        if str(d.get("metric", "")).endswith("_12m"):
            return d
    return dists[0]


def _build_plan_set(
    run: RecommendationRun,
    feasible_candidates: List[Dict[str, Any]],
    precedent_matches: List[Dict[str, Any]],
    registry: ActionSchemaRegistry,
    top_plans: int,
) -> Dict[str, Any]:
    from .planner_brain import build_plan_set

    return build_plan_set(
        run=run,
        feasible_candidates=feasible_candidates,
        precedent_matches=precedent_matches,
        registry=registry,
        top_plans=top_plans,
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = ["execute_recommendation_run", "create_and_execute_recommendation_run"]
