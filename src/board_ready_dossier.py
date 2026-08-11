from __future__ import annotations

from datetime import datetime
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .model_feature_bundle import feature_view_from_snapshot
from .runtime_feature_adapter import resolve_feature_value


_OBJECTIVE_FIELDS = (
    "value_creation",
    "risk_reduction",
    "growth",
    "rating_preservation",
    "optionality",
)


def build_board_ready_dossier(
    *,
    run: Any,
    snapshot: Dict[str, Any],
    plan_set: Dict[str, Any],
    feasible_candidates: Sequence[Dict[str, Any]],
    precedent_matches: Sequence[Dict[str, Any]],
    registry: Any,
) -> Dict[str, Any]:
    plans = list(plan_set.get("plans", []) or [])
    top_plan = dict(plans[0] or {}) if plans else {}
    top_steps = list(top_plan.get("steps", []) or [])
    top_actions = list(top_plan.get("actions", []) or [])
    if not top_plan:
        return {
            "run_id": str(getattr(run, "run_id", "") or ""),
            "company_id": str(getattr(run, "company_id", "") or ""),
            "as_of_time": str(getattr(run, "as_of_time", "") or ""),
            "generated_at": _now_iso(),
            "status": "no_plan",
            "executive_summary": "No feasible plan was generated.",
        }

    candidate_by_action = _best_candidate_by_action(feasible_candidates)
    precedent_by_action = _best_precedent_by_action(precedent_matches)
    diagnosed = _diagnose_context(snapshot=snapshot, top_plan=top_plan, registry=registry)
    step_theses = [
        _build_step_thesis(
            step=step,
            action_candidate=_resolve_action_candidate(
                step_action_id=str(step.get("action_id", "") or ""),
                top_actions=top_actions,
                candidate_by_action=candidate_by_action,
            ),
            precedent_pack=precedent_by_action.get(str(step.get("action_id", "") or ""), {}),
            snapshot=snapshot,
            registry=registry,
            plan=top_plan,
            diagnosed=diagnosed,
        )
        for step in top_steps
    ]
    supporting_evidence = _build_supporting_evidence(
        snapshot=snapshot,
        top_plan=top_plan,
        step_theses=step_theses,
        precedent_by_action=precedent_by_action,
    )
    status_quo_view = _build_status_quo_view(
        top_plan=top_plan,
        step_theses=step_theses,
        snapshot=snapshot,
        diagnosed=diagnosed,
    )
    alternative_analysis = _build_alternative_analysis(
        top_plan=top_plan,
        other_plans=plans[1:4],
        diagnosed=diagnosed,
        snapshot=snapshot,
        candidate_by_action=candidate_by_action,
        precedent_by_action=precedent_by_action,
    )
    risk_case = _build_risk_case(
        top_plan=top_plan,
        step_theses=step_theses,
        snapshot=snapshot,
    )
    confidence_posture = _confidence_posture(
        top_plan=top_plan,
        step_theses=step_theses,
        precedent_by_action=precedent_by_action,
    )
    recommendation_thesis = _build_recommendation_thesis(
        top_plan=top_plan,
        step_theses=step_theses,
        diagnosed=diagnosed,
        snapshot=snapshot,
        confidence_posture=confidence_posture,
        status_quo_view=status_quo_view,
    )
    sizing_guidance = _build_plan_sizing_guidance(
        top_plan=top_plan,
        step_theses=step_theses,
        snapshot=snapshot,
    )
    parameter_optimization = _build_parameter_optimization(
        top_plan=top_plan,
        snapshot=snapshot,
        registry=registry,
        sizing_guidance=sizing_guidance,
    )
    sizing_guidance["scenario_overrides"] = _build_scenario_sizing(
        top_plan=top_plan,
        snapshot=snapshot,
        base_sizing=sizing_guidance,
    )
    sizing_guidance["parameter_optimization"] = parameter_optimization
    regret_analysis = _build_regret_analysis(
        top_plan=top_plan,
        step_theses=step_theses,
        snapshot=snapshot,
        status_quo_view=status_quo_view,
    )
    rating_cliff_analysis = _build_rating_cliff_analysis(
        top_plan=top_plan,
        snapshot=snapshot,
    )
    signaling_analysis = _build_signaling_analysis(
        top_plan=top_plan,
        snapshot=snapshot,
        status_quo_view=status_quo_view,
    )
    recommendation_thesis["sizing_summary"] = sizing_guidance
    recommendation_thesis["parameter_summary"] = parameter_optimization.get("summary")
    recommendation_thesis["regret_balance"] = regret_analysis.get("regret_balance")
    recommendation_thesis["rating_constraint_posture"] = rating_cliff_analysis.get("constraint_posture")
    recommendation_thesis["market_signal_posture"] = signaling_analysis.get("signal_posture")
    ranked_action_views = _build_ranked_action_views(
        plans=plans[:3],
        snapshot=snapshot,
        registry=registry,
        diagnosed=diagnosed,
        candidate_by_action=candidate_by_action,
        precedent_by_action=precedent_by_action,
    )
    first_action = str((top_steps[0].get("action_id", "") if top_steps else "") or "")
    monitoring_triggers = _dedupe_trigger_rows(_humanize_triggers(list(top_plan.get("triggers", []) or [])))
    if not monitoring_triggers:
        monitoring_triggers = _fallback_monitoring_triggers(first_action=first_action, snapshot=snapshot)
    monitoring = {
        "triggers": monitoring_triggers,
        "branches": _humanize_branches(list(top_plan.get("branches", []) or [])),
        "kill_criteria": list(risk_case.get("kill_criteria", []) or []),
    }
    scorecard = _build_scorecard(
        top_plan=top_plan,
        step_theses=step_theses,
        precedent_by_action=precedent_by_action,
    )

    return {
        "run_id": str(getattr(run, "run_id", "") or ""),
        "company_id": str(getattr(run, "company_id", "") or ""),
        "as_of_time": str(getattr(run, "as_of_time", "") or ""),
        "generated_at": _now_iso(),
        "plan_id": str(top_plan.get("plan_id", "") or ""),
        "confidence_posture": confidence_posture,
        "executive_summary": recommendation_thesis["executive_summary"],
        "situation_assessment": diagnosed,
        "recommendation_thesis": recommendation_thesis,
        "sizing_guidance": sizing_guidance,
        "parameter_optimization": parameter_optimization,
        "regret_analysis": regret_analysis,
        "rating_cliff_analysis": rating_cliff_analysis,
        "signaling_analysis": signaling_analysis,
        "status_quo_view": status_quo_view,
        "ranked_action_views": ranked_action_views,
        "supporting_evidence": supporting_evidence,
        "step_theses": step_theses,
        "alternative_analysis": alternative_analysis,
        "risk_case": risk_case,
        "monitoring": monitoring,
        "scorecard": scorecard,
    }


def _resolve_action_candidate(
    *,
    step_action_id: str,
    top_actions: Sequence[Dict[str, Any]],
    candidate_by_action: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    for action in top_actions:
        if str((action or {}).get("action_id", "") or "") == step_action_id:
            return dict(action or {})
    return dict(candidate_by_action.get(step_action_id) or {})


def _best_candidate_by_action(candidates: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for candidate in candidates:
        cand = dict(candidate or {})
        action_id = str(cand.get("action_id", "") or "")
        if not action_id:
            continue
        current = out.get(action_id)
        if current is None or _candidate_support_score(cand) > _candidate_support_score(current):
            out[action_id] = cand
    return out


def _best_precedent_by_action(precedent_matches: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in precedent_matches:
        candidate = dict(row.get("candidate", {}) or {})
        action_id = str(candidate.get("action_id", "") or "")
        if not action_id:
            continue
        pack = dict(row['precedent_pack'] or {})
        current = out.get(action_id)
        if current is None or _precedent_confidence(pack) > _precedent_confidence(current):
            out[action_id] = pack
    return out


def _candidate_support_score(candidate: Dict[str, Any]) -> float:
    impact = dict(candidate.get("impact_distribution", {}) or {})
    uncertainty = float(impact.get("uncertainty_score", 1.0) or 1.0)
    return float(candidate.get("evaluation_confidence", 0.0) or 0.0) - (0.25 * uncertainty)


def _candidate_expected_utility(candidate: Dict[str, Any]) -> float:
    objectives = dict(((candidate.get("impact_distribution", {}) or {}).get("objectives", {}) or {}))
    total = 0.0
    seen = 0
    for objective in _OBJECTIVE_FIELDS:
        value = _safe_float(((objectives.get(objective, {}) or {}).get("median")))
        if value is None:
            continue
        total += value
        seen += 1
    if not seen:
        return 0.0
    return 0.5 + (0.5 * total)


