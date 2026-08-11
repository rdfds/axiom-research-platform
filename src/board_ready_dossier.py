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


def _candidate_tail_penalty(candidate: Dict[str, Any]) -> float:
    risks = list(candidate.get("risks", []) or [])
    return min(0.25, 0.05 * float(len(risks)))


def _diagnose_context(snapshot: Dict[str, Any], top_plan: Dict[str, Any], registry: Any) -> Dict[str, Any]:
    steps = list(top_plan.get("steps", []) or [])
    action_ids = [str(step.get("action_id", "") or "") for step in steps]
    first_action = action_ids[0] if action_ids else ""
    available_for_actions = _safe_float(_feature_value(snapshot, "liquidity.available_for_actions"))
    market_cap = _safe_float(_feature_value(snapshot, "market.market_cap"))
    net_leverage = _safe_float(_feature_value(snapshot, "capital_structure.net_leverage"))
    maturity_wall = _safe_float(_feature_value(snapshot, "capital_structure.maturity_wall_ratio_24m"))
    fcf_conversion = _safe_float(_feature_value(snapshot, "operating.fcf_conversion"))
    revenue_yoy = _safe_float(_feature_value(snapshot, "operating.revenue_yoy_last_q"))
    equity_window = _safe_float(_feature_value(snapshot, "market.equity_window_proxy"))
    credit_window = _safe_float(_feature_value(snapshot, "market.credit_window_proxy"))
    return_capital_priority = _safe_float(_feature_value(snapshot, "strategic.intent.return_capital_priority"))
    pursue_mna_priority = _safe_float(_feature_value(snapshot, "strategic.intent.pursue_mna_priority"))
    focus_on_core = _safe_float(_feature_value(snapshot, "strategic.intent.focus_on_core"))
    liquidity_to_mcap = None
    if available_for_actions is not None and market_cap not in (None, 0.0):
        liquidity_to_mcap = available_for_actions / market_cap

    primary_problem = "Capital allocation is not yet crisply diagnosed."
    timing_posture = "deliberate"
    interaction_logic: List[str] = []

    if _is_capacity_then_return_sequence(action_ids):
        primary_problem = (
            "The company can support shareholder return, but it first has to create enough balance-sheet "
            "capacity to fund that return from strength rather than necessity."
        )
        timing_posture = "sequence_now"
        interaction_logic.append("The plan is sequencing-driven: financing flexibility is created before capital is returned.")
    elif _is_capital_return_action(first_action):
        primary_problem = _capital_return_problem_statement(
            liquidity_to_mcap=liquidity_to_mcap,
            net_leverage=net_leverage,
            revenue_yoy=revenue_yoy,
            fcf_conversion=fcf_conversion,
        )
        if liquidity_to_mcap is not None:
            interaction_logic.append(
                f"Deployable liquidity is {_fmt_pct(liquidity_to_mcap)} of market value, which is large enough to move capital allocation."
            )
        if net_leverage is not None:
            interaction_logic.append(f"Net leverage is {_fmt_x(net_leverage)}, so payout capacity is not obviously constrained.")
        timing_posture = "opportunistic_now" if (liquidity_to_mcap or 0.0) >= 0.03 else "measured_now"
    elif _is_balance_sheet_action(first_action):
        primary_problem = _balance_sheet_problem_statement(
            maturity_wall=maturity_wall,
            net_leverage=net_leverage,
            credit_window=credit_window,
        )
        if maturity_wall is not None:
            interaction_logic.append(f"The near-term maturity wall proxy is {_fmt_pct(maturity_wall)}, which directly affects timing.")
        if net_leverage is not None and net_leverage >= 3.0:
            interaction_logic.append(f"Net leverage is already {_fmt_x(net_leverage)}, which narrows room for discretionary moves.")
        timing_posture = "urgent_now" if (maturity_wall or 0.0) >= 0.20 else "window_sensitive"
    elif _is_mna_action(first_action):
        primary_problem = _mna_problem_statement(
            pursue_mna_priority=pursue_mna_priority,
            net_leverage=net_leverage,
        )
        if pursue_mna_priority is not None:
            interaction_logic.append(f"Management intent to pursue M&A scores {_fmt_score(pursue_mna_priority)}.")
        timing_posture = "window_sensitive"
    elif _is_divestiture_action(first_action):
        primary_problem = _divestiture_problem_statement(
            focus_on_core=focus_on_core,
            net_leverage=net_leverage,
        )
        if focus_on_core is not None:
            interaction_logic.append(f"Focus-on-core intent scores {_fmt_score(focus_on_core)}.")
        timing_posture = "deliberate_now"

    if (maturity_wall or 0.0) >= 0.20:
        interaction_logic.append("Waiting increases the risk that financing decisions are made under worse terms.")
    if (credit_window or 0.0) >= 0.65 and _uses_credit_markets(action_ids):
        interaction_logic.append(f"Credit window proxy is {_fmt_score(credit_window)}, so market access is supportive rather than punitive.")
    if (equity_window or 0.0) >= 0.65 and _uses_equity_markets(action_ids):
        interaction_logic.append(f"Equity window proxy is {_fmt_score(equity_window)}, so the issuance window is open enough to act.")
    if (return_capital_priority or 0.0) >= 0.75 and _has_capital_return(action_ids):
        interaction_logic.append("Management signaling already leans toward capital return, which lowers execution surprise.")
    if revenue_yoy is not None and revenue_yoy <= 0.02 and _has_capital_return(action_ids):
        interaction_logic.append(f"Revenue growth is only {_fmt_pct(revenue_yoy)}, which weakens the case for keeping excess cash idle.")
    if fcf_conversion is not None and fcf_conversion >= 0.7 and _has_capital_return(action_ids):
        interaction_logic.append(f"FCF conversion is {_fmt_ratio(fcf_conversion)}, supporting a durable payout or repurchase posture.")

    return {
        "primary_problem": primary_problem,
        "timing_posture": timing_posture,
        "sequence_logic": interaction_logic[:4],
        "context_metrics": {
            "available_for_actions_usd": available_for_actions,
            "liquidity_to_market_cap": liquidity_to_mcap,
            "net_leverage": net_leverage,
            "maturity_wall_ratio_24m": maturity_wall,
            "fcf_conversion": fcf_conversion,
            "revenue_yoy_last_q": revenue_yoy,
            "equity_window_proxy": equity_window,
            "credit_window_proxy": credit_window,
        },
    }


def _capital_return_problem_statement(
    *,
    liquidity_to_mcap: Optional[float],
    net_leverage: Optional[float],
    revenue_yoy: Optional[float],
    fcf_conversion: Optional[float],
) -> str:
    if (liquidity_to_mcap or 0.0) >= 0.05 and (revenue_yoy is not None and revenue_yoy <= 0.02):
        return "The core issue is excess deployable capital relative to near-term operating demand, so the board has to choose the least-regrettable way to return it."
    if (liquidity_to_mcap or 0.0) >= 0.03 and (fcf_conversion or 0.0) >= 0.70 and (net_leverage or 0.0) < 2.5:
        return "The company has enough cash generation and balance-sheet room to support shareholder return; the decision is which return path preserves the most flexibility."
    return "The core issue is capital allocation discipline: deployable capital is available and the question is how to return it without damaging flexibility."


def _balance_sheet_problem_statement(
    *,
    maturity_wall: Optional[float],
    net_leverage: Optional[float],
    credit_window: Optional[float],
) -> str:
    if (maturity_wall or 0.0) >= 0.20 and (credit_window or 0.0) >= 0.60:
        return "The immediate issue is to term out an elevated near-term maturity wall while debt markets are still open enough to do it on acceptable terms."
    if (maturity_wall or 0.0) >= 0.20:
        return "The immediate issue is to manage an elevated near-term maturity wall before financing conditions become materially worse."
    if (net_leverage or 0.0) >= 3.0:
        return "The main issue is protecting financing flexibility at elevated leverage, so discretionary moves should wait until the balance sheet is steadier."
    return "The main issue is preserving financing flexibility so later strategic moves are funded from strength rather than necessity."


def _mna_problem_statement(*, pursue_mna_priority: Optional[float], net_leverage: Optional[float]) -> str:
    if (pursue_mna_priority or 0.0) >= 0.70 and (net_leverage or 0.0) < 2.5:
        return "The decision is whether the current balance sheet and strategic mandate justify using external growth now rather than keeping dry powder."
    return "The decision is about external growth: whether current balance-sheet capacity and strategic intent justify acting on M&A now."


def _divestiture_problem_statement(*, focus_on_core: Optional[float], net_leverage: Optional[float]) -> str:
    if (focus_on_core or 0.0) >= 0.70:
        return "The issue is portfolio focus: a divestiture has to simplify the story enough to justify selling now rather than waiting."
    if (net_leverage or 0.0) >= 2.75:
        return "The issue is capital release: a divestiture only makes sense if it improves flexibility faster than financing alone."
    return "The issue is portfolio focus and capital release: a divestiture only makes sense if it improves focus, flexibility, or both."


def _humanize_use_of_proceeds(raw_value: str) -> str:
    value = str(raw_value or "").strip().lower()
    mapping = {
        "refinancing": "term out upcoming maturities",
        "general_corporate": "fund general corporate needs without using scarce cash",
        "liquidity_buffer": "rebuild liquidity reserves",
        "deleveraging": "reduce leverage and protect the rating envelope",
        "buyback": "fund shareholder return from a position of strength",
        "reinvestment": "reinvest into core priorities",
    }
    return mapping.get(value, _humanize_text(value) or "support the balance sheet")


def _for_phrase(text: str) -> str:
    clean = str(text or "").strip()
    if not clean:
        return ""
    return f" for {clean}"


def _build_recommendation_thesis(
    *,
    top_plan: Dict[str, Any],
    step_theses: Sequence[Dict[str, Any]],
    diagnosed: Dict[str, Any],
    snapshot: Dict[str, Any],
    confidence_posture: str,
    status_quo_view: Dict[str, Any],
) -> Dict[str, Any]:
    steps = list(top_plan.get("steps", []) or [])
    action_ids = [str(step.get("action_id", "") or "") for step in steps]
    first_step = step_theses[0] if step_theses else {}
    first_action = action_ids[0] if action_ids else ""
    sequence_text = " then ".join(_humanize_action_id(action_id) for action_id in action_ids[:3])
    if len(action_ids) > 3:
        sequence_text += " before later follow-ons"
    plan_score = float(top_plan.get("score", 0.0) or 0.0)
    support_score = _safe_float(((top_plan.get("score_components", {}) or {}).get("support_factor")))
    feasibility_chain = _safe_float(((top_plan.get("score_components", {}) or {}).get("feasibility_chain")))
    why_now = str(first_step.get("why_now", "") or "")
    problem_statement = str(diagnosed.get("primary_problem", "") or "")
    why_this_plan = str(first_step.get("why_this_step", "") or "")
    if len(action_ids) > 1:
        why_this_plan = f"{why_this_plan} The sequence matters: {sequence_text}."

    recommended_posture = str(status_quo_view.get("recommended_posture", "") or "conditional_action")
    if recommended_posture == "wait":
        why_this_plan = (
            f"No immediate action clears the act-now bar. The highest-ranked conditional path is {sequence_text}, "
            f"but the better current posture is to preserve flexibility."
        )
        why_now = str(status_quo_view.get("why_wait", "") or why_now)
        executive_summary = (
            f"{problem_statement} The better current posture is to wait rather than force an action. "
            f"{why_now} The leading conditional path is {sequence_text}, but it only becomes attractive if the "
            f"current objections ease. The case for action is currently weaker than waiting, and the overall posture remains "
            f"{confidence_posture.replace('_', ' ')}."
        ).strip()
    else:
        action_descriptor = "recommended path" if recommended_posture == "act_now" else "best conditional path"
        executive_summary = (
            f"{problem_statement} The {action_descriptor} is {sequence_text}. "
            f"{why_this_plan} {why_now} {_edge_summary(float(status_quo_view.get('edge_vs_status_quo', 0.0) or 0.0), recommended_posture)} "
            f"Empirical support is {'strong' if (support_score or 0.0) >= 0.8 else 'mixed'}, and the overall posture is "
            f"{confidence_posture.replace('_', ' ')}."
        ).strip()

    what_has_to_be_true = _decision_preconditions(first_action=first_action, snapshot=snapshot, top_plan=top_plan)
    what_changes_mind = _decision_boundaries(first_action=first_action, snapshot=snapshot, top_plan=top_plan)

    return {
        "problem_statement": problem_statement,
        "why_this_plan": why_this_plan,
        "why_now": why_now,
        "recommended_posture": recommended_posture,
        "case_for_action": list(status_quo_view.get("case_for_action", []) or []),
        "case_for_wait": list(status_quo_view.get("case_for_wait", []) or []),
        "sizing_summary": {},
        "what_has_to_be_true": what_has_to_be_true,
        "what_would_change_our_mind": what_changes_mind,
        "executive_summary": executive_summary,
        "plan_score": plan_score,
        "support_factor": support_score,
        "feasibility_chain": feasibility_chain,
    }


