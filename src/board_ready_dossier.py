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
        pack = dict(row.get("precedent_pack", {}) or {})
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


def _build_supporting_evidence(
    *,
    snapshot: Dict[str, Any],
    top_plan: Dict[str, Any],
    step_theses: Sequence[Dict[str, Any]],
    precedent_by_action: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    metrics = [
        ("Deployable liquidity", "liquidity.available_for_actions", _fmt_currency),
        ("Net leverage", "capital_structure.net_leverage", _fmt_x),
        ("Maturity wall (24m)", "capital_structure.maturity_wall_ratio_24m", _fmt_pct),
        ("FCF conversion", "operating.fcf_conversion", _fmt_ratio),
        ("Revenue growth", "operating.revenue_yoy_last_q", _fmt_pct),
        ("Equity-market conditions", "market.equity_window_proxy", _fmt_score),
        ("Debt-market conditions", "market.credit_window_proxy", _fmt_score),
    ]
    for label, key, formatter in metrics:
        value = _feature_value(snapshot, key)
        if value is None:
            continue
        out.append(
            {
                "label": label,
                "metric": key,
                "value": value,
                "formatted_value": formatter(value),
                "text": f"{label} is {formatter(value)}.",
                "source": "snapshot",
            }
        )

    for thesis in step_theses[:2]:
        out.extend(list(thesis.get("supporting_facts", []) or [])[:3])

    for action_id, pack in precedent_by_action.items():
        if len(out) >= 12:
            break
        confidence = _precedent_confidence(pack)
        tier = str(((pack.get("mismatch_diagnostics", {}) or {}).get("retrieval_tier", "")) or "")
        sample_n = _precedent_sample_size(pack)
        if confidence <= 0.0:
            continue
        out.append(
            {
                "label": f"Precedent for {_humanize_action_id(action_id)}",
                "metric": "precedent_confidence",
                "value": confidence,
                "formatted_value": f"{confidence:.3f}",
                "text": f"Precedent confidence is {confidence:.3f} on a {tier or 'unknown'} cohort with n={sample_n}.",
                "source": "precedent",
            }
        )
    return out[:12]


def _build_status_quo_view(
    *,
    top_plan: Dict[str, Any],
    step_theses: Sequence[Dict[str, Any]],
    snapshot: Dict[str, Any],
    diagnosed: Dict[str, Any],
) -> Dict[str, Any]:
    first_step = dict(step_theses[0] or {}) if step_theses else {}
    evaluation = _evaluate_plan_vs_status_quo(
        plan=top_plan,
        first_step_thesis=first_step,
        snapshot=snapshot,
        diagnosed=diagnosed,
    )
    return {
        "recommended_posture": evaluation["recommended_posture"],
        "status_quo_preferred": evaluation["recommended_posture"] == "wait",
        "edge_vs_status_quo": evaluation["edge_vs_status_quo"],
        "edge_vs_status_quo_formatted": f"{evaluation['edge_vs_status_quo']:+.3f}",
        "status_quo_score": evaluation["status_quo_score"],
        "why_act_now": evaluation["why_act_now"],
        "why_wait": evaluation["why_wait"],
        "case_for_action": evaluation["case_for_action"],
        "case_for_wait": evaluation["case_for_wait"],
        "key_counterarguments": evaluation["case_for_wait"][:3],
        "reassessment_triggers": evaluation["reassessment_triggers"],
    }


def _build_ranked_action_views(
    *,
    plans: Sequence[Dict[str, Any]],
    snapshot: Dict[str, Any],
    registry: Any,
    diagnosed: Dict[str, Any],
    candidate_by_action: Dict[str, Dict[str, Any]],
    precedent_by_action: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx, plan in enumerate(plans, start=1):
        plan = dict(plan or {})
        steps = list(plan.get("steps", []) or [])
        action_ids = [str(step.get("action_id", "") or "") for step in steps]
        first_action = str((action_ids[0] if action_ids else "") or "")
        action_candidate = _resolve_plan_action_candidate(plan=plan, action_id=first_action, candidate_by_action=candidate_by_action)
        step_thesis = _build_step_thesis(
            step=dict(steps[0] or {}) if steps else {},
            action_candidate=action_candidate,
            precedent_pack=precedent_by_action.get(first_action, {}),
            snapshot=snapshot,
            registry=registry,
            plan=plan,
            diagnosed=diagnosed,
        ) if steps else {}
        evaluation = _evaluate_plan_vs_status_quo(
            plan=plan,
            first_step_thesis=step_thesis,
            snapshot=snapshot,
            diagnosed=diagnosed,
        )
        out.append(
            {
                "rank": idx,
                "plan_id": str(plan.get("plan_id", "") or ""),
                "action_ids": action_ids,
                "recommended_posture": evaluation["recommended_posture"],
                "edge_vs_status_quo": evaluation["edge_vs_status_quo"],
                "edge_vs_status_quo_formatted": f"{evaluation['edge_vs_status_quo']:+.3f}",
                "case_for": str(step_thesis.get("why_this_step", "") or ""),
                "case_against": list(step_thesis.get("tradeoffs", []) or [])[:3] or list(evaluation["case_for_wait"][:2]),
                "why_now": str(step_thesis.get("why_now", "") or ""),
                "support_type": str(step_thesis.get("support_type", "") or ""),
                "plan_score": float(plan.get("score", 0.0) or 0.0),
                "support_factor": float(((plan.get("score_components", {}) or {}).get("support_factor", 0.0) or 0.0)),
                "sizing_guidance": _build_step_sizing_guidance(
                    action_id=first_action,
                    parameters=dict(((steps[0] if steps else {}) or {}).get("parameters", {}) or {}),
                    snapshot=snapshot,
                ),
                "parameter_optimization": _build_step_parameter_optimization(
                    action_id=first_action,
                    parameters=dict(((steps[0] if steps else {}) or {}).get("parameters", {}) or {}),
                    snapshot=snapshot,
                    registry=registry,
                ),
                "regret_balance": evaluation["regret_balance"],
                "rating_constraint_posture": _build_rating_cliff_analysis(top_plan=plan, snapshot=snapshot).get("constraint_posture"),
                "signal_posture": _build_signaling_analysis(top_plan=plan, snapshot=snapshot, status_quo_view=evaluation).get("signal_posture"),
            }
        )
    return out


def _build_step_thesis(
    *,
    step: Dict[str, Any],
    action_candidate: Dict[str, Any],
    precedent_pack: Dict[str, Any],
    snapshot: Dict[str, Any],
    registry: Any,
    plan: Dict[str, Any],
    diagnosed: Dict[str, Any],
) -> Dict[str, Any]:
    action_id = str(step.get("action_id", "") or "")
    schema = registry.get_action(action_id) or {}
    explanation = dict(step.get("explanation", {}) or {})
    parameters = dict(step.get("parameters", {}) or {})
    mechanisms = list(((action_candidate.get("mechanism_activation", {}) or {}).get("mechanisms", []) or []))
    strongest_mech = max(mechanisms, key=lambda m: float(m.get("activation_strength", 0.0) or 0.0), default={})
    strongest_mech_name = _humanize_mechanism_id(str(strongest_mech.get("mechanism_id", "") or ""))

    objective_signal = _best_objective_signal(action_candidate)
    precedent_confidence = _precedent_confidence(precedent_pack)
    sample_n = _precedent_sample_size(precedent_pack)
    support_type = _support_type(action_candidate=action_candidate, precedent_pack=precedent_pack)
    role_text = _step_role_text(action_id=action_id, parameters=parameters, snapshot=snapshot, diagnosed=diagnosed)
    why_this_step = role_text
    if strongest_mech_name:
        why_this_step += f" The dominant mechanism is {strongest_mech_name}."
    if objective_signal is not None:
        why_this_step += (
            f" The clearest modeled benefit is {_humanize_objective_name(objective_signal[0])} "
            f"({objective_signal[1]:+.3f})."
        )
    if schema.get("description") and role_text.endswith("addresses the current strategic bottleneck."):
        why_this_step += f" {str(schema.get('description')).strip()}"

    why_now = _timing_thesis(
        action_id=action_id,
        step=step,
        snapshot=snapshot,
        diagnosed=diagnosed,
        plan=plan,
    )

    tradeoffs = _step_tradeoffs(action_candidate=action_candidate, action_id=action_id, snapshot=snapshot)
    supporting_facts = _step_supporting_facts(
        action_id=action_id,
        action_candidate=action_candidate,
        precedent_pack=precedent_pack,
        snapshot=snapshot,
        support_type=support_type,
        sample_n=sample_n,
        precedent_confidence=precedent_confidence,
    )
    return {
        "action_id": action_id,
        "action_label": _humanize_action_id(action_id),
        "role": role_text,
        "why_this_step": why_this_step.strip(),
        "why_now": why_now,
        "sizing_guidance": _build_step_sizing_guidance(
            action_id=action_id,
            parameters=parameters,
            snapshot=snapshot,
        ),
        "parameter_optimization": _build_step_parameter_optimization(
            action_id=action_id,
            parameters=parameters,
            snapshot=snapshot,
            registry=registry,
        ),
        "support_type": support_type,
        "precedent_confidence": precedent_confidence,
        "precedent_sample_n": sample_n,
        "supporting_facts": supporting_facts,
        "tradeoffs": tradeoffs[:4],
        "tail_descriptions": _tail_descriptions(precedent_pack),
        "probability_of_success": float(step.get("probability_of_success", 0.0) or 0.0),
        "lead_time_days": int(((step.get("expected_duration", {}) or {}).get("median_days", 0) or 0)),
    }


def _evaluate_plan_vs_status_quo(
    *,
    plan: Dict[str, Any],
    first_step_thesis: Dict[str, Any],
    snapshot: Dict[str, Any],
    diagnosed: Dict[str, Any],
) -> Dict[str, Any]:
    steps = list(plan.get("steps", []) or [])
    first_action = str((((steps[0] if steps else {}) or {}).get("action_id", "") or ""))
    components = dict(plan.get("score_components", {}) or {})
    plan_score = float(plan.get("score", 0.0) or 0.0)
    expected_utility = float(components.get("expected_utility", 0.0) or 0.0)
    support_factor = float(components.get("support_factor", 0.0) or 0.0)
    feasibility_chain = float(components.get("feasibility_chain", 0.0) or 0.0)
    tail_penalty = float(components.get("tail_risk_penalty", 0.0) or 0.0)
    net_leverage = _safe_float(_feature_value(snapshot, "capital_structure.net_leverage"))
    maturity_wall = _safe_float(_feature_value(snapshot, "capital_structure.maturity_wall_ratio_24m"))
    liquidity = _safe_float(_feature_value(snapshot, "liquidity.available_for_actions"))
    market_cap = _safe_float(_feature_value(snapshot, "market.market_cap"))
    credit_window = _safe_float(_feature_value(snapshot, "market.credit_window_proxy"))
    equity_window = _safe_float(_feature_value(snapshot, "market.equity_window_proxy"))
    pursue_mna_priority = _safe_float(_feature_value(snapshot, "strategic.intent.pursue_mna_priority"))
    focus_on_core = _safe_float(_feature_value(snapshot, "strategic.intent.focus_on_core"))
    liquidity_to_mcap = (liquidity / market_cap) if liquidity is not None and market_cap not in (None, 0.0) else None

    status_quo_score = 0.18
    case_for_action: List[str] = []
    case_for_wait: List[str] = []

    why_this_step = str(first_step_thesis.get("why_this_step", "") or "").strip()
    why_now = str(first_step_thesis.get("why_now", "") or "").strip()
    if why_this_step:
        case_for_action.append(why_this_step)
    if why_now:
        case_for_action.append(why_now)

    if support_factor < 0.75:
        status_quo_score += 0.05
        case_for_wait.append("Empirical support is not yet strong enough for a clean act-now call.")
    if feasibility_chain < 0.80:
        status_quo_score += 0.04
        case_for_wait.append("Execution still depends on a relatively fragile chain of assumptions.")
    if tail_penalty > 0.08:
        status_quo_score += 0.05
        case_for_wait.append("Downside tails are still heavy enough that preserving optionality matters.")
    if expected_utility < 0.52:
        status_quo_score += 0.03
        case_for_wait.append("The incremental benefit over waiting is still modest.")

    if _has_capital_return([first_action]):
        if (liquidity_to_mcap or 0.0) >= 0.03:
            status_quo_score -= 0.06
            case_for_action.append(f"Deployable liquidity already equals {_fmt_pct(liquidity_to_mcap)} of market value, so inactivity has an opportunity cost.")
        if net_leverage is not None and net_leverage >= 2.75:
            status_quo_score += 0.06
            case_for_wait.append(f"Net leverage is already {_fmt_x(net_leverage)}, which makes immediate payout easier to regret.")
        if first_action in {"capital_return.dividend_increase", "capital_return.dividend_initiate", "capital_return.special_dividend"}:
            status_quo_score += 0.03
            case_for_wait.append("A dividend step is stickier than waiting, so the hurdle to act should be higher.")
        if _is_buyback_action(first_action):
            status_quo_score -= 0.02
            case_for_action.append("Repurchases are more reversible than a permanent payout reset.")

    if _is_balance_sheet_action(first_action):
        if (maturity_wall or 0.0) >= 0.20:
            status_quo_score -= 0.10
            case_for_action.append(f"A {_fmt_pct(maturity_wall)} 24-month maturity wall makes delay more expensive.")
        if credit_window is not None and credit_window >= 0.60:
            status_quo_score -= 0.04
            case_for_action.append(f"Credit conditions are currently workable at {_fmt_score(credit_window)}.")
        if _uses_equity_markets([first_action]) and (equity_window or 0.0) < 0.50:
            status_quo_score += 0.05
            case_for_wait.append("The equity window is not attractive enough to force issuance now.")

    if _is_mna_action(first_action):
        status_quo_score += 0.04
        case_for_wait.append("M&A is less reversible than waiting, so it needs a wider edge before acting.")
        if pursue_mna_priority is not None and pursue_mna_priority >= 0.75:
            status_quo_score -= 0.04
            case_for_action.append(f"Strategic intent to pursue M&A is already high at {_fmt_score(pursue_mna_priority)}.")

    if _is_divestiture_action(first_action):
        if focus_on_core is not None and focus_on_core >= 0.70:
            status_quo_score -= 0.04
            case_for_action.append(f"Focus-on-core pressure is elevated at {_fmt_score(focus_on_core)}.")
        else:
            status_quo_score += 0.02
            case_for_wait.append("If strategic focus is not clearly impaired, waiting is a real alternative to selling.")

    status_quo_score = _clip(status_quo_score, 0.05, 0.40)
    edge_vs_status_quo = round(plan_score - status_quo_score, 6)
    if edge_vs_status_quo >= 0.05 and support_factor >= 0.75 and feasibility_chain >= 0.80:
        recommended_posture = "act_now"
    elif edge_vs_status_quo >= 0.02 and expected_utility >= 0.50 and support_factor >= 0.70 and feasibility_chain >= 0.75:
        recommended_posture = "conditional_action"
    else:
        recommended_posture = "wait"

    case_for_wait = _posture_adjust_case_for_wait(
        recommended_posture=recommended_posture,
        first_action=first_action,
        snapshot=snapshot,
        case_for_wait=case_for_wait,
    )
    if not case_for_wait:
        case_for_wait.append("Waiting preserves flexibility until the edge versus status quo becomes clearer.")
    why_wait = " ".join(case_for_wait[:2])
    why_act_now = " ".join(case_for_action[:2]) if case_for_action else "No action-specific reason is strong enough to justify moving immediately."
    reassessment_triggers = _decision_boundaries(first_action=first_action, snapshot=snapshot, top_plan=plan)
    return {
        "recommended_posture": recommended_posture,
        "status_quo_score": round(status_quo_score, 6),
        "edge_vs_status_quo": edge_vs_status_quo,
        "why_act_now": why_act_now,
        "why_wait": why_wait,
        "case_for_action": _dedupe(case_for_action)[:4],
        "case_for_wait": _dedupe(case_for_wait)[:4],
        "reassessment_triggers": reassessment_triggers,
        "regret_balance": _regret_balance(first_action=first_action, recommended_posture=recommended_posture, snapshot=snapshot),
    }


def _posture_adjust_case_for_wait(
    *,
    recommended_posture: str,
    first_action: str,
    snapshot: Dict[str, Any],
    case_for_wait: Sequence[str],
) -> List[str]:
    items = _dedupe([str(item or "").strip() for item in case_for_wait if str(item or "").strip()])
    if recommended_posture == "act_now":
        filtered = [
            item for item in items
            if "incremental benefit over waiting is still modest" not in item.lower()
            and "not yet strong enough for a clean act-now call" not in item.lower()
        ]
        if filtered:
            return filtered[:3]
        if _is_buyback_action(first_action):
            return ["Waiting preserves liquidity if a clearly better use of capital appears quickly."]
        if _is_balance_sheet_action(first_action):
            return ["Waiting avoids locking in financing if the need proves less durable than it currently appears."]
        return ["Waiting preserves flexibility if the current thesis weakens quickly."]
    if recommended_posture == "conditional_action":
        adjusted: List[str] = []
        for item in items:
            if "incremental benefit over waiting is still modest" in item.lower():
                adjusted.append("The edge over waiting is real, but not yet wide enough to force immediate execution.")
            else:
                adjusted.append(item)
        return _dedupe(adjusted)[:3]
    return items[:3]


def _build_plan_sizing_guidance(
    *,
    top_plan: Dict[str, Any],
    step_theses: Sequence[Dict[str, Any]],
    snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    steps = list(top_plan.get("steps", []) or [])
    if not steps:
        return {}
    first_step = dict(steps[0] or {})
    sizing = _build_step_sizing_guidance(
        action_id=str(first_step.get("action_id", "") or ""),
        parameters=dict(first_step.get("parameters", {}) or {}),
        snapshot=snapshot,
    )
    if step_theses:
        sizing["execution_notes"] = list((step_theses[0] or {}).get("tradeoffs", []) or [])[:2]
    return sizing


def _build_parameter_optimization(
    *,
    top_plan: Dict[str, Any],
    snapshot: Dict[str, Any],
    registry: Any,
    sizing_guidance: Dict[str, Any],
) -> Dict[str, Any]:
    steps = list(top_plan.get("steps", []) or [])
    if not steps:
        return {}
    first_step = dict(steps[0] or {})
    return _build_step_parameter_optimization(
        action_id=str(first_step.get("action_id", "") or ""),
        parameters=dict(first_step.get("parameters", {}) or {}),
        snapshot=snapshot,
        registry=registry,
        sizing_guidance=sizing_guidance,
    )


def _build_step_parameter_optimization(
    *,
    action_id: str,
    parameters: Dict[str, Any],
    snapshot: Dict[str, Any],
    registry: Any,
    sizing_guidance: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    action = (registry.get_action(action_id) or {}) if registry is not None else {}
    schema = dict(action.get("parameter_schema", {}) or {})
    if not schema:
        return {}

    context = _parameter_context(snapshot=snapshot)
    recommended_parameters: Dict[str, Dict[str, Any]] = {}
    for parameter_name, parameter_schema in schema.items():
        recommendation = _optimize_parameter_recommendation(
            action_id=action_id,
            parameter_name=str(parameter_name),
            parameter_schema=dict(parameter_schema or {}),
            parameters=parameters,
            context=context,
        )
        if recommendation:
            recommended_parameters[str(parameter_name)] = recommendation

    if not recommended_parameters:
        return {}

    guardrails = _parameter_guardrails(
        action_id=action_id,
        context=context,
        recommended_parameters=recommended_parameters,
    )
    rejected_variants = _parameter_rejected_variants(
        action_id=action_id,
        recommended_parameters=recommended_parameters,
        context=context,
    )
    summary = _parameter_optimization_summary(
        action_id=action_id,
        recommended_parameters=recommended_parameters,
        sizing_guidance=sizing_guidance or {},
    )
    return {
        "action_id": action_id,
        "objective": _parameter_optimization_objective(action_id),
        "summary": summary,
        "recommended_parameters": recommended_parameters,
        "guardrails": guardrails,
        "rejected_variants": rejected_variants,
    }


def _parameter_context(*, snapshot: Dict[str, Any]) -> Dict[str, Optional[float]]:
    liquidity = _safe_float(_feature_value(snapshot, "liquidity.available_for_actions"))
    market_cap = _safe_float(_feature_value(snapshot, "market.market_cap"))
    net_leverage = _safe_float(_feature_value(snapshot, "capital_structure.net_leverage"))
    maturity_wall = _safe_float(_feature_value(snapshot, "capital_structure.maturity_wall_ratio_24m"))
    credit_window = _safe_float(_feature_value(snapshot, "market.credit_window_proxy"))
    equity_window = _safe_float(_feature_value(snapshot, "market.equity_window_proxy"))
    credit_spread_pct = _safe_float(_feature_value(snapshot, "market.credit_spread_percentile_2y"))
    return {
        "liquidity": liquidity,
        "market_cap": market_cap,
        "net_leverage": net_leverage,
        "maturity_wall": maturity_wall,
        "credit_window": credit_window,
        "equity_window": equity_window,
        "credit_spread_pct": credit_spread_pct,
        "liquidity_to_market_cap": (liquidity / market_cap) if liquidity is not None and market_cap not in (None, 0.0) else None,
    }


def _optimize_parameter_recommendation(
    *,
    action_id: str,
    parameter_name: str,
    parameter_schema: Dict[str, Any],
    parameters: Dict[str, Any],
    context: Dict[str, Optional[float]],
) -> Optional[Dict[str, Any]]:
    parameter_type = str(parameter_schema.get("type", "") or "")
    current_value = parameters.get(parameter_name)

    if parameter_name == "funding_mix":
        mix = _recommended_funding_mix(action_id=action_id, context=context)
        return {
            "parameter_type": parameter_type,
            "current_value": current_value,
            "current_value_formatted": _format_parameter_value(parameter_type, current_value),
            "recommended_value": mix,
            "recommended_value_formatted": _format_parameter_value(parameter_type, mix),
            "why": _funding_mix_reason(action_id=action_id, context=context, mix=mix),
        }

    if parameter_name in {"size_pct_market_cap", "target_size_pct_ev", "percent_divested", "premium_pct", "discount_pct", "conversion_premium_pct", "initial_yield_pct", "percent_change"}:
        target, lower, upper, why = _optimize_percent_parameter(
            action_id=action_id,
            parameter_name=parameter_name,
            current_value=current_value,
            parameter_schema=parameter_schema,
            context=context,
        )
        return {
            "parameter_type": parameter_type,
            "current_value": _safe_float(current_value),
            "current_value_formatted": _format_parameter_value(parameter_type, current_value),
            "recommended_value": target,
            "recommended_value_formatted": _format_parameter_value(parameter_type, target),
            "recommended_range": f"{_fmt_pct(lower)} to {_fmt_pct(upper)}",
            "why": why,
        }

    if parameter_name in {"size_absolute_usd", "amount_usd", "amount_refinanced_usd", "draw_amount_usd", "resize_amount_usd", "estimated_ev_usd", "annualized_cash_commitment_usd"}:
        target, lower, upper, why = _optimize_amount_parameter(
            action_id=action_id,
            parameter_name=parameter_name,
            current_value=current_value,
            context=context,
        )
        return {
            "parameter_type": parameter_type,
            "current_value": _safe_float(current_value),
            "current_value_formatted": _format_parameter_value(parameter_type, current_value, parameter_name=parameter_name),
            "recommended_value": target,
            "recommended_value_formatted": _format_parameter_value(parameter_type, target, parameter_name=parameter_name),
            "recommended_range": f"{_fmt_currency(lower)} to {_fmt_currency(upper)}",
            "why": why,
        }

    if parameter_name in {"tenor_years", "new_tenor_years", "call_protection_years", "leverage_post_close"}:
        normalized_current = _normalize_numeric_current(
            parameter_name=parameter_name,
            current_value=current_value,
            parameter_schema=parameter_schema,
        )
        target, lower, upper, why = _optimize_numeric_parameter(
            action_id=action_id,
            parameter_name=parameter_name,
            current_value=normalized_current,
            parameter_schema=parameter_schema,
            context=context,
        )
        return {
            "parameter_type": parameter_type,
            "current_value": normalized_current,
            "current_value_formatted": _format_parameter_value(parameter_type, normalized_current, parameter_name=parameter_name),
            "recommended_value": target,
            "recommended_value_formatted": _format_parameter_value(parameter_type, target, parameter_name=parameter_name),
            "recommended_range": _format_numeric_range(parameter_name=parameter_name, lower=lower, upper=upper),
            "why": why,
        }

    if parameter_name in {"pace", "use_of_proceeds", "fixed_vs_floating", "rate_structure", "instrument_type", "offering_type", "intent", "target_sector_match", "synergy_case_strength", "geography_overlap", "regulatory_risk", "effective_quarter"}:
        value, why = _optimize_enum_parameter(
            action_id=action_id,
            parameter_name=parameter_name,
            current_value=current_value,
            context=context,
        )
        return {
            "parameter_type": parameter_type,
            "current_value": current_value,
            "current_value_formatted": _format_parameter_value(parameter_type, current_value),
            "recommended_value": value,
            "recommended_value_formatted": _format_parameter_value(parameter_type, value),
            "why": why,
        }

    if parameter_name == "secured_flag":
        value, why = _optimize_boolean_parameter(
            action_id=action_id,
            parameter_name=parameter_name,
            current_value=current_value,
            context=context,
        )
        return {
            "parameter_type": parameter_type,
            "current_value": current_value,
            "current_value_formatted": _format_parameter_value(parameter_type, current_value),
            "recommended_value": value,
            "recommended_value_formatted": _format_parameter_value(parameter_type, value),
            "why": why,
        }

    return None


def _optimize_percent_parameter(
    *,
    action_id: str,
    parameter_name: str,
    current_value: Any,
    parameter_schema: Dict[str, Any],
    context: Dict[str, Optional[float]],
) -> Tuple[float, float, float, str]:
    minimum = float(parameter_schema.get("min", 0.0) or 0.0)
    maximum = float(parameter_schema.get("max", 1.0) or 1.0)
    current = _safe_float(current_value)
    net_leverage = context.get("net_leverage")
    maturity_wall = context.get("maturity_wall")
    liquidity_to_market_cap = context.get("liquidity_to_market_cap")
    equity_window = context.get("equity_window")

    if parameter_name == "size_pct_market_cap":
        base = 0.03
        if liquidity_to_market_cap is not None:
            base += min(0.04, liquidity_to_market_cap * 0.35)
        if (equity_window or 0.0) >= 0.65:
            base += 0.01
        if (net_leverage or 0.0) >= 2.5:
            base -= 0.015
        if (maturity_wall or 0.0) >= 0.20:
            base -= 0.015
        target = current if current is not None else base
        if current is not None:
            target = (0.6 * current) + (0.4 * base)
        target = _clip(target, minimum, maximum)
        lower = _clip(target * 0.85, minimum, maximum)
        upper = _clip(target * 1.15, minimum, maximum)
        why = "Keep the buyback large enough to matter, but cap it where leverage or maturity pressure would start to crowd out flexibility."
        return target, lower, upper, why

    if parameter_name == "target_size_pct_ev":
        base = 0.06 if action_id == "mna.tuck_in_acquisition" else 0.12
        if liquidity_to_market_cap is not None:
            base = max(base, min(0.18 if action_id == "mna.tuck_in_acquisition" else 0.22, liquidity_to_market_cap * 0.75))
        if (net_leverage or 0.0) >= 2.5:
            base -= 0.03
        if (maturity_wall or 0.0) >= 0.20:
            base -= 0.02
        target = current if current is not None else base
        if current is not None:
            target = (0.7 * current) + (0.3 * base)
        target = _clip(target, minimum, maximum)
        lower = _clip(target * 0.8, minimum, maximum)
        upper = _clip(target * 1.2, minimum, maximum)
        why = "Keep deal size inside a range the balance sheet can absorb without turning the strategic thesis into a financing thesis."
        return target, lower, upper, why

    if parameter_name == "percent_divested":
        base = 0.15
        if (net_leverage or 0.0) >= 2.75 or (maturity_wall or 0.0) >= 0.20:
            base = 0.25
        target = current if current is not None else base
        if current is not None:
            target = (0.7 * current) + (0.3 * base)
        target = _clip(target, minimum, maximum)
        lower = _clip(target * 0.8, minimum, maximum)
        upper = _clip(target * 1.25, minimum, maximum)
        why = "Bias the sale toward the smallest package that meaningfully simplifies the portfolio or releases capital."
        return target, lower, upper, why

    if parameter_name == "premium_pct":
        base = 0.03 if (equity_window or 0.0) < 0.60 else 0.05
        target = _clip(current if current is not None else base, minimum, maximum)
        lower = _clip(max(minimum, target - 0.01), minimum, maximum)
        upper = _clip(min(maximum, target + 0.02), minimum, maximum)
        why = "Keep the tender premium high enough to secure participation but low enough to preserve per-share economics."
        return target, lower, upper, why

    if parameter_name == "initial_yield_pct":
        base = 0.015 if (net_leverage or 0.0) >= 2.5 or (maturity_wall or 0.0) >= 0.20 else 0.02
        target = _clip(current if current is not None else base, minimum, maximum)
        lower = _clip(max(minimum, target - 0.005), minimum, maximum)
        upper = _clip(min(maximum, target + 0.005), minimum, maximum)
        why = "Start any new recurring dividend at a yield the company can defend through a weaker operating patch."
        return target, lower, upper, why

    if parameter_name == "percent_change":
        base = 0.05 if (net_leverage or 0.0) >= 2.5 or (maturity_wall or 0.0) >= 0.20 else 0.08
        target = _clip(current if current is not None else base, minimum, maximum)
        lower = _clip(max(minimum, target - 0.02), minimum, maximum)
        upper = _clip(min(maximum, target + 0.03), minimum, maximum)
        why = "Keep the increase inside a band that signals confidence without turning the payout into the dominant capital-allocation commitment."
        return target, lower, upper, why

    if parameter_name == "discount_pct":
        base = 0.02 if (equity_window or 0.0) >= 0.60 else 0.05
        target = _clip(current if current is not None else base, minimum, maximum)
        lower = _clip(max(minimum, target - 0.01), minimum, maximum)
        upper = _clip(min(maximum, target + 0.02), minimum, maximum)
        why = "Keep issuance discount narrow enough to avoid unnecessary dilution while still clearing the book."
        return target, lower, upper, why

    if parameter_name == "conversion_premium_pct":
        base = 0.25 if (equity_window or 0.0) >= 0.60 else 0.18
        target = _clip(current if current is not None else base, minimum, maximum)
        lower = _clip(max(minimum, target - 0.05), minimum, maximum)
        upper = _clip(min(maximum, target + 0.05), minimum, maximum)
        why = "A mid-range conversion premium preserves some equity optionality without making the instrument too expensive to place."
        return target, lower, upper, why

    target = _clip(current if current is not None else minimum, minimum, maximum)
    return target, target, target, "Keep the parameter inside the supported schema bounds."


def _optimize_amount_parameter(
    *,
    action_id: str,
    parameter_name: str,
    current_value: Any,
    context: Dict[str, Optional[float]],
) -> Tuple[float, float, float, str]:
    current = _safe_float(current_value)
    liquidity = context.get("liquidity") or 0.0
    market_cap = context.get("market_cap") or 0.0
    net_leverage = context.get("net_leverage") or 0.0
    maturity_wall = context.get("maturity_wall") or 0.0

    if parameter_name == "annualized_cash_commitment_usd":
        base = min(liquidity * 0.18, market_cap * 0.012) if liquidity and market_cap else max(liquidity * 0.12, 0.0)
        if net_leverage >= 2.5 or maturity_wall >= 0.20:
            base *= 0.8
        target = current if current is not None else base
        if current is not None and base > 0.0:
            target = (0.7 * current) + (0.3 * base)
        lower, upper = _bounded_amount_band(target)
        why = "Set recurring cash commitment from defendable annual free-cash-flow capacity rather than a single strong quarter."
        return target, lower, upper, why

    if parameter_name in {"size_absolute_usd"} and _has_capital_return([action_id]):
        base = min(liquidity * 0.45, market_cap * 0.06) if liquidity and market_cap else max(liquidity * 0.35, 0.0)
        if net_leverage >= 2.5 or maturity_wall >= 0.20:
            base *= 0.8
        target = current if current is not None else base
        if current is not None and base > 0.0:
            target = (0.65 * current) + (0.35 * base)
        lower, upper = _bounded_amount_band(target)
        why = "Size the return against true excess liquidity rather than the full cash balance."
        return target, lower, upper, why

    if parameter_name in {"amount_usd", "amount_refinanced_usd", "draw_amount_usd", "resize_amount_usd"}:
        base_ratio = 0.04
        if maturity_wall >= 0.20:
            base_ratio += 0.04
        if net_leverage >= 3.0:
            base_ratio += 0.02
        if _uses_equity_markets([action_id]):
            base_ratio = max(0.03, base_ratio - 0.01)
        base = market_cap * base_ratio if market_cap else liquidity * 0.35
        if current is not None:
            target = (0.7 * current) + (0.3 * base)
        else:
            target = base
        lower, upper = _bounded_amount_band(target)
        why = "Anchor proceeds to the identified balance-sheet need plus a buffer, not to maximum available market appetite."
        return target, lower, upper, why

    if parameter_name == "estimated_ev_usd":
        target = current if current is not None else max(market_cap * 0.15, liquidity * 0.5)
        lower, upper = _bounded_amount_band(target)
        why = "Frame divestiture value around a targeted non-core package rather than a forced headline disposal."
        return target, lower, upper, why

    target = current if current is not None else 0.0
    return target, target, target, "Size the notional to the minimum amount that solves the problem."


def _numeric_parameter_bounds(*, parameter_name: str, parameter_schema: Dict[str, Any]) -> Tuple[float, float]:
    minimum = float(parameter_schema.get("min", 0.0) or 0.0)
    maximum = parameter_schema.get("max")
    if maximum is not None:
        return minimum, float(maximum)
    if parameter_name in {"tenor_years", "new_tenor_years"}:
        return minimum, 10.0
    if parameter_name == "call_protection_years":
        return minimum, 5.0
    if parameter_name == "leverage_post_close":
        return minimum, 4.0
    return minimum, max(minimum, 1_000_000_000.0)


def _normalize_numeric_current(
    *,
    parameter_name: str,
    current_value: Any,
    parameter_schema: Dict[str, Any],
) -> Optional[float]:
    current = _safe_float(current_value)
    if current is None:
        return None
    minimum, maximum = _numeric_parameter_bounds(parameter_name=parameter_name, parameter_schema=parameter_schema)
    if current < minimum or current > maximum:
        return None
    return current


def _optimize_numeric_parameter(
    *,
    action_id: str,
    parameter_name: str,
    current_value: Any,
    parameter_schema: Dict[str, Any],
    context: Dict[str, Optional[float]],
) -> Tuple[float, float, float, str]:
    current = _safe_float(current_value)
    maturity_wall = context.get("maturity_wall") or 0.0
    credit_window = context.get("credit_window") or 0.0
    net_leverage = context.get("net_leverage") or 0.0
    minimum, maximum = _numeric_parameter_bounds(parameter_name=parameter_name, parameter_schema=parameter_schema)

    if parameter_name in {"tenor_years", "new_tenor_years"}:
        base = 5.0
        if maturity_wall >= 0.20:
            base += 1.0
        if credit_window >= 0.70:
            base += 1.0
        target = current if current is not None else base
        if current is not None:
            target = (0.65 * current) + (0.35 * base)
        target = _clip(target, minimum, maximum)
        lower = _clip(max(3.0, target - 1.0), minimum, maximum)
        upper = _clip(min(10.0, target + 1.0), minimum, maximum)
        why = "Extend tenor enough to move the maturity wall, but not so far that the company pays for duration it does not need."
        return target, lower, upper, why

    if parameter_name == "leverage_post_close":
        base = 2.5 if action_id == "mna.tuck_in_acquisition" else 3.0
        if maturity_wall >= 0.20 or net_leverage >= 2.5:
            base -= 0.25
        target = current if current is not None else base
        if current is not None:
            target = (0.6 * current) + (0.4 * base)
        target = _clip(target, minimum, maximum)
        lower = _clip(max(1.5, target - 0.25), minimum, maximum)
        upper = _clip(min(4.0, target + 0.25), minimum, maximum)
        why = "Keep pro forma leverage inside a range that preserves financing flexibility after the transaction."
        return target, lower, upper, why

    if parameter_name == "call_protection_years":
        target = current if current is not None else 3.0
        target = _clip(target, minimum, maximum)
        lower = _clip(max(1.0, target - 1.0), minimum, maximum)
        upper = _clip(min(5.0, target + 1.0), minimum, maximum)
        why = "Use enough call protection to clear the security cleanly without overpaying for rigidity."
        return target, lower, upper, why

    target = _clip(current if current is not None else minimum, minimum, maximum)
    return target, target, target, "Keep the numeric parameter near the center of the feasible range."


def _optimize_enum_parameter(
    *,
    action_id: str,
    parameter_name: str,
    current_value: Any,
    context: Dict[str, Optional[float]],
) -> Tuple[str, str]:
    net_leverage = context.get("net_leverage") or 0.0
    maturity_wall = context.get("maturity_wall") or 0.0
    credit_window = context.get("credit_window") or 0.0
    equity_window = context.get("equity_window") or 0.0

    if parameter_name == "pace":
        value = "front_loaded" if equity_window >= 0.65 and net_leverage < 2.25 and maturity_wall < 0.15 else "gradual"
        why = "Front-load only when the valuation window is open and the balance sheet can absorb the faster capital return."
        return value, why
    if parameter_name in {"fixed_vs_floating", "rate_structure"}:
        value = "fixed" if maturity_wall >= 0.20 or credit_window < 0.55 else "mixed"
        why = "Bias the liability profile toward fixed-rate certainty when refinancing risk matters more than carry optimization."
        return value, why
    if parameter_name == "instrument_type":
        value = "term_loan" if credit_window < 0.50 else "bond"
        why = "Use the instrument that is most likely to clear reliably in the current financing window."
        return value, why
    if parameter_name == "use_of_proceeds":
        if _uses_equity_markets([action_id]):
            value = "deleveraging" if net_leverage >= 2.5 or maturity_wall >= 0.20 else "liquidity_buffer"
        elif _is_balance_sheet_action(action_id):
            value = "refinancing" if maturity_wall >= 0.15 else "liquidity_buffer"
        elif _is_divestiture_action(action_id):
            value = "deleveraging" if net_leverage >= 2.5 or maturity_wall >= 0.20 else "reinvestment"
        else:
            value = "general_corporate"
        why = "Direct proceeds first to the binding balance-sheet problem, then to optionality."
        return value, why
    if parameter_name == "offering_type":
        value = "at_the_market" if equity_window >= 0.65 and net_leverage < 2.75 else "follow_on"
        why = "Use a slower ATM only when the window is supportive; otherwise clear the financing in one transaction."
        return value, why
    if parameter_name == "effective_quarter":
        value = "Q2" if (maturity_wall or 0.0) < 0.20 and net_leverage < 2.5 else "Q3"
        why = "Only pull the effective quarter forward when balance-sheet pressure is modest enough to support the commitment immediately."
        return value, why
    if parameter_name == "intent":
        value = "precautionary_draw" if credit_window < 0.45 or maturity_wall >= 0.20 else "resize"
        why = "Use the revolver first as insurance when the financing window is shaky; resize only when liquidity architecture is the issue."
        return value, why
    if parameter_name == "target_sector_match":
        return "high", "Tuck-in logic works best when adjacency risk is low and synergies are easier to underwrite."
    if parameter_name == "synergy_case_strength":
        return ("high" if net_leverage < 2.25 else "medium"), "Require a stronger synergy case as balance-sheet tolerance narrows."
    if parameter_name == "geography_overlap":
        return "high", "Higher geographic overlap reduces execution complexity and integration regret."
    if parameter_name == "regulatory_risk":
        return "low" if action_id == "mna.tuck_in_acquisition" else "medium", "Prefer transactions whose strategic value does not depend on taking large regulatory risk."
    return str(current_value or ""), "Keep the enum choice aligned with the current financing and execution environment."


def _optimize_boolean_parameter(
    *,
    action_id: str,
    parameter_name: str,
    current_value: Any,
    context: Dict[str, Optional[float]],
) -> Tuple[bool, str]:
    net_leverage = context.get("net_leverage") or 0.0
    credit_window = context.get("credit_window") or 0.0
    maturity_wall = context.get("maturity_wall") or 0.0
    if parameter_name == "secured_flag":
        value = bool(net_leverage >= 3.25 or (credit_window < 0.40 and maturity_wall >= 0.20))
        why = "Use secured structure only when clearing the financing reliably is more valuable than preserving unencumbered flexibility."
        return value, why
    return bool(current_value), "Preserve the current boolean posture unless the financing constraint clearly changes."


def _recommended_funding_mix(
    *,
    action_id: str,
    context: Dict[str, Optional[float]],
) -> Dict[str, float]:
    net_leverage = context.get("net_leverage") or 0.0
    maturity_wall = context.get("maturity_wall") or 0.0
    credit_window = context.get("credit_window") or 0.0
    liquidity_to_market_cap = context.get("liquidity_to_market_cap") or 0.0

    if _has_capital_return([action_id]):
        if net_leverage < 1.75 and maturity_wall < 0.15 and credit_window >= 0.60:
            return {"cash": 0.6, "debt": 0.4, "equity": 0.0}
        if liquidity_to_market_cap >= 0.08:
            return {"cash": 0.8, "debt": 0.2, "equity": 0.0}
        return {"cash": 1.0, "debt": 0.0, "equity": 0.0}
    if _is_mna_action(action_id):
        if net_leverage < 2.0 and credit_window >= 0.60:
            return {"cash": 0.5, "debt": 0.5, "equity": 0.0}
        if net_leverage < 2.75:
            return {"cash": 0.5, "debt": 0.35, "equity": 0.15}
        return {"cash": 0.4, "debt": 0.3, "equity": 0.3}
    return {"cash": 1.0, "debt": 0.0, "equity": 0.0}


def _funding_mix_reason(
    *,
    action_id: str,
    context: Dict[str, Optional[float]],
    mix: Dict[str, float],
) -> str:
    net_leverage = context.get("net_leverage")
    maturity_wall = context.get("maturity_wall")
    if _has_capital_return([action_id]):
        return (
            f"Keep return-of-capital funding mostly cash-backed; net leverage at {_fmt_x(net_leverage)}"
            f" and a {_fmt_pct(maturity_wall)} maturity wall do not justify a debt-heavy payout."
            if net_leverage is not None and maturity_wall is not None
            else "Keep return-of-capital funding mostly cash-backed unless the balance sheet is exceptionally underlevered."
        )
    if _is_mna_action(action_id):
        return "Use a mixed funding stack only up to the point where pro forma leverage stays inside the target band."
    return f"Recommended mix is {_format_parameter_value('funding_mix_object', mix)}."


def _parameter_optimization_objective(action_id: str) -> str:
    if _has_capital_return([action_id]):
        return "Maximize per-share value while preserving balance-sheet flexibility."
    if _is_balance_sheet_action(action_id):
        return "Solve the financing problem with the smallest durable increase in risk or cost."
    if _is_mna_action(action_id):
        return "Keep the strategic upside while capping financing and integration regret."
    if _is_divestiture_action(action_id):
        return "Release capital and simplify the portfolio without forcing strategic over-disposal."
    return "Tune parameters to solve the diagnosed problem with minimal irreversible regret."


def _parameter_guardrails(
    *,
    action_id: str,
    context: Dict[str, Optional[float]],
    recommended_parameters: Dict[str, Dict[str, Any]],
) -> List[str]:
    net_leverage = context.get("net_leverage")
    maturity_wall = context.get("maturity_wall")
    guardrails: List[str] = []
    if _has_capital_return([action_id]):
        guardrails.append("Keep return-of-capital funding primarily cash-backed unless leverage is clearly below target.")
        if maturity_wall is not None and maturity_wall >= 0.20:
            guardrails.append(f"Do not size the payout as if the {_fmt_pct(maturity_wall)} near-term maturity wall does not exist.")
    if _is_balance_sheet_action(action_id):
        guardrails.append("Do not raise materially more capital than the identified coverage need plus a buffer.")
        guardrails.append("Bias financing structure toward certainty before carry optimization.")
    if _is_mna_action(action_id):
        guardrails.append("Keep pro forma leverage inside the recommended post-close band.")
        guardrails.append("Do not solve a strategic case by overusing equity or balance-sheet stretch.")
    if _is_divestiture_action(action_id):
        guardrails.append("Keep the sold package targeted; scale only if strategic coherence improves.")
    if net_leverage is not None and net_leverage >= 3.0:
        guardrails.append(f"Current net leverage at {_fmt_x(net_leverage)} leaves little room for parameter drift.")
    return guardrails[:4]


def _parameter_rejected_variants(
    *,
    action_id: str,
    recommended_parameters: Dict[str, Dict[str, Any]],
    context: Dict[str, Optional[float]],
) -> List[str]:
    rejected: List[str] = []
    if _has_capital_return([action_id]):
        rejected.append("Debt-heavy capital return that relies on a still-open credit window.")
        rejected.append("Token sizing that leaves the capital-allocation problem essentially unchanged.")
    if _is_balance_sheet_action(action_id):
        rejected.append("Max-size issuance that creates future leverage or dilution regret after the immediate problem is solved.")
    if _is_mna_action(action_id):
        rejected.append("Acquisition sizing that only works if synergies or financing terms are perfect.")
    if _is_divestiture_action(action_id):
        rejected.append("Over-broad divestiture simply to maximize proceeds in one step.")
    if not rejected and recommended_parameters:
        rejected.append("Parameter choices that maximize size before proving the case on flexibility, risk, and timing.")
    return rejected[:3]


def _parameter_optimization_summary(
    *,
    action_id: str,
    recommended_parameters: Dict[str, Dict[str, Any]],
    sizing_guidance: Dict[str, Any],
) -> str:
    parts: List[str] = []
    if "size_pct_market_cap" in recommended_parameters:
        parts.append(f"Target {_humanize_action_id(action_id).lower()} around {recommended_parameters['size_pct_market_cap'].get('recommended_range')}.")
    elif "size_absolute_usd" in recommended_parameters:
        parts.append(f"Target notional around {recommended_parameters['size_absolute_usd'].get('recommended_range')}.")
    elif "amount_usd" in recommended_parameters:
        parts.append(f"Target proceeds around {recommended_parameters['amount_usd'].get('recommended_range')}.")
    elif "amount_refinanced_usd" in recommended_parameters:
        parts.append(f"Target refinanced notional around {recommended_parameters['amount_refinanced_usd'].get('recommended_range')}.")
    elif "initial_yield_pct" in recommended_parameters:
        parts.append(f"Start the payout around {recommended_parameters['initial_yield_pct'].get('recommended_range')} of yield.")
    elif "percent_change" in recommended_parameters:
        parts.append(f"Keep the dividend change around {recommended_parameters['percent_change'].get('recommended_range')}.")
    if "funding_mix" in recommended_parameters:
        parts.append(f"Fund it with {recommended_parameters['funding_mix'].get('recommended_value_formatted')}.")
    if "annualized_cash_commitment_usd" in recommended_parameters:
        parts.append(f"Keep annualized cash commitment around {recommended_parameters['annualized_cash_commitment_usd'].get('recommended_range')}.")
    if "pace" in recommended_parameters:
        parts.append(f"Execution pace should be {recommended_parameters['pace'].get('recommended_value_formatted')}.")
    if "tenor_years" in recommended_parameters:
        parts.append(f"Tenor should center on {recommended_parameters['tenor_years'].get('recommended_value_formatted')}.")
    if "new_tenor_years" in recommended_parameters:
        parts.append(f"Tenor should center on {recommended_parameters['new_tenor_years'].get('recommended_value_formatted')}.")
    if not parts and sizing_guidance.get("recommended_range"):
        parts.append(f"Use the sizing posture of {sizing_guidance.get('recommended_range')}.")
    return " ".join(part for part in parts if part).strip()


def _bounded_amount_band(target: float) -> Tuple[float, float]:
    lower = max(0.0, target * 0.85)
    upper = max(lower, target * 1.15)
    return lower, upper


def _format_numeric_range(*, parameter_name: str, lower: float, upper: float) -> str:
    if parameter_name in {"tenor_years", "new_tenor_years", "call_protection_years"}:
        return f"{lower:.1f} to {upper:.1f} years"
    if parameter_name == "leverage_post_close":
        return f"{lower:.2f}x to {upper:.2f}x"
    return f"{lower:.2f} to {upper:.2f}"


