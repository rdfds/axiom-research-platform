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


def _format_parameter_value(parameter_type: str, value: Any, *, parameter_name: str = "") -> str:
    if value is None or value == "":
        return "n/a"
    if parameter_type == "percent":
        return _fmt_pct(value)
    if parameter_type == "numeric":
        if parameter_name.endswith("_usd") or parameter_name.startswith("amount_") or parameter_name in {"size_absolute_usd", "draw_amount_usd", "resize_amount_usd", "estimated_ev_usd"}:
            return _fmt_currency(value)
        if parameter_name in {"tenor_years", "new_tenor_years", "call_protection_years"}:
            num = _safe_float(value)
            return f"{num:.1f} years" if num is not None else "n/a"
        if parameter_name == "leverage_post_close":
            return _fmt_x(value)
        num = _safe_float(value)
        return f"{num:.2f}" if num is not None else "n/a"
    if parameter_type == "funding_mix_object":
        if not isinstance(value, dict):
            return "n/a"
        parts = [f"{key} {_fmt_pct(val)}" for key, val in value.items()]
        return " / ".join(parts)
    if parameter_type == "boolean":
        return "yes" if bool(value) else "no"
    return _humanize_text(str(value))

def _build_scenario_sizing(
    *,
    top_plan: Dict[str, Any],
    snapshot: Dict[str, Any],
    base_sizing: Dict[str, Any],
) -> List[Dict[str, Any]]:
    steps = list(top_plan.get("steps", []) or [])
    if not steps:
        return []
    first_action = str((steps[0].get("action_id", "") or ""))
    base_range = str(base_sizing.get("recommended_range", "") or "")
    if _is_buyback_action(first_action):
        return [
            {
                "scenario": "tight_credit_or_higher_vol",
                "sizing_adjustment": "Use the lower end of the range or phase execution.",
                "reason": "Repurchase regret rises if balance-sheet flexibility tightens after launch.",
            },
            {
                "scenario": "stronger_cash_build_or_deeper_discount",
                "sizing_adjustment": "Use the upper end of the range.",
                "reason": "A wider discount or larger cash build increases the cost of waiting.",
            },
        ]
    if first_action in {"capital_return.dividend_increase", "capital_return.dividend_initiate", "capital_return.special_dividend"}:
        return [
            {
                "scenario": "weaker_operating_outlook",
                "sizing_adjustment": "Stay at the floor of the range or defer.",
                "reason": "Sticky payout commitments are hardest to reverse cleanly.",
            },
            {
                "scenario": "sustained_cash_generation",
                "sizing_adjustment": "Move toward the upper end only after the stronger run rate proves durable.",
                "reason": "Dividend sizing should follow repeatable cash generation, not one strong quarter.",
            },
        ]
    if _is_balance_sheet_action(first_action):
        return [
            {
                "scenario": "tighter_financing_window",
                "sizing_adjustment": "Front-load execution but keep size to coverage-first minimums.",
                "reason": "The priority becomes securing resilience, not maximizing proceeds.",
            },
            {
                "scenario": "better_credit_or_equity_window",
                "sizing_adjustment": f"Use the current base range: {base_range}" if base_range else "Extend tenor or prefund modestly while terms remain constructive.",
                "reason": "A better window supports cleaner financing, not necessarily larger financing.",
            },
        ]
    if _is_mna_action(first_action):
        return [
            {
                "scenario": "higher_volatility_or_weaker_financing",
                "sizing_adjustment": "Bias toward a smaller bolt-on or defer.",
                "reason": "Deal regret rises quickly when financing and integration conditions worsen.",
            },
            {
                "scenario": "supportive_financing_and_high_conviction_target",
                "sizing_adjustment": "Move up only if leverage and integration remain contained.",
                "reason": "Larger deals require both strategic conviction and financing room.",
            },
        ]
    if _is_divestiture_action(first_action):
        return [
            {
                "scenario": "weak_bid_environment",
                "sizing_adjustment": "Sell the narrowest non-core package or wait.",
                "reason": "Forced scale in a weak market increases the odds of selling too cheap.",
            },
            {
                "scenario": "strong_bid_environment",
                "sizing_adjustment": "Expand scope only if strategic focus improves with the larger package.",
                "reason": "Higher bids justify more scale only if portfolio quality remains coherent afterward.",
            },
        ]
    return [
        {
            "scenario": "weaker_case",
            "sizing_adjustment": "Use the lower end of the sizing posture or defer.",
            "reason": "When the setup weakens, preserving flexibility should dominate scale.",
        },
        {
            "scenario": "stronger_case",
            "sizing_adjustment": f"Use the base range: {base_range}" if base_range else "Use the upper end only if the action edge widens.",
            "reason": "Larger action size should follow a clearer edge, not optimism alone.",
        },
    ]


def _build_regret_analysis(
    *,
    top_plan: Dict[str, Any],
    step_theses: Sequence[Dict[str, Any]],
    snapshot: Dict[str, Any],
    status_quo_view: Dict[str, Any],
) -> Dict[str, Any]:
    steps = list(top_plan.get("steps", []) or [])
    first_action = str((steps[0].get("action_id", "") or "")) if steps else ""
    regret_if_act = _regret_if_act(first_action=first_action, snapshot=snapshot)
    regret_if_wait = _regret_if_wait(first_action=first_action, snapshot=snapshot)
    return {
        "regret_balance": _regret_balance(
            first_action=first_action,
            recommended_posture=str(status_quo_view.get("recommended_posture", "") or ""),
            snapshot=snapshot,
        ),
        "if_we_act_and_are_wrong": regret_if_act,
        "if_we_wait_and_are_wrong": regret_if_wait,
        "decision_rule": "Prefer the path with lower irreversible regret unless the current edge versus status quo is clear.",
        "counterfactual_summary": (
            f"If we act and are wrong: {regret_if_act} "
            f"If we wait and are wrong: {regret_if_wait}"
        ),
    }


def _build_rating_cliff_analysis(
    *,
    top_plan: Dict[str, Any],
    snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    steps = list(top_plan.get("steps", []) or [])
    first_action = str((steps[0].get("action_id", "") or "")) if steps else ""
    rating_state = _feature_value(snapshot, "capital_structure.rating_state")
    net_leverage = _safe_float(_feature_value(snapshot, "capital_structure.net_leverage"))
    maturity_wall = _safe_float(_feature_value(snapshot, "capital_structure.maturity_wall_ratio_24m"))
    credit_window = _safe_float(_feature_value(snapshot, "market.credit_window_proxy"))
    credit_spread_pct = _safe_float(_feature_value(snapshot, "market.credit_spread_percentile_2y"))
    rating = ""
    outlook = ""
    is_ig: Optional[bool] = None
    if isinstance(rating_state, dict):
        rating = str(rating_state.get("rating", "") or "")
        outlook = str(rating_state.get("outlook", "") or "")
        score = rating_state.get("score")
        upper = rating.upper()
        if upper:
            is_ig = not upper.startswith("BB") and not upper.startswith("B") and not upper.startswith("CCC")
        elif score is not None:
            try:
                is_ig = float(score) <= 10.5
            except Exception:
                is_ig = None

    pressure = 0.0
    why_it_matters: List[str] = []
    constraints_to_watch: List[str] = []
    if is_ig is False:
        pressure += 0.18
        why_it_matters.append("The issuer already screens as non-investment-grade, so incremental balance-sheet stress is punished faster.")
    if outlook.lower().startswith("neg"):
        pressure += 0.08
        why_it_matters.append("The rating outlook is already negative, so the downgrade path is shorter than normal.")
    if net_leverage is not None and net_leverage >= 3.0:
        pressure += 0.12
        why_it_matters.append(f"Net leverage at {_fmt_x(net_leverage)} leaves limited downgrade buffer.")
    elif net_leverage is not None and net_leverage >= 2.5:
        pressure += 0.06
    if maturity_wall is not None and maturity_wall >= 0.20:
        pressure += 0.10
        why_it_matters.append(f"A {_fmt_pct(maturity_wall)} near-term maturity wall can force financing under pressure.")
    if credit_spread_pct is not None and credit_spread_pct >= 75.0:
        pressure += 0.08
        why_it_matters.append(f"Credit spreads are already wide versus history at the {credit_spread_pct:.0f}th percentile.")
    if credit_window is not None and credit_window <= 0.40:
        pressure += 0.06
        why_it_matters.append(
            f"Debt-market conditions are {_market_window_description(credit_window, 'debt')}, so rating damage would be expensive."
        )

    covenant_pressure = 0.0
    if is_ig is False:
        covenant_pressure += 0.10
    if net_leverage is not None and net_leverage >= 3.5:
        covenant_pressure += 0.18
    elif net_leverage is not None and net_leverage >= 3.0:
        covenant_pressure += 0.10
    if maturity_wall is not None and maturity_wall >= 0.20:
        covenant_pressure += 0.06

    pressure_level = _pressure_label(pressure)
    covenant_level = _pressure_label(covenant_pressure)

    if _has_capital_return([first_action]):
        if pressure >= 0.18:
            constraint_posture = "binding_against_payout"
            action_interaction = "The rating/covenant profile raises the hurdle for immediate capital return."
        else:
            constraint_posture = "not_binding"
            action_interaction = "Rating and covenant pressure do not appear to be the main reason to avoid the payout."
    elif _is_balance_sheet_action(first_action):
        if pressure >= 0.12 or covenant_pressure >= 0.12:
            constraint_posture = "supports_balance_sheet_action"
            action_interaction = "Rating and covenant pressure reinforce the case for a financing or liability-management step."
        else:
            constraint_posture = "monitor_but_not_binding"
            action_interaction = "Balance-sheet action is not purely being forced by rating or covenant stress."
    else:
        constraint_posture = "monitor_but_not_binding" if pressure >= 0.12 else "not_binding"
        action_interaction = "Rating and covenant pressure are a real constraint but not obviously the sole decision driver."

    if is_ig is False:
        constraints_to_watch.append("Avoid actions that could push the company deeper into sub-investment-grade financing costs.")
    if covenant_pressure >= 0.12:
        constraints_to_watch.append("Do not assume covenant headroom is abundant once leverage or EBITDA softens.")
    if maturity_wall is not None and maturity_wall >= 0.20:
        constraints_to_watch.append("Do not spend flexibility ahead of the near-term maturity burden.")
    if not constraints_to_watch:
        constraints_to_watch.append("Monitor rating and financing conditions even if they are not the lead constraint today.")

    return {
        "rating": rating or None,
        "outlook": outlook or None,
        "is_investment_grade": is_ig,
        "rating_pressure_level": pressure_level,
        "covenant_pressure_level": covenant_level,
        "constraint_posture": constraint_posture,
        "why_it_matters": why_it_matters[:3] or ["Rating and covenant headroom are not obviously the binding constraint."],
        "action_interaction": action_interaction,
        "constraints_to_watch": constraints_to_watch[:4],
    }


def _build_signaling_analysis(
    *,
    top_plan: Dict[str, Any],
    snapshot: Dict[str, Any],
    status_quo_view: Dict[str, Any],
) -> Dict[str, Any]:
    steps = list(top_plan.get("steps", []) or [])
    first_action = str((steps[0].get("action_id", "") or "")) if steps else ""
    return_capital_priority = _safe_float(_feature_value(snapshot, "strategic.intent.return_capital_priority"))
    pursue_mna_priority = _safe_float(_feature_value(snapshot, "strategic.intent.pursue_mna_priority"))
    focus_on_core = _safe_float(_feature_value(snapshot, "strategic.intent.focus_on_core"))
    activist_signal = _safe_float(_feature_value(snapshot, "ownership_governance.activist_signal"))

    favorable: List[str] = []
    adverse: List[str] = []
    belief_tests: List[str] = []
    signal_posture = "mixed"

    if _is_buyback_action(first_action):
        signal_posture = "positive_if_disciplined"
        favorable.append("The market can read a buyback as evidence that excess capital exists and management sees limited better uses.")
        adverse.append("The market can also read a buyback as a lack of growth ideas if the operating backdrop is soft.")
        belief_tests.append("Investors have to believe liquidity is truly excess and not needed for resilience.")
    elif first_action in {"capital_return.dividend_increase", "capital_return.dividend_initiate", "capital_return.special_dividend"}:
        signal_posture = "positive_but_sticky"
        favorable.append("A dividend step can signal confidence in durable cash generation.")
        adverse.append("If the payout later proves hard to sustain, the signaling damage is worse than for a buyback.")
        belief_tests.append("Investors have to believe free cash flow is repeatable enough to support the payout.")
    elif first_action == "capital_return.dividend_cut":
        signal_posture = "negative_but_honest"
        favorable.append("A cut can be read positively only if it clearly protects balance-sheet resilience.")
        adverse.append("Absent a strong repair story, a dividend cut is usually read as distress.")
        belief_tests.append("Investors have to believe the cut fixes a real problem rather than confirms deeper deterioration.")
    elif _uses_equity_markets([first_action]):
        signal_posture = "negative_unless_proactive"
        favorable.append("Equity issuance can signal prudence if it is clearly prefunding resilience or a high-return use of capital.")
        adverse.append("The default market read is dilution, funding stress, or overvaluation capture.")
        belief_tests.append("Investors have to believe the proceeds solve a concrete need that justifies the dilution.")
    elif _is_balance_sheet_action(first_action):
        signal_posture = "constructive_if_preemptive"
        favorable.append("Refinancing or liability management can signal proactive balance-sheet discipline.")
        adverse.append("If the company looks forced into the deal, the same action can signal fragility.")
        belief_tests.append("Investors have to believe the company is acting from choice rather than desperation.")
    elif _is_mna_action(first_action):
        signal_posture = "high_variance"
        favorable.append("M&A can signal confidence, ambition, and a differentiated growth path.")
        adverse.append("It can also signal empire-building or overpayment if the strategic fit is not obvious.")
        belief_tests.append("Investors have to believe the return on the deal exceeds the next-best use of capital.")
    elif _is_divestiture_action(first_action):
        signal_posture = "positive_if_focus"
        favorable.append("Divestiture can signal discipline, focus, and willingness to exit low-value complexity.")
        adverse.append("It can also signal that the company is a forced seller if the balance-sheet story looks weak.")
        belief_tests.append("Investors have to believe the sale improves the remaining business rather than just raises cash.")

    if (return_capital_priority or 0.0) >= 0.75 and _has_capital_return([first_action]):
        favorable.append("Management signaling already leans toward capital return, so the action is less likely to shock the market.")
    if (pursue_mna_priority or 0.0) >= 0.75 and _is_mna_action(first_action):
        favorable.append("Management has already been signaling external growth, which lowers surprise risk.")
    if (focus_on_core or 0.0) >= 0.70 and _is_divestiture_action(first_action):
        favorable.append("The portfolio-focus narrative is already available to investors.")
    if (activist_signal or 0.0) >= 0.5:
        favorable.append("Elevated activist pressure means the market is already primed for visible action.")
        adverse.append("Visible action under activist pressure can be read as reactive rather than strategic if the rationale is thin.")

    if str(status_quo_view.get("recommended_posture", "") or "") == "wait":
        adverse.append("Because the edge versus waiting is not decisive, the market could read the action as forced or premature.")

    return {
        "signal_posture": signal_posture,
        "favorable_interpretations": favorable[:4] or ["The action does not create an obviously strong external signal."],
        "adverse_interpretations": adverse[:4] or ["The action does not carry a strong adverse signal by itself."],
        "what_market_has_to_believe": belief_tests[:3] or ["Investors have to believe the action solves a real problem more cleanly than waiting."],
    }


def _build_step_sizing_guidance(
    *,
    action_id: str,
    parameters: Dict[str, Any],
    snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    liquidity = _safe_float(_feature_value(snapshot, "liquidity.available_for_actions"))
    market_cap = _safe_float(_feature_value(snapshot, "market.market_cap"))
    net_leverage = _safe_float(_feature_value(snapshot, "capital_structure.net_leverage"))
    maturity_wall = _safe_float(_feature_value(snapshot, "capital_structure.maturity_wall_ratio_24m"))
    liquidity_to_mcap = (liquidity / market_cap) if liquidity is not None and market_cap not in (None, 0.0) else None

    if _is_buyback_action(action_id):
        explicit = _safe_float(parameters.get("size_pct_market_cap"))
        if explicit is not None:
            lower = max(0.01, explicit * 0.85)
            upper = explicit * 1.15
            if net_leverage is not None and net_leverage >= 2.5:
                upper = min(upper, explicit)
            recommended_range = f"{_fmt_pct(lower)} to {_fmt_pct(upper)} of market value"
            why_not_larger = "A larger repurchase would start to trade away flexibility too aggressively."
            if maturity_wall is not None and maturity_wall >= 0.20:
                why_not_larger = f"A larger repurchase would spend capital ahead of a {_fmt_pct(maturity_wall)} maturity wall."
            why_not_smaller = "A meaningfully smaller program would not solve the idle-capital problem as directly."
            return {
                "sizing_posture": "moderate" if explicit < 0.08 else "assertive",
                "recommended_range": recommended_range,
                "rationale": [
                    f"Current plan parameter is {_fmt_pct(explicit)} of market value.",
                    f"Net leverage is {_fmt_x(net_leverage)}." if net_leverage is not None else "Balance-sheet capacity still matters more than gross authorization size.",
                ],
                "why_not_larger": why_not_larger,
                "why_not_smaller": why_not_smaller,
            }
        if liquidity_to_mcap is not None:
            upper = min(0.10, max(0.03, liquidity_to_mcap * 0.5))
            lower = max(0.02, min(upper - 0.01, upper * 0.6))
            if net_leverage is not None and net_leverage >= 2.5:
                upper = min(upper, 0.06)
                lower = min(lower, 0.04)
            return {
                "sizing_posture": "moderate",
                "recommended_range": f"{_fmt_pct(lower)} to {_fmt_pct(upper)} of market value",
                "rationale": [
                    f"Deployable liquidity is {_fmt_pct(liquidity_to_mcap)} of market value.",
                    f"Net leverage is {_fmt_x(net_leverage)}." if net_leverage is not None else "Sizing should preserve room for downside volatility.",
                ],
                "why_not_larger": "Going larger would reduce optionality faster than the current setup justifies.",
                "why_not_smaller": "Going much smaller would leave the capital-allocation problem mostly unresolved.",
            }

    if action_id in {"capital_return.dividend_increase", "capital_return.dividend_initiate"}:
        return {
            "sizing_posture": "conservative",
            "recommended_range": "Start with a modest recurring payout and leave room to scale only after results hold.",
            "rationale": [
                "Recurring dividends are harder to reverse than buybacks.",
                f"Net leverage is {_fmt_x(net_leverage)}." if net_leverage is not None else "The recurring commitment should be sized below the business's downside cash-generation case.",
            ],
            "why_not_larger": "A larger recurring payout raises the odds of later reversal and signaling damage.",
            "why_not_smaller": "A token increase would not create a meaningful capital-allocation signal.",
        }

    if action_id == "capital_return.special_dividend":
        return {
            "sizing_posture": "measured",
            "recommended_range": "Keep the one-time payout below the full excess-cash position and retain a liquidity buffer.",
            "rationale": [
                f"Deployable liquidity is {_fmt_currency(liquidity)}." if liquidity is not None else "One-time payouts should be sized against true excess liquidity, not gross cash.",
                "A special dividend is less sticky than a recurring payout but still consumes optionality immediately.",
            ],
            "why_not_larger": "Distributing too much cash now would reduce flexibility with no chance to scale back afterward.",
            "why_not_smaller": "A de minimis special dividend would not solve the distribution question cleanly.",
        }

    if action_id in {"capital_structure.refinancing", "capital_structure.new_debt_issuance", "capital_structure.revolver_draw_or_resize"}:
        return {
            "sizing_posture": "coverage_first",
            "recommended_range": "Size the financing to cover the near-term need plus a buffer, not to maximize gross proceeds.",
            "rationale": [
                f"24-month maturity wall is {_fmt_pct(maturity_wall)}." if maturity_wall is not None else "The financing should be anchored to a concrete need rather than headline size.",
                f"Net leverage is {_fmt_x(net_leverage)}." if net_leverage is not None else "Additional debt should improve resilience rather than just expand gross leverage.",
            ],
            "why_not_larger": "Over-issuing debt can fix a timing problem by creating a later leverage problem.",
            "why_not_smaller": "Under-sizing the transaction can force a second financing under worse conditions.",
        }

    if _uses_equity_markets([action_id]):
        return {
            "sizing_posture": "minimum_necessary",
            "recommended_range": "Raise only the amount required to solve the funding problem with an explicit dilution ceiling.",
            "rationale": [
                "Equity is the most visible and often the most expensive source of permanent capital.",
                f"Net leverage is {_fmt_x(net_leverage)}." if net_leverage is not None else "Dilution should be justified by a clear resilience or growth need.",
            ],
            "why_not_larger": "Excess equity issuance dilutes existing holders without proportional strategic benefit.",
            "why_not_smaller": "A too-small issuance can leave the balance-sheet problem unresolved and force another raise.",
        }

    if _is_mna_action(action_id):
        return {
            "sizing_posture": "bolt_on_bias",
            "recommended_range": "Prefer a deal size the company can absorb without compromising financing flexibility.",
            "rationale": [
                "The hurdle for M&A should rise with irreversibility and integration risk.",
                f"Net leverage is {_fmt_x(net_leverage)}." if net_leverage is not None else "Keep transaction size inside the company's proven integration capacity.",
            ],
            "why_not_larger": "A larger acquisition increases integration, financing, and regret risk all at once.",
            "why_not_smaller": "A too-small deal can consume management attention without moving the strategic outcome.",
        }

    if _is_divestiture_action(action_id):
        return {
            "sizing_posture": "targeted",
            "recommended_range": "Sell the least strategic or lowest-return assets first rather than forcing a large disposal.",
            "rationale": [
                "Divestiture sizing should follow strategic fit, not just headline proceeds.",
                "The first sale should prove portfolio simplification and capital release before scaling further.",
            ],
            "why_not_larger": "A larger sale can destroy strategic coherence if it is driven by urgency rather than asset quality.",
            "why_not_smaller": "A very small sale may not create enough focus or flexibility to justify the effort.",
        }

    return {
        "sizing_posture": "measured",
        "recommended_range": "Size the action to solve the diagnosed problem without giving away future flexibility.",
        "rationale": ["The action should clear a concrete need test rather than maximize gross scale."],
        "why_not_larger": "Larger size would increase regret risk without enough added benefit.",
        "why_not_smaller": "Smaller size would risk failing to solve the problem cleanly.",
    }


def _regret_if_act(*, first_action: str, snapshot: Dict[str, Any]) -> str:
    maturity_wall = _safe_float(_feature_value(snapshot, "capital_structure.maturity_wall_ratio_24m"))
    if _has_capital_return([first_action]):
        if maturity_wall is not None and maturity_wall >= 0.20:
            return f"We commit capital before resolving a {_fmt_pct(maturity_wall)} near-term maturity burden."
        return "We return capital now and later discover that the cash had a better strategic or defensive use."
    if _is_balance_sheet_action(first_action):
        return "We lock in financing that solves a timing problem but leaves the company with avoidable cost or dilution."
    if _is_mna_action(first_action):
        return "We commit to an irreversible deal and then find the strategic fit or integration case was overstated."
    if _is_divestiture_action(first_action):
        return "We sell too much or sell too cheaply and regret the loss of the asset later."
    return "We act too aggressively before the edge versus waiting is truly proven."


def _regret_if_wait(*, first_action: str, snapshot: Dict[str, Any]) -> str:
    maturity_wall = _safe_float(_feature_value(snapshot, "capital_structure.maturity_wall_ratio_24m"))
    liquidity = _safe_float(_feature_value(snapshot, "liquidity.available_for_actions"))
    market_cap = _safe_float(_feature_value(snapshot, "market.market_cap"))
    liquidity_to_mcap = (liquidity / market_cap) if liquidity is not None and market_cap not in (None, 0.0) else None
    if _has_capital_return([first_action]):
        if liquidity_to_mcap is not None:
            return f"We leave {_fmt_pct(liquidity_to_mcap)} of market value idle and miss a clean capital-return window."
        return "We leave excess capital idle and fail to solve the capital-allocation problem."
    if _is_balance_sheet_action(first_action):
        if maturity_wall is not None and maturity_wall >= 0.20:
            return f"We are forced to refinance later under worse conditions while a {_fmt_pct(maturity_wall)} maturity burden is still approaching."
        return "We lose the current financing window and end up solving the balance-sheet problem on worse terms."
    if _is_mna_action(first_action):
        return "We preserve flexibility but miss a genuine strategic opening that does not come back on similar terms."
    if _is_divestiture_action(first_action):
        return "We keep a low-quality or non-core asset too long and preserve complexity that should have been removed."
    return "We preserve optionality but allow a fixable problem to linger longer than necessary."


def _regret_balance(*, first_action: str, recommended_posture: str, snapshot: Dict[str, Any]) -> str:
    maturity_wall = _safe_float(_feature_value(snapshot, "capital_structure.maturity_wall_ratio_24m"))
    if recommended_posture == "wait":
        return "bias_to_wait"
    if _is_balance_sheet_action(first_action) and (maturity_wall or 0.0) >= 0.20:
        return "bias_to_action"
    if _is_mna_action(first_action) or first_action in {"capital_return.dividend_increase", "capital_return.dividend_initiate", "capital_return.special_dividend"}:
        return "bias_to_wait"
    if _is_buyback_action(first_action):
        return "balanced_with_reversible_bias"
    return "balanced"


def _build_alternative_analysis(
    *,
    top_plan: Dict[str, Any],
    other_plans: Sequence[Dict[str, Any]],
    diagnosed: Dict[str, Any],
    snapshot: Dict[str, Any],
    candidate_by_action: Dict[str, Dict[str, Any]],
    precedent_by_action: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    top_components = dict(top_plan.get("score_components", {}) or {})
    top_action_ids = [str(step.get("action_id", "") or "") for step in list(top_plan.get("steps", []) or [])]
    top_first = str((top_action_ids[0] if top_action_ids else "") or "")
    top_candidate = _resolve_plan_action_candidate(plan=top_plan, action_id=top_first, candidate_by_action=candidate_by_action)
    for alt in other_plans:
        alt = dict(alt or {})
        alt_components = dict(alt.get("score_components", {}) or {})
        alt_action_ids = [str(step.get("action_id", "") or "") for step in list(alt.get("steps", []) or [])]
        alt_first = str((alt_action_ids[0] if alt_action_ids else "") or "")
        alt_candidate = _resolve_plan_action_candidate(plan=alt, action_id=alt_first, candidate_by_action=candidate_by_action)
        reasons = _build_alternative_rebuttal_reasons(
            top_plan=top_plan,
            alt_plan=alt,
            top_components=top_components,
            alt_components=alt_components,
            top_action_ids=top_action_ids,
            alt_action_ids=alt_action_ids,
            top_candidate=top_candidate,
            alt_candidate=alt_candidate,
            snapshot=snapshot,
            diagnosed=diagnosed,
            precedent_by_action=precedent_by_action,
        )
        out.append(
            {
                "plan_id": str(alt.get("plan_id", "") or ""),
                "action_ids": alt_action_ids,
                "why_not_preferred": _format_alternative_rebuttal(reasons),
                "comparison_reasons": reasons,
                "score_delta": round(float(top_plan.get("score", 0.0) or 0.0) - float(alt.get("score", 0.0) or 0.0), 6),
                "problem_alignment": diagnosed.get("primary_problem", ""),
            }
        )
    if not out:
        out = _fallback_alternative_analysis(
            top_plan=top_plan,
            diagnosed=diagnosed,
            snapshot=snapshot,
            candidate_by_action=candidate_by_action,
            precedent_by_action=precedent_by_action,
        )
    return out


def _fallback_alternative_analysis(
    *,
    top_plan: Dict[str, Any],
    diagnosed: Dict[str, Any],
    snapshot: Dict[str, Any],
    candidate_by_action: Dict[str, Dict[str, Any]],
    precedent_by_action: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    top_components = dict(top_plan.get("score_components", {}) or {})
    top_action_ids = [str(step.get("action_id", "") or "") for step in list(top_plan.get("steps", []) or [])]
    top_first = str((top_action_ids[0] if top_action_ids else "") or "")
    top_candidate = _resolve_plan_action_candidate(plan=top_plan, action_id=top_first, candidate_by_action=candidate_by_action)
    ranked: List[Tuple[float, str, Dict[str, Any]]] = []
    for action_id, candidate in candidate_by_action.items():
        if not action_id or action_id == top_first:
            continue
        support = _candidate_support_score(candidate) + (0.25 * _precedent_confidence(precedent_by_action.get(action_id, {})))
        ranked.append((support, action_id, dict(candidate or {})))
    ranked.sort(key=lambda item: item[0], reverse=True)

    out: List[Dict[str, Any]] = []
    for _, alt_action_id, alt_candidate in ranked[:3]:
        alt_components = {
            "expected_utility": _candidate_expected_utility(alt_candidate),
            "support_factor": float(alt_candidate.get("evaluation_confidence", 0.0) or 0.0),
            "tail_risk_penalty": _candidate_tail_penalty(alt_candidate),
            "time_discount_factor": 1.0,
        }
        reasons = _build_alternative_rebuttal_reasons(
            top_plan=top_plan,
            alt_plan={"plan_id": f"candidate::{alt_action_id}", "steps": [{"action_id": alt_action_id}], "actions": [alt_candidate]},
            top_components=top_components,
            alt_components=alt_components,
            top_action_ids=top_action_ids,
            alt_action_ids=[alt_action_id],
            top_candidate=top_candidate,
            alt_candidate=alt_candidate,
            snapshot=snapshot,
            diagnosed=diagnosed,
            precedent_by_action=precedent_by_action,
        )
        out.append(
            {
                "plan_id": f"candidate::{alt_action_id}",
                "action_ids": [alt_action_id],
                "why_not_preferred": _format_alternative_rebuttal(reasons),
                "comparison_reasons": reasons,
                "score_delta": None,
                "problem_alignment": diagnosed.get("primary_problem", ""),
            }
        )
    return out


def _build_risk_case(
    *,
    top_plan: Dict[str, Any],
    step_theses: Sequence[Dict[str, Any]],
    snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    risks = dict(top_plan.get("risks", {}) or {})
    first_action = str((((top_plan.get("steps", []) or [{}])[0] or {}).get("action_id", "") or ""))
    main_failure_modes = _dedupe([str(x).strip() for x in list(risks.get("main_failure_modes", []) or []) if str(x).strip()])[:4]
    regime_sensitivity = _dedupe([str(x).strip() for x in list(risks.get("regime_sensitivity", []) or []) if str(x).strip()])[:4]
    execution_risks = _dedupe([str(x).strip() for x in list(risks.get("execution_risks", []) or []) if str(x).strip()])[:4]
    tail_descriptions = _dedupe(
        desc
        for thesis in step_theses
        for desc in list(thesis.get("tail_descriptions", []) or [])
        if str(desc).strip()
    )
    adverse_tails = [desc for desc in tail_descriptions if str(desc).startswith("Adverse")]
    generic_failure_modes = {
        "Bottom decile historical outcome.",
        "Top decile historical outcome.",
    }
    if not main_failure_modes or set(main_failure_modes).issubset(generic_failure_modes):
        main_failure_modes = _fallback_failure_modes(first_action=first_action, snapshot=snapshot)
    elif _has_only_generic_tail_descriptions(main_failure_modes):
        main_failure_modes = _dedupe(_fallback_failure_modes(first_action=first_action, snapshot=snapshot) + main_failure_modes)[:4]
    if not main_failure_modes:
        main_failure_modes = _fallback_failure_modes(first_action=first_action, snapshot=snapshot)
    execution_risks = _dedupe(execution_risks + [tradeoff for thesis in step_theses for tradeoff in list(thesis.get("tradeoffs", []) or [])])[:4]
    if not execution_risks or all(_looks_generic_execution_risk(x) for x in execution_risks):
        execution_risks = _fallback_execution_risks(first_action=first_action, snapshot=snapshot)
    why_acceptable: List[str] = []
    feasibility_chain = float(((top_plan.get("score_components", {}) or {}).get("feasibility_chain", 0.0) or 0.0))
    if feasibility_chain >= 0.9:
        why_acceptable.append(f"Plan feasibility chain is {feasibility_chain:.3f}, so execution is not being forced through a weak step.")
    if list(top_plan.get("triggers", []) or []):
        why_acceptable.append("The plan has explicit monitoring triggers rather than assuming static conditions.")
    if list(top_plan.get("branches", []) or []):
        why_acceptable.append("The plan includes contingency branches, so it is not reliant on one path only.")
    if not why_acceptable:
        why_acceptable.append("Risks are acceptable only if the current operating and market posture holds.")

    kill_criteria = _decision_boundaries(
        first_action=first_action,
        snapshot=snapshot,
        top_plan=top_plan,
    )
    return {
        "main_failure_modes": [_humanize_text(x) for x in main_failure_modes],
        "regime_sensitivity": [_humanize_text(x) for x in regime_sensitivity],
        "execution_risks": [_humanize_text(x) for x in execution_risks],
        "why_risks_acceptable": why_acceptable[:4],
        "kill_criteria": kill_criteria,
    }


def _build_scorecard(
    *,
    top_plan: Dict[str, Any],
    step_theses: Sequence[Dict[str, Any]],
    precedent_by_action: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    action_ids = [str(step.get("action_id", "") or "") for step in list(top_plan.get("steps", []) or [])]
    precedent_scores = [
        _precedent_confidence(precedent_by_action.get(action_id, {}))
        for action_id in action_ids
        if action_id in precedent_by_action
    ]
    eval_scores = [float(((top_plan.get("actions", []) or [{}])[idx].get("evaluation_confidence", 0.0) or 0.0)) for idx in range(len(action_ids)) if idx < len(list(top_plan.get("actions", []) or []))]
    return {
        "plan_score": float(top_plan.get("score", 0.0) or 0.0),
        "expected_utility": float(((top_plan.get("score_components", {}) or {}).get("expected_utility", 0.0) or 0.0)),
        "feasibility_chain": float(((top_plan.get("score_components", {}) or {}).get("feasibility_chain", 0.0) or 0.0)),
        "support_factor": float(((top_plan.get("score_components", {}) or {}).get("support_factor", 0.0) or 0.0)),
        "tail_risk_penalty": float(((top_plan.get("score_components", {}) or {}).get("tail_risk_penalty", 0.0) or 0.0)),
        "average_precedent_confidence": round(sum(precedent_scores) / len(precedent_scores), 6) if precedent_scores else 0.0,
        "average_evaluation_confidence": round(sum(eval_scores) / len(eval_scores), 6) if eval_scores else 0.0,
        "causal_supported_steps": sum(1 for thesis in step_theses if str(thesis.get("support_type", "")).startswith("causal")),
    }


def _confidence_posture(
    *,
    top_plan: Dict[str, Any],
    step_theses: Sequence[Dict[str, Any]],
    precedent_by_action: Dict[str, Dict[str, Any]],
) -> str:
    plan_score = float(top_plan.get("score", 0.0) or 0.0)
    support_factor = float(((top_plan.get("score_components", {}) or {}).get("support_factor", 0.0) or 0.0))
    tail_penalty = float(((top_plan.get("score_components", {}) or {}).get("tail_risk_penalty", 0.0) or 0.0))
    avg_precedent = 0.0
    if step_theses:
        vals = [float(thesis.get("precedent_confidence", 0.0) or 0.0) for thesis in step_theses]
        avg_precedent = sum(vals) / len(vals)
    if plan_score >= 0.30 and support_factor >= 0.80 and avg_precedent >= 0.35 and tail_penalty <= 0.08:
        return "high_conviction"
    if plan_score >= 0.18 and support_factor >= 0.70:
        return "supported_but_conditional"
    return "conditional"


def _step_role_text(action_id: str, parameters: Dict[str, Any], snapshot: Dict[str, Any], diagnosed: Dict[str, Any]) -> str:
    if _is_buyback_action(action_id):
        size_pct = _safe_float(parameters.get("size_pct_market_cap"))
        size_abs = _safe_float(parameters.get("size_absolute_usd"))
        funding_mix = dict(parameters.get("funding_mix", {}) or {})
        if size_pct is not None:
            size_text = f" at about {_fmt_pct(size_pct)} of market value"
        elif size_abs is not None:
            size_text = f" for roughly {_fmt_currency(size_abs)}"
        else:
            size_text = ""
        funding_text = ""
        cash_share = _safe_float(funding_mix.get("cash"))
        debt_share = _safe_float(funding_mix.get("debt"))
        if cash_share is not None and cash_share >= 0.75:
            funding_text = " funded primarily from cash"
        elif debt_share is not None and debt_share >= 0.25:
            funding_text = " while preserving flexibility on the funding mix"
        return f"Repurchase stock{size_text}{funding_text} to absorb excess capital and improve per-share value."
    if action_id in {"capital_return.special_dividend", "capital_return.dividend_increase", "capital_return.dividend_initiate"}:
        yield_pct = _safe_float(parameters.get("initial_yield_pct"))
        annual_cash = _safe_float(parameters.get("annualized_cash_commitment_usd"))
        if action_id == "capital_return.special_dividend":
            return "Return excess cash through a one-time distribution without permanently resetting payout policy."
        if yield_pct is not None:
            return f"Reset recurring payout policy around {_fmt_pct(yield_pct)} of yield so cash return is explicit rather than residual."
        if annual_cash is not None:
            return f"Reset recurring payout policy around {_fmt_currency(annual_cash)} of annual cash commitment."
        return "Reset payout policy to return excess cash through a recurring distribution."
    if action_id == "capital_structure.refinancing":
        amount = _safe_float(parameters.get("amount_refinanced_usd"))
        tenor = _normalize_numeric_current(
            parameter_name="new_tenor_years",
            current_value=parameters.get("new_tenor_years"),
            parameter_schema={},
        )
        targeted = dict(parameters.get("maturities_targeted", {}) or {})
        targeted_max = _safe_float(targeted.get("max"))
        amount_text = f" about {_fmt_currency(amount)}" if amount is not None else ""
        tenor_text = f" into roughly {tenor:.1f}-year paper" if tenor is not None else ""
        targeted_text = ""
        if targeted_max is not None:
            targeted_text = f" with a focus on obligations coming due inside {targeted_max:.0f} years"
        return (
            f"Refinance{amount_text}{tenor_text} to push out maturities{targeted_text} before optionality is used elsewhere."
        ).replace("  ", " ").strip()
    if action_id in {"capital_structure.new_debt_issuance", "capital_structure.revolver_draw_or_resize"}:
        amount = _safe_float(
            parameters.get("amount_usd")
            or parameters.get("draw_amount_usd")
            or parameters.get("resize_amount_usd")
        )
        tenor = _normalize_numeric_current(
            parameter_name="tenor_years",
            current_value=parameters.get("tenor_years"),
            parameter_schema={},
        )
        use_of_proceeds = _humanize_use_of_proceeds(str(parameters.get("use_of_proceeds", "") or ""))
        amount_text = f" about {_fmt_currency(amount)}" if amount is not None else ""
        tenor_text = f" with roughly {tenor:.1f}-year tenor" if tenor is not None else ""
        if action_id == "capital_structure.revolver_draw_or_resize":
            return (
                f"Use the revolver{amount_text} to secure liquidity insurance{_for_phrase(use_of_proceeds)} "
                f"before relying on a less forgiving market."
            )
        return (
            f"Raise{amount_text} of new debt{tenor_text} to {use_of_proceeds}, so later steps are not funded from a weaker position."
        ).replace("  ", " ").strip()
    if action_id in {"capital_structure.tender_offer_debt", "capital_structure.exchange_offer", "capital_structure.liability_management_exercise"}:
        return "Actively reshape liabilities to reduce refinancing or spread risk."
    if action_id in {"capital_structure.equity_issuance", "capital_structure.convertible_issuance", "capital_structure.preferred_issuance"}:
        amount = _safe_float(parameters.get("amount_usd"))
        use_of_proceeds = _humanize_use_of_proceeds(str(parameters.get("use_of_proceeds", "") or ""))
        amount_text = f" about {_fmt_currency(amount)} of" if amount is not None else ""
        instrument = "convertible capital" if action_id == "capital_structure.convertible_issuance" else "equity capital"
        if action_id == "capital_structure.preferred_issuance":
            instrument = "preferred capital"
        return (
            f"Raise {amount_text} {instrument} to {use_of_proceeds}, accepting dilution only because flexibility is the binding issue."
        ).replace("  ", " ").strip()
    if _is_mna_action(action_id):
        size = _safe_float(parameters.get("estimated_ev_usd"))
        leverage_post_close = _normalize_numeric_current(
            parameter_name="leverage_post_close",
            current_value=parameters.get("leverage_post_close"),
            parameter_schema={},
        )
        size_text = f" at roughly {_fmt_currency(size)} of enterprise value" if size is not None else ""
        leverage_text = f" while keeping pro forma leverage near {leverage_post_close:.1f}x" if leverage_post_close is not None else ""
        return (
            f"Use external action{size_text} to add growth, scale, or strategic control that internal deployment would not create{leverage_text}."
        ).replace("  ", " ").strip()
    if _is_divestiture_action(action_id):
        pct_divested = _safe_float(parameters.get("percent_divested"))
        use_of_proceeds = _humanize_use_of_proceeds(str(parameters.get("use_of_proceeds", "") or ""))
        pct_text = f" by selling roughly {_fmt_pct(pct_divested)} of the asset base" if pct_divested is not None else ""
        return f"Release capital{pct_text} and simplify the portfolio before using the proceeds to {use_of_proceeds}."
    return f"{_humanize_action_id(action_id)} addresses the current strategic bottleneck."


def _timing_thesis(
    *,
    action_id: str,
    step: Dict[str, Any],
    snapshot: Dict[str, Any],
    diagnosed: Dict[str, Any],
    plan: Dict[str, Any],
) -> str:
    prerequisites = list(step.get("prerequisites", []) or [])
    median_days = int(((step.get("expected_duration", {}) or {}).get("median_days", 0) or 0))
    maturity_wall = _safe_float(_feature_value(snapshot, "capital_structure.maturity_wall_ratio_24m"))
    net_leverage = _safe_float(_feature_value(snapshot, "capital_structure.net_leverage"))
    equity_window = _safe_float(_feature_value(snapshot, "market.equity_window_proxy"))
    credit_window = _safe_float(_feature_value(snapshot, "market.credit_window_proxy"))
    liquidity_to_mcap = None
    liq = _safe_float(_feature_value(snapshot, "liquidity.available_for_actions"))
    mcap = _safe_float(_feature_value(snapshot, "market.market_cap"))
    if liq is not None and mcap not in (None, 0.0):
        liquidity_to_mcap = liq / mcap

    parts: List[str] = []
    if prerequisites:
        parts.append(f"This step should happen only after {', '.join(_humanize_action_id(x) for x in prerequisites)} has landed.")
    else:
        parts.append("This is a front-of-plan action rather than a follow-on clean-up step.")
    if _is_balance_sheet_action(action_id) and (maturity_wall or 0.0) >= 0.12:
        parts.append(f"The 24-month maturity wall is already {_fmt_pct(maturity_wall)}, so waiting does not improve the liability profile.")
    if _is_balance_sheet_action(action_id) and (net_leverage or 0.0) >= 3.0:
        parts.append(f"Net leverage is {_fmt_x(net_leverage)}, so preserving financing flexibility matters before doing anything more discretionary.")
    if _uses_credit_markets([action_id]) and (credit_window or 0.0) >= 0.50:
        parts.append(
            f"Debt markets are currently {_market_window_description(credit_window, 'debt')}, "
            "so it is safer to address the financing need before conditions worsen."
        )
    elif _uses_credit_markets([action_id]) and credit_window is not None and credit_window <= 0.35:
        parts.append("Debt markets are already fragile enough that waiting risks losing the remaining financing window.")
    if _uses_equity_markets([action_id]) and (equity_window or 0.0) >= 0.60:
        parts.append(f"Equity markets are open enough to issue now and issuance conditions are {_market_window_description(equity_window, 'equity')}.")
    if _has_capital_return([action_id]) and (liquidity_to_mcap or 0.0) >= 0.03:
        parts.append(f"Waiting mainly leaves {_fmt_pct(liquidity_to_mcap)} of market value idle rather than improving the setup.")
    if median_days > 0:
        parts.append(f"Lead time is about {median_days} days, so the plan captures value on a near-term rather than distant timeline.")
    return " ".join(parts[:4]).strip()


def _step_tradeoffs(action_candidate: Dict[str, Any], action_id: str, snapshot: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for risk in list(action_candidate.get("risks", []) or []):
        explanation = str(risk.get("explanation", "") or "").strip()
        if explanation:
            out.append(explanation)
    for flag in list(action_candidate.get("structural_sanity_flags", []) or []):
        if str(flag.get("status", "") or "") == "warning":
            explanation = str(flag.get("explanation", "") or "").strip()
            if explanation:
                out.append(explanation)
    if _has_capital_return([action_id]):
        net_leverage = _safe_float(_feature_value(snapshot, "capital_structure.net_leverage"))
        if net_leverage is not None and net_leverage >= 2.5:
            out.append("Capital return would be more controversial at current leverage and should not crowd out balance-sheet repair.")
    return _dedupe(out)


def _step_supporting_facts(
    *,
    action_id: str,
    action_candidate: Dict[str, Any],
    precedent_pack: Dict[str, Any],
    snapshot: Dict[str, Any],
    support_type: str,
    sample_n: int,
    precedent_confidence: float,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    pass_probability = float(((action_candidate.get("feasibility", {}) or {}).get("pass_probability", 0.0) or 0.0))
    if pass_probability > 0.0:
        out.append(
            {
                "label": "Feasibility",
                "metric": "pass_probability",
                "value": pass_probability,
                "formatted_value": f"{pass_probability:.3f}",
                "text": f"Modeled pass probability is {pass_probability:.3f}.",
                "source": "feasibility",
            }
        )

    objective_signal = _best_objective_signal(action_candidate)
    if objective_signal is not None:
        out.append(
            {
                "label": "Modeled objective",
                "metric": objective_signal[0],
                "value": objective_signal[1],
                "formatted_value": f"{objective_signal[1]:+.3f}",
                "text": f"The strongest median objective contribution is {objective_signal[0]} at {objective_signal[1]:+.3f}.",
                "source": support_type,
            }
        )

    if precedent_confidence > 0.0:
        tier = str(((precedent_pack.get("mismatch_diagnostics", {}) or {}).get("retrieval_tier", "")) or "")
        out.append(
            {
                "label": "Precedent quality",
                "metric": "precedent_confidence",
                "value": precedent_confidence,
                "formatted_value": f"{precedent_confidence:.3f}",
                "text": f"Historical analog support is {precedent_confidence:.3f} on a {tier or 'matched'} cohort with n={sample_n}.",
                "source": "precedent",
            }
        )

    for metric_key, label, formatter in _family_metrics_for_action(action_id):
        value = _feature_value(snapshot, metric_key)
        if value is None:
            continue
        out.append(
            {
                "label": label,
                "metric": metric_key,
                "value": value,
                "formatted_value": formatter(value),
                "text": f"{label} is {formatter(value)}.",
                "source": "snapshot",
            }
        )
    return out[:5]


def _family_metrics_for_action(action_id: str) -> List[Tuple[str, str, Any]]:
    if _has_capital_return([action_id]):
        return [
            ("liquidity.available_for_actions", "Deployable liquidity", _fmt_currency),
            ("market.market_cap", "Market value", _fmt_currency),
            ("capital_structure.net_leverage", "Net leverage", _fmt_x),
            ("operating.fcf_conversion", "FCF conversion", _fmt_ratio),
        ]
    if _is_balance_sheet_action(action_id):
        return [
            ("capital_structure.maturity_wall_ratio_24m", "Maturity wall", _fmt_pct),
            ("capital_structure.net_leverage", "Net leverage", _fmt_x),
            ("market.credit_window_proxy", "Credit window", _fmt_score),
            ("liquidity.available_for_actions", "Liquidity", _fmt_currency),
        ]
    if _is_mna_action(action_id):
        return [
            ("liquidity.available_for_actions", "Deployable liquidity", _fmt_currency),
            ("capital_structure.net_leverage", "Net leverage", _fmt_x),
            ("strategic.intent.pursue_mna_priority", "M&A intent", _fmt_score),
            ("market.equity_window_proxy", "Equity window", _fmt_score),
        ]
    if _is_divestiture_action(action_id):
        return [
            ("capital_structure.net_leverage", "Net leverage", _fmt_x),
            ("strategic.intent.focus_on_core", "Focus on core", _fmt_score),
            ("ownership_governance.activist_signal", "Activist pressure", _fmt_score),
        ]
    return [
        ("capital_structure.net_leverage", "Net leverage", _fmt_x),
        ("liquidity.available_for_actions", "Liquidity", _fmt_currency),
    ]


def _decision_preconditions(first_action: str, snapshot: Dict[str, Any], top_plan: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    if _has_capital_return([first_action]):
        out.append("Liquidity after the action must remain comfortably above minimum operating needs.")
        out.append("There cannot be a hidden maturity or funding issue that makes capital return premature.")
    if _is_balance_sheet_action(first_action):
        out.append("The capital-markets window must remain open enough to transact on acceptable terms.")
        out.append("The transaction has to improve flexibility, not just add gross debt.")
    if _is_mna_action(first_action):
        out.append("The deal has to clear a return hurdle that is better than the next-best use of capital.")
        out.append("Financing cannot compromise balance-sheet resilience.")
    if _is_divestiture_action(first_action):
        out.append("The asset sold has to be non-core or low-return relative to the capital it frees up.")
    if not out:
        out.append("The factual diagnosis and sequencing assumptions have to remain intact.")
    return out[:3]


def _decision_boundaries(first_action: str, snapshot: Dict[str, Any], top_plan: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    net_leverage = _safe_float(_feature_value(snapshot, "capital_structure.net_leverage"))
    maturity_wall = _safe_float(_feature_value(snapshot, "capital_structure.maturity_wall_ratio_24m"))
    credit_window = _safe_float(_feature_value(snapshot, "market.credit_window_proxy"))
    equity_window = _safe_float(_feature_value(snapshot, "market.equity_window_proxy"))
    if _has_capital_return([first_action]):
        out.append("Pause if liquidity is reallocated to a higher-return strategic use or leverage drifts materially higher.")
        if net_leverage is not None:
            out.append(f"Pause if net leverage moves materially above the current {_fmt_x(net_leverage)} baseline.")
    if _is_balance_sheet_action(first_action):
        out.append("Do not force a financing if the market window closes and terms stop improving the balance sheet.")
        if credit_window is not None and credit_window > 0.05:
            out.append(
                f"Reassess if debt-market conditions weaken meaningfully from the current "
                f"{_market_window_description(credit_window, 'debt')} backdrop."
            )
        else:
            out.append("Reassess if credit conditions deteriorate materially from here.")
    if _uses_equity_markets([first_action]) and equity_window is not None:
        if equity_window > 0.05:
            out.append(
                f"Reassess if equity-market conditions weaken from the current "
                f"{_market_window_description(equity_window, 'equity')} backdrop."
            )
        else:
            out.append("Reassess if the equity window weakens further.")
    if maturity_wall is not None and maturity_wall >= 0.20:
        out.append("Move faster, not slower, if near-term maturity pressure rises further.")
    triggers = list(top_plan.get("triggers", []) or [])
    for trigger in triggers[:2]:
        condition = str(trigger.get("condition", "") or "").strip()
        if condition:
            out.append(_humanize_condition(condition))
    return _dedupe(out)[:4]


def _precedent_confidence(pack: Dict[str, Any]) -> float:
    if not isinstance(pack, dict):
        return 0.0
    return float(pack.get("precedent_confidence", pack.get("calibration_confidence", 0.0)) or 0.0)


def _precedent_sample_size(pack: Dict[str, Any]) -> int:
    if not isinstance(pack, dict):
        return 0
    out = pack.get("outcome_distributions") or {}
    if isinstance(out, dict):
        h12 = dict(out.get("horizon_12m", {}) or {})
        val = dict(h12.get("valuation_multiple_change", {}) or {})
        n = val.get("sample_size")
        if n is not None:
            return int(n or 0)
    legacy = list(pack.get("legacy_distributions", []) or [])
    for item in legacy:
        if str((item or {}).get("metric", "")) == "outcome_pe_12m":
            return int((item or {}).get("n", 0) or 0)
    return 0


def _support_type(action_candidate: Dict[str, Any], precedent_pack: Dict[str, Any]) -> str:
    impact = dict(action_candidate.get("impact_distribution", {}) or {})
    key_drivers = list(impact.get("key_drivers", []) or [])
    if any(str((driver or {}).get("driver_name", "")).startswith("causal_model_") for driver in key_drivers):
        return "causal_and_precedent" if _precedent_confidence(precedent_pack) > 0 else "causal"
    if _precedent_confidence(precedent_pack) > 0:
        return "precedent"
    return "model_only"


def _best_objective_signal(action_candidate: Dict[str, Any]) -> Optional[Tuple[str, float]]:
    objectives = dict(((action_candidate.get("impact_distribution", {}) or {}).get("objectives", {}) or {}))
    best: Optional[Tuple[str, float]] = None
    for objective in _OBJECTIVE_FIELDS:
        median = _safe_float(((objectives.get(objective, {}) or {}).get("median")))
        if median is None:
            continue
        if best is None or median > best[1]:
            best = (objective, median)
    return best


def _objective_median(action_candidate: Dict[str, Any], objective: str) -> Optional[float]:
    objectives = dict(((action_candidate.get("impact_distribution", {}) or {}).get("objectives", {}) or {}))
    return _safe_float(((objectives.get(objective, {}) or {}).get("median")))


def _humanize_action_id(action_id: str) -> str:
    if not action_id:
        return ""
    leaf = action_id.split(".")[-1]
    return leaf.replace("_", " ")


def _humanize_mechanism_id(mechanism_id: str) -> str:
    if not mechanism_id:
        return ""
    return mechanism_id.replace("_", " ")


def _humanize_objective_name(objective_name: str) -> str:
    mapping = {
        "value_creation": "value creation",
        "risk_reduction": "risk reduction",
        "growth": "growth",
        "rating_preservation": "rating preservation",
        "optionality": "optionality",
    }
    return mapping.get(str(objective_name or ""), _humanize_text(str(objective_name or "")))


def _market_window_description(value: Any, market_type: str) -> str:
    score = _safe_float(value)
    if score is None:
        return "uncertain"
    if score < 0.25:
        return "tight"
    if score < 0.45:
        return "only partially open"
    if score < 0.65:
        return "open enough"
    return "very supportive"


def _humanize_condition(condition: str) -> str:
    raw = str(condition or "").strip()
    if not raw:
        return ""
    text = _humanize_text(raw)
    if text.startswith("follow-on capacity remains available after "):
        suffix = text[len("follow-on capacity remains available after ") :].strip()
        return f"Only continue if capacity still exists after {suffix}."
    enum_match = re.fullmatch(r"\s*([a-zA-Z0-9_.]+)\s+in\s+\[(.+)\]\s*", raw)
    if enum_match:
        field = _humanize_field_name(enum_match.group(1))
        values = _humanize_condition_values(enum_match.group(2))
        if field == "use of proceeds":
            return f"Only continue if the use of proceeds remains limited to {values}."
        return f"Only continue if {field} remains within {values}."
    compare_match = re.fullmatch(r"\s*([a-zA-Z0-9_.]+)\s*(==|>=|<=|>|<)\s*['\"]?([^'\"]+)['\"]?\s*", raw)
    if compare_match:
        field = _humanize_field_name(compare_match.group(1))
        operator = compare_match.group(2)
        value = _humanize_text(compare_match.group(3).strip())
        operator_text = {
            "==": "is",
            ">=": "stays at or above",
            "<=": "stays at or below",
            ">": "stays above",
            "<": "stays below",
        }.get(operator, operator)
        return f"Only continue if {field} {operator_text} {value}."
    directional_match = re.fullmatch(r"\s*([a-zA-Z0-9_.]+)\s+(drops below|falls below|rises above|moves above|improves)\s+(.+)\s*", raw)
    if directional_match:
        field = _humanize_field_name(directional_match.group(1))
        verb = directional_match.group(2)
        value = _humanize_text(directional_match.group(3).strip())
        return f"Only continue if {field} {verb} {value}."
    return text


def _humanize_text(text: str) -> str:
    if not text:
        return ""
    def repl(match: re.Match[str]) -> str:
        return _humanize_action_id(match.group(0))
    out = re.sub(r"\b[a-z_]+\.[a-z0-9_]+\b", repl, str(text or ""))
    out = out.replace("post refi", "post-refinancing")
    out = out.replace("_", " ")
    return re.sub(r"\s+", " ", out).strip()


def _edge_summary(edge_vs_wait: float, recommended_posture: str) -> str:
    if recommended_posture == "conditional_action":
        if edge_vs_wait >= 0.12:
            return "The edge over waiting is meaningful, but the action still depends on conditions holding."
        return "The edge over waiting is modest, so the action should stay conditional rather than automatic."
    if edge_vs_wait >= 0.12:
        return "The edge over waiting is meaningful enough to justify acting now."
    if edge_vs_wait >= 0.05:
        return "The edge over waiting is positive, though not overwhelming."
    return "The edge over waiting is thin, so execution discipline matters."


def _humanize_field_name(field_name: str) -> str:
    field = str(field_name or "").strip()
    if not field:
        return ""
    explicit = {
        "use_of_proceeds": "use of proceeds",
        "capital_structure.maturity_wall_ratio_24m": "the near-term maturity wall ratio",
        "capital_structure.net_leverage": "net leverage",
        "capital_structure.interest_coverage": "interest coverage",
        "market.credit_window_proxy": "debt-market conditions",
        "market.equity_window_proxy": "equity-market conditions",
        "liquidity.runway_months": "the liquidity runway",
    }
    if field in explicit:
        return explicit[field]
    if "." in field:
        field = field.split(".", 1)[1]
    return field.replace("_", " ")


def _humanize_condition_values(raw_values: str) -> str:
    parts = [
        _humanize_text(part.strip().strip("\"'"))
        for part in str(raw_values or "").split(",")
        if part.strip()
    ]
    if not parts:
        return "the allowed set"
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} or {parts[1]}"
    return ", ".join(parts[:-1]) + f", or {parts[-1]}"


def _humanize_triggers(triggers: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for trigger in triggers:
        row = dict(trigger or {})
        row["condition"] = _humanize_condition(str(row.get("condition", "") or ""))
        row["explanation"] = _humanize_explanation(str(row.get("explanation", "") or ""))
        out.append(row)
    return out


def _humanize_branches(branches: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for branch in branches:
        row = dict(branch or {})
        row["branch_condition"] = _humanize_condition(str(row.get("branch_condition", "") or ""))
        row["branch_plan_steps"] = [_humanize_action_id(str(x or "")) for x in list(row.get("branch_plan_steps", []) or [])]
        row["explanation"] = _humanize_explanation(str(row.get("explanation", "") or ""))
        out.append(row)
    return out


def _humanize_explanation(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    follow_on = re.fullmatch(
        r"Historical follow-on frequency supports ([a-z_]+\.[a-z0-9_]+) after ([a-z_]+\.[a-z0-9_]+)\.",
        raw,
    )
    if follow_on:
        return _follow_on_explanation(
            target_action_id=follow_on.group(1),
            source_action_id=follow_on.group(2),
        )
    unlock = re.fullmatch(r"Incremental debt capacity can unlock buyback actions\.", raw)
    if unlock:
        return "Additional debt capacity can support later share repurchases, but only if it clearly strengthens the balance-sheet plan first."
    return _humanize_text(raw)


def _follow_on_explanation(*, target_action_id: str, source_action_id: str) -> str:
    target_leaf = str(target_action_id or "").split(".")[-1]
    source = _humanize_action_id(source_action_id)
    if target_leaf == "dividend_increase":
        return f"After {source}, boards often revisit whether a higher recurring payout is supportable."
    if target_leaf == "dividend_cut":
        return f"After {source}, boards may need to revisit whether the current recurring payout still fits the balance sheet."
    if target_leaf == "dividend_initiate":
        return f"After {source}, a new recurring payout only becomes credible if clear excess capacity emerges."
    if target_leaf == "special_dividend":
        return f"After {source}, a one-time payout can be considered if surplus capital remains genuinely excess."
    if target_leaf in {"open_market_buyback", "tender_offer_buyback", "accelerated_share_repurchase"}:
        return f"After {source}, a follow-on share repurchase can become more credible if surplus capacity remains."
    if target_leaf in {"new_debt_issuance", "refinancing", "revolver_draw_or_resize", "equity_issuance"}:
        return f"After {source}, the board may revisit the financing mix if balance-sheet flexibility still needs work."
    if target_leaf in {"tuck_in_acquisition", "platform_acquisition", "transformational_acquisition"}:
        return f"After {source}, acquisitions become more realistic only if the balance sheet is still strong enough to support them."
    return f"After {source}, the board may revisit {_humanize_action_id(target_action_id)} if conditions improve."


def _humanize_tail_metric(*, metric: str, horizon: str, direction: str) -> str:
    metric_key = str(metric or "").strip()
    horizon_text = f"{str(horizon).strip()} " if str(horizon or "").strip() else ""
    mapping = {
        "equity_return_vs_sector": "relative share performance",
        "valuation_multiple_change": "valuation multiple performance",
        "credit_spread_change": "credit spread performance",
        "ebitda_change": "EBITDA performance",
        "fcf_change": "free-cash-flow performance",
    }
    label = mapping.get(metric_key, metric_key.replace("_", " "))
    return f"{direction} tail in {horizon_text}{label}."


def _dedupe_trigger_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen_conditions: set[str] = set()
    for row in rows:
        condition = str((row or {}).get("condition", "") or "").strip()
        if not condition or condition in seen_conditions:
            continue
        seen_conditions.add(condition)
        out.append(dict(row or {}))
    return out


def _tail_descriptions(precedent_pack: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for tail in list((precedent_pack.get("tail_events", []) or [])):
        description = str((tail or {}).get("description", "") or "").strip()
        metric = str((tail or {}).get("metric", "") or "").strip()
        horizon = str((tail or {}).get("horizon", "") or "").strip()
        generic = description in {"Bottom decile historical outcome.", "Top decile historical outcome."}
        if generic and metric:
            direction = "Adverse" if "Bottom" in description else "Favorable"
            out.append(_humanize_tail_metric(metric=metric, horizon=horizon, direction=direction))
            continue
        if description:
            out.append(_humanize_text(description))
        elif metric:
            out.append(_humanize_tail_metric(metric=metric, horizon=horizon, direction="Adverse"))
    return _dedupe(out)


def _fallback_monitoring_triggers(first_action: str, snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    credit_window = _safe_float(_feature_value(snapshot, "market.credit_window_proxy"))
    if _has_capital_return([first_action]):
        out.append(
            {
                "trigger_type": "balance_sheet_condition",
                "condition": "Pause if leverage rises materially from the current baseline.",
                "evaluation_frequency": "monthly",
                "trigger_probability": 0.35,
                "explanation": "Capital return should stop if balance-sheet capacity erodes.",
            }
        )
        out.append(
            {
                "trigger_type": "capital_allocation_condition",
                "condition": "Pause if a higher-return strategic use for capital appears.",
                "evaluation_frequency": "monthly",
                "trigger_probability": 0.25,
                "explanation": "The recommendation assumes no better use of cash emerges.",
            }
        )
    if _is_balance_sheet_action(first_action):
        out.append(
            {
                "trigger_type": "market_window_condition",
                "condition": "Only proceed if financing terms still improve flexibility rather than just add gross debt.",
                "evaluation_frequency": "weekly",
                "trigger_probability": 0.4,
                "explanation": "The financing step is only justified while the market window is constructive.",
            }
        )
        if credit_window is not None and credit_window > 0.0:
            out.append(
                {
                    "trigger_type": "market_window_condition",
                    "condition": (
                        "Reassess if debt-market conditions weaken meaningfully from the current "
                        f"{_market_window_description(credit_window, 'debt')} backdrop."
                    ),
                    "evaluation_frequency": "weekly",
                    "trigger_probability": 0.3,
                    "explanation": "Closing credit markets can invalidate the financing thesis.",
                }
            )
    if _is_mna_action(first_action):
        out.append(
            {
                "trigger_type": "valuation_condition",
                "condition": "Proceed only if the deal still clears the internal return hurdle after financing costs.",
                "evaluation_frequency": "weekly",
                "trigger_probability": 0.3,
                "explanation": "The acquisition case depends on the spread between deal returns and the next-best use of capital.",
            }
        )
    if _is_divestiture_action(first_action):
        out.append(
            {
                "trigger_type": "execution_condition",
                "condition": "Proceed only if the asset can be sold at a price that clearly improves focus or flexibility.",
                "evaluation_frequency": "monthly",
                "trigger_probability": 0.3,
                "explanation": "A divestiture is only worth doing if the sale price and strategic simplification are both credible.",
            }
        )
    return out[:3]


def _has_only_generic_tail_descriptions(items: Sequence[str]) -> bool:
    cleaned = [str(item or "").strip() for item in items if str(item or "").strip()]
    if not cleaned:
        return False
    return all(item.startswith("Adverse tail in ") for item in cleaned)


def _looks_generic_execution_risk(text: str) -> bool:
    lower = str(text or "").strip().lower()
    if not lower:
        return True
    generic_markers = (
        "requires disciplined execution",
        "requires disciplined market timing",
        "plan has explicit monitoring triggers",
        "not reliant on one path only",
    )
    return any(marker in lower for marker in generic_markers)


def _fallback_failure_modes(first_action: str, snapshot: Dict[str, Any]) -> List[str]:
    if _has_capital_return([first_action]):
        return [
            "Capital is returned just before the company needs that cash for resilience, refinancing, or reinvestment.",
            "The market does not reward the payout decision enough to offset the lost balance-sheet flexibility.",
        ]
    if _is_balance_sheet_action(first_action):
        return [
            "Funding arrives on terms that add gross debt but do not improve real financing flexibility.",
            "The transaction buys time without moving enough maturity pressure or leverage risk off the balance sheet.",
        ]
    if _is_mna_action(first_action):
        return [
            "The transaction clears strategically but destroys value through price or integration risk.",
            "Financing the deal leaves the company with less resilience than the growth is worth.",
        ]
    if _is_divestiture_action(first_action):
        return [
            "The asset is sold too cheaply relative to its strategic or cash-flow value.",
            "The sale simplifies the portfolio but fails to create enough flexibility to justify the loss of earnings.",
        ]
    return ["The recommendation fails because the expected strategic benefit does not survive real execution conditions."]


def _fallback_execution_risks(first_action: str, snapshot: Dict[str, Any]) -> List[str]:
    if _has_capital_return([first_action]):
        return [
            "Execution has to preserve enough liquidity that the company does not regret returning cash into a weaker backdrop.",
            "Management has to show that capital return is the best remaining use of cash, not the default after growth options ran thin.",
        ]
    if _is_balance_sheet_action(first_action):
        return [
            "Management has to avoid issuing financing that solves a short-term need but leaves the company with the same structural problem later.",
            "Terms have to improve maturity profile or liquidity headroom, not just increase gross funding.",
        ]
    if _is_mna_action(first_action):
        return ["Execution requires valuation discipline, financing discipline, and integration discipline at the same time."]
    if _is_divestiture_action(first_action):
        return ["Execution requires clean buyer interest and a sale process that does not weaken the remaining business."]
    return ["Execution assumptions have to hold through completion, not just at announcement."]


def _resolve_plan_action_candidate(
    *,
    plan: Dict[str, Any],
    action_id: str,
    candidate_by_action: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    for action in list(plan.get("actions", []) or []):
        row = dict(action or {})
        if str(row.get("action_id", "") or "") == action_id:
            return row
    return dict(candidate_by_action.get(action_id) or {})


def _build_alternative_rebuttal_reasons(
    *,
    top_plan: Dict[str, Any],
    alt_plan: Dict[str, Any],
    top_components: Dict[str, Any],
    alt_components: Dict[str, Any],
    top_action_ids: Sequence[str],
    alt_action_ids: Sequence[str],
    top_candidate: Dict[str, Any],
    alt_candidate: Dict[str, Any],
    snapshot: Dict[str, Any],
    diagnosed: Dict[str, Any],
    precedent_by_action: Dict[str, Dict[str, Any]],
) -> List[str]:
    reasons: List[str] = []
    top_utility = float(top_components.get("expected_utility", 0.0) or 0.0)
    alt_utility = float(alt_components.get("expected_utility", 0.0) or 0.0)
    if top_utility > alt_utility + 0.03:
        reasons.append(f"It carries higher modeled expected utility ({top_utility:.3f} vs {alt_utility:.3f}).")

    top_support = float(top_components.get("support_factor", 0.0) or 0.0)
    alt_support = float(alt_components.get("support_factor", 0.0) or 0.0)
    if top_support > alt_support + 0.03:
        reasons.append(f"It has stronger empirical support ({top_support:.3f} vs {alt_support:.3f}).")

    top_tail = float(top_components.get("tail_risk_penalty", 0.0) or 0.0)
    alt_tail = float(alt_components.get("tail_risk_penalty", 0.0) or 0.0)
    if alt_tail > top_tail + 0.02:
        reasons.append(f"It takes more downside risk (tail penalty {alt_tail:.3f} vs {top_tail:.3f}).")

    top_time = float(top_components.get("time_discount_factor", 1.0) or 1.0)
    alt_time = float(alt_components.get("time_discount_factor", 1.0) or 1.0)
    if alt_time < top_time - 0.03:
        reasons.append(f"It pushes value further out in time (discount factor {alt_time:.3f} vs {top_time:.3f}).")

    top_precedent = _precedent_confidence(precedent_by_action.get(str((top_action_ids[0] if top_action_ids else "") or ""), {}))
    alt_precedent = _precedent_confidence(precedent_by_action.get(str((alt_action_ids[0] if alt_action_ids else "") or ""), {}))
    if top_precedent > alt_precedent + 0.03:
        reasons.append(f"It has a cleaner precedent match ({top_precedent:.3f} vs {alt_precedent:.3f}).")

    reasons.extend(
        _alternative_specific_reasons(
            top_action_ids=top_action_ids,
            alt_action_ids=alt_action_ids,
            top_candidate=top_candidate,
            alt_candidate=alt_candidate,
            snapshot=snapshot,
            diagnosed=diagnosed,
        )
    )
    reasons = _dedupe(reasons)
    if not reasons:
        reasons.append("It addresses the diagnosed problem less directly than the top plan.")
    return reasons[:4]


def _format_alternative_rebuttal(reasons: Sequence[str]) -> str:
    items = [str(reason or "").strip() for reason in reasons if str(reason or "").strip()]
    if not items:
        return "Not preferred because it addresses the diagnosed problem less directly than the top plan."
    if len(items) == 1:
        return f"Not preferred because {items[0][0].lower() + items[0][1:] if len(items[0]) > 1 else items[0].lower()}"
    first = items[0]
    second = items[1]
    connector = " It also " if not second.lower().startswith(("it ", "this ")) else " Also, "
    sentence = f"Not preferred because {first[0].lower() + first[1:] if len(first) > 1 else first.lower()}"
    sentence += connector + (second if connector == " Also, " else second[0].lower() + second[1:] if len(second) > 1 else second.lower())
    return sentence


def _alternative_specific_reasons(
    *,
    top_action_ids: Sequence[str],
    alt_action_ids: Sequence[str],
    top_candidate: Dict[str, Any],
    alt_candidate: Dict[str, Any],
    snapshot: Dict[str, Any],
    diagnosed: Dict[str, Any],
) -> List[str]:
    top_first = str((top_action_ids[0] if top_action_ids else "") or "")
    alt_first = str((alt_action_ids[0] if alt_action_ids else "") or "")
    reasons: List[str] = []
    liquidity = _safe_float(_feature_value(snapshot, "liquidity.available_for_actions"))
    market_cap = _safe_float(_feature_value(snapshot, "market.market_cap"))
    liquidity_to_mcap = (liquidity / market_cap) if liquidity is not None and market_cap not in (None, 0.0) else None
    net_leverage = _safe_float(_feature_value(snapshot, "capital_structure.net_leverage"))
    maturity_wall = _safe_float(_feature_value(snapshot, "capital_structure.maturity_wall_ratio_24m"))
    credit_window = _safe_float(_feature_value(snapshot, "market.credit_window_proxy"))
    pursue_mna_priority = _safe_float(_feature_value(snapshot, "strategic.intent.pursue_mna_priority"))
    if top_first == "capital_structure.new_debt_issuance" and alt_first == "capital_structure.equity_issuance":
        reasons.append("It is more dilutive for a similar financing outcome.")
    if top_first == "capital_structure.new_debt_issuance" and alt_first == "capital_structure.refinancing":
        reasons.append("It creates less fresh capacity to solve the financing problem.")
    if _is_buyback_action(top_first) and alt_first == "capital_return.open_market_buyback":
        reasons.append("It is a less decisive capital-return mechanism.")
    if _is_capital_return_action(top_first) and _is_balance_sheet_action(alt_first) and (liquidity_to_mcap or 0.0) >= 0.03:
        reasons.append(f"It solves the capital-return question less directly even though deployable liquidity is already {_fmt_pct(liquidity_to_mcap)} of market value.")
    if _is_balance_sheet_action(top_first) and _has_capital_return(top_action_ids[1:]) and not _has_capital_return(alt_action_ids):
        reasons.append("It does not unlock the planned return-of-capital step.")
    if _is_divestiture_action(top_first) and not _is_divestiture_action(alt_first):
        reasons.append("It raises or redeploys capital without simplifying the portfolio.")
    if _is_mna_action(top_first) and not _is_mna_action(alt_first) and pursue_mna_priority is not None:
        reasons.append(f"It does not address the external-growth problem despite an M&A-priority score of {_fmt_score(pursue_mna_priority)}.")
    if "balance-sheet capacity" in str(diagnosed.get("primary_problem", "")).lower() and _is_balance_sheet_action(top_first) and not _is_balance_sheet_action(alt_first):
        if maturity_wall is not None and maturity_wall >= 0.15:
            reasons.append(f"It leaves a {_fmt_pct(maturity_wall)} 24-month maturity wall unresolved before using capacity elsewhere.")
        else:
            reasons.append("It addresses the balance-sheet-capacity problem less directly.")
    if _is_balance_sheet_action(top_first) and _has_capital_return(alt_action_ids) and maturity_wall is not None and maturity_wall >= 0.15:
        reasons.append(f"It spends capital before a {_fmt_pct(maturity_wall)} maturity wall is repaired.")
    if _is_buyback_action(top_first) and alt_first in {"capital_return.dividend_increase", "capital_return.special_dividend", "capital_return.dividend_initiate"}:
        reasons.append("It locks the company into a stickier payout instead of a more reversible repurchase program.")
    if _is_buyback_action(top_first) and _uses_equity_markets([alt_first]) and liquidity_to_mcap is not None:
        reasons.append(f"It raises dilutive capital even though deployable liquidity already equals {_fmt_pct(liquidity_to_mcap)} of market value.")
    if _is_mna_action(top_first) and _has_capital_return([alt_first]) and pursue_mna_priority is not None:
        reasons.append(f"It returns capital instead of pursuing the external-growth agenda implied by the {_fmt_score(pursue_mna_priority)} M&A-priority signal.")
    if _is_balance_sheet_action(top_first) and credit_window is not None and _uses_equity_markets([alt_first]):
        reasons.append(
            f"It prefers equity even though debt markets are currently "
            f"{_market_window_description(credit_window, 'debt')}."
        )
    if _is_capital_return_action(top_first) and _is_capital_return_action(alt_first):
        top_value = _objective_median(top_candidate, "value_creation")
        alt_value = _objective_median(alt_candidate, "value_creation")
        if top_value is not None and alt_value is not None and top_value > alt_value + 0.03:
            reasons.append(f"It is the weaker capital-return tool on value creation ({alt_value:+.3f} vs {top_value:+.3f}).")
    if _uses_equity_markets([alt_first]) and net_leverage is not None and net_leverage < 2.5 and not _uses_equity_markets([top_first]):
        reasons.append(f"It adds dilution even though net leverage is only {_fmt_x(net_leverage)}.")
    return reasons


def _has_capital_return(action_ids: Sequence[str]) -> bool:
    return any(_is_capital_return_action(action_id) for action_id in action_ids)


def _is_capital_return_action(action_id: str) -> bool:
    return str(action_id or "").startswith("capital_return.")


def _is_buyback_action(action_id: str) -> bool:
    aid = str(action_id or "")
    return aid in {
        "capital_return.open_market_buyback",
        "capital_return.accelerated_share_repurchase",
        "capital_return.tender_offer_buyback",
    }


def _is_balance_sheet_action(action_id: str) -> bool:
    aid = str(action_id or "")
    return aid.startswith("capital_structure.")


def _is_mna_action(action_id: str) -> bool:
    return str(action_id or "").startswith("mna.")


def _is_divestiture_action(action_id: str) -> bool:
    return str(action_id or "").startswith("portfolio.")


def _uses_credit_markets(action_ids: Sequence[str]) -> bool:
    return any(
        action_id in {
            "capital_structure.refinancing",
            "capital_structure.new_debt_issuance",
            "capital_structure.revolver_draw_or_resize",
            "capital_structure.tender_offer_debt",
            "capital_structure.exchange_offer",
            "capital_structure.liability_management_exercise",
        }
        for action_id in action_ids
    )


def _uses_equity_markets(action_ids: Sequence[str]) -> bool:
    return any(
        action_id in {
            "capital_structure.equity_issuance",
            "capital_structure.convertible_issuance",
            "capital_structure.preferred_issuance",
        }
        for action_id in action_ids
    )


def _is_capacity_then_return_sequence(action_ids: Sequence[str]) -> bool:
    if len(action_ids) < 2:
        return False
    return _is_balance_sheet_action(action_ids[0]) and _has_capital_return(action_ids[1:])


def _feature_value(snapshot: Dict[str, Any], key: str) -> Any:
    features = feature_view_from_snapshot(snapshot, view_name="dossier")
    return resolve_feature_value(features, key)


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _fmt_currency(value: Any) -> str:
    num = _safe_float(value)
    if num is None:
        return "n/a"
    abs_num = abs(num)
    if abs_num >= 1.0e9:
        return f"${num / 1.0e9:.1f}B"
    if abs_num >= 1.0e6:
        return f"${num / 1.0e6:.1f}M"
    if abs_num >= 1.0e3:
        return f"${num / 1.0e3:.1f}K"
    return f"${num:.0f}"


def _fmt_pct(value: Any) -> str:
    num = _safe_float(value)
    if num is None:
        return "n/a"
    return f"{num * 100.0:.1f}%"


def _fmt_x(value: Any) -> str:
    num = _safe_float(value)
    if num is None:
        return "n/a"
    return f"{num:.2f}x"


def _fmt_ratio(value: Any) -> str:
    num = _safe_float(value)
    if num is None:
        return "n/a"
    return f"{num:.2f}x"


def _fmt_score(value: Any) -> str:
    num = _safe_float(value)
    if num is None:
        return "n/a"
    return f"{num:.2f}/1.00"


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _pressure_label(value: float) -> str:
    if value >= 0.28:
        return "high"
    if value >= 0.16:
        return "elevated"
    if value >= 0.08:
        return "moderate"
    return "low"


def _dedupe(values: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"
