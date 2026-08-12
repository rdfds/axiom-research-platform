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


