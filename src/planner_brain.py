from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .candidate_generation import PlaybookRegistry
from .planner_types import (
    ActionDependencyGraph,
    DependencyEdge,
    Plan,
    PlanBranch,
    PlanExplanation,
    PlanRisk,
    PlanScoreBreakdown,
    PlanStep,
    PlanTimeline,
    PlanTrigger,
)
from .recommendation_run import RecommendationRun, validate_plan_hard_constraints


_OBJECTIVE_FIELDS = (
    "value_creation",
    "risk_reduction",
    "growth",
    "rating_preservation",
    "optionality",
)

_TRIGGER_KEYWORDS = {
    "credit": "credit_condition",
    "spread": "credit_condition",
    "valuation": "valuation_condition",
    "discount": "valuation_condition",
    "earnings": "earnings_outcome",
    "peer": "peer_activity",
    "vol": "market_regime",
    "risk_off": "market_regime",
    "risk_on": "market_regime",
    "liquidity": "liquidity_condition",
}


@dataclass(frozen=True)
class PlannerNode:
    action_id: str
    candidate: Dict[str, Any]
    precedent_pack: Dict[str, Any]
    schema: Dict[str, Any]
    lead_time: Dict[str, Any]
    base_rank_score: float


def build_plan_set(
    run: RecommendationRun,
    precedent_matches: List[Dict[str, Any]],
    registry: Any,
    top_plans: int,
    feasible_candidates: Optional[List[Dict[str, Any]]] = None,
    beam_width: int = 10,
    max_depth: int = 4,
) -> Dict[str, Any]:
    seed = int(run.planner_random_seed if run.planner_random_seed is not None else 0)
    nodes = _select_nodes(
        feasible_candidates=feasible_candidates or [],
        precedent_matches=precedent_matches,
        registry=registry,
        run=run,
    )
    node_by_action = {node.action_id: node for node in nodes}
    dep_graph = _build_dependency_graph(node_by_action=node_by_action, registry=registry)

    sequences = _search_sequences(
        run=run,
        node_by_action=node_by_action,
        dep_graph=dep_graph,
        beam_width=max(1, int(beam_width)),
        max_depth=max(1, int(max_depth)),
    )

    plans: List[Dict[str, Any]] = []
    for sequence in sequences:
        plan = _assemble_plan(
            run=run,
            sequence=sequence,
            node_by_action=node_by_action,
            dep_graph=dep_graph,
        )
        if plan is None:
            continue
        plans.append(plan)

    plans.sort(
        key=lambda item: (
            -float((item.get("score_components") or {}).get("raw_total_score", item.get("score", 0.0)) or 0.0),
            -float(item.get("score", 0.0) or 0.0),
            str(item.get("plan_id", "")),
        )
    )
    plans = plans[: max(1, int(top_plans))]

    return {
        "run_id": run.run_id,
        "generated_at": _now_iso(),
        "planner_random_seed": seed,
        "search_metadata": {
            "beam_width": max(1, int(beam_width)),
            "max_depth": max(1, int(max_depth)),
            "candidate_action_count": len(node_by_action),
            "sequence_count_considered": len(sequences),
            "feasible_candidate_count": len(feasible_candidates or []),
            "precedent_candidate_count": len(precedent_matches),
        },
        "dependency_graph": dep_graph.to_dict(),
        "plans": plans,
    }


def _select_nodes(
    feasible_candidates: List[Dict[str, Any]],
    precedent_matches: List[Dict[str, Any]],
    registry: Any,
    run: RecommendationRun,
) -> List[PlannerNode]:
    precedent_by_candidate_id: Dict[str, Dict[str, Any]] = {}
    precedent_by_action_id: Dict[str, Dict[str, Any]] = {}
    for row in precedent_matches:
        candidate = _normalize_candidate(dict(row.get("candidate", {}) or {}))
        action_id = str(candidate.get("action_id", "") or "")
        candidate_id = str(candidate.get("candidate_id", "") or "")
        pack = dict(row.get("precedent_pack", {}) or {})
        if candidate_id:
            current = precedent_by_candidate_id.get(candidate_id)
            if current is None or _precedent_confidence(pack) > _precedent_confidence(current):
                precedent_by_candidate_id[candidate_id] = pack
        if action_id:
            current = precedent_by_action_id.get(action_id)
            if current is None or _precedent_confidence(pack) > _precedent_confidence(current):
                precedent_by_action_id[action_id] = pack

    best_by_action: Dict[str, PlannerNode] = {}
    source_candidates: List[Dict[str, Any]] = []
    for cand in feasible_candidates:
        source_candidates.append(_normalize_candidate(dict(cand or {})))
    if not source_candidates:
        for row in precedent_matches:
            source_candidates.append(_normalize_candidate(dict(row.get("candidate", {}) or {})))

    for candidate in source_candidates:
        action_id = str(candidate.get("action_id", "") or "")
        if not action_id:
            continue
        schema = registry.get_action(action_id) or {}
        candidate_id = str(candidate.get("candidate_id", "") or "")
        precedent_pack = dict(precedent_by_candidate_id.get(candidate_id) or precedent_by_action_id.get(action_id) or {})
        node = PlannerNode(
            action_id=action_id,
            candidate=candidate,
            precedent_pack=precedent_pack,
            schema=schema,
            lead_time=registry.fetch_planner_lead_time_distribution(action_id),
            base_rank_score=_base_rank_score(candidate=candidate, precedent_pack=precedent_pack, run=run),
        )
        current = best_by_action.get(action_id)
        if current is None or node.base_rank_score > current.base_rank_score:
            best_by_action[action_id] = node
    return sorted(best_by_action.values(), key=lambda node: (-node.base_rank_score, node.action_id))


def _normalize_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(candidate)
    if "parameters" not in out and "params" in out:
        out["parameters"] = dict(out.get("params", {}) or {})
    if "params" not in out and "parameters" in out:
        out["params"] = dict(out.get("parameters", {}) or {})
    return out


def _base_rank_score(candidate: Dict[str, Any], precedent_pack: Dict[str, Any], run: RecommendationRun) -> float:
    impact_distribution = dict(candidate.get("impact_distribution", {}) or {})
    weighted_components = _weighted_objective_components(impact_distribution, run)
    utility = _aggregate_weighted_objectives(weighted_components)
    pass_probability = float(((candidate.get("feasibility", {}) or {}).get("pass_probability", 0.0) or 0.0))
    feasibility_factor = _feasibility_factor(pass_probability)
    precedent_confidence = _precedent_confidence(precedent_pack)
    confidence = float(candidate.get("evaluation_confidence", 0.0) or 0.0)
    utility_score = _bounded_signal(utility)
    strategic_penalty = _action_specific_penalty(candidate=candidate, precedent_pack=precedent_pack)
    negative_utility_penalty = _negative_utility_penalty(
        weighted_utility=utility,
        weighted_components=weighted_components,
        action_ids=[str(candidate.get("action_id", "") or "")],
    )
    status_quo_hurdle = _status_quo_hurdle(
        weighted_utility=utility,
        weighted_components=weighted_components,
        action_ids=[str(candidate.get("action_id", "") or "")],
        candidates=[candidate],
    )
    structural_bonus = _structural_action_bonus(candidate=candidate, precedent_pack=precedent_pack)
    return round(
        utility_score
        * feasibility_factor
        * max(0.25, 0.55 + (0.45 * confidence))
        * max(0.25, 0.55 + (0.45 * precedent_confidence)),
        6,
    ) - strategic_penalty - negative_utility_penalty - status_quo_hurdle + structural_bonus


def _build_dependency_graph(node_by_action: Dict[str, PlannerNode], registry: Any) -> ActionDependencyGraph:
    available_actions = set(node_by_action)
    edges: List[DependencyEdge] = []
    seen: set[Tuple[str, str, str]] = set()
    for action_id in sorted(available_actions):
        for raw in registry.fetch_planner_dependency_edges(action_id):
            src = str(raw.get("source_action") or "")
            dst = str(raw.get("target_action") or "")
            rel = str(raw.get("relationship_type") or "")
            key = (src, dst, rel)
            if not src or not dst or dst not in available_actions or key in seen:
                continue
            seen.add(key)
            edges.append(
                DependencyEdge(
                    source_action=src,
                    target_action=dst,
                    relationship_type=rel,
                    condition=raw.get("condition"),
                    strength=raw.get("strength"),
                    explanation=raw.get("explanation"),
                    original_rule_type=raw.get("original_rule_type"),
                )
            )
    return ActionDependencyGraph(nodes=sorted(available_actions), edges=sorted(edges, key=lambda e: (e.source_action, e.target_action, e.relationship_type)))


def _search_sequences(
    run: RecommendationRun,
    node_by_action: Dict[str, PlannerNode],
    dep_graph: ActionDependencyGraph,
    beam_width: int,
    max_depth: int,
) -> List[Tuple[str, ...]]:
    seeds = _initial_sequences(node_by_action=node_by_action, dep_graph=dep_graph, max_depth=max_depth)
    ranked = _rank_sequences(run=run, sequences=seeds, node_by_action=node_by_action, dep_graph=dep_graph)
    beam = [seq for seq, _ in ranked[:beam_width]]
    all_sequences = set(beam)

    for _ in range(1, max_depth):
        expanded: set[Tuple[str, ...]] = set()
        for seq in beam:
            for action_id in sorted(node_by_action):
                if action_id in seq:
                    continue
                if not _can_append(seq=seq, action_id=action_id, dep_graph=dep_graph):
                    continue
                expanded.add(seq + (action_id,))
        if not expanded:
            break
        ranked = _rank_sequences(run=run, sequences=expanded, node_by_action=node_by_action, dep_graph=dep_graph)
        beam = [seq for seq, _ in ranked[:beam_width]]
        all_sequences.update(beam)

    return [seq for seq, _ in _rank_sequences(run=run, sequences=all_sequences, node_by_action=node_by_action, dep_graph=dep_graph)]


def _initial_sequences(node_by_action: Dict[str, PlannerNode], dep_graph: ActionDependencyGraph, max_depth: int) -> set[Tuple[str, ...]]:
    sequences: set[Tuple[str, ...]] = {(action_id,) for action_id in node_by_action}
    templates = PlaybookRegistry.default().templates
    available = set(node_by_action)

    for template in templates:
        actions = [action_id for action_id in template.action_sequence_template if action_id in available]
        for start in range(len(actions)):
            window = actions[start : start + max_depth]
            for length in range(2, len(window) + 1):
                seq = tuple(window[:length])
                if _sequence_is_valid(seq, dep_graph):
                    sequences.add(seq)

    for edge in dep_graph.edges:
        if edge.relationship_type == "unlocks":
            seq = (edge.source_action, edge.target_action)
        elif edge.relationship_type in {"requires", "recommended_after"}:
            seq = (edge.target_action, edge.source_action)
        else:
            continue
        if len(seq) <= max_depth and _sequence_is_valid(seq, dep_graph):
            sequences.add(seq)
    return sequences


def _rank_sequences(
    run: RecommendationRun,
    sequences: Iterable[Tuple[str, ...]],
    node_by_action: Dict[str, PlannerNode],
    dep_graph: ActionDependencyGraph,
) -> List[Tuple[Tuple[str, ...], float]]:
    ranked: List[Tuple[Tuple[str, ...], float]] = []
    for seq in sequences:
        score = _score_sequence_prefix(run=run, sequence=seq, node_by_action=node_by_action, dep_graph=dep_graph)
        ranked.append((tuple(seq), score))
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return ranked


def _score_sequence_prefix(
    run: RecommendationRun,
    sequence: Sequence[str],
    node_by_action: Dict[str, PlannerNode],
    dep_graph: ActionDependencyGraph,
) -> float:
    utilities = []
    pass_probs = []
    confidences = []
    for action_id in sequence:
        node = node_by_action[action_id]
        utilities.append(_weighted_objective_sum(node.candidate.get("impact_distribution", {}), run))
        pass_probs.append(float(((node.candidate.get("feasibility", {}) or {}).get("pass_probability", 0.0) or 0.0)))
        confidences.append(_precedent_confidence(node.precedent_pack))

    expected_utility = _bounded_signal(sum(utilities))
    feasibility_chain = _feasibility_factor(min(pass_probs) if pass_probs else 0.0)
    confidence_factor = sum(confidences) / len(confidences) if confidences else 0.0
    ordering_bonus = _transition_bonus(sequence=sequence, node_by_action=node_by_action, dep_graph=dep_graph)
    order_penalty = _ordering_penalty(sequence=sequence, dep_graph=dep_graph)
    structural_bonus = sum(
        _structural_action_bonus(
            candidate=node_by_action[action_id].candidate,
            precedent_pack=node_by_action[action_id].precedent_pack,
        )
        for action_id in sequence
    )
    return round(
        (expected_utility * max(0.2, feasibility_chain) * max(0.2, 0.6 + (0.4 * confidence_factor)))
        + ordering_bonus
        + structural_bonus
        - order_penalty,
        6,
    )


def _assemble_plan(
    run: RecommendationRun,
    sequence: Sequence[str],
    node_by_action: Dict[str, PlannerNode],
    dep_graph: ActionDependencyGraph,
) -> Optional[Dict[str, Any]]:
    nodes = [node_by_action[action_id] for action_id in sequence]
    plan_actions = [dict(node.candidate) for node in nodes]
    hard_violations = validate_plan_hard_constraints(plan_actions=plan_actions, constraints=run.constraints, projected_state={})
    if hard_violations:
        return None

    steps = _build_steps(run=run, sequence=sequence, node_by_action=node_by_action, dep_graph=dep_graph)
    timeline = _build_timeline(run=run, steps=steps)
    branches, triggers = _build_branches_and_triggers(sequence=sequence, node_by_action=node_by_action, dep_graph=dep_graph)
    score_breakdown = _score_plan(run=run, sequence=sequence, node_by_action=node_by_action, dep_graph=dep_graph, timeline=timeline)
    risks = _build_plan_risk(sequence=sequence, node_by_action=node_by_action, dep_graph=dep_graph)

    plan = Plan(
        plan_id="plan_" + "__".join(sequence),
        run_id=run.run_id,
        steps=steps,
        timeline=timeline,
        triggers=triggers,
        branches=branches,
        score_breakdown=score_breakdown,
        risks=risks,
        summary_explanation=_plan_summary(sequence=sequence, node_by_action=node_by_action),
    )
    payload = plan.to_dict()
    payload["score"] = score_breakdown.total_score
    payload["score_components"] = dict(score_breakdown.components)
    payload["actions"] = [dict(node.candidate) for node in nodes]
    payload["hard_constraint_violations"] = hard_violations
    return payload


def _build_steps(
    run: RecommendationRun,
    sequence: Sequence[str],
    node_by_action: Dict[str, PlannerNode],
    dep_graph: ActionDependencyGraph,
) -> List[PlanStep]:
    steps: List[PlanStep] = []
    current_dt = _parse_run_time(run.as_of_time)
    for idx, action_id in enumerate(sequence, start=1):
        node = node_by_action[action_id]
        lead = dict(node.lead_time)
        prerequisites = _prerequisites_for_step(action_id=action_id, sequence=sequence, dep_graph=dep_graph)
        step = PlanStep(
            step_id=f"step_{idx:02d}",
            action_id=action_id,
            parameters=dict(node.candidate.get("parameters", {}) or {}),
            earliest_start=current_dt.isoformat(),
            expected_duration=lead,
            prerequisites=prerequisites,
            probability_of_success=float(((node.candidate.get("feasibility", {}) or {}).get("pass_probability", 0.0) or 0.0)),
            impact_contribution=_impact_contribution(node.candidate),
            explanation=_build_step_explanation(
                action_id=action_id,
                node=node,
                sequence=sequence,
                dep_graph=dep_graph,
            ),
        )
        steps.append(step)
        current_dt = current_dt + timedelta(days=int(lead.get("median_days", 0) or 0))
    return steps


def _build_timeline(run: RecommendationRun, steps: Sequence[PlanStep]) -> PlanTimeline:
    schedule: List[Dict[str, Any]] = []
    current_dt = _parse_run_time(run.as_of_time)
    for step in steps:
        lead = dict(step.expected_duration or {})
        start_dt = current_dt
        end_dt = start_dt + timedelta(days=int(lead.get("median_days", 0) or 0))
        schedule.append(
            {
                "step_id": step.step_id,
                "action_id": step.action_id,
                "start_time": start_dt.isoformat(),
                "expected_completion_time": end_dt.isoformat(),
                "minimum_days": int(lead.get("minimum_days", 0) or 0),
                "median_days": int(lead.get("median_days", 0) or 0),
                "p90_days": int(lead.get("p90_days", 0) or 0),
            }
        )
        current_dt = end_dt
    return PlanTimeline(start_time=_parse_run_time(run.as_of_time).isoformat(), step_schedule=schedule)


def _build_branches_and_triggers(
    sequence: Sequence[str],
    node_by_action: Dict[str, PlannerNode],
    dep_graph: ActionDependencyGraph,
) -> Tuple[List[PlanBranch], List[PlanTrigger]]:
    in_plan = set(sequence)
    candidates: List[Tuple[str, float, str, str]] = []
    for edge in dep_graph.edges:
        if edge.source_action not in in_plan or edge.target_action in in_plan:
            continue
        if edge.relationship_type not in {"unlocks", "recommended_after"}:
            continue
        probability = _branch_probability(source=node_by_action[edge.source_action], target_action=edge.target_action)
        condition = str(edge.condition or f"{edge.target_action} becomes attractive after {edge.source_action}")
        explanation = str(edge.explanation or f"{edge.source_action} creates an opening for {edge.target_action}.")
        candidates.append((edge.target_action, probability, condition, explanation))

    for source_action in sequence:
        source_node = node_by_action[source_action]
        for outcome in list(source_node.precedent_pack.get("second_order_effects", []) or []):
            target_action = str(outcome.get("follow_on_action_id", "") or "")
            if not target_action or target_action in in_plan or target_action not in node_by_action:
                continue
            frequency = float(outcome.get("frequency", 0.0) or 0.0)
            if frequency <= 0.1:
                continue
            condition = f"follow-on capacity remains available after {source_action}"
            explanation = f"Historical follow-on frequency supports {target_action} after {source_action}."
            candidates.append((target_action, _clip(0.2 + frequency, 0.2, 0.7), condition, explanation))

    branch_rows: List[Tuple[str, float, str, str]] = []
    seen_actions: set[str] = set()
    for target_action, probability, condition, explanation in sorted(candidates, key=lambda row: (-row[1], row[0])):
        if target_action in seen_actions:
            continue
        seen_actions.add(target_action)
        branch_rows.append((target_action, probability, condition, explanation))
        if len(branch_rows) >= 2:
            break

    branches: List[PlanBranch] = []
    triggers: List[PlanTrigger] = []
    for target_action, probability, condition, explanation in branch_rows:
        branches.append(
            PlanBranch(
                branch_condition=condition,
                branch_plan_steps=[target_action],
                branch_probability=probability,
                explanation=explanation,
            )
        )
        triggers.append(
            PlanTrigger(
                trigger_type=_trigger_type(condition),
                condition=condition,
                evaluation_frequency=_trigger_frequency(condition),
                trigger_probability=probability,
                explanation=explanation,
            )
        )
    return branches, triggers


def _score_plan(
    run: RecommendationRun,
    sequence: Sequence[str],
    node_by_action: Dict[str, PlannerNode],
    dep_graph: ActionDependencyGraph,
    timeline: PlanTimeline,
) -> PlanScoreBreakdown:
    objective_components = _objective_components(sequence=sequence, node_by_action=node_by_action, run=run)
    weighted_net_utility = sum(
        _weighted_objective_sum(node_by_action[action_id].candidate.get("impact_distribution", {}), run)
        for action_id in sequence
    )
    expected_utility = _bounded_signal(weighted_net_utility)
    negative_utility_penalty = _negative_utility_penalty(
        weighted_utility=weighted_net_utility,
        weighted_components=objective_components,
        action_ids=sequence,
    )
    feasibility_chain = _feasibility_factor(
        min(float(((node_by_action[action_id].candidate.get("feasibility", {}) or {}).get("pass_probability", 0.0) or 0.0)) for action_id in sequence)
    )
    robustness_score = _robustness_score(sequence=sequence, node_by_action=node_by_action)
    tail_risk_penalty = _tail_risk_penalty(sequence=sequence, node_by_action=node_by_action)
    complexity_penalty = _complexity_penalty(sequence=sequence, node_by_action=node_by_action)
    total_days = 0
    if timeline.step_schedule:
        first = timeline.step_schedule[0]
        last = timeline.step_schedule[-1]
        start_dt = _parse_run_time(first["start_time"])
        end_dt = _parse_run_time(last["expected_completion_time"])
        total_days = max(0, (end_dt - start_dt).days)
    time_discount_factor = round(math.exp(-0.001 * float(total_days)), 6)
    transition_bonus = _transition_bonus(sequence=sequence, node_by_action=node_by_action, dep_graph=dep_graph)
    support_factor = _support_factor(sequence=sequence, node_by_action=node_by_action)
    action_penalty = sum(
        _action_specific_penalty(
            candidate=node_by_action[action_id].candidate,
            precedent_pack=node_by_action[action_id].precedent_pack,
        )
        for action_id in sequence
    )
    structural_bonus = sum(
        _structural_action_bonus(
            candidate=node_by_action[action_id].candidate,
            precedent_pack=node_by_action[action_id].precedent_pack,
        )
        for action_id in sequence
    )
    status_quo_hurdle = _status_quo_hurdle(
        weighted_utility=weighted_net_utility,
        weighted_components=objective_components,
        action_ids=sequence,
        support_factor=support_factor,
        transition_bonus=transition_bonus,
        structural_bonus=structural_bonus,
        candidates=[node_by_action[action_id].candidate for action_id in sequence],
    )
    raw_total_score = (
        (expected_utility * feasibility_chain * robustness_score * time_discount_factor * support_factor)
        + transition_bonus
        + structural_bonus
        - tail_risk_penalty
    )
    raw_total_score = raw_total_score - action_penalty - negative_utility_penalty - status_quo_hurdle
    total_score = _clip(raw_total_score, 0.0, 1.0)
    components = {
        **objective_components,
        "weighted_net_utility": round(weighted_net_utility, 6),
        "expected_utility": round(expected_utility, 6),
        "feasibility_chain": round(feasibility_chain, 6),
        "robustness_score": round(robustness_score, 6),
        "time_discount_factor": round(time_discount_factor, 6),
        "support_factor": round(support_factor, 6),
        "transition_bonus": round(transition_bonus, 6),
        "structural_bonus": round(structural_bonus, 6),
        "tail_risk_penalty": round(tail_risk_penalty, 6),
        "complexity_penalty": round(complexity_penalty, 6),
        "action_specific_penalty": round(action_penalty, 6),
        "negative_utility_penalty": round(negative_utility_penalty, 6),
        "status_quo_hurdle": round(status_quo_hurdle, 6),
        "raw_total_score": round(raw_total_score, 6),
    }
    return PlanScoreBreakdown(
        expected_utility=round(expected_utility, 6),
        feasibility_chain=round(feasibility_chain, 6),
        robustness_score=round(robustness_score, 6),
        tail_risk_penalty=round(tail_risk_penalty, 6),
        complexity_penalty=round(complexity_penalty, 6),
        time_discount_factor=round(time_discount_factor, 6),
        total_score=round(total_score, 6),
        components=components,
    )


def _objective_components(sequence: Sequence[str], node_by_action: Dict[str, PlannerNode], run: RecommendationRun) -> Dict[str, float]:
    totals = {name: 0.0 for name in _OBJECTIVE_FIELDS}
    for action_id in sequence:
        objectives = dict(((node_by_action[action_id].candidate.get("impact_distribution", {}) or {}).get("objectives", {}) or {}))
        totals["value_creation"] += run.objectives.value_creation_weight * float((objectives.get("value_creation", {}) or {}).get("median", 0.0) or 0.0)
        totals["risk_reduction"] += run.objectives.risk_reduction_weight * float((objectives.get("risk_reduction", {}) or {}).get("median", 0.0) or 0.0)
        totals["growth"] += run.objectives.growth_weight * float((objectives.get("growth", {}) or {}).get("median", 0.0) or 0.0)
        totals["rating_preservation"] += run.objectives.rating_preservation_weight * float((objectives.get("rating_preservation", {}) or {}).get("median", 0.0) or 0.0)
        totals["optionality"] += run.objectives.optionality_weight * float((objectives.get("optionality", {}) or {}).get("median", 0.0) or 0.0)
    return {key: round(value, 6) for key, value in totals.items()}


def _build_plan_risk(sequence: Sequence[str], node_by_action: Dict[str, PlannerNode], dep_graph: ActionDependencyGraph) -> PlanRisk:
    failure_modes: List[str] = []
    regime_risks: List[str] = []
    execution_risks: List[str] = []

    for action_id in sequence:
        node = node_by_action[action_id]
        for risk in list(node.candidate.get("risks", []) or []):
            explanation = str(risk.get("explanation", "") or "").strip()
            if explanation:
                failure_modes.append(explanation)
        for tail in list(node.precedent_pack.get("tail_events", []) or []):
            description = str(tail.get("description") or tail.get("explanation") or "").strip()
            if description and _is_adverse_tail(tail):
                failure_modes.append(description)
        for regime in list((node.candidate.get("impact_distribution", {}) or {}).get("regime_sensitivity", []) or []):
            effect_shift = float(regime.get("effect_shift", 0.0) or 0.0)
            if effect_shift < 0:
                regime_risks.append(f"{action_id} weakens under {regime.get('regime_condition')}.")
        complexity = float((node.schema.get("execution_complexity_prior", {}) or {}).get("base_complexity_score", 3) or 3)
        if complexity >= 4:
            execution_risks.append(f"{action_id} has elevated execution complexity.")

    for edge in dep_graph.edges:
        if edge.source_action in sequence and edge.target_action in sequence and edge.relationship_type == "conflicts":
            execution_risks.append(f"{edge.source_action} conflicts with {edge.target_action}.")

    return PlanRisk(
        main_failure_modes=_dedupe_keep_order(failure_modes)[:5],
        regime_sensitivity=_dedupe_keep_order(regime_risks)[:4],
        execution_risks=_dedupe_keep_order(execution_risks)[:4],
    )


def _plan_summary(sequence: Sequence[str], node_by_action: Dict[str, PlannerNode]) -> str:
    if not sequence:
        return ""
    parts = []
    for action_id in sequence:
        narrative = str((((node_by_action[action_id].candidate.get("mechanism_activation", {}) or {}).get("narrative_explanation", "")) or "")).strip()
        if narrative:
            parts.append(narrative)
    if not parts:
        return " -> ".join(sequence)
    return " ".join(_dedupe_keep_order(parts)[:2])


def _build_step_explanation(action_id: str, node: PlannerNode, sequence: Sequence[str], dep_graph: ActionDependencyGraph) -> PlanExplanation:
    narrative = str((((node.candidate.get("mechanism_activation", {}) or {}).get("narrative_explanation", "")) or "")).strip()
    driver_facts = []
    for driver in list(((node.candidate.get("impact_distribution", {}) or {}).get("key_drivers", []) or [])):
        explanation = str(driver.get("explanation", "") or "").strip()
        if explanation:
            driver_facts.append(explanation)
    tradeoffs = [str(risk.get("explanation", "") or "").strip() for risk in list(node.candidate.get("risks", []) or []) if str(risk.get("explanation", "") or "").strip()]
    alternatives = []
    for edge in dep_graph.edges:
        if edge.source_action == action_id and edge.relationship_type == "conflicts":
            alternatives.append(f"Avoid combining with {edge.target_action}.")
    prerequisites = _prerequisites_for_step(action_id=action_id, sequence=sequence, dep_graph=dep_graph)
    why_now = "Dependencies are satisfied." if prerequisites else "This step can start immediately under the current plan."
    if prerequisites:
        why_now = f"Sequence after {', '.join(prerequisites)} so the prerequisite actions land first."
    problem_statement = narrative or f"{action_id} addresses a current strategic constraint."
    why_this_action = driver_facts[0] if driver_facts else f"{action_id} offers a favorable trade-off versus nearby alternatives."
    return PlanExplanation(
        problem_statement=problem_statement,
        why_this_action=why_this_action,
        why_now=why_now,
        key_supporting_facts=_dedupe_keep_order(driver_facts)[:3],
        main_tradeoffs=_dedupe_keep_order(tradeoffs)[:3],
        why_not_alternatives=_dedupe_keep_order(alternatives)[:3],
    )


def _impact_contribution(candidate: Dict[str, Any]) -> Dict[str, Any]:
    impact = dict(candidate.get("impact_distribution", {}) or {})
    objectives = {}
    for objective, payload in dict(impact.get("objectives", {}) or {}).items():
        objectives[objective] = float(payload.get("median", 0.0) or 0.0)
    return {
        "objectives": objectives,
        "uncertainty_score": float(impact.get("uncertainty_score", 0.0) or 0.0),
        "evaluation_confidence": float(candidate.get("evaluation_confidence", 0.0) or 0.0),
    }


def _prerequisites_for_step(action_id: str, sequence: Sequence[str], dep_graph: ActionDependencyGraph) -> List[str]:
    seen: List[str] = []
    current_index = sequence.index(action_id)
    prior_actions = set(sequence[:current_index])
    for edge in dep_graph.edges:
        if edge.source_action != action_id:
            continue
        if edge.relationship_type not in {"requires", "recommended_after"}:
            continue
        if edge.target_action in prior_actions:
            seen.append(edge.target_action)
    return _dedupe_keep_order(seen)


def _can_append(seq: Sequence[str], action_id: str, dep_graph: ActionDependencyGraph) -> bool:
    for edge in dep_graph.edges:
        if edge.relationship_type == "conflicts":
            if (edge.source_action == action_id and edge.target_action in seq) or (
                edge.target_action == action_id and edge.source_action in seq
            ):
                return False
        if edge.source_action == action_id and edge.relationship_type == "requires" and edge.target_action not in seq:
            return False
    return _sequence_is_valid(tuple(seq) + (action_id,), dep_graph)


def _sequence_is_valid(seq: Sequence[str], dep_graph: ActionDependencyGraph) -> bool:
    if len(set(seq)) != len(seq):
        return False
    position = {action_id: idx for idx, action_id in enumerate(seq)}
    for edge in dep_graph.edges:
        src = edge.source_action
        dst = edge.target_action
        if src not in position or dst not in position:
            continue
        if edge.relationship_type == "conflicts":
            return False
        if edge.relationship_type in {"requires", "recommended_after"} and position[dst] > position[src]:
            return False
    return True


def _ordering_penalty(sequence: Sequence[str], dep_graph: ActionDependencyGraph) -> float:
    position = {action_id: idx for idx, action_id in enumerate(sequence)}
    penalty = 0.0
    for edge in dep_graph.edges:
        if edge.relationship_type != "recommended_after":
            continue
        src = edge.source_action
        dst = edge.target_action
        if src in position and dst not in position:
            penalty += 0.015
        elif src in position and dst in position and position[dst] > position[src]:
            penalty += 0.03
    return round(min(0.2, penalty), 6)


def _transition_bonus(sequence: Sequence[str], node_by_action: Dict[str, PlannerNode], dep_graph: ActionDependencyGraph) -> float:
    bonus = 0.0
    position = {action_id: idx for idx, action_id in enumerate(sequence)}
    for edge in dep_graph.edges:
        src = edge.source_action
        dst = edge.target_action
        if src not in position or dst not in position:
            continue
        if position[dst] <= position[src]:
            continue
        if edge.relationship_type == "unlocks":
            bonus += 0.04
        elif edge.relationship_type == "requires":
            bonus += 0.03
        elif edge.relationship_type == "recommended_after":
            bonus += 0.02
    for left, right in zip(sequence, sequence[1:]):
        for outcome in list(node_by_action[left].precedent_pack.get("second_order_effects", []) or []):
            if str(outcome.get("follow_on_action_id", "")) == right:
                bonus += min(0.05, 0.05 * float(outcome.get("frequency", 0.0) or 0.0))
    return round(min(0.2, bonus), 6)


def _support_factor(sequence: Sequence[str], node_by_action: Dict[str, PlannerNode]) -> float:
    vals = []
    for action_id in sequence:
        node = node_by_action[action_id]
        precedent_confidence = _precedent_confidence(node.precedent_pack)
        eval_conf = float(node.candidate.get("evaluation_confidence", 0.0) or 0.0)
        vals.append((precedent_confidence + eval_conf) / 2.0)
    avg = sum(vals) / len(vals) if vals else 0.0
    return round(max(0.25, min(1.0, 0.6 + (0.4 * avg))), 6)


def _branch_probability(source: PlannerNode, target_action: str) -> float:
    freq = 0.0
    for outcome in list(source.precedent_pack.get("second_order_effects", []) or []):
        if str(outcome.get("follow_on_action_id", "")) == target_action:
            freq = max(freq, float(outcome.get("frequency", 0.0) or 0.0))
    return round(_clip(max(0.2, min(0.7, 0.2 + freq)), 0.2, 0.7), 6)


def _trigger_type(condition: str) -> str:
    text = str(condition or "").lower()
    for keyword, trigger_type in _TRIGGER_KEYWORDS.items():
        if keyword in text:
            return trigger_type
    return "strategic_condition"


def _trigger_frequency(condition: str) -> str:
    trigger_type = _trigger_type(condition)
    if trigger_type in {"credit_condition", "valuation_condition", "market_regime"}:
        return "weekly"
    return "monthly"


def _robustness_score(sequence: Sequence[str], node_by_action: Dict[str, PlannerNode]) -> float:
    worst_penalties = []
    for action_id in sequence:
        regime_rows = list(((node_by_action[action_id].candidate.get("impact_distribution", {}) or {}).get("regime_sensitivity", []) or []))
        if not regime_rows:
            worst_penalties.append(0.1)
            continue
        worst_shift = min(float(row.get("effect_shift", 0.0) or 0.0) for row in regime_rows)
        worst_penalties.append(max(0.0, -worst_shift))
    penalty = max(worst_penalties) if worst_penalties else 0.0
    return round(_clip(1.0 - min(0.6, penalty), 0.35, 1.0), 6)


def _tail_risk_penalty(sequence: Sequence[str], node_by_action: Dict[str, PlannerNode]) -> float:
    penalties: List[float] = []
    for action_id in sequence:
        node = node_by_action[action_id]
        tails = list(node.precedent_pack.get("tail_events", []) or [])
        if not tails:
            penalties.append(0.02 if bool((node.precedent_pack.get("mismatch_diagnostics", {}) or {}).get("out_of_sample_flag")) else 0.0)
            continue
        adverse = [tail for tail in tails if _is_adverse_tail(tail)]
        if not adverse:
            penalties.append(0.0)
            continue
        severities = [_tail_severity(tail) for tail in adverse]
        penalties.append((len(adverse) / max(1, len(tails))) * (sum(severities) / len(severities)) * 0.2)
    return round(min(0.35, sum(penalties)), 6)


def _complexity_penalty(sequence: Sequence[str], node_by_action: Dict[str, PlannerNode]) -> float:
    complexities = []
    for action_id in sequence:
        complexity = float((node_by_action[action_id].schema.get("execution_complexity_prior", {}) or {}).get("base_complexity_score", 3) or 3)
        complexities.append((complexity - 1.0) / 4.0)
    avg_complexity = sum(complexities) / len(complexities) if complexities else 0.0
    penalty = (0.05 * max(0, len(sequence) - 1)) + (0.12 * avg_complexity)
    return round(min(0.35, penalty), 6)


def _weighted_objective_sum(impact_distribution: Dict[str, Any], run: RecommendationRun) -> float:
    return _aggregate_weighted_objectives(_weighted_objective_components(impact_distribution, run))


def _weighted_objective_components(impact_distribution: Dict[str, Any], run: RecommendationRun) -> Dict[str, float]:
    objectives = dict((impact_distribution or {}).get("objectives", {}) or {})
    return {
        "value_creation": run.objectives.value_creation_weight * float((objectives.get("value_creation", {}) or {}).get("median", 0.0) or 0.0),
        "risk_reduction": run.objectives.risk_reduction_weight * float((objectives.get("risk_reduction", {}) or {}).get("median", 0.0) or 0.0),
        "growth": run.objectives.growth_weight * float((objectives.get("growth", {}) or {}).get("median", 0.0) or 0.0),
        "rating_preservation": run.objectives.rating_preservation_weight * float((objectives.get("rating_preservation", {}) or {}).get("median", 0.0) or 0.0),
        "optionality": run.objectives.optionality_weight * float((objectives.get("optionality", {}) or {}).get("median", 0.0) or 0.0),
    }


def _aggregate_weighted_objectives(weighted: Dict[str, float]) -> float:
    positive = sum(value for value in weighted.values() if value > 0.0)
    negative = sum(-value for value in weighted.values() if value < 0.0)
    positive_count = sum(1 for value in weighted.values() if value > 0.01)
    negative_count = sum(1 for value in weighted.values() if value < -0.01)
    breadth_bonus = 0.02 * max(0, positive_count - 1)
    narrowness_penalty = 0.015 * max(0, negative_count - positive_count)
    return positive - (2.5 * negative) + breadth_bonus - narrowness_penalty


def _negative_utility_penalty(weighted_utility: float, weighted_components: Dict[str, float], action_ids: Sequence[str]) -> float:
    if weighted_utility >= 0.0:
        return 0.0
    penalty = 0.02 + min(0.03, abs(float(weighted_utility)))
    positive_count = sum(1 for value in weighted_components.values() if float(value) > 0.01)
    negative_count = sum(1 for value in weighted_components.values() if float(value) < -0.01)
    if negative_count >= positive_count:
        penalty += 0.015
    recurring_payout_actions = {
        "capital_return.dividend_increase",
        "capital_return.dividend_initiate",
        "capital_return.dividend_cut",
    }
    if action_ids and all(action_id in recurring_payout_actions for action_id in action_ids):
        penalty += 0.015
    return round(min(0.09, penalty), 6)


def _impact_driver_contribution(candidate: Dict[str, Any], driver_name: str) -> float:
    impact = dict(candidate.get("impact_distribution", {}) or {})
    for driver in list(impact.get("key_drivers", []) or []):
        if str(driver.get("driver_name", "") or "") != driver_name:
            continue
        try:
            return float(driver.get("contribution", 0.0) or 0.0)
        except Exception:
            return 0.0
    return 0.0


def _dividend_initiate_causal_relief(candidate: Dict[str, Any]) -> Dict[str, float]:
    if str(candidate.get("action_id", "") or "") != "capital_return.dividend_initiate":
        return {"action_specific_penalty_relief": 0.0, "status_quo_hurdle_relief": 0.0}
    if not _has_causal_support(candidate):
        return {"action_specific_penalty_relief": 0.0, "status_quo_hurdle_relief": 0.0}

    blend_weight = _impact_driver_contribution(candidate, "causal_model_blend_weight")
    model_quality = max(
        _impact_driver_contribution(candidate, "causal_model_quality"),
        _impact_driver_contribution(candidate, "causal_model_min_oos_r2"),
    )
    support_score = _impact_driver_contribution(candidate, "causal_model_support_score")
    if blend_weight < 0.2 or model_quality < 0.1 or support_score < 0.85:
        return {"action_specific_penalty_relief": 0.0, "status_quo_hurdle_relief": 0.0}

    impact = dict(candidate.get("impact_distribution", {}) or {})
    objectives = dict(impact.get("objectives", {}) or {})
    value_creation = float((objectives.get("value_creation", {}) or {}).get("median", 0.0) or 0.0)
    positive_count = sum(
        1
        for objective in _OBJECTIVE_FIELDS
        if float((objectives.get(objective, {}) or {}).get("median", 0.0) or 0.0) > 0.01
    )

    action_relief = 0.01
    status_quo_relief = 0.005
    if blend_weight >= 0.24:
        action_relief += 0.01
        status_quo_relief += 0.005
    if support_score >= 0.9:
        action_relief += 0.01
        status_quo_relief += 0.005
    if model_quality >= 0.12:
        action_relief += 0.005
        status_quo_relief += 0.005
    if value_creation >= 0.04:
        action_relief += 0.005
    if positive_count >= 1:
        status_quo_relief += 0.005
    return {
        "action_specific_penalty_relief": round(min(0.04, action_relief), 6),
        "status_quo_hurdle_relief": round(min(0.02, status_quo_relief), 6),
    }


def _buyback_maturity_wall_relief(candidate: Dict[str, Any]) -> float:
    action_id = str(candidate.get("action_id", "") or "")
    if action_id != "capital_return.open_market_buyback":
        return 0.0

    feasibility = dict(candidate.get("feasibility", {}) or {})
    pass_probability = float(feasibility.get("pass_probability", 0.0) or 0.0)
    if pass_probability < 0.5:
        return 0.0

    params = dict(candidate.get("parameters", {}) or {})
    size_pct_market_cap = float(params.get("size_pct_market_cap", 0.0) or 0.0)
    if size_pct_market_cap <= 0.0 or size_pct_market_cap > 0.02:
        return 0.0

    blockers = list(feasibility.get("blockers", []) or [])
    if not blockers:
        return 0.0
    if any(str(blocker.get("severity", "") or "").lower() == "hard" for blocker in blockers):
        return 0.0
    blocker_types = {str(blocker.get("blocker_type", "") or "") for blocker in blockers}
    if blocker_types != {"maturity_wall_conflict"}:
        return 0.0

    runway_months = 0.0
    proforma_interest_coverage = 0.0
    maturity_wall_ratio = 0.0
    for signal in list(feasibility.get("gating_signals", []) or []):
        name = str(signal.get("feature_name", "") or "")
        try:
            value = float(signal.get("value", 0.0) or 0.0)
        except Exception:
            continue
        if name == "liquidity.runway_months_proforma":
            runway_months = value
        elif name == "capital_structure.proforma_interest_coverage":
            proforma_interest_coverage = value
        elif name == "capital_structure.maturity_wall_ratio_24m":
            maturity_wall_ratio = value

    if runway_months < 18.0 or proforma_interest_coverage < 10.0:
        return 0.0
    if maturity_wall_ratio <= 0.25 or maturity_wall_ratio > 0.3:
        return 0.0

    objectives = dict(dict(candidate.get("impact_distribution", {}) or {}).get("objectives", {}) or {})
    value_creation = float((objectives.get("value_creation", {}) or {}).get("median", 0.0) or 0.0)
    rating_preservation = float((objectives.get("rating_preservation", {}) or {}).get("median", 0.0) or 0.0)
    if value_creation < 0.04 or rating_preservation < -0.02:
        return 0.0

    return 0.04


def _status_quo_hurdle(
    weighted_utility: float,
    weighted_components: Dict[str, float],
    action_ids: Sequence[str],
    support_factor: float = 0.0,
    transition_bonus: float = 0.0,
    structural_bonus: float = 0.0,
    candidates: Optional[Sequence[Dict[str, Any]]] = None,
) -> float:
    positive_count = sum(1 for value in weighted_components.values() if float(value) > 0.01)
    if weighted_utility >= 0.02 and positive_count >= 2:
        return 0.0
    if transition_bonus >= 0.03 or structural_bonus >= 0.02:
        return 0.0

    soft_actions = {
        "capital_return.dividend_initiate",
        "capital_return.dividend_increase",
        "capital_return.dividend_cut",
        "capital_return.special_dividend",
        "governance.board_refresh",
        "governance.activist_settlement",
        "governance.stock_split",
        "restructuring.working_capital_program",
    }
    thin_financing_actions = {
        "capital_structure.equity_issuance",
        "capital_structure.new_debt_issuance",
        "capital_structure.refinancing",
        "capital_structure.revolver_draw_or_resize",
    }

    if action_ids and all(action_id in soft_actions for action_id in action_ids):
        penalty = 0.025
        if weighted_utility < 0.02:
            penalty += 0.015
        if positive_count <= 1:
            penalty += 0.015
        if support_factor and support_factor < 0.88:
            penalty += 0.01
        if candidates and all(str(candidate.get("action_id", "") or "") == "capital_return.dividend_initiate" for candidate in candidates):
            relief = max(
                (_dividend_initiate_causal_relief(candidate).get("status_quo_hurdle_relief", 0.0) or 0.0)
                for candidate in candidates
            )
            penalty = max(0.0, penalty - float(relief))
        return round(min(0.06, penalty), 6)

    if action_ids and all(action_id in thin_financing_actions for action_id in action_ids):
        penalty = 0.0
        if weighted_utility < 0.03:
            penalty += 0.02
        if positive_count <= 2:
            penalty += 0.015
        if support_factor and support_factor < 0.88:
            penalty += 0.01
        return round(min(0.05, penalty), 6)

    return 0.0


def _is_deleveraging_equity_recap(candidate: Dict[str, Any]) -> bool:
    action_id = str(candidate.get("action_id", "") or "")
    if action_id != "capital_structure.equity_issuance":
        return False

    params = dict(candidate.get("parameters") or candidate.get("params") or {})
    use_of_proceeds = str(params.get("use_of_proceeds", "") or "").strip().lower()
    if use_of_proceeds != "deleveraging":
        return False

    return True


def _action_specific_penalty(candidate: Dict[str, Any], precedent_pack: Optional[Dict[str, Any]] = None) -> float:
    action_id = str(candidate.get("action_id", "") or "")
    action_type = str(candidate.get("action_type", "") or "")
    precedent_pack = dict(precedent_pack or {})
    impact = dict(candidate.get("impact_distribution", {}) or {})
    objectives = dict(impact.get("objectives", {}) or {})
    medians = {
        objective: float((objectives.get(objective, {}) or {}).get("median", 0.0) or 0.0)
        for objective in _OBJECTIVE_FIELDS
    }
    recurring_dividend_actions = {
        "capital_return.dividend_increase",
        "capital_return.dividend_initiate",
        "capital_return.dividend_cut",
    }
    if action_id in recurring_dividend_actions:
        # Recurring dividend policy moves are narrower than capital-structure,
        # portfolio, or acquisition plans. Require a clearer edge before they
        # dominate the top plan.
        penalty = 0.06 if action_id == "capital_return.dividend_initiate" else 0.05
        positive = 0
        negative = 0
        for objective in _OBJECTIVE_FIELDS:
            median = medians[objective]
            if median > 0.01:
                positive += 1
            elif median < -0.01:
                negative += 1
        if positive <= 1 and negative >= 2:
            penalty += 0.04
        if float((impact.get("uncertainty_score", 0.0) or 0.0)) > 0.3:
            penalty += 0.01
        if action_id == "capital_return.dividend_initiate":
            penalty -= float(_dividend_initiate_causal_relief(candidate).get("action_specific_penalty_relief", 0.0) or 0.0)
            penalty = max(0.03, penalty)
    elif action_id == "capital_return.special_dividend":
        # One-time payouts are less sticky than dividend policy changes but
        # still need a clearer edge than default financing or portfolio moves.
        penalty = 0.035
    elif action_id == "capital_structure.equity_issuance":
        # External equity is dilutive and should only rank highly when it
        # clearly solves a balance-sheet constraint or funds a strong strategic
        # opportunity. Mildly positive generic financing effects are not enough.
        structural_need = max(medians["risk_reduction"], medians["rating_preservation"])
        strategic_need = max(medians["growth"], medians["optionality"])
        value_creation = medians["value_creation"]
        recap_context = _is_deleveraging_equity_recap(candidate)

        penalty = 0.03
        if structural_need < 0.06 and strategic_need < 0.06:
            penalty += 0.05
        elif structural_need < 0.08 and strategic_need < 0.08:
            penalty += 0.03
        if value_creation < 0.03:
            penalty += 0.02
        if float((impact.get("uncertainty_score", 0.0) or 0.0)) > 0.3:
            penalty += 0.01
        if structural_need >= 0.12:
            penalty -= 0.03
        elif strategic_need >= 0.12 and value_creation >= 0.05:
            penalty -= 0.02
        if recap_context:
            if structural_need >= 0.04:
                penalty -= 0.04
            elif structural_need >= 0.03:
                penalty -= 0.025
            if value_creation >= 0.0 and strategic_need >= 0.02:
                penalty -= 0.01
        penalty = max(0.0, penalty)
    elif action_type in {"governance", "restructuring"}:
        # Generic governance and restructuring actions are valid, but they
        # should not beat clearer capital allocation or financing actions on a
        # thin edge. Require either a broad measurable benefit or unusually
        # strong evidence.
        structural_need = max(medians["risk_reduction"], medians["rating_preservation"])
        strategic_need = max(medians["growth"], medians["optionality"])
        value_creation = medians["value_creation"]
        positive_count = sum(1 for value in medians.values() if value > 0.01)

        penalty = 0.03
        if positive_count <= 1:
            penalty += 0.04
        if max(value_creation, structural_need, strategic_need) < 0.05:
            penalty += 0.04
        if float((impact.get("uncertainty_score", 0.0) or 0.0)) > 0.3:
            penalty += 0.01
        if max(value_creation, structural_need, strategic_need) >= 0.1 and positive_count >= 2:
            penalty -= 0.03
        penalty = max(0.0, penalty)
    else:
        penalty = 0.0

    has_precedent = _precedent_confidence(precedent_pack) > 0.0
    has_causal = _has_causal_support(candidate)
    if not has_precedent and not has_causal:
        unsupported_penalty = 0.15
        unsupported_penalty -= _buyback_maturity_wall_relief(candidate)
        penalty += max(0.0, unsupported_penalty)
    if action_type in {"governance", "restructuring"} and not has_precedent and not has_causal:
        penalty += 0.08
    if action_id == "capital_structure.revolver_draw_or_resize" and not has_precedent:
        penalty += 0.1
    return round(penalty, 6)


def _structural_action_bonus(candidate: Dict[str, Any], precedent_pack: Optional[Dict[str, Any]] = None) -> float:
    action_id = str(candidate.get("action_id", "") or "")
    action_type = str(candidate.get("action_type", "") or "")
    if action_id == "capital_structure.equity_issuance" and _is_deleveraging_equity_recap(candidate):
        impact = dict(candidate.get("impact_distribution", {}) or {})
        objectives = dict(impact.get("objectives", {}) or {})
        structural_need = max(
            float((objectives.get("risk_reduction", {}) or {}).get("median", 0.0) or 0.0),
            float((objectives.get("rating_preservation", {}) or {}).get("median", 0.0) or 0.0),
        )
        optionality = float((objectives.get("optionality", {}) or {}).get("median", 0.0) or 0.0)
        if structural_need >= 0.04 or optionality >= 0.03:
            return 0.025
        if structural_need >= 0.03:
            return 0.015
        return 0.0

    if action_type != "portfolio" or action_id not in {
        "portfolio.divestiture_partial",
        "portfolio.divestiture_full",
        "portfolio.asset_sale",
    }:
        return 0.0

    precedent_confidence = _precedent_confidence(dict(precedent_pack or {}))
    if precedent_confidence < 0.25:
        return 0.0

    impact = dict(candidate.get("impact_distribution", {}) or {})
    objectives = dict(impact.get("objectives", {}) or {})
    medians = {
        objective: float((objectives.get(objective, {}) or {}).get("median", 0.0) or 0.0)
        for objective in _OBJECTIVE_FIELDS
    }
    positive_count = sum(1 for value in medians.values() if value > 0.01)
    negative_count = sum(1 for value in medians.values() if value < -0.01)
    if positive_count < 3:
        return 0.0

    bonus = 0.0
    if medians["risk_reduction"] > 0.05 and (medians["rating_preservation"] > 0.03 or medians["optionality"] > 0.05):
        bonus += 0.03
    if negative_count == 0 and positive_count >= 4:
        bonus += 0.015
    return round(min(0.05, bonus), 6)


def _has_causal_support(candidate: Dict[str, Any]) -> bool:
    impact = dict(candidate.get("impact_distribution", {}) or {})
    for driver in list(impact.get("key_drivers", []) or []):
        driver_name = str(driver.get("driver_name", "") or "")
        try:
            contribution = float(driver.get("contribution", 0.0) or 0.0)
        except Exception:
            contribution = 0.0
        if driver_name == "causal_model_mode" and contribution >= 0.5:
            return True
        if driver_name == "causal_model_blend_weight" and contribution > 0.0:
            return True
    return False


def _precedent_confidence(precedent_pack: Dict[str, Any]) -> float:
    return float(
        precedent_pack.get("precedent_confidence")
        or precedent_pack.get("calibration_confidence")
        or 0.0
    )


def _tail_severity(tail: Dict[str, Any]) -> float:
    value = float(tail.get("value") or tail.get("outcome_value") or 0.0)
    return min(1.0, abs(value))


def _is_adverse_tail(tail: Dict[str, Any]) -> bool:
    metric = str(tail.get("metric") or tail.get("outcome_metric") or "").lower()
    value = float(tail.get("value") or tail.get("outcome_value") or 0.0)
    if metric in {"credit_spread_change", "volatility_change"}:
        return value > 0
    if metric == "rating_migration":
        return value < 0
    return value < 0


def _bounded_signal(value: float) -> float:
    return round(0.5 + (0.5 * math.tanh(float(value))), 6)


def _feasibility_factor(pass_probability: float) -> float:
    return round(_clip(0.55 + (0.45 * float(pass_probability)), 0.55, 1.0), 6)


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _parse_run_time(raw: str) -> datetime:
    value = str(raw)
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dedupe_keep_order(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for value in values:
        token = str(value or "").strip()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


__all__ = ["build_plan_set"]
