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


