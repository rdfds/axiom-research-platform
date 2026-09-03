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
            -float((item.get("score_components") or {}).get("raw_total_score", item['score']) or 0.0),
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


