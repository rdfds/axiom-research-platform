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


