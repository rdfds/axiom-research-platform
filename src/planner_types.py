from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DependencyEdge:
    source_action: str
    target_action: str
    relationship_type: str
    condition: Optional[str] = None
    strength: Optional[str] = None
    explanation: Optional[str] = None
    original_rule_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ActionDependencyGraph:
    nodes: List[str]
    edges: List[DependencyEdge] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": list(self.nodes),
            "edges": [edge.to_dict() for edge in self.edges],
        }


@dataclass
class LeadTimeDistribution:
    action_id: str
    minimum_days: int
    mean_days: float
    median_days: int
    p25_days: int
    p75_days: int
    p90_days: int
    conditional_adjustments: List[Dict[str, Any]] = field(default_factory=list)
    source: str = "schema_prior_interpolated"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PlanExplanation:
    problem_statement: str
    why_this_action: str
    why_now: str
    key_supporting_facts: List[str] = field(default_factory=list)
    main_tradeoffs: List[str] = field(default_factory=list)
    why_not_alternatives: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PlanTrigger:
    trigger_type: str
    condition: str
    evaluation_frequency: str
    trigger_probability: float
    explanation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PlanBranch:
    branch_condition: str
    branch_plan_steps: List[str]
    branch_probability: float
    explanation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PlanRisk:
    main_failure_modes: List[str] = field(default_factory=list)
    regime_sensitivity: List[str] = field(default_factory=list)
    execution_risks: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PlanTimeline:
    start_time: Optional[str]
    step_schedule: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PlanStep:
    step_id: str
    action_id: str
    parameters: Dict[str, Any]
    earliest_start: Optional[str]
    expected_duration: Dict[str, Any]
    prerequisites: List[str] = field(default_factory=list)
    probability_of_success: float = 0.0
    impact_contribution: Dict[str, Any] = field(default_factory=dict)
    explanation: Optional[PlanExplanation] = None

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        if self.explanation is not None:
            out["explanation"] = self.explanation.to_dict()
        return out


@dataclass
class PlanScoreBreakdown:
    expected_utility: float = 0.0
    feasibility_chain: float = 0.0
    robustness_score: float = 0.0
    tail_risk_penalty: float = 0.0
    complexity_penalty: float = 0.0
    time_discount_factor: float = 1.0
    total_score: float = 0.0
    components: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Plan:
    plan_id: str
    run_id: str
    steps: List[PlanStep]
    timeline: PlanTimeline
    triggers: List[PlanTrigger] = field(default_factory=list)
    branches: List[PlanBranch] = field(default_factory=list)
    score_breakdown: PlanScoreBreakdown = field(default_factory=PlanScoreBreakdown)
    risks: PlanRisk = field(default_factory=PlanRisk)
    summary_explanation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "plan_id": self.plan_id,
            "run_id": self.run_id,
            "steps": [step.to_dict() for step in self.steps],
            "timeline": self.timeline.to_dict(),
            "triggers": [trigger.to_dict() for trigger in self.triggers],
            "branches": [branch.to_dict() for branch in self.branches],
            "score_breakdown": self.score_breakdown.to_dict(),
            "risks": self.risks.to_dict(),
            "summary_explanation": self.summary_explanation,
        }
        return out


__all__ = [
    "ActionDependencyGraph",
    "DependencyEdge",
    "LeadTimeDistribution",
    "Plan",
    "PlanBranch",
    "PlanExplanation",
    "PlanRisk",
    "PlanScoreBreakdown",
    "PlanStep",
    "PlanTimeline",
    "PlanTrigger",
]
