"""Mechanism Brain: evaluates action candidates without ranking.

Transforms ActionCandidateDraft-like records into ActionCandidate evaluations with:
- feasibility gating
- mechanism activation
- probabilistic counterfactual impact
- structural sanity flags
- risks and assumptions
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .causal_impact_model import (
    CausalImpactModel,
    get_causal_action_policy,
    load_default_causal_impact_model,
)
from .model_feature_bundle import feature_view_from_snapshot
from .runtime_feature_adapter import resolve_feature_value


_HARD_ACTIONS_HIGH_COMPLEXITY = {
    "mna.go_private_lbo",
    "mna.transformational_acquisition",
    "portfolio.spin_off",
    "portfolio.carve_out_ipo",
    "restructuring.chapter_pathway",
    "restructuring.out_of_court_restructuring",
}

_CASH_CONSUMING_ACTION_PREFIXES = (
    "capital_return.",
    "mna.",
)

_DEBT_REQUIRING_ACTION_IDS = {
    "capital_structure.new_debt_issuance",
    "capital_structure.refinancing",
    "capital_structure.tender_offer_debt",
    "capital_structure.exchange_offer",
    "capital_structure.liability_management_exercise",
    "capital_structure.revolver_draw_or_resize",
    "capital_structure.convertible_issuance",
}

_EQUITY_REQUIRING_ACTION_IDS = {
    "capital_structure.equity_issuance",
    "capital_structure.convertible_issuance",
    "capital_structure.preferred_issuance",
    "portfolio.carve_out_ipo",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _to_float(v: Any, default: Optional[float] = None) -> Optional[float]:
    if v is None:
        return default
    if isinstance(v, bool):
        return float(v)
    try:
        out = float(v)
    except Exception:
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def _feature_value(raw: Any) -> Any:
    if isinstance(raw, dict):
        return raw.get("value")
    return raw


def _extract_feature(features: Dict[str, Any], name: str, default: Any = None) -> Any:
    return resolve_feature_value(features, name, default=default)


def _nested_get(obj: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _parse_action_id_tokens(raw: str) -> set[str]:
    tokens: set[str] = set()
    text = str(raw or "").strip()
    if not text:
        return tokens
    for part in re.split(r"[,\s]+", text):
        tok = str(part or "").strip().lower()
        if tok:
            tokens.add(tok)
    return tokens


def _load_action_id_tokens_from_file(path_value: str) -> set[str]:
    path = Path(str(path_value or "").strip())
    if not str(path):
        return set()
    if not path.exists() or not path.is_file():
        return set()
    try:
        body = path.read_text()
    except Exception:
        return set()
    out: set[str] = set()
    for line in body.splitlines():
        line_clean = str(line).split("#", 1)[0].strip()
        if not line_clean:
            continue
        out.update(_parse_action_id_tokens(line_clean))
    return out


@dataclass
class Signal:
    feature_name: str
    value: Any
    threshold: Any
    interpretation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Blocker:
    blocker_type: str
    severity: str
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Remediation:
    action_required: str
    expected_effect: str
    estimated_delay_days: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FeasibilityResult:
    feasibility_status: str
    pass_probability: float
    blockers: List[Blocker] = field(default_factory=list)
    remediation_steps: List[Remediation] = field(default_factory=list)
    lead_time_prior_days: int = 0
    gating_signals: List[Signal] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feasibility_status": self.feasibility_status,
            "pass_probability": float(self.pass_probability),
            "blockers": [b.to_dict() for b in self.blockers],
            "remediation_steps": [r.to_dict() for r in self.remediation_steps],
            "lead_time_prior_days": int(self.lead_time_prior_days),
            "gating_signals": [s.to_dict() for s in self.gating_signals],
        }


@dataclass
class Mechanism:
    mechanism_id: str
    channel_type: str
    activation_strength: float
    positive_signals: List[Signal] = field(default_factory=list)
    negative_signals: List[Signal] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mechanism_id": self.mechanism_id,
            "channel_type": self.channel_type,
            "activation_strength": float(self.activation_strength),
            "positive_signals": [s.to_dict() for s in self.positive_signals],
            "negative_signals": [s.to_dict() for s in self.negative_signals],
        }


@dataclass
class Interaction:
    feature_combination: str
    direction: str
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MechanismActivation:
    mechanisms: List[Mechanism] = field(default_factory=list)
    key_interactions: List[Interaction] = field(default_factory=list)
    narrative_explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mechanisms": [m.to_dict() for m in self.mechanisms],
            "key_interactions": [k.to_dict() for k in self.key_interactions],
            "narrative_explanation": self.narrative_explanation,
        }


@dataclass
class Distribution:
    median: float
    p10: float
    p25: float
    p75: float
    p90: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RegimeImpact:
    regime_condition: str
    effect_shift: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Driver:
    driver_name: str
    contribution: float
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ImpactDistribution:
    objectives: Dict[str, Distribution]
    regime_sensitivity: List[RegimeImpact]
    key_drivers: List[Driver]
    uncertainty_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "objectives": {k: v.to_dict() for k, v in self.objectives.items()},
            "regime_sensitivity": [r.to_dict() for r in self.regime_sensitivity],
            "key_drivers": [d.to_dict() for d in self.key_drivers],
            "uncertainty_score": float(self.uncertainty_score),
        }


@dataclass
class SanityCheck:
    check_type: str
    status: str
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RiskItem:
    risk_type: str
    probability: float
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_type": self.risk_type,
            "probability": float(self.probability),
            "explanation": self.explanation,
        }


@dataclass
class Assumption:
    assumption_type: str
    description: str
    sensitivity: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ActionCandidate:
    candidate_id: str
    run_id: str
    action_id: str
    action_type: str
    action_subtype: str
    parameters: Dict[str, Any]
    feasibility: FeasibilityResult
    mechanism_activation: MechanismActivation
    impact_distribution: ImpactDistribution
    structural_sanity_flags: List[SanityCheck]
    risks: List[RiskItem]
    assumptions: List[Assumption]
    evaluation_confidence: float
    created_at: str
    evaluation_profile: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "run_id": self.run_id,
            "action_id": self.action_id,
            "action_type": self.action_type,
            "action_subtype": self.action_subtype,
            "parameters": self.parameters,
            "feasibility": self.feasibility.to_dict(),
            "mechanism_activation": self.mechanism_activation.to_dict(),
            "impact_distribution": self.impact_distribution.to_dict(),
            "structural_sanity_flags": [x.to_dict() for x in self.structural_sanity_flags],
            "risks": [x.to_dict() for x in self.risks],
            "assumptions": [x.to_dict() for x in self.assumptions],
            "evaluation_confidence": float(self.evaluation_confidence),
            "created_at": self.created_at,
            "evaluation_profile": dict(self.evaluation_profile or {}),
        }


