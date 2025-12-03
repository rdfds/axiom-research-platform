from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class CompanyStateSnapshot:
    company_id: str
    as_of_time: datetime
    features: Dict[str, Any]
    regime: Dict[str, Any] = field(default_factory=dict)
    constraint_set: List[Dict[str, Any]] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "company_id": self.company_id,
            "as_of_time": self.as_of_time.isoformat(),
            "features": self.features,
            "regime": self.regime,
            "constraint_set": self.constraint_set,
            "provenance": self.provenance,
        }


@dataclass
class ActionCandidate:
    action_type: str
    params: Dict[str, Any]
    action_subtype: Optional[str] = None
    action_id: Optional[str] = None
    assumed_preconditions: List[str] = field(default_factory=list)
    rationale_refs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type,
            "action_subtype": self.action_subtype,
            "action_id": self.action_id,
            "params": self.params,
            "assumed_preconditions": self.assumed_preconditions,
            "rationale_refs": self.rationale_refs,
        }


@dataclass
class ImpactDistribution:
    metric: str
    horizon_months: int
    p25: Optional[float]
    p50: Optional[float]
    p75: Optional[float]
    n: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "horizon_months": self.horizon_months,
            "p25": self.p25,
            "p50": self.p50,
            "p75": self.p75,
            "n": self.n,
        }


@dataclass
class SimilarityScore:
    precedent_id: str
    score: float
    state_similarity: float
    regime_similarity: float
    parameter_similarity: float
    action_match_score: float
    sector_similarity: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "precedent_id": self.precedent_id,
            "score": self.score,
            "state_similarity": self.state_similarity,
            "regime_similarity": self.regime_similarity,
            "parameter_similarity": self.parameter_similarity,
            "action_match_score": self.action_match_score,
            "sector_similarity": self.sector_similarity,
        }


@dataclass
class DistributionStats:
    mean: Optional[float]
    median: Optional[float]
    p10: Optional[float]
    p25: Optional[float]
    p75: Optional[float]
    p90: Optional[float]
    sample_size: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean": self.mean,
            "median": self.median,
            "p10": self.p10,
            "p25": self.p25,
            "p75": self.p75,
            "p90": self.p90,
            "sample_size": self.sample_size,
        }


@dataclass
class MetricDistributionSet:
    valuation_multiple_change: DistributionStats
    equity_return_vs_sector: DistributionStats
    credit_spread_change: DistributionStats
    rating_migration: DistributionStats
    leverage_change: DistributionStats
    fcf_change: DistributionStats
    volatility_change: DistributionStats

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valuation_multiple_change": self.valuation_multiple_change.to_dict(),
            "equity_return_vs_sector": self.equity_return_vs_sector.to_dict(),
            "credit_spread_change": self.credit_spread_change.to_dict(),
            "rating_migration": self.rating_migration.to_dict(),
            "leverage_change": self.leverage_change.to_dict(),
            "fcf_change": self.fcf_change.to_dict(),
            "volatility_change": self.volatility_change.to_dict(),
        }


@dataclass
class OutcomeDistributions:
    horizon_1m: MetricDistributionSet
    horizon_6m: MetricDistributionSet
    horizon_12m: MetricDistributionSet
    horizon_24m: MetricDistributionSet

    def to_dict(self) -> Dict[str, Any]:
        return {
            "horizon_1m": self.horizon_1m.to_dict(),
            "horizon_6m": self.horizon_6m.to_dict(),
            "horizon_12m": self.horizon_12m.to_dict(),
            "horizon_24m": self.horizon_24m.to_dict(),
        }


@dataclass
class PrecedentCase:
    precedent_id: str
    company_id: str
    decision_time: str
    action_id: str
    parameters: Dict[str, Any]
    regime: Dict[str, Any]
    similarity_score: float
    key_state_features: Dict[str, Any]
    source_event_id: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "precedent_id": self.precedent_id,
            "company_id": self.company_id,
            "decision_time": self.decision_time,
            # Step-8 schema alias
            "event_time": self.decision_time,
            "action_id": self.action_id,
            "parameters": self.parameters,
            "regime": self.regime,
            "similarity_score": self.similarity_score,
            "key_state_features": self.key_state_features,
            "source_event_id": self.source_event_id,
        }


@dataclass
class RegimeDistribution:
    regime_label: str
    outcome_distributions: OutcomeDistributions
    sample_size: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime_label": self.regime_label,
            "outcome_distributions": self.outcome_distributions.to_dict(),
            "sample_size": self.sample_size,
        }


@dataclass
class TailEvent:
    precedent_id: str
    outcome_metric: str
    outcome_value: float
    horizon: str
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "precedent_id": self.precedent_id,
            "outcome_metric": self.outcome_metric,
            "outcome_value": self.outcome_value,
            "horizon": self.horizon,
            "explanation": self.explanation,
            # Step-8 schema aliases
            "metric": self.outcome_metric,
            "value": self.outcome_value,
            "description": self.explanation,
        }


