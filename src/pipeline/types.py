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


@dataclass
class FollowOnOutcome:
    follow_on_action_id: str
    frequency: float
    average_time_to_follow_on: Optional[float]
    median_time_to_follow_on: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "follow_on_action_id": self.follow_on_action_id,
            "frequency": self.frequency,
            "average_time_to_follow_on": self.average_time_to_follow_on,
            # Step-8 schema name
            "median_time_to_follow_on": (
                self.median_time_to_follow_on
                if self.median_time_to_follow_on is not None
                else self.average_time_to_follow_on
            ),
        }


@dataclass
class FeatureMismatch:
    feature_name: str
    candidate_value: Any
    cohort_range: str
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "candidate_value": self.candidate_value,
            "cohort_range": self.cohort_range,
            "explanation": self.explanation,
        }


@dataclass
class MismatchDiagnostics:
    feature_mismatches: List[FeatureMismatch] = field(default_factory=list)
    regime_mismatch: bool = False
    parameter_scale_mismatch: bool = False
    narrative_mismatch: bool = False
    out_of_sample_flag: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_mismatches": [m.to_dict() for m in self.feature_mismatches],
            "regime_mismatch": self.regime_mismatch,
            "parameter_scale_mismatch": self.parameter_scale_mismatch,
            "narrative_mismatch": self.narrative_mismatch,
            "out_of_sample_flag": self.out_of_sample_flag,
        }


@dataclass
class PrecedentPack:
    # Legacy fields (kept for backward compatibility)
    matches: List[Dict[str, Any]] = field(default_factory=list)
    distributions: List[ImpactDistribution] = field(default_factory=list)

    # Precedent Brain v2 fields
    candidate_id: str = ""
    run_id: str = ""
    retrieved_cohorts: List[PrecedentCase] = field(default_factory=list)
    similarity_scores: List[SimilarityScore] = field(default_factory=list)
    outcome_distributions: Optional[OutcomeDistributions] = None
    regime_splits: List[RegimeDistribution] = field(default_factory=list)
    tail_events: List[TailEvent] = field(default_factory=list)
    second_order_effects: List[FollowOnOutcome] = field(default_factory=list)
    mismatch_diagnostics: Dict[str, Any] = field(default_factory=dict)
    calibration_confidence: float = 0.0
    profiling: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        mismatch = self.mismatch_diagnostics
        if isinstance(mismatch, MismatchDiagnostics):
            mismatch = mismatch.to_dict()
        cohorts = [c.to_dict() for c in self.retrieved_cohorts]
        dists_obj = (
            self.outcome_distributions.to_dict()
            if isinstance(self.outcome_distributions, OutcomeDistributions)
            else self.outcome_distributions
        )
        tails = [t.to_dict() for t in self.tail_events]
        second_order = [s.to_dict() for s in self.second_order_effects]
        legacy_dists = [d.to_dict() for d in self.distributions]
        return {
            "candidate_id": self.candidate_id,
            "run_id": self.run_id,
            "retrieved_cohorts": cohorts,
            # Step-8 schema alias
            "cohorts": cohorts,
            "similarity_scores": [s.to_dict() for s in self.similarity_scores],
            "outcome_distributions": dists_obj,
            # Step-8 schema alias
            "distributions": dists_obj,
            "regime_splits": [r.to_dict() for r in self.regime_splits],
            "tail_events": tails,
            # Step-8 schema alias
            "tails": tails,
            "second_order_effects": second_order,
            "calibration_confidence": float(self.calibration_confidence),
            # Step-8 schema alias
            "precedent_confidence": float(self.calibration_confidence),
            "profiling": dict(self.profiling or {}),
            # Legacy keys consumed by existing downstream modules
            "matches": self.matches,
            "legacy_distributions": legacy_dists,
            "mismatch_diagnostics": mismatch,
        }
