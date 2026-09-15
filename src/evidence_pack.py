"""
Evidence Pack
=============
The core data structure that grounds ALL narrative generation.

This is the heart of the system - before any text is generated,
we assemble a complete, auditable EvidencePack containing:

1. State Summary - SignalProfile with drivers and explanations
2. Regime Context - Current market environment
3. Cohorts - Grouped analog cases with outcomes and key splits
4. Objections - Pre-computed Q&A with grounded answers
5. Citations - Pointers to source data for every claim

The LLM/templates can ONLY use data from the EvidencePack.
No hallucination, no invented facts.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json

from .recommendation_runtime_config import DEFAULT_PRECEDENT_RETRIEVAL_VERSION
from .snapshot import AsOfSnapshotBuilder, DATA_DIR
from .asof_store import AsOfWarehouse


# =============================================================================
# SCHEMA DEFINITIONS
# =============================================================================

@dataclass
class FeatureContribution:
    """A single feature's contribution to a signal."""
    feature_name: str
    feature_value: float
    contribution: float  # How much this feature drove the signal
    direction: str  # "positive" or "negative"
    human_label: str  # e.g., "Net Debt / EBITDA"


@dataclass
class SignalDetail:
    """Full detail for a single signal."""
    name: str
    value: str  # "high", "medium", "low"
    score: float  # 0-100
    confidence: float  # 0-1
    drivers: List[FeatureContribution]
    explanation: str  # 1-sentence banker-friendly explanation
    evidence_refs: Dict[str, List[str]]  # warehouse_rows, doc_chunks


@dataclass
class SignalProfileV2:
    """Enhanced signal profile with full audit trail."""
    company_id: str
    company_name: str
    as_of_time: str
    signal_schema_version: str
    signals: Dict[str, SignalDetail]
    composite_score: float
    vector: List[float]  # For similarity computation


@dataclass
class RegimeContext:
    """Market regime with transition context."""
    regime_id: str  # "LOOSE", "SELECTIVE", "TIGHT"
    confidence: float
    since: Optional[str]  # When this regime started
    is_transitioning: bool
    transition_direction: Optional[str]  # "tightening" or "loosening"
    characteristics: Dict[str, str]


@dataclass
class OutcomeDistribution:
    """Outcome distribution for a cohort."""
    metric_name: str  # e.g., "tsr_12m"
    n: int
    p10: Optional[float]
    p25: Optional[float]
    p50: Optional[float]  # median
    p75: Optional[float]
    p90: Optional[float]
    mean: Optional[float]
    std: Optional[float]
    pct_positive: Optional[float]  # % with positive outcome
    pct_beat_benchmark: Optional[float]


@dataclass
class KeySplit:
    """A condition that materially changes outcomes."""
    condition: str  # e.g., "valuation_dislocation = high"
    condition_human: str  # e.g., "When stock is undervalued vs peers"
    effect: str  # e.g., "+4% median TSR"
    effect_magnitude: float
    n_with_condition: int
    n_without_condition: int
    outcomes_with: OutcomeDistribution
    outcomes_without: OutcomeDistribution
    statistical_significance: float  # p-value or confidence


@dataclass
class ExampleCase:
    """A specific historical case for narrative use."""
    case_id: str
    company_name: str
    date: str
    action_type: str
    similarity_score: float
    signal_profile_summary: Dict[str, float]  # key signals at time of action
    outcome_tsr_12m: Optional[float]
    context_summary: str  # 1-2 sentence description


@dataclass
class Cohort:
    """A group of similar historical cases."""
    action_type: str  # e.g., "bolt_on_mna", "buyback"
    action_human: str  # e.g., "Bolt-on M&A"
    filters: Dict[str, Any]  # How this cohort was defined
    n: int
    outcomes: Dict[str, OutcomeDistribution]  # tsr_1m, tsr_3m, tsr_12m, etc.
    key_splits: List[KeySplit]
    example_cases: List[ExampleCase]
    quality_flags: List[str]  # e.g., ["small_sample", "regime_mismatch"]


@dataclass
class Objection:
    """A pre-computed objection with grounded rebuttal."""
    question: str  # e.g., "Why not wait 6 months?"
    question_category: str  # "timing", "sizing", "alternative", "risk"
    grounded_answer_points: List[str]
    evidence_refs: List[str]
    confidence: float


@dataclass
class ActionCard:
    """Full evidence for one potential action."""
    action_type: str
    action_human: str
    recommendation_score: int  # 0-100
    recommendation_label: str  # "HIGH", "MEDIUM", etc.
    status: str  # "NEWLY_ACTIONABLE", "PERSISTENT", "WINDOW_OPEN"

    # Thesis
    thesis_summary: str
    value_lever: str
    economic_impact: str
    share_price_impact_range: Optional[str]

    # Why Now bullets (grounded)
    why_now_bullets: List[Dict[str, Any]]  # Each has text, metric, source

    # Conditions
    works_when: List[str]
    fails_when: List[str]

    # Evidence
    cohort: Cohort
    comparable_cohorts: List[Cohort]  # Alternative filters for comparison

    # Objections
    objections: List[Objection]


@dataclass
class EvidencePack:
    """
    The complete grounded evidence package.

    This is what the narrative layer receives - nothing else.
    Every claim must trace back to something in here.
    """
    # Metadata
    pack_id: str
    generated_at: str
    company_id: str
    company_name: str
    as_of_time: str

    # Version tracking (for audit)
    data_snapshot_version: str
    signal_schema_version: str
    regime_model_version: str
    retrieval_version: str

    # Core content
    state_summary: SignalProfileV2
    regime: RegimeContext
    company_metrics: Dict[str, Any]  # Revenue, margins, etc.
    peer_comparison: Dict[str, Any]

    # Action analysis
    action_cards: List[ActionCard]

    # Cross-cutting
    binding_constraints: List[Dict[str, str]]
    timing_posture: Dict[str, str]

    # Historical context
    company_action_history: Dict[str, Any]

    def to_dict(self) -> Dict:
        """Serialize to dictionary for storage/API."""
        # This would be a full serialization - simplified here
        return {
            'pack_id': self.pack_id,
            'company_id': self.company_id,
            'as_of_time': self.as_of_time,
            'generated_at': self.generated_at,
            # ... full serialization
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), default=str)


# =============================================================================
# EVIDENCE PACK BUILDER
# =============================================================================

