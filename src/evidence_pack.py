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


