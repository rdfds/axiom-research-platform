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


