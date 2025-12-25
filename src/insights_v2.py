"""
Insights Generator V2
=====================
Enhanced insights generation with:
1. Company-specific action history
2. Quantified "Why Now" bullets with actual data
3. TSR ranges with sample sizes
4. Time-since-trigger tracking
5. Peer/sector valuation comparisons

This gets us closer to the Nike mock quality.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .snapshot import AsOfSnapshotBuilder, DATA_DIR


@dataclass
class WhyNowBullet:
    """A single 'Why Now' bullet with supporting data."""
    text: str
    metric_name: Optional[str] = None
    current_value: Optional[float] = None
    comparison_value: Optional[float] = None
    comparison_label: Optional[str] = None  # "vs 5-year avg", "vs peers", etc.
    source: Optional[str] = None  # "fundamentals", "precedent", "regime", "transcript"


@dataclass
class IdeaV2:
    """Enhanced capital allocation idea with full evidence."""
    name: str
    score: int  # 0-100
    priority: str  # HIGH, MEDIUM-HIGH, MEDIUM, LOW-MEDIUM, LOW
    status: str  # NEWLY_ACTIONABLE, PERSISTENT, WINDOW_OPEN
    active_since: Optional[str] = None  # "Q2 2025 (6 months)"
    value_lever: str = ""
    economic_impact: str = ""
    why_now: List[WhyNowBullet] = field(default_factory=list)
    share_price_impact: Optional[str] = None  # "+8-12%"
    share_price_basis: Optional[str] = None  # "Based on 8 precedent transactions..."
    precedent_n: int = 0  # Number of similar cases
    tsr_p25: Optional[float] = None
    tsr_p50: Optional[float] = None
    tsr_p75: Optional[float] = None


