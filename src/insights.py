"""
Insights Generator
==================
Generates actionable insights from state profiles and historical analysis.

This module produces:
1. "Why Now" bullets - specific reasons supporting each action type
2. Ideas Rankings - scored recommendations based on signals + precedent
3. Decision Logic - if/then rules for capital allocation
4. Peer Diagnostics - comparison to sector peers

No ML required - this is all rule-based logic derived from:
- Company's current state profile (7 signals)
- Historical precedent (what similar companies did)
- Market regime
- Peer comparison
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .snapshot import AsOfSnapshotBuilder, DATA_DIR


@dataclass
class Idea:
    """A capital allocation idea with supporting evidence."""
    name: str
    score: int  # 0-100
    priority: str  # HIGH, MEDIUM-HIGH, MEDIUM, LOW-MEDIUM, LOW
    status: str  # NEWLY_ACTIONABLE, PERSISTENT, WINDOW_OPEN, etc.
    value_lever: str  # What value it creates
    economic_impact: str  # Quantified impact
    why_now: List[str]  # Bullet points supporting the idea
    share_price_impact: Optional[str] = None  # Expected TSR range


