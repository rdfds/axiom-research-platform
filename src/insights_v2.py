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


