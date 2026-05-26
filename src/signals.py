"""
Signal Engine
=============
Computes the 5-7 interpretable signals that form the "state profile"
for V1's analog retrieval system.

V1 Signals:
1. Balance Sheet Optionality - Can this company act? (cash, debt capacity)
2. Growth Momentum - Is the business growing or shrinking?
3. Valuation Dislocation - Is it cheap/expensive vs history/peers?
4. Margin Trend - Are margins expanding or compressing?
5. Refinancing Pressure - Is there near-term debt to address?
6. Size Factor - Absolute scale matters for M&A
7. Asset Intensity - Capital structure and asset base

All signals are designed to be:
- Interpretable (a banker can explain them)
- Point-in-time (using rdq, not datadate)
- Deterministic (no ML, just math)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Union
from datetime import datetime

from .snapshot import AsOfSnapshotBuilder, DATA_DIR
from .asof_store import AsOfWarehouse


def safe_float(value, default=0):
    """Safely convert a value to float, handling pandas NA/NaN."""
    if pd.isna(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


