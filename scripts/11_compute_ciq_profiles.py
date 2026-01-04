"""
Compute State Profiles for Linked CIQ Events
=============================================
Now that we've linked 5,550 CIQ events to Compustat, compute their
state profiles. This adds dividends, divestitures, etc. to our database.

Usage:
  python scripts/11_compute_ciq_profiles.py

Output: data/ciq_profiles_all.parquet
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import re
from pathlib import Path
from datetime import datetime

from src.snapshot import AsOfSnapshotBuilder, DATA_DIR
from src.signals import SignalEngine


def classify_action(headline):
    """Classify CIQ headline into action type."""
    if pd.isna(headline):
        return 'other'

    h = headline.lower()

    if 'dividend' in h:
        if any(w in h for w in ['increase', 'raise', 'hike']):
            return 'dividend_increase'
        elif any(w in h for w in ['cut', 'reduce', 'lower']):
            return 'dividend_cut'
        elif any(w in h for w in ['suspend', 'eliminate', 'omit']):
            return 'dividend_suspend'
        else:
            return 'dividend'
    elif any(w in h for w in ['buyback', 'repurchase']):
        return 'buyback'
    elif any(w in h for w in ['acqui', 'merger', 'takeover', 'purchase of']):
        return 'acquisition'
    elif any(w in h for w in ['divest', 'sell', 'dispose', 'asset sale']):
        return 'divestiture'
    elif any(w in h for w in ['offering', 'ipo', 'secondary']):
        return 'equity_offering'
    elif 'spin' in h and any(w in h for w in ['off', 'out']):
        return 'spinoff'

    return 'other'


