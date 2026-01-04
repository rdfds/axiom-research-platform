"""
Link CIQ Key Developments to Compustat
======================================
The 39K CIQ events (dividends, equity offerings, divestitures) only have
company names in headlines. This script links them to Compustat gvkeys
so we can compute state profiles.

Approach:
1. Extract company names from CIQ headlines
2. Build lookup table from Compustat (name -> gvkey, ticker -> gvkey)
3. Match via exact ticker, then fuzzy company name

Usage:
  python scripts/10_link_ciq_to_compustat.py

Output: data/ciq_linked.parquet
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import re
from pathlib import Path
from datetime import datetime
from difflib import SequenceMatcher

from src.snapshot import DATA_DIR


def clean_company_name(name):
    """Normalize company name for matching."""
    if pd.isna(name):
        return None

    name = str(name).upper().strip()

    # Remove common suffixes
    suffixes = [
        ' INC', ' INCORPORATED', ' CORP', ' CORPORATION', ' CO', ' COMPANY',
        ' LTD', ' LIMITED', ' LLC', ' LP', ' PLC', ' SA', ' AG', ' NV',
        ' GROUP', ' HOLDINGS', ' HOLDING', ' INTERNATIONAL', ' INTL',
        ' & CO', ' AND CO', ' THE', ', THE',
    ]
    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[:-len(suffix)]

    # Remove punctuation
    name = re.sub(r'[^\w\s]', '', name)
    name = re.sub(r'\s+', ' ', name).strip()

    return name


