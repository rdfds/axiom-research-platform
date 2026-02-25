"""
Pull All Corporate Actions & Data from Refinitiv
================================================
Comprehensive data pull for Axiom V1.
Requires Refinitiv Workspace/Eikon terminal to be running.

Run with: python -u scripts/pull_refinitiv_all.py
"""

import os
import refinitiv.data as rd
import pandas as pd
from pathlib import Path
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).parent.parent / 'data' / 'refinitiv'
DATA_DIR.mkdir(exist_ok=True)

START_DATE = '2020-01-01'
END_DATE = '2025-12-31'
MNA_START_YEAR = int(os.getenv("MNA_START_YEAR", "2000"))
MNA_END_YEAR = int(os.getenv("MNA_END_YEAR", str(datetime.now().year)))
ONLY_MNA = os.getenv("ONLY_MNA", "0") == "1"
MNA_PROBE_FIELDS = os.getenv("MNA_PROBE_FIELDS", "1") == "1"
MNA_PROBE_MAX_YEARS = int(os.getenv("MNA_PROBE_MAX_YEARS", "6"))
MNA_SAVE_BY_YEAR = os.getenv("MNA_SAVE_BY_YEAR", "1") == "1"
MNA_SKIP_EXISTING = os.getenv("MNA_SKIP_EXISTING", "1") == "1"
MNA_FALLBACK_MONTHLY = os.getenv("MNA_FALLBACK_MONTHLY", "1") == "1"
MNA_MONTH_SKIP_EXISTING = os.getenv("MNA_MONTH_SKIP_EXISTING", "1") == "1"

MNA_YEARLY_DIR = DATA_DIR / "mna_yearly"
MNA_YEARLY_DIR.mkdir(parents=True, exist_ok=True)

MNA_FIELD_BASE = [
    "TR.MnADealValue(Scale=6)",
    "TR.MnAAnnDate",
    "TR.MnACompDate",
    "TR.MnAStatus",
    "TR.MnADealType",
    "TR.MnATargetNation",
    "TR.MnAAcquirorNation",
]

# Candidate fields to probe; only working ones will be used.
MNA_FIELD_CANDIDATES = [
    # Deal identifiers
    "TR.MnASDCDealNo",
    "TR.MnADealNo",

    # Core dates/status
    "TR.MnAAnnDate",
    "TR.MnACompDate",
    "TR.MnAStatus",
    "TR.MnADealType",

    # Value and terms
    "TR.MnADealValue(Scale=6)",
    "TR.MnADealValue",
    "TR.MnAPaymentType",
    "TR.MnAPctCash",
    "TR.MnAPctStock",
    "TR.MnAPremium1Day",
    "TR.MnAPremium1Week",
    "TR.MnAPremium4Week",
    "TR.MnATransactionNature",
    "TR.MnADealSynopsis",

    # Target identifiers
    "TR.MnATarget",
    "TR.MnATargetName",
    "TR.MnATargetTicker",
    "TR.MnATargetRIC",
    "TR.MnATargetPermID",
    "TR.MnATargetCUSIP",
    "TR.MnATargetISIN",

    # Acquiror identifiers
    "TR.MnAAcquiror",
    "TR.MnAAcquirorName",
    "TR.MnAAcquirorTicker",
    "TR.MnAAcquirorRIC",
    "TR.MnAAcquirorPermID",
    "TR.MnAAcquirorCUSIP",
    "TR.MnAAcquirorISIN",

    # Classification
    "TR.MnATargetNation",
    "TR.MnAAcquirorNation",
    "TR.MnATargetPrimarySICCode",
    "TR.MnAAcquirorPrimarySICCode",
    "TR.MnATargetGICSSubIndustry",
]

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def save_parquet(df, name):
    """Save dataframe to parquet with logging."""
    path = DATA_DIR / f'{name}.parquet'
    df.to_parquet(path, index=False)
    log(f"  Saved {len(df):,} rows to {path.name}")
    return path


def ensure_session() :
    """Verify Refinitiv Desktop/Workspace session is actually usable."""
    try:
        _ = rd.get_data(universe='0#.SPX', fields=['TR.CommonName'])
        return True
    except Exception as e:
        log(f"Refinitiv session check failed: {e}")
        log("Make sure Refinitiv Workspace/Desktop is running and you are logged in.")
        log("Then rerun this script in the same terminal.")
        return False


