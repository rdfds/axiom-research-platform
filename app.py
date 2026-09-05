"""
Axiom V1 - Decision Intelligence for Capital Allocation
========================================================
Full EvidencePack-based UI.

Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import plotly.graph_objects as go

# Must be first Streamlit command
st.set_page_config(
    page_title="Axiom",
    page_icon="📊",
    layout="wide",
)

# Import modules
import sys
sys.path.insert(0, '.')

from src.snapshot import AsOfSnapshotBuilder
from src.signals import SignalEngine
from src.regimes import RegimeClassifier
from src.corporate_actions import CorporateActionsDB, ActionAnalyzer
from src.evidence_pack import EvidencePackBuilder, EvidencePack, ActionCard
from src.market_data import MarketDataProvider


@st.cache_resource
def load_components():
    """Load all components (cached)."""
    snapshot = AsOfSnapshotBuilder()
    engine = SignalEngine(snapshot)
    regime = RegimeClassifier()
    actions_db = CorporateActionsDB()
    analyzer = ActionAnalyzer(actions_db)
    evidence_builder = EvidencePackBuilder()
    return snapshot, engine, regime, actions_db, analyzer, evidence_builder


@st.cache_resource
def get_market_provider():
    """Create a single Refinitiv market data provider."""
    return MarketDataProvider()


@st.cache_data(ttl=60)
def fetch_quote(ric: str) -> Dict:
    return get_market_provider().get_quote(ric)


@st.cache_data(ttl=60)
def fetch_intraday(ric: str) -> Dict:
    return get_market_provider().get_intraday_quote(ric)


@st.cache_data
def load_ric_map():
    """Load RIC map for ticker -> RIC suggestions."""
    path = Path("data/refinitiv/ric_to_cusip_map.parquet")
    if not path.exists():
        return pd.DataFrame(columns=["ric", "ticker"])
    df = pd.read_parquet(path, columns=["ric", "ticker"])
    df["ticker"] = df["ticker"].astype("string").str.upper().str.strip()
    return df


