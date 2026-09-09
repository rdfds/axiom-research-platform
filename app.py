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


def pick_best_ric(candidates: List[str]) -> Optional[str]:
    def rank_ric(ric: str) :
        if not isinstance(ric, str):
            return 99
        ric = ric.upper()
        if ric.endswith(".N"):
            return 0
        if ric.endswith(".OQ"):
            return 1
        if ric.endswith(".Q"):
            return 2
        if ric.endswith(".A"):
            return 3
        if ric.endswith(".K"):
            return 4
        if ric.endswith(".P"):
            return 5
        return 9

    if not candidates:
        return None
    return sorted(candidates, key=rank_ric)[0]


def guess_ric_from_ticker(ticker: Optional[str]) -> Optional[str]:
    if ticker is None:
        return None
    if pd.isna(ticker):
        return None
    df = load_ric_map()
    if df.empty:
        return None
    matches = df[df["ticker"] == str(ticker).upper()].copy()
    if matches.empty:
        return None
    return pick_best_ric(matches["ric"].dropna().unique().tolist())


def render_live_quote(ric: str):
    if not ric:
        st.sidebar.info("Enter a RIC to load live data.")
        return

    quote = fetch_quote(ric)
    intraday = fetch_intraday(ric)

    if quote.get("error") and intraday.get("error"):
        st.sidebar.warning(f"No live data for {ric}.")
        return

    price = intraday.get("last") or quote.get("price")
    change_pct = quote.get("change_pct")
    volume = intraday.get("volume") or quote.get("volume")

    st.sidebar.markdown("**Live Quote**")
    st.sidebar.metric("Last Price", f"{price:.2f}" if price is not None else "N/A")
    st.sidebar.metric("Change (1D)", f"{change_pct:+.2f}%" if change_pct is not None else "N/A")
    st.sidebar.metric("Volume", f"{int(volume):,}" if volume is not None else "N/A")

    st.sidebar.caption("Source: Refinitiv (RDP)")


