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
    def rank_ric(ric: str) -> int:
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


def render_metrics_header(pack: EvidencePack):
    """Render the metrics header bar."""
    metrics = pack.company_metrics

    st.markdown(f"""
    <div style="padding: 12px 0; border-bottom: 1px solid #e5e7eb; margin-bottom: 16px;">
        <div style="display: flex; align-items: baseline; gap: 12px;">
            <span style="font-size: 1.3em; font-weight: 600;">{pack.company_name}</span>
            <span style="color: #6b7280;">As of {pack.as_of_time}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(6)

    with cols[0]:
        rev = metrics.get('revenue')
        st.metric("Revenue", f"${rev:.0f}B" if rev else "N/A")

    with cols[1]:
        growth = metrics.get('revenue_growth_yoy')
        st.metric("Growth", f"{growth:+.0f}%" if growth is not None else "N/A")

    with cols[2]:
        margin = metrics.get('ebitda_margin')
        st.metric("EBITDA Margin", f"{margin:.0f}%" if margin else "N/A")

    with cols[3]:
        fcf = metrics.get('fcf')
        st.metric("FCF", f"${fcf:.1f}B" if fcf else "N/A")

    with cols[4]:
        lev = metrics.get('net_leverage')
        st.metric("Leverage", f"{lev:.1f}x" if lev is not None else "N/A")

    with cols[5]:
        st.metric("EV/EBITDA", "N/A")  # Would need market cap


def render_action_card(card: ActionCard, index: int):
    """Render full action card with all evidence."""

    # Status colors
    status_colors = {
        'NEWLY_ACTIONABLE': '#22c55e',
        'PERSISTENT': '#3b82f6',
        'WINDOW_OPEN': '#f59e0b',
    }
    status_color = status_colors.get(card.status, '#6b7280')

    # Card header
    st.markdown(f"""
    <div style="border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px; margin-bottom: 16px; background: white;">
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
            <span style="background: {status_color}; color: white; padding: 2px 10px; border-radius: 4px; font-size: 0.75em; font-weight: 500; text-transform: uppercase;">{card.status.replace('_', ' ')}</span>
        </div>
        <div style="display: flex; align-items: baseline; gap: 8px;">
            <span style="color: #6b7280; font-size: 0.9em;">#{index}</span>
            <span style="font-size: 1.3em; font-weight: 600;">{card.action_human}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Why Now section
    with st.expander("**Why Now**", expanded=True):
        for bullet in card.why_now_bullets:
            text = bullet.get('text', '')
            source = bullet.get('source', '')

            # Add source indicator
            if source == 'fundamentals':
                icon = "📊"
            elif source == 'precedent':
                icon = "📈"
            elif source == 'regime':
                icon = "🌐"
            else:
                icon = "✓"

            st.markdown(f"{icon} {text}")

        # Active since (if available)
        # st.caption("Active since Q2 2025 (6 months)")

    # Conditions section
    col1, col2 = st.columns(2)

    with col1:
        if card.works_when:
            st.markdown("**Works When:**")
            for condition in card.works_when:
                st.markdown(f"<span style='color: #22c55e;'>✓</span> {condition}", unsafe_allow_html=True)

    with col2:
        if card.fails_when:
            st.markdown("**Fails When:**")
            for condition in card.fails_when:
                st.markdown(f"<span style='color: #ef4444;'>✗</span> {condition}", unsafe_allow_html=True)

    # Value / Impact row
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Value Lever**")
        st.caption(card.value_lever)

    with col2:
        st.markdown("**Economic Impact**")
        st.caption(card.economic_impact)

    with col3:
        st.markdown("**Share Price Impact**")
        if card.share_price_impact_range:
            st.markdown(f"<span style='color: #22c55e; font-weight: 600;'>{card.share_price_impact_range}</span>", unsafe_allow_html=True)
            if card.cohort and card.cohort.n >= 5:
                st.caption(f"Based on {card.cohort.n} precedent transactions")
        else:
            st.caption(card.share_price_impact_range or "Depends on execution")

    # Objections section (collapsed)
    if card.objections:
        with st.expander("**Objection Prep**"):
            for obj in card.objections:
                st.markdown(f"**Q: {obj.question}**")
                for point in obj.grounded_answer_points:
                    st.markdown(f"- {point}")


def render_historical_precedent(pack: EvidencePack):
    """Render historical precedent summary."""

    # Gather all cohorts
    all_cohorts = []
    for card in pack.action_cards:
        if card.cohort:
            all_cohorts.append(card.cohort)

    if not all_cohorts:
        return

    total_n = sum(c.n for c in all_cohorts)

    st.markdown(f"**HISTORICAL PRECEDENT** ({total_n} similar companies, 2010-2024)")

    cols = st.columns(min(len(all_cohorts) + 1, 4))

    for i, cohort in enumerate(all_cohorts[:3]):
        with cols[i]:
            tsr = cohort.outcomes.get('tsr_12m')
            if tsr:
                tsr_color = '#22c55e' if tsr.p50 and tsr.p50 > 0 else '#ef4444'
                tsr_str = f"+{tsr.p50:.0f}%" if tsr.p50 and tsr.p50 > 0 else f"{tsr.p50:.0f}%" if tsr.p50 else "N/A"
                pct = cohort.n / total_n * 100 if total_n > 0 else 0

                st.markdown(f"""
                <div style="background: #f9fafb; border-radius: 8px; padding: 12px; text-align: center;">
                    <div style="font-weight: 600; margin-bottom: 4px;">{cohort.action_human}</div>
                    <div style="color: #6b7280; font-size: 0.85em;">{pct:.0f}% did this → <span style="color: {tsr_color}">{tsr_str} avg return</span></div>
                </div>
                """, unsafe_allow_html=True)

    st.caption("See Logic tab for full precedent analysis")


def render_signal_chart(pack: EvidencePack):
    """Render radar chart of signals."""
    signals = pack.state_summary.signals

    categories = [s.replace('_', ' ').title() for s in signals.keys()]
    values = [signals[s].score for s in signals.keys()]

    # Close the radar chart
    categories = categories + [categories[0]]
    values = values + [values[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='State Profile',
        line_color='#1f77b4',
        fillcolor='rgba(31, 119, 180, 0.3)',
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        height=350,
        margin=dict(l=60, r=60, t=30, b=30),
    )

    return fig


def render_decision_table(pack: EvidencePack):
    """Render decision table based on signals."""

    st.markdown("**DECISION TABLE**")
    st.markdown('<div style="color: #6b7280; font-size: 0.85em; margin-bottom: 12px;">IF THIS HAPPENS → DOMINANT IDEA</div>', unsafe_allow_html=True)

    signals = pack.state_summary.signals

    rules = []

    # Growth-based rules
    growth_sig = signals.get('growth_momentum')
    val_sig = signals.get('valuation_dislocation')

    if growth_sig and growth_sig.score <= 40:
        if val_sig and val_sig.score >= 60:
            rules.append(("Growth slows + valuation premium persists", "Buybacks", None))
        else:
            rules.append(("Growth slows + valuation compresses", "Buybacks", "(more attractive)"))

    # Regime-based rules
    if pack.regime.regime_id == 'LOOSE':
        rules.append(("Credit spreads remain tight", "M&A Advisory", "(window strengthens)"))
    elif pack.regime.regime_id == 'TIGHT':
        rules.append(("Credit spreads widen further", "Defensive positioning", "(preserve optionality)"))

    # Margin-based rules
    margin_sig = signals.get('margin_trend')
    if margin_sig:
        if margin_sig.score >= 60:
            rules.append(("Margins stabilize at current levels", "Reinvestment", "(M&A urgency fades)"))
        elif margin_sig.score <= 40:
            rules.append(("Margin pressure continues", "Cost rationalization", "(divestitures possible)"))

    for condition, action, note in rules:
        note_str = f' <span style="color: #6b7280;">{note}</span>' if note else ''
        st.markdown(f"- {condition} → **{action}**{note_str}", unsafe_allow_html=True)


def render_signal_explanations(pack: EvidencePack):
    """Render signal explanations with drivers."""

    for sig_name, sig_detail in pack.state_summary.signals.items():
        score = sig_detail.score
        color = "#22c55e" if score >= 70 else "#f59e0b" if score >= 40 else "#ef4444"

        with st.expander(f"**{sig_name.replace('_', ' ').title()}** — {sig_detail.value.upper()} ({score:.0f}/100)"):
            st.markdown(sig_detail.explanation)

            if sig_detail.drivers:
                st.markdown("**Key Drivers:**")
                for driver in sig_detail.drivers:
                    direction_icon = "↑" if driver.direction == 'positive' else "↓"
                    st.markdown(f"- {direction_icon} {driver.human_label}")


