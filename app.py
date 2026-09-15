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


def main():
    # Load components
    with st.spinner("Loading data..."):
        snapshot, engine, regime_classifier, actions_db, analyzer, evidence_builder = load_components()

    # Sidebar
    st.sidebar.header("Query Parameters")

    as_of_date = st.sidebar.date_input(
        "As-of Date",
        value=datetime(2023, 6, 30),
        min_value=datetime(2010, 1, 1),
        max_value=datetime(2024, 12, 31),
    )
    as_of_str = as_of_date.strftime('%Y-%m-%d')

    with st.spinner("Loading universe..."):
        universe = snapshot.get_universe_snapshot(as_of_str, min_assets=500, min_revenue=100)

    if len(universe) == 0:
        st.error("No companies found for this date.")
        return

    company_options = universe[['gvkey', 'tic', 'conm']].copy()
    company_options['display'] = company_options['tic'].fillna('') + ' - ' + company_options['conm']
    company_options = company_options.sort_values('display')

    selected_display = st.sidebar.selectbox(
        "Select Company",
        options=company_options['display'].tolist(),
        index=0,
    )

    selected_row = company_options[company_options['display'] == selected_display].iloc[0]
    gvkey = selected_row['gvkey']
    ticker = selected_row['tic']
    company_name = selected_row['conm']

    similarity_threshold = st.sidebar.slider(
        "Similarity Threshold",
        min_value=0.70,
        max_value=0.99,
        value=0.85,
        step=0.01,
    )

    st.sidebar.markdown("---")
    use_sector_filter = st.sidebar.checkbox("Match within same sector", value=False)
    use_action_weights = st.sidebar.checkbox("Apply action weighting", value=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Universe:** {len(universe):,} companies")

    # Live Market Data
    st.sidebar.markdown("---")
    st.sidebar.header("Live Market Data")
    ric_guess = guess_ric_from_ticker(ticker)
    ticker_str = "" if (isinstance(ticker, float) and pd.isna(ticker)) or ticker is None else str(ticker)
    ric_default = ric_guess or (f"{ticker_str}.OQ" if ticker_str else "")
    ric_input = st.sidebar.text_input("RIC (Refinitiv)", value=ric_default)

    if st.sidebar.button("Refresh Live Quote"):
        fetch_quote.clear()
        fetch_intraday.clear()

    render_live_quote(ric_input)

    # Main content
    st.markdown("# Axiom")

    # Build EvidencePack
    with st.spinner("Building evidence pack..."):
        # Compute profile
        profile = engine.compute_state_profile(gvkey, as_of_str)
        if profile is None:
            st.error("Could not compute state profile.")
            return

        # Get regime
        regime_result = regime_classifier.classify_regime(as_of_str)

        # Get similar cases
        company_sector = analyzer.get_company_sector(gvkey)
        sector_filter = company_sector if use_sector_filter else None

        similar_result = analyzer.analyze_similar_states(
            profile,
            min_similarity=similarity_threshold,
            sector_filter=sector_filter,
            weight_actions=use_action_weights,
        )

        # Build evidence pack
        pack = evidence_builder.build(
            gvkey=gvkey,
            company_name=company_name,
            as_of_date=as_of_str,
            signal_profile=profile,
            regime_result=regime_result,
            similar_cases=similar_result.get('similar_cases', []),
        )

    # Tabs
    tab_overview, tab_logic, tab_evidence = st.tabs(["Overview", "Logic", "Evidence"])

    # ============== OVERVIEW TAB ==============
    with tab_overview:
        render_metrics_header(pack)

        # Summary
        n_similar = similar_result.get('n_similar', 0)
        top_actions = [card.action_human for card in pack.action_cards[:2]]
        summary = f"{company_name}'s configuration suggests {' or '.join(top_actions)}. {n_similar} similar companies in historical data."
        st.info(summary)

        # Top Ideas
        st.markdown("### TOP IDEAS")

        for i, card in enumerate(pack.action_cards[:3], 1):
            render_action_card(card, i)

        # Historical Precedent
        st.markdown("---")
        render_historical_precedent(pack)

    # ============== LOGIC TAB ==============
    with tab_logic:
        col1, col2 = st.columns([1, 1])

        with col1:
            render_decision_table(pack)

            st.markdown("---")
            st.markdown("**ALTERNATIVE INTERPRETATIONS**")
            st.caption("Credible paths evaluated but deprioritized based on configuration.")

            # Sensitivity box
            st.markdown("""
            <div style="background: #fef3c7; border-left: 4px solid #f59e0b; padding: 12px; margin-top: 12px;">
                <div style="font-weight: 600; color: #92400e;">Sensitivity & Dependencies</div>
                <ul style="color: #92400e; margin: 8px 0; padding-left: 20px;">
                    <li>Margin recovery → inventory normalization</li>
                    <li>M&A viability → target availability at reasonable multiples</li>
                    <li>Valuation → narrative consistency, not growth acceleration</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("**BINDING CONSTRAINTS**")
            if pack.binding_constraints:
                for constraint in pack.binding_constraints:
                    with st.expander(constraint['name']):
                        st.markdown(f"**Severity:** {constraint['severity']}")
                        st.markdown(f"**Implication:** {constraint['implication']}")
            else:
                st.caption("No significant constraints identified.")

        with col2:
            st.markdown("**IDEAS RANKED**")
            for i, card in enumerate(pack.action_cards, 1):
                score_color = '#22c55e' if card.recommendation_score >= 70 else '#f59e0b' if card.recommendation_score >= 50 else '#ef4444'

                st.markdown(f"""
                <div style="display: flex; align-items: center; gap: 12px; padding: 12px; border: 1px solid #e5e7eb; border-radius: 8px; margin-bottom: 8px;">
                    <div>
                        <span style="font-size: 1.8em; font-weight: bold; color: {score_color};">{card.recommendation_score}</span>
                        <span style="color: #6b7280; font-size: 0.8em;">/100</span>
                    </div>
                    <div>
                        <span style="background: #e5e7eb; padding: 2px 8px; border-radius: 4px; font-size: 0.7em;">#{i} {card.recommendation_label}</span>
                        <div style="font-weight: 500; margin-top: 4px;">{card.action_human}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.caption(f"VALUE: {card.value_lever} | IMPACT: {card.economic_impact}")

            st.markdown("---")
            st.markdown("**TIMING POSTURE**")
            timing = pack.timing_posture
            posture_icon = "⚡" if timing['posture'] == "Urgent" else "⏱️" if timing['posture'] == "Opportunistic" else "⏳"
            st.markdown(f"### {posture_icon} {timing['posture']}")
            st.caption(timing['description'])

    # ============== EVIDENCE TAB ==============
    with tab_evidence:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("**SIGNAL ANALYSIS**")
            render_signal_explanations(pack)

            st.markdown("---")
            st.markdown("**DRIVERS**")

            drivers_data = []
            for sig_name, sig_detail in pack.state_summary.signals.items():
                trend = "→ Stable" if 40 <= sig_detail.score <= 60 else "↑ Strong" if sig_detail.score > 60 else "↓ Weak"
                drivers_data.append({
                    'Signal': sig_name.replace('_', ' ').title(),
                    'Value': sig_detail.value.upper(),
                    'Score': f"{sig_detail.score:.0f}/100",
                    'Trend': trend,
                })

            df = pd.DataFrame(drivers_data)
            st.dataframe(df, use_container_width=True, hide_index=True)

        with col2:
            st.markdown("**STATE PROFILE**")
            fig = render_signal_chart(pack)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            st.markdown("**MARKET REGIME**")
            regime_colors = {'LOOSE': '🟢', 'SELECTIVE': '🟡', 'TIGHT': '🔴'}
            st.markdown(f"### {regime_colors.get(pack.regime.regime_id, '')} {pack.regime.regime_id}")
            st.caption(pack.regime.characteristics.get('deal_activity', ''))

            st.markdown("---")
            st.markdown("**CAPITAL POSTURE**")
            st.markdown("**Funding Hierarchy:**")

            bs_score = pack.state_summary.signals.get('balance_sheet_optionality')
            if bs_score and bs_score.score >= 60:
                st.markdown("1. **Free Cash Flow** — Primary")
                st.progress(0.9)
                st.markdown("2. **Incremental Debt** — Secondary")
                st.progress(0.6)
                st.markdown("3. **Equity** — Conditional")
                st.progress(0.2)
            else:
                st.markdown("1. **Debt** — Primary")
                st.progress(0.7)
                st.markdown("2. **Equity** — Secondary")
                st.progress(0.5)
                st.markdown("3. **FCF** — Limited")
                st.progress(0.3)

    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.caption(
            f"Axiom V1 | EvidencePack ID: {pack.pack_id[:8]}... | "
            f"Profiles: {len(analyzer.deal_profiles) if analyzer.deal_profiles is not None else 0:,}"
        )
    with col2:
        st.caption("Export: PDF | PowerPoint")
    with col3:
        st.caption("Share")


if __name__ == "__main__":
    main()
