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


class InsightsGenerator:
    """
    Generates actionable insights from company analysis.

    This is the "brain" that turns raw signals into recommendations.
    """

    def __init__(self):
        """Initialize the insights generator."""
        self.snapshot = AsOfSnapshotBuilder()

    def generate_insights(
        self,
        gvkey: str,
        profile: Dict,
        similar_analysis: Dict,
        regime: Dict,
        peer_stats: Optional[Dict] = None,
    ) -> Dict:
        """
        Generate full insights package for a company.

        Parameters
        ----------
        gvkey : str
            Company identifier
        profile : dict
            State profile from SignalEngine
        similar_analysis : dict
            Results from ActionAnalyzer.analyze_similar_states()
        regime : dict
            Market regime from RegimeClassifier
        peer_stats : dict, optional
            Peer comparison statistics

        Returns
        -------
        dict with:
        - 'summary': One-line configuration summary
        - 'ideas': List of Idea objects, ranked
        - 'decision_table': If/then decision rules
        - 'constraints': Binding constraints
        - 'timing': Timing posture (urgent/opportunistic/patient)
        """
        # Extract key data
        signals = profile['signals']
        composite = profile['composite_score']
        action_dist = similar_analysis.get('action_distribution', {})
        n_similar = similar_analysis.get('n_similar', 0)

        # Generate insights
        summary = self._generate_summary(signals, action_dist, n_similar)
        ideas = self._generate_ideas(signals, action_dist, similar_analysis, regime)
        decision_table = self._generate_decision_table(signals, regime)
        constraints = self._identify_constraints(signals, profile)
        timing = self._assess_timing(signals, regime)

        return {
            'summary': summary,
            'ideas': ideas,
            'decision_table': decision_table,
            'constraints': constraints,
            'timing': timing,
            'n_similar': n_similar,
        }

    def _generate_summary(
        self,
        signals: Dict,
        action_dist: Dict,
        n_similar: int
    ) -> str:
        """
        Generate one-line configuration summary.

        Example: "Nike's configuration suggests M&A or buybacks.
                 14 of 18 similar companies acted within 24 months"
        """
        if not action_dist or n_similar == 0:
            return "Insufficient historical data for configuration analysis."

        # Find top 2 actions
        sorted_actions = sorted(action_dist.items(), key=lambda x: -x[1])[:2]

        if len(sorted_actions) >= 2:
            top_actions = f"{self._format_action_name(sorted_actions[0][0])} or {self._format_action_name(sorted_actions[1][0])}"
        elif len(sorted_actions) == 1:
            top_actions = self._format_action_name(sorted_actions[0][0])
        else:
            return "Configuration analysis in progress."

        # Estimate how many "acted" (took a capital action vs status quo)
        active_pct = sum(
            pct for action, pct in action_dist.items()
            if action not in ['status_quo', 'no_action']
        )
        n_acted = int(n_similar * active_pct / 100)

        return f"Configuration suggests {top_actions}. {n_acted} of {n_similar} similar companies took action."

    def _generate_ideas(
        self,
        signals: Dict,
        action_dist: Dict,
        similar_analysis: Dict,
        regime: Dict,
    ) -> List[Idea]:
        """
        Generate ranked capital allocation ideas.

        Each idea is scored based on:
        - Signal alignment (does the profile support this action?)
        - Historical precedent (how often did similar companies do this?)
        - Regime fit (is the market environment supportive?)
        - TSR outcome (what happened when others did this?)
        """
        ideas = []

        # Get TSR outcomes by action from similar cases
        tsr_by_action = self._compute_tsr_by_action(similar_analysis)

        # 1. Buy-Side M&A
        mna_score = self._score_mna_idea(signals, action_dist, regime, tsr_by_action)
        if mna_score['score'] >= 40:
            ideas.append(Idea(
                name="Buy-Side M&A",
                score=mna_score['score'],
                priority=self._score_to_priority(mna_score['score']),
                status=mna_score['status'],
                value_lever="Growth durability",
                economic_impact=mna_score['impact'],
                why_now=mna_score['why_now'],
                share_price_impact=mna_score.get('tsr_range'),
            ))

        # 2. Capital Allocation (Buybacks)
        buyback_score = self._score_buyback_idea(signals, action_dist, regime, tsr_by_action)
        if buyback_score['score'] >= 40:
            ideas.append(Idea(
                name="Capital Allocation",
                score=buyback_score['score'],
                priority=self._score_to_priority(buyback_score['score']),
                status=buyback_score['status'],
                value_lever="FCF per share",
                economic_impact=buyback_score['impact'],
                why_now=buyback_score['why_now'],
                share_price_impact=buyback_score.get('tsr_range'),
            ))

        # 3. Incremental Debt Issuance
        debt_score = self._score_debt_idea(signals, action_dist, regime, tsr_by_action)
        if debt_score['score'] >= 40:
            ideas.append(Idea(
                name="Incremental Debt Issuance",
                score=debt_score['score'],
                priority=self._score_to_priority(debt_score['score']),
                status=debt_score['status'],
                value_lever="Strategic flexibility",
                economic_impact=debt_score['impact'],
                why_now=debt_score['why_now'],
            ))

        # 4. Dividend Action
        div_score = self._score_dividend_idea(signals, action_dist, regime, tsr_by_action)
        if div_score['score'] >= 40:
            ideas.append(Idea(
                name="Dividend Policy",
                score=div_score['score'],
                priority=self._score_to_priority(div_score['score']),
                status=div_score['status'],
                value_lever="Shareholder yield",
                economic_impact=div_score['impact'],
                why_now=div_score['why_now'],
            ))

        # Sort by score descending
        ideas.sort(key=lambda x: -x.score)

        return ideas

    def _score_mna_idea(
        self,
        signals: Dict,
        action_dist: Dict,
        regime: Dict,
        tsr_by_action: Dict,
    ) -> Dict:
        """Score the M&A idea based on signals and precedent."""
        score = 50  # Start neutral
        why_now = []
        status = "PERSISTENT"

        # Balance sheet optionality - key for M&A
        bs_score = signals.get('balance_sheet_optionality', {}).get('score', 50)
        if bs_score >= 70:
            score += 15
            why_now.append("Strong balance sheet with capacity for acquisitions")
        elif bs_score <= 30:
            score -= 20

        # Valuation - can we afford targets?
        val_score = signals.get('valuation_dislocation', {}).get('score', 50)
        if val_score >= 60:
            score += 10
            why_now.append("Valuation provides acquisition currency advantage")

        # Size factor - larger companies can do M&A more easily
        size_score = signals.get('size_factor', {}).get('score', 50)
        if size_score >= 60:
            score += 5

        # Historical precedent
        mna_pct = sum(
            pct for action, pct in action_dist.items()
            if 'acquisition' in action.lower() or 'mna' in action.lower() or action in ['Takeover', 'LBO']
        )
        if mna_pct >= 30:
            score += 15
            status = "NEWLY_ACTIONABLE"
            why_now.append(f"{mna_pct:.0f}% of similar companies pursued M&A")
        elif mna_pct >= 15:
            score += 8

        # Market regime
        if regime.get('regime') == 'LOOSE':
            score += 10
            why_now.append("Favorable credit environment for deal financing")
            status = "WINDOW_OPEN"
        elif regime.get('regime') == 'TIGHT':
            score -= 10

        # TSR outcome bonus
        mna_tsr = tsr_by_action.get('acquisition', tsr_by_action.get('acquired'))
        if mna_tsr and mna_tsr > 5:
            score += 5
            why_now.append(f"Similar M&A transactions delivered +{mna_tsr:.0f}% median TSR")

        # Cap score
        score = max(0, min(100, score))

        return {
            'score': score,
            'status': status,
            'impact': "+100-200bps revenue growth equivalent",
            'why_now': why_now if why_now else ["Strategic optionality available"],
            'tsr_range': f"+{max(0, mna_tsr-5):.0f}-{mna_tsr+5:.0f}%" if mna_tsr else None,
        }

    def _score_buyback_idea(
        self,
        signals: Dict,
        action_dist: Dict,
        regime: Dict,
        tsr_by_action: Dict,
    ) -> Dict:
        """Score the buyback/capital allocation idea."""
        score = 50
        why_now = []
        status = "PERSISTENT"

        # Balance sheet - need cash/capacity
        bs_score = signals.get('balance_sheet_optionality', {}).get('score', 50)
        if bs_score >= 60:
            score += 10
            why_now.append("Cash position supports capital return")

        # Valuation - buybacks work best when undervalued
        val_score = signals.get('valuation_dislocation', {}).get('score', 50)
        if val_score <= 40:  # Undervalued
            score += 20
            why_now.append("Stock undervaluation makes buybacks accretive")
            status = "NEWLY_ACTIONABLE"
        elif val_score >= 70:  # Overvalued
            score -= 15

        # Growth - buybacks for slow-growth companies
        growth_score = signals.get('growth_momentum', {}).get('score', 50)
        if growth_score <= 40:
            score += 10
            why_now.append("Limited organic growth opportunities favor capital return")

        # Historical precedent
        buyback_pct = action_dist.get('buyback', 0)
        if buyback_pct >= 30:
            score += 10
            why_now.append(f"{buyback_pct:.0f}% of similar companies executed buybacks")

        # TSR outcome
        buyback_tsr = tsr_by_action.get('buyback')
        if buyback_tsr and buyback_tsr > 0:
            score += 5

        score = max(0, min(100, score))

        return {
            'score': score,
            'status': status,
            'impact': "3-4% annual FCF/share accretion",
            'why_now': why_now if why_now else ["Capital return program available"],
            'tsr_range': f"+{max(0, buyback_tsr-3):.0f}-{buyback_tsr+3:.0f}%" if buyback_tsr else None,
        }

    def _score_debt_idea(
        self,
        signals: Dict,
        action_dist: Dict,
        regime: Dict,
        tsr_by_action: Dict,
    ) -> Dict:
        """Score the debt issuance idea."""
        score = 40  # Start slightly lower
        why_now = []
        status = "PERSISTENT"

        # Balance sheet - low leverage = capacity
        bs_score = signals.get('balance_sheet_optionality', {}).get('score', 50)
        if bs_score >= 70:
            score += 15
            why_now.append("Underutilized leverage capacity available")
        elif bs_score <= 30:
            score -= 20  # Already levered

        # Refinancing pressure - opportunity if debt maturing
        refi_score = signals.get('refinancing_pressure', {}).get('score', 50)
        if refi_score <= 40:
            score += 10
            why_now.append("Near-term maturities create refinancing opportunity")

        # Market regime - critical for debt
        if regime.get('regime') == 'LOOSE':
            score += 20
            why_now.append("Credit spreads at favorable levels")
            status = "WINDOW_OPEN"
        elif regime.get('regime') == 'TIGHT':
            score -= 25

        score = max(0, min(100, score))

        return {
            'score': score,
            'status': status,
            'impact': "$2-5B dry powder for opportunistic moves",
            'why_now': why_now if why_now else ["Debt capacity available"],
        }

    def _score_dividend_idea(
        self,
        signals: Dict,
        action_dist: Dict,
        regime: Dict,
        tsr_by_action: Dict,
    ) -> Dict:
        """Score dividend action ideas."""
        score = 45
        why_now = []
        status = "PERSISTENT"

        # Balance sheet
        bs_score = signals.get('balance_sheet_optionality', {}).get('score', 50)
        if bs_score >= 60:
            score += 10

        # Margin trend - stable margins support dividends
        margin_score = signals.get('margin_trend', {}).get('score', 50)
        if margin_score >= 60:
            score += 10
            why_now.append("Stable margins support dividend commitment")

        # Historical precedent
        div_increase_pct = action_dist.get('dividend_increase', 0)
        div_initiate_pct = action_dist.get('dividend_initiate', 0)
        total_div_pct = div_increase_pct + div_initiate_pct

        if total_div_pct >= 20:
            score += 10
            why_now.append(f"{total_div_pct:.0f}% of similar companies increased/initiated dividends")

        # TSR outcome
        div_tsr = tsr_by_action.get('dividend_increase', tsr_by_action.get('dividend_initiate'))
        if div_tsr and div_tsr > 0:
            score += 5

        score = max(0, min(100, score))

        return {
            'score': score,
            'status': status,
            'impact': "2-3% yield enhancement",
            'why_now': why_now if why_now else ["Dividend policy review warranted"],
        }

    def _compute_tsr_by_action(self, similar_analysis: Dict) -> Dict[str, float]:
        """Compute median TSR by action type from similar cases."""
        cases = similar_analysis.get('similar_cases', [])
        if not cases:
            return {}

        df = pd.DataFrame(cases)
        if 'tsr_12m' not in df.columns or 'action_type' not in df.columns:
            return {}

        return df.groupby('action_type')['tsr_12m'].median().to_dict()

    def _generate_decision_table(self, signals: Dict, regime: Dict) -> List[Dict]:
        """
        Generate if/then decision rules.

        Example:
        IF growth <4% + multiple stays premium → Buybacks
        IF credit spreads tighten → M&A Advisory (window strengthens)
        """
        rules = []

        # Growth-based rules
        growth_score = signals.get('growth_momentum', {}).get('score', 50)
        val_score = signals.get('valuation_dislocation', {}).get('score', 50)

        if growth_score <= 40:
            if val_score >= 60:
                rules.append({
                    'condition': "Growth slows + valuation premium persists",
                    'action': "Buybacks",
                    'note': None,
                })
            else:
                rules.append({
                    'condition': "Growth slows + valuation compresses",
                    'action': "Buybacks",
                    'note': "(more attractive)",
                })

        # Credit-based rules
        if regime.get('regime') == 'LOOSE':
            rules.append({
                'condition': "Credit spreads remain tight",
                'action': "M&A Advisory",
                'note': "(window strengthens)",
            })
        elif regime.get('regime') == 'TIGHT':
            rules.append({
                'condition': "Credit spreads widen further",
                'action': "Defensive positioning",
                'note': "(preserve optionality)",
            })

        # Margin-based rules
        margin_score = signals.get('margin_trend', {}).get('score', 50)
        if margin_score >= 60:
            rules.append({
                'condition': "Margins stabilize at current levels",
                'action': "Reinvestment",
                'note': "(M&A urgency fades)",
            })
        elif margin_score <= 40:
            rules.append({
                'condition': "Margin pressure continues",
                'action': "Cost rationalization",
                'note': "(divestitures possible)",
            })

        return rules

    def _identify_constraints(self, signals: Dict, profile: Dict) -> List[Dict]:
        """Identify binding constraints that limit strategic options."""
        constraints = []

        # Leverage constraint
        bs_score = signals.get('balance_sheet_optionality', {}).get('score', 50)
        if bs_score <= 30:
            constraints.append({
                'name': "Leverage limits acquisition capacity",
                'severity': "HIGH",
                'implication': "Debt reduction or equity financing required for large M&A",
            })

        # Valuation constraint
        val_score = signals.get('valuation_dislocation', {}).get('score', 50)
        if val_score <= 30:
            constraints.append({
                'name': "Valuation limits equity currency",
                'severity': "MEDIUM",
                'implication': "Cash or debt financing preferred for M&A",
            })

        # Growth constraint
        growth_score = signals.get('growth_momentum', {}).get('score', 50)
        if growth_score <= 30:
            constraints.append({
                'name': "Growth narrative must be addressed",
                'severity': "MEDIUM",
                'implication': "FCF deployment must reinforce growth story",
            })

        # Refinancing constraint
        refi_score = signals.get('refinancing_pressure', {}).get('score', 50)
        if refi_score <= 30:
            constraints.append({
                'name': "Near-term debt maturities",
                'severity': "HIGH",
                'implication': "Refinancing takes priority over new capital deployment",
            })

        return constraints

    def _assess_timing(self, signals: Dict, regime: Dict) -> Dict:
        """Assess timing posture for action."""
        # Base timing on regime and signals
        regime_name = regime.get('regime', 'SELECTIVE')
        bs_score = signals.get('balance_sheet_optionality', {}).get('score', 50)
        refi_score = signals.get('refinancing_pressure', {}).get('score', 50)

        if refi_score <= 30:
            return {
                'posture': "Urgent",
                'description': "Near-term maturities require action",
            }
        elif regime_name == 'LOOSE' and bs_score >= 60:
            return {
                'posture': "Opportunistic",
                'description': "Favorable window, not urgent",
            }
        elif regime_name == 'TIGHT':
            return {
                'posture': "Patient",
                'description': "Wait for better entry point",
            }
        else:
            return {
                'posture': "Opportunistic",
                'description': "Selective action warranted",
            }

    def _score_to_priority(self, score: int) -> str:
        """Convert numeric score to priority label."""
        if score >= 80:
            return "HIGH"
        elif score >= 65:
            return "MEDIUM-HIGH"
        elif score >= 50:
            return "MEDIUM"
        elif score >= 35:
            return "LOW-MEDIUM"
        else:
            return "LOW"

    def _format_action_name(self, action: str) -> str:
        """Format action type for display."""
        name_map = {
            'buyback': 'buybacks',
            'acquisition': 'M&A',
            'acquired': 'M&A',
            'acquisition_merger': 'M&A',
            'dividend_increase': 'dividend increases',
            'dividend_cut': 'dividend adjustments',
            'dividend_initiate': 'dividend initiation',
            'Takeover': 'M&A',
            'LBO': 'M&A',
            'debt_refinance': 'debt refinancing',
        }
        return name_map.get(action, action.replace('_', ' '))


class PeerAnalyzer:
    """
    Compares company to sector peers.

    Generates the peer diagnostics table from the mock.
    """

    def __init__(self):
        self.snapshot = AsOfSnapshotBuilder()

    def get_peer_comparison(
        self,
        gvkey: str,
        as_of_date: str,
        n_peers: int = 6,
    ) -> Dict:
        """
        Get peer comparison table.

        Returns metrics for the company and top peers in the same sector.
        """
        # Get company data
        company_data = self.snapshot.get_snapshot(gvkey, as_of_date, lookback_quarters=4)
        if company_data is None or len(company_data) == 0:
            return {'company': None, 'peers': [], 'median': None}

        company = company_data.iloc[0]
        sic = company.get('sic')

        if pd.isna(sic):
            return {'company': self._extract_metrics(company), 'peers': [], 'median': None}

        # Get peer universe (same 2-digit SIC)
        sic_2digit = str(int(sic))[:2]
        universe = self.snapshot.get_universe_snapshot(as_of_date, min_assets=100)

        if universe is None or len(universe) == 0:
            return {'company': self._extract_metrics(company), 'peers': [], 'median': None}

        # Filter to same sector
        universe['sic_2'] = universe['sic'].apply(
            lambda x: str(int(x))[:2] if pd.notna(x) else None
        )
        peers = universe[
            (universe['sic_2'] == sic_2digit) &
            (universe['gvkey'] != gvkey)
        ].copy()

        if len(peers) == 0:
            return {'company': self._extract_metrics(company), 'peers': [], 'median': None}

        # Sort by revenue and take top N
        peers = peers.nlargest(n_peers, 'revtq')

        # Extract metrics for each peer
        peer_metrics = []
        for _, peer in peers.iterrows():
            metrics = self._extract_metrics(peer)
            if metrics:
                peer_metrics.append(metrics)

        # Compute median
        if peer_metrics:
            median = self._compute_median(peer_metrics)
        else:
            median = None

        return {
            'company': self._extract_metrics(company),
            'peers': peer_metrics,
            'median': median,
        }

    def _extract_metrics(self, row) -> Optional[Dict]:
        """Extract key metrics from a data row."""
        try:
            # Revenue (annualized from quarterly)
            revtq = float(row.get('revtq', 0) or 0)
            revenue = revtq * 4 / 1000  # Convert to $B

            # Get prior year revenue for growth (if available)
            # For now, use a placeholder
            growth = None

            # Gross margin
            gross_margin = None
            if row.get('revtq') and row.get('cogsq'):
                revtq = float(row.get('revtq', 0) or 0)
                cogsq = float(row.get('cogsq', 0) or 0)
                if revtq > 0:
                    gross_margin = (revtq - cogsq) / revtq * 100

            # FCF margin (operating income / revenue as proxy)
            fcf_margin = None
            if row.get('oibdpq') and row.get('revtq'):
                oibdpq = float(row.get('oibdpq', 0) or 0)
                revtq = float(row.get('revtq', 0) or 0)
                if revtq > 0:
                    fcf_margin = oibdpq / revtq * 100

            # EV/EBITDA
            ev_ebitda = None
            # Would need market cap data for this

            # Net leverage
            net_leverage = None
            cash = float(row.get('cheq', 0) or 0)
            debt = float(row.get('dlttq', 0) or 0) + float(row.get('dlcq', 0) or 0)
            ebitda = float(row.get('oibdpq', 0) or 0) * 4  # Annualize
            if ebitda > 0:
                net_leverage = (debt - cash) / ebitda

            return {
                'name': row.get('conm', 'Unknown'),
                'ticker': row.get('tic', ''),
                'revenue': revenue,
                'growth': growth,
                'gross_margin': gross_margin,
                'fcf_margin': fcf_margin,
                'ev_ebitda': ev_ebitda,
                'net_leverage': net_leverage,
            }
        except Exception:
            return None

    def _compute_median(self, peer_metrics: List[Dict]) -> Dict:
        """Compute median values across peers."""
        df = pd.DataFrame(peer_metrics)

        return {
            'name': 'Peer Median',
            'ticker': '',
            'revenue': df['revenue'].median(),
            'growth': df['growth'].median() if 'growth' in df else None,
            'gross_margin': df['gross_margin'].median(),
            'fcf_margin': df['fcf_margin'].median(),
            'ev_ebitda': df['ev_ebitda'].median() if 'ev_ebitda' in df else None,
            'net_leverage': df['net_leverage'].median(),
        }
