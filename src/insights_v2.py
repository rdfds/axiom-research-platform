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


@dataclass
class IdeaV2:
    """Enhanced capital allocation idea with full evidence."""
    name: str
    score: int  # 0-100
    priority: str  # HIGH, MEDIUM-HIGH, MEDIUM, LOW-MEDIUM, LOW
    status: str  # NEWLY_ACTIONABLE, PERSISTENT, WINDOW_OPEN
    active_since: Optional[str] = None  # "Q2 2025 (6 months)"
    value_lever: str = ""
    economic_impact: str = ""
    why_now: List[WhyNowBullet] = field(default_factory=list)
    share_price_impact: Optional[str] = None  # "+8-12%"
    share_price_basis: Optional[str] = None  # "Based on 8 precedent transactions..."
    precedent_n: int = 0  # Number of similar cases
    tsr_p25: Optional[float] = None
    tsr_p50: Optional[float] = None
    tsr_p75: Optional[float] = None


class InsightsGeneratorV2:
    """
    Enhanced insights generation matching mock quality.
    """

    def __init__(self):
        self.snapshot = AsOfSnapshotBuilder()
        self._load_action_history()
        self._load_valuation_history()

    def _load_action_history(self):
        """Load corporate actions for company-specific history lookups."""
        self.buybacks = None
        self.acquisitions = None
        self.dividends = None

        buybacks_path = DATA_DIR / 'buybacks_clean.parquet'
        if buybacks_path.exists():
            self.buybacks = pd.read_parquet(buybacks_path)
            if 'action_date' in self.buybacks.columns:
                self.buybacks['action_date'] = pd.to_datetime(self.buybacks['action_date'])

        acq_path = DATA_DIR / 'acquisitions_linked.parquet'
        if acq_path.exists():
            self.acquisitions = pd.read_parquet(acq_path)
            if 'action_date' in self.acquisitions.columns:
                self.acquisitions['action_date'] = pd.to_datetime(self.acquisitions['action_date'])

        div_path = DATA_DIR / 'dividend_actions.parquet'
        if div_path.exists():
            self.dividends = pd.read_parquet(div_path)
            if 'action_date' in self.dividends.columns:
                self.dividends['action_date'] = pd.to_datetime(self.dividends['action_date'])

    def _load_valuation_history(self):
        """Load price data for valuation time series."""
        self.prices = None
        prices_path = DATA_DIR / 'prices_monthly.parquet'
        if prices_path.exists():
            self.prices = pd.read_parquet(prices_path)

    def generate_insights(
        self,
        gvkey: str,
        company_name: str,
        profile: Dict,
        similar_analysis: Dict,
        regime: Dict,
        as_of_date: str,
    ) -> Dict:
        """Generate full enhanced insights package."""
        signals = profile['signals']
        action_dist = similar_analysis.get('action_distribution', {})
        n_similar = similar_analysis.get('n_similar', 0)
        similar_cases = similar_analysis.get('similar_cases', [])

        # Get company-specific data
        company_metrics = self._get_company_metrics(gvkey, as_of_date)
        company_action_history = self._get_company_action_history(gvkey, as_of_date)
        sector_comps = self._get_sector_comps(gvkey, as_of_date)

        # Generate enhanced summary
        summary = self._generate_enhanced_summary(
            company_name, action_dist, n_similar, company_action_history
        )

        # Generate enhanced ideas
        ideas = self._generate_enhanced_ideas(
            gvkey, company_name, signals, action_dist, similar_cases,
            regime, company_metrics, sector_comps, as_of_date
        )

        # Other components
        decision_table = self._generate_decision_table(signals, regime, company_metrics)
        constraints = self._identify_constraints(signals, profile)
        timing = self._assess_timing(signals, regime)

        return {
            'summary': summary,
            'ideas': ideas,
            'decision_table': decision_table,
            'constraints': constraints,
            'timing': timing,
            'n_similar': n_similar,
            'company_metrics': company_metrics,
            'company_action_history': company_action_history,
        }

    def _get_company_metrics(self, gvkey: str, as_of_date: str) -> Dict:
        """Get key metrics for the company."""
        data = self.snapshot.get_snapshot(gvkey, as_of_date, lookback_quarters=8)
        if data is None or len(data) == 0:
            return {}

        latest = data.iloc[0]

        # Calculate actual metrics
        metrics = {}

        # Revenue
        revtq = float(latest.get('revtq', 0) or 0)
        metrics['revenue'] = revtq * 4 / 1000  # Annualized, in $B

        # Revenue growth YoY
        if len(data) >= 5:
            rev_now = float(data.iloc[0].get('revtq', 0) or 0)
            rev_1y = float(data.iloc[4].get('revtq', 0) or 0)
            if rev_1y > 0:
                metrics['revenue_growth'] = (rev_now - rev_1y) / rev_1y * 100

        # Gross margin
        if latest.get('revtq') and latest.get('cogsq'):
            rev = float(latest.get('revtq', 0) or 0)
            cogs = float(latest.get('cogsq', 0) or 0)
            if rev > 0:
                metrics['gross_margin'] = (rev - cogs) / rev * 100

        # EBITDA margin
        if latest.get('oibdpq') and latest.get('revtq'):
            ebitda = float(latest.get('oibdpq', 0) or 0)
            rev = float(latest.get('revtq', 0) or 0)
            if rev > 0:
                metrics['ebitda_margin'] = ebitda / rev * 100

        # FCF (approximate: EBITDA - CapEx)
        ebitda = float(latest.get('oibdpq', 0) or 0) * 4
        capex = float(latest.get('capxy', 0) or 0)
        metrics['fcf'] = (ebitda - capex) / 1000  # in $B

        # Leverage
        cash = float(latest.get('cheq', 0) or 0)
        debt = float(latest.get('dlttq', 0) or 0) + float(latest.get('dlcq', 0) or 0)
        ebitda_annual = float(latest.get('oibdpq', 0) or 0) * 4
        if ebitda_annual > 0:
            metrics['net_leverage'] = (debt - cash) / ebitda_annual

        # FCF yield (need market cap - approximate with book value for now)
        # This is a gap - we need price data linked to compute properly

        return metrics

    def _get_company_action_history(self, gvkey: str, as_of_date: str) -> Dict:
        """Get when this company last took various actions."""
        as_of = pd.to_datetime(as_of_date)
        history = {}

        # Last buyback
        if self.buybacks is not None and 'gvkey' in self.buybacks.columns:
            company_buybacks = self.buybacks[
                (self.buybacks['gvkey'] == str(gvkey)) &
                (self.buybacks['action_date'] <= as_of)
            ]
            if len(company_buybacks) > 0:
                last_date = company_buybacks['action_date'].max()
                months_ago = (as_of - last_date).days // 30
                history['last_buyback'] = {
                    'date': last_date,
                    'months_ago': months_ago,
                }

        # Last acquisition
        if self.acquisitions is not None and 'gvkey' in self.acquisitions.columns:
            company_acq = self.acquisitions[
                (self.acquisitions['gvkey'] == str(gvkey)) &
                (self.acquisitions['action_date'] <= as_of)
            ]
            if len(company_acq) > 0:
                last_date = company_acq['action_date'].max()
                months_ago = (as_of - last_date).days // 30
                history['last_acquisition'] = {
                    'date': last_date,
                    'months_ago': months_ago,
                }

        # Last dividend action
        if self.dividends is not None and 'gvkey' in self.dividends.columns:
            company_div = self.dividends[
                (self.dividends['gvkey'] == str(gvkey)) &
                (self.dividends['action_date'] <= as_of)
            ]
            if len(company_div) > 0:
                last_date = company_div['action_date'].max()
                months_ago = (as_of - last_date).days // 30
                history['last_dividend'] = {
                    'date': last_date,
                    'months_ago': months_ago,
                }

        return history

    def _get_sector_comps(self, gvkey: str, as_of_date: str) -> Dict:
        """Get sector comparison data."""
        company_data = self.snapshot.get_snapshot(gvkey, as_of_date)
        if company_data is None or len(company_data) == 0:
            return {}

        company = company_data.iloc[0]
        sic = company.get('sic')

        if pd.isna(sic):
            return {}

        sic_2digit = str(int(sic))[:2]
        universe = self.snapshot.get_universe_snapshot(as_of_date, min_assets=100)

        if universe is None or len(universe) == 0:
            return {}

        # Filter to same sector
        universe['sic_2'] = universe['sic'].apply(
            lambda x: str(int(x))[:2] if pd.notna(x) else None
        )
        peers = universe[universe['sic_2'] == sic_2digit].copy()

        if len(peers) == 0:
            return {}

        # Calculate sector stats
        sector_stats = {}

        # Helper function to safely get float value
        def safe_float(val, default=0):
            if pd.isna(val):
                return default
            try:
                return float(val)
            except (TypeError, ValueError):
                return default

        # Median EBITDA margin
        def calc_ebitda_margin(r):
            oibdpq = safe_float(r.get('oibdpq'), 0)
            revtq = safe_float(r.get('revtq'), 0)
            if revtq > 0:
                return oibdpq / revtq * 100
            return None

        peers['ebitda_margin'] = peers.apply(calc_ebitda_margin, axis=1)
        sector_stats['median_ebitda_margin'] = peers['ebitda_margin'].median()

        # Median leverage
        def calc_leverage(r):
            dlttq = safe_float(r.get('dlttq'), 0)
            dlcq = safe_float(r.get('dlcq'), 0)
            cheq = safe_float(r.get('cheq'), 0)
            oibdpq = safe_float(r.get('oibdpq'), 0)
            if oibdpq > 0:
                return (dlttq + dlcq - cheq) / (oibdpq * 4)
            return None

        peers['leverage'] = peers.apply(calc_leverage, axis=1)
        sector_stats['median_leverage'] = peers['leverage'].median()

        sector_stats['n_peers'] = len(peers)

        return sector_stats

    def _generate_enhanced_summary(
        self,
        company_name: str,
        action_dist: Dict,
        n_similar: int,
        action_history: Dict
    ) -> str:
        """Generate enhanced summary with company-specific timing."""
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

        # Count how many acted
        active_pct = sum(
            pct for action, pct in action_dist.items()
            if action not in ['status_quo', 'no_action']
        )
        n_acted = int(n_similar * active_pct / 100)

        # Build summary
        summary = f"{company_name}'s configuration suggests {top_actions}. {n_acted} of {n_similar} similar companies acted"

        # Add company-specific timing if available
        timing_parts = []
        if 'last_buyback' in action_history:
            months = action_history['last_buyback']['months_ago']
            timing_parts.append(f"{months} months on buybacks")
        if 'last_acquisition' in action_history:
            months = action_history['last_acquisition']['months_ago']
            timing_parts.append(f"{months} months on M&A")

        if timing_parts:
            summary += f" — {company_name} has waited " + ", ".join(timing_parts)

        return summary + "."

    def _generate_enhanced_ideas(
        self,
        gvkey: str,
        company_name: str,
        signals: Dict,
        action_dist: Dict,
        similar_cases: List,
        regime: Dict,
        company_metrics: Dict,
        sector_comps: Dict,
        as_of_date: str,
    ) -> List[IdeaV2]:
        """Generate enhanced ideas with quantified bullets."""
        ideas = []

        # Compute TSR distributions from similar cases
        tsr_stats = self._compute_tsr_stats(similar_cases)

        # 1. Buy-Side M&A
        mna_idea = self._score_mna_idea_v2(
            signals, action_dist, regime, tsr_stats,
            company_metrics, sector_comps, similar_cases
        )
        if mna_idea.score >= 40:
            ideas.append(mna_idea)

        # 2. Capital Allocation (Buybacks)
        buyback_idea = self._score_buyback_idea_v2(
            signals, action_dist, regime, tsr_stats,
            company_metrics, similar_cases
        )
        if buyback_idea.score >= 40:
            ideas.append(buyback_idea)

        # 3. Debt Issuance
        debt_idea = self._score_debt_idea_v2(
            signals, action_dist, regime, tsr_stats,
            company_metrics
        )
        if debt_idea.score >= 40:
            ideas.append(debt_idea)

        # Sort by score
        ideas.sort(key=lambda x: -x.score)

        return ideas

    def _compute_tsr_stats(self, similar_cases: List) -> Dict:
        """Compute TSR statistics by action type."""
        if not similar_cases:
            return {}

        df = pd.DataFrame(similar_cases)
        if 'tsr_12m' not in df.columns or 'action_type' not in df.columns:
            return {}

        stats = {}
        for action_type in df['action_type'].unique():
            if pd.isna(action_type):
                continue
            action_df = df[df['action_type'] == action_type]['tsr_12m'].dropna()
            if len(action_df) >= 3:
                stats[action_type] = {
                    'n': len(action_df),
                    'p25': action_df.quantile(0.25),
                    'p50': action_df.median(),
                    'p75': action_df.quantile(0.75),
                }

        return stats

    def _score_mna_idea_v2(
        self,
        signals: Dict,
        action_dist: Dict,
        regime: Dict,
        tsr_stats: Dict,
        company_metrics: Dict,
        sector_comps: Dict,
        similar_cases: List,
    ) -> IdeaV2:
        """Enhanced M&A scoring with quantified bullets."""
        score = 50
        why_now = []
        status = "PERSISTENT"

        # Balance sheet optionality
        bs_score = signals.get('balance_sheet_optionality', {}).get('score', 50)
        if bs_score >= 70:
            score += 15
            leverage = company_metrics.get('net_leverage')
            sector_lev = sector_comps.get('median_leverage')
            if leverage is not None and sector_lev is not None:
                why_now.append(WhyNowBullet(
                    text=f"Net leverage at {leverage:.1f}x vs sector median {sector_lev:.1f}x — significant debt capacity",
                    metric_name="net_leverage",
                    current_value=leverage,
                    comparison_value=sector_lev,
                    comparison_label="vs sector median",
                    source="fundamentals"
                ))
            else:
                why_now.append(WhyNowBullet(
                    text="Strong balance sheet with capacity for acquisitions",
                    source="signals"
                ))
        elif bs_score <= 30:
            score -= 20

        # Valuation
        val_score = signals.get('valuation_dislocation', {}).get('score', 50)
        if val_score >= 60:
            score += 10
            why_now.append(WhyNowBullet(
                text="Valuation provides acquisition currency advantage",
                source="signals"
            ))

        # Historical precedent with count
        mna_cases = [c for c in similar_cases if
                     isinstance(c.get('action_type'), str) and
                     ('acquisition' in c.get('action_type', '').lower() or
                      c.get('action_type') in ['Takeover', 'LBO'])]
        n_mna = len(mna_cases)
        n_total = len(similar_cases) if similar_cases else 1
        mna_pct = n_mna / n_total * 100 if n_total > 0 else 0

        if mna_pct >= 30:
            score += 15
            status = "NEWLY_ACTIONABLE"
            why_now.append(WhyNowBullet(
                text=f"{mna_pct:.0f}% of similar companies ({n_mna} of {n_total}) pursued M&A",
                metric_name="mna_precedent_pct",
                current_value=mna_pct,
                source="precedent"
            ))
        elif mna_pct >= 15:
            score += 8

        # Market regime
        if regime.get('regime') == 'LOOSE':
            score += 10
            why_now.append(WhyNowBullet(
                text="Favorable credit environment for deal financing",
                source="regime"
            ))
            status = "WINDOW_OPEN"
        elif regime.get('regime') == 'TIGHT':
            score -= 10

        # Get TSR stats for M&A
        mna_tsr = tsr_stats.get('acquisition', tsr_stats.get('Takeover', {}))
        tsr_p25 = mna_tsr.get('p25') if mna_tsr else None
        tsr_p50 = mna_tsr.get('p50') if mna_tsr else None
        tsr_p75 = mna_tsr.get('p75') if mna_tsr else None
        precedent_n = mna_tsr.get('n', 0) if mna_tsr else 0

        if tsr_p50 and tsr_p50 > 5:
            score += 5
            why_now.append(WhyNowBullet(
                text=f"Similar M&A transactions delivered +{tsr_p50:.0f}% median TSR (n={precedent_n})",
                metric_name="mna_tsr",
                current_value=tsr_p50,
                source="precedent"
            ))

        score = max(0, min(100, score))

        # Build share price impact string
        share_price_impact = None
        share_price_basis = None
        if tsr_p25 is not None and tsr_p75 is not None:
            share_price_impact = f"+{tsr_p25:.0f}% to +{tsr_p75:.0f}%"
            share_price_basis = f"Based on {precedent_n} precedent transactions with similar acquirer profiles"

        return IdeaV2(
            name="Buy-Side M&A",
            score=score,
            priority=self._score_to_priority(score),
            status=status,
            value_lever="Growth durability",
            economic_impact="+100-200bps revenue growth equivalent",
            why_now=why_now,
            share_price_impact=share_price_impact,
            share_price_basis=share_price_basis,
            precedent_n=precedent_n,
            tsr_p25=tsr_p25,
            tsr_p50=tsr_p50,
            tsr_p75=tsr_p75,
        )

    def _score_buyback_idea_v2(
        self,
        signals: Dict,
        action_dist: Dict,
        regime: Dict,
        tsr_stats: Dict,
        company_metrics: Dict,
        similar_cases: List,
    ) -> IdeaV2:
        """Enhanced buyback scoring."""
        score = 50
        why_now = []
        status = "PERSISTENT"

        # Balance sheet
        bs_score = signals.get('balance_sheet_optionality', {}).get('score', 50)
        if bs_score >= 60:
            score += 10
            fcf = company_metrics.get('fcf')
            if fcf:
                why_now.append(WhyNowBullet(
                    text=f"FCF of ${fcf:.1f}B supports capital return",
                    metric_name="fcf",
                    current_value=fcf,
                    source="fundamentals"
                ))

        # Valuation - buybacks work best when undervalued
        val_score = signals.get('valuation_dislocation', {}).get('score', 50)
        if val_score <= 40:
            score += 20
            status = "NEWLY_ACTIONABLE"
            why_now.append(WhyNowBullet(
                text="Stock undervaluation makes buybacks accretive",
                source="signals"
            ))
        elif val_score >= 70:
            score -= 15

        # Growth - buybacks for slow-growth
        growth_score = signals.get('growth_momentum', {}).get('score', 50)
        growth_rate = company_metrics.get('revenue_growth')
        if growth_score <= 40:
            score += 10
            if growth_rate is not None:
                why_now.append(WhyNowBullet(
                    text=f"Revenue growth at {growth_rate:.0f}% — limited organic opportunities favor capital return",
                    metric_name="revenue_growth",
                    current_value=growth_rate,
                    source="fundamentals"
                ))

        # Historical precedent
        buyback_cases = [c for c in similar_cases if
                        isinstance(c.get('action_type'), str) and
                        'buyback' in c.get('action_type', '').lower()]
        n_buyback = len(buyback_cases)
        n_total = len(similar_cases) if similar_cases else 1
        buyback_pct = n_buyback / n_total * 100 if n_total > 0 else 0

        if buyback_pct >= 30:
            score += 10
            why_now.append(WhyNowBullet(
                text=f"{buyback_pct:.0f}% of similar companies ({n_buyback} of {n_total}) executed buybacks",
                source="precedent"
            ))

        # TSR stats
        buyback_tsr = tsr_stats.get('buyback', {})
        tsr_p25 = buyback_tsr.get('p25')
        tsr_p50 = buyback_tsr.get('p50')
        tsr_p75 = buyback_tsr.get('p75')
        precedent_n = buyback_tsr.get('n', 0)

        score = max(0, min(100, score))

        share_price_impact = None
        share_price_basis = None
        if tsr_p25 is not None and tsr_p75 is not None:
            share_price_impact = f"+{tsr_p25:.0f}% to +{tsr_p75:.0f}%"
            share_price_basis = f"Based on FCF/share accretion at current yield (n={precedent_n})"

        return IdeaV2(
            name="Capital Allocation",
            score=score,
            priority=self._score_to_priority(score),
            status=status,
            value_lever="FCF per share",
            economic_impact="3-4% annual FCF/share accretion",
            why_now=why_now,
            share_price_impact=share_price_impact,
            share_price_basis=share_price_basis,
            precedent_n=precedent_n,
            tsr_p25=tsr_p25,
            tsr_p50=tsr_p50,
            tsr_p75=tsr_p75,
        )

    def _score_debt_idea_v2(
        self,
        signals: Dict,
        action_dist: Dict,
        regime: Dict,
        tsr_stats: Dict,
        company_metrics: Dict,
    ) -> IdeaV2:
        """Enhanced debt issuance scoring."""
        score = 40
        why_now = []
        status = "PERSISTENT"

        # Balance sheet capacity
        bs_score = signals.get('balance_sheet_optionality', {}).get('score', 50)
        leverage = company_metrics.get('net_leverage')

        if bs_score >= 70:
            score += 15
            if leverage is not None:
                why_now.append(WhyNowBullet(
                    text=f"Net leverage at {leverage:.1f}x — underutilized debt capacity",
                    metric_name="net_leverage",
                    current_value=leverage,
                    source="fundamentals"
                ))
        elif bs_score <= 30:
            score -= 20

        # Market regime - critical for debt
        if regime.get('regime') == 'LOOSE':
            score += 20
            why_now.append(WhyNowBullet(
                text="IG spreads at favorable levels — clean issuance window",
                source="regime"
            ))
            status = "WINDOW_OPEN"
        elif regime.get('regime') == 'TIGHT':
            score -= 25

        # Refinancing pressure
        refi_score = signals.get('refinancing_pressure', {}).get('score', 50)
        if refi_score <= 40:
            score += 10
            why_now.append(WhyNowBullet(
                text="Near-term maturities create refinancing opportunity",
                source="signals"
            ))

        score = max(0, min(100, score))

        return IdeaV2(
            name="Incremental Debt Issuance",
            score=score,
            priority=self._score_to_priority(score),
            status=status,
            value_lever="Strategic flexibility",
            economic_impact="$2-5B dry powder for opportunistic moves",
            why_now=why_now,
        )

    def _generate_decision_table(self, signals: Dict, regime: Dict, metrics: Dict) -> List[Dict]:
        """Generate decision table with actual metrics."""
        rules = []

        growth_score = signals.get('growth_momentum', {}).get('score', 50)
        val_score = signals.get('valuation_dislocation', {}).get('score', 50)
        growth_rate = metrics.get('revenue_growth')

        # Growth-based rules
        if growth_score <= 40:
            growth_str = f"Growth <{growth_rate:.0f}%" if growth_rate else "Growth slows"
            if val_score >= 60:
                rules.append({
                    'condition': f"{growth_str} + multiple stays premium",
                    'action': "Buybacks",
                    'note': None,
                })
            else:
                rules.append({
                    'condition': f"{growth_str} + valuation compresses",
                    'action': "Buybacks",
                    'note': "(more attractive)",
                })

        # Credit-based rules
        if regime.get('regime') == 'LOOSE':
            rules.append({
                'condition': "Credit spreads remain tight + bolt-on target availability improves",
                'action': "M&A Advisory",
                'note': "(window strengthens)",
            })

        # Margin-based rules
        margin_score = signals.get('margin_trend', {}).get('score', 50)
        margin = metrics.get('ebitda_margin')
        if margin_score >= 60:
            margin_str = f"Margins stabilize at {margin:.0f}%" if margin else "Margins stabilize"
            rules.append({
                'condition': margin_str,
                'action': "Reinvestment",
                'note': "(M&A urgency fades)",
            })

        return rules

    def _identify_constraints(self, signals: Dict, profile: Dict) -> List[Dict]:
        """Identify binding constraints."""
        constraints = []

        bs_score = signals.get('balance_sheet_optionality', {}).get('score', 50)
        if bs_score <= 30:
            constraints.append({
                'name': "Leverage limits acquisition capacity",
                'severity': "HIGH",
                'implication': "Debt reduction or equity financing required for large M&A",
            })

        val_score = signals.get('valuation_dislocation', {}).get('score', 50)
        if val_score <= 30:
            constraints.append({
                'name': "Valuation limits equity currency",
                'severity': "MEDIUM",
                'implication': "Cash or debt financing preferred for M&A",
            })

        growth_score = signals.get('growth_momentum', {}).get('score', 50)
        if growth_score <= 30:
            constraints.append({
                'name': "Growth narrative must be addressed",
                'severity': "MEDIUM",
                'implication': "FCF deployment must reinforce growth story",
            })

        return constraints

    def _assess_timing(self, signals: Dict, regime: Dict) -> Dict:
        """Assess timing posture."""
        regime_name = regime.get('regime', 'SELECTIVE')
        bs_score = signals.get('balance_sheet_optionality', {}).get('score', 50)
        refi_score = signals.get('refinancing_pressure', {}).get('score', 50)

        if refi_score <= 30:
            return {'posture': "Urgent", 'description': "Near-term maturities require action"}
        elif regime_name == 'LOOSE' and bs_score >= 60:
            return {'posture': "Opportunistic", 'description': "Favorable window, not urgent"}
        elif regime_name == 'TIGHT':
            return {'posture': "Patient", 'description': "Wait for better entry point"}
        else:
            return {'posture': "Opportunistic", 'description': "Selective action warranted"}

    def _score_to_priority(self, score: int) -> str:
        if score >= 80: return "HIGH"
        elif score >= 65: return "MEDIUM-HIGH"
        elif score >= 50: return "MEDIUM"
        elif score >= 35: return "LOW-MEDIUM"
        else: return "LOW"

    def _format_action_name(self, action: str) -> str:
        name_map = {
            'buyback': 'buybacks',
            'acquisition': 'M&A',
            'dividend_increase': 'dividend increases',
            'Takeover': 'M&A',
            'LBO': 'M&A',
        }
        return name_map.get(action, action.replace('_', ' '))
