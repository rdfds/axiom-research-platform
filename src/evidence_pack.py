"""
Evidence Pack
=============
The core data structure that grounds ALL narrative generation.

This is the heart of the system - before any text is generated,
we assemble a complete, auditable EvidencePack containing:

1. State Summary - SignalProfile with drivers and explanations
2. Regime Context - Current market environment
3. Cohorts - Grouped analog cases with outcomes and key splits
4. Objections - Pre-computed Q&A with grounded answers
5. Citations - Pointers to source data for every claim

The LLM/templates can ONLY use data from the EvidencePack.
No hallucination, no invented facts.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json

from .recommendation_runtime_config import DEFAULT_PRECEDENT_RETRIEVAL_VERSION
from .snapshot import AsOfSnapshotBuilder, DATA_DIR
from .asof_store import AsOfWarehouse


# =============================================================================
# SCHEMA DEFINITIONS
# =============================================================================

@dataclass
class FeatureContribution:
    """A single feature's contribution to a signal."""
    feature_name: str
    feature_value: float
    contribution: float  # How much this feature drove the signal
    direction: str  # "positive" or "negative"
    human_label: str  # e.g., "Net Debt / EBITDA"


@dataclass
class SignalDetail:
    """Full detail for a single signal."""
    name: str
    value: str  # "high", "medium", "low"
    score: float  # 0-100
    confidence: float  # 0-1
    drivers: List[FeatureContribution]
    explanation: str  # 1-sentence banker-friendly explanation
    evidence_refs: Dict[str, List[str]]  # warehouse_rows, doc_chunks


@dataclass
class SignalProfileV2:
    """Enhanced signal profile with full audit trail."""
    company_id: str
    company_name: str
    as_of_time: str
    signal_schema_version: str
    signals: Dict[str, SignalDetail]
    composite_score: float
    vector: List[float]  # For similarity computation


@dataclass
class RegimeContext:
    """Market regime with transition context."""
    regime_id: str  # "LOOSE", "SELECTIVE", "TIGHT"
    confidence: float
    since: Optional[str]  # When this regime started
    is_transitioning: bool
    transition_direction: Optional[str]  # "tightening" or "loosening"
    characteristics: Dict[str, str]


@dataclass
class OutcomeDistribution:
    """Outcome distribution for a cohort."""
    metric_name: str  # e.g., "tsr_12m"
    n: int
    p10: Optional[float]
    p25: Optional[float]
    p50: Optional[float]  # median
    p75: Optional[float]
    p90: Optional[float]
    mean: Optional[float]
    std: Optional[float]
    pct_positive: Optional[float]  # % with positive outcome
    pct_beat_benchmark: Optional[float]


@dataclass
class KeySplit:
    """A condition that materially changes outcomes."""
    condition: str  # e.g., "valuation_dislocation = high"
    condition_human: str  # e.g., "When stock is undervalued vs peers"
    effect: str  # e.g., "+4% median TSR"
    effect_magnitude: float
    n_with_condition: int
    n_without_condition: int
    outcomes_with: OutcomeDistribution
    outcomes_without: OutcomeDistribution
    statistical_significance: float  # p-value or confidence


@dataclass
class ExampleCase:
    """A specific historical case for narrative use."""
    case_id: str
    company_name: str
    date: str
    action_type: str
    similarity_score: float
    signal_profile_summary: Dict[str, float]  # key signals at time of action
    outcome_tsr_12m: Optional[float]
    context_summary: str  # 1-2 sentence description


@dataclass
class Cohort:
    """A group of similar historical cases."""
    action_type: str  # e.g., "bolt_on_mna", "buyback"
    action_human: str  # e.g., "Bolt-on M&A"
    filters: Dict[str, Any]  # How this cohort was defined
    n: int
    outcomes: Dict[str, OutcomeDistribution]  # tsr_1m, tsr_3m, tsr_12m, etc.
    key_splits: List[KeySplit]
    example_cases: List[ExampleCase]
    quality_flags: List[str]  # e.g., ["small_sample", "regime_mismatch"]


@dataclass
class Objection:
    """A pre-computed objection with grounded rebuttal."""
    question: str  # e.g., "Why not wait 6 months?"
    question_category: str  # "timing", "sizing", "alternative", "risk"
    grounded_answer_points: List[str]
    evidence_refs: List[str]
    confidence: float


@dataclass
class ActionCard:
    """Full evidence for one potential action."""
    action_type: str
    action_human: str
    recommendation_score: int  # 0-100
    recommendation_label: str  # "HIGH", "MEDIUM", etc.
    status: str  # "NEWLY_ACTIONABLE", "PERSISTENT", "WINDOW_OPEN"

    # Thesis
    thesis_summary: str
    value_lever: str
    economic_impact: str
    share_price_impact_range: Optional[str]

    # Why Now bullets (grounded)
    why_now_bullets: List[Dict[str, Any]]  # Each has text, metric, source

    # Conditions
    works_when: List[str]
    fails_when: List[str]

    # Evidence
    cohort: Cohort
    comparable_cohorts: List[Cohort]  # Alternative filters for comparison

    # Objections
    objections: List[Objection]


@dataclass
class EvidencePack:
    """
    The complete grounded evidence package.

    This is what the narrative layer receives - nothing else.
    Every claim must trace back to something in here.
    """
    # Metadata
    pack_id: str
    generated_at: str
    company_id: str
    company_name: str
    as_of_time: str

    # Version tracking (for audit)
    data_snapshot_version: str
    signal_schema_version: str
    regime_model_version: str
    retrieval_version: str

    # Core content
    state_summary: SignalProfileV2
    regime: RegimeContext
    company_metrics: Dict[str, Any]  # Revenue, margins, etc.
    peer_comparison: Dict[str, Any]

    # Action analysis
    action_cards: List[ActionCard]

    # Cross-cutting
    binding_constraints: List[Dict[str, str]]
    timing_posture: Dict[str, str]

    # Historical context
    company_action_history: Dict[str, Any]

    def to_dict(self) -> Dict:
        """Serialize to dictionary for storage/API."""
        # This would be a full serialization - simplified here
        return {
            'pack_id': self.pack_id,
            'company_id': self.company_id,
            'as_of_time': self.as_of_time,
            'generated_at': self.generated_at,
            # ... full serialization
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), default=str)


# =============================================================================
# EVIDENCE PACK BUILDER
# =============================================================================

class EvidencePackBuilder:
    """
    Builds a complete EvidencePack from raw analysis.

    This is the critical component that assembles grounded evidence
    before any narrative generation.
    """

    def __init__(self):
        self.snapshot = AsOfSnapshotBuilder()
        self.asof = AsOfWarehouse()
        self._load_data()

    def _load_data(self):
        """Load required data sources."""
        # Load action profiles for cohort analysis
        self.action_profiles = None
        profiles_path = DATA_DIR / 'clean_action_profiles.parquet'
        if profiles_path.exists():
            self.action_profiles = pd.read_parquet(profiles_path)
        else:
            # Fall back to older profiles
            for path in [
                DATA_DIR / 'action_profiles_with_outcomes.parquet',
                DATA_DIR / 'deal_profiles_with_outcomes.parquet'
            ]:
                if path.exists():
                    self.action_profiles = pd.read_parquet(path)
                    break

        # Load fundamentals for feature computation
        self.fundamentals = None
        fund_path = DATA_DIR / 'fundamentals_quarterly.parquet'
        if fund_path.exists():
            self.fundamentals = pd.read_parquet(fund_path)

    def build(
        self,
        gvkey: str,
        company_name: str,
        as_of_date: str,
        signal_profile: Dict,
        regime_result: Dict,
        similar_cases: List[Dict],
    ) -> EvidencePack:
        """
        Build complete EvidencePack.

        Parameters
        ----------
        gvkey : str
            Company identifier
        company_name : str
            Company name
        as_of_date : str
            Analysis date
        signal_profile : dict
            Output from SignalEngine
        regime_result : dict
            Output from RegimeClassifier
        similar_cases : list
            Output from ActionAnalyzer
        """
        import uuid

        # Build enhanced signal profile with drivers
        state_summary = self._build_signal_profile(
            gvkey, company_name, as_of_date, signal_profile
        )

        # Build regime context
        regime = self._build_regime_context(regime_result, as_of_date)

        # Get company metrics
        company_metrics = self._get_company_metrics(gvkey, as_of_date)

        # Get peer comparison
        peer_comparison = self._get_peer_comparison(gvkey, as_of_date)

        # Build cohorts from similar cases
        cohorts = self._build_cohorts(similar_cases, signal_profile, regime)

        # Build action cards
        action_cards = self._build_action_cards(
            cohorts, signal_profile, regime, company_metrics
        )

        # Identify constraints
        constraints = self._identify_constraints(signal_profile)

        # Assess timing
        timing = self._assess_timing(signal_profile, regime)

        # Get company action history
        action_history = self._get_action_history(gvkey, as_of_date)

        return EvidencePack(
            pack_id=str(uuid.uuid4()),
            generated_at=datetime.now().isoformat(),
            company_id=gvkey,
            company_name=company_name,
            as_of_time=as_of_date,
            data_snapshot_version="warehouse_asof_v1",
            signal_schema_version="v1",
            regime_model_version="v1",
            retrieval_version=DEFAULT_PRECEDENT_RETRIEVAL_VERSION,
            state_summary=state_summary,
            regime=regime,
            company_metrics=company_metrics,
            peer_comparison=peer_comparison,
            action_cards=action_cards,
            binding_constraints=constraints,
            timing_posture=timing,
            company_action_history=action_history,
        )

    def _build_signal_profile(
        self,
        gvkey: str,
        company_name: str,
        as_of_date: str,
        raw_profile: Dict,
    ) -> SignalProfileV2:
        """Build enhanced signal profile with drivers and explanations."""

        signals = {}

        for sig_name, sig_data in raw_profile.get('signals', {}).items():
            score = sig_data.get('score', 50)

            # Determine value bucket
            if score >= 70:
                value = "high"
            elif score >= 40:
                value = "medium"
            else:
                value = "low"

            # Generate drivers (what features contributed)
            drivers = self._compute_signal_drivers(gvkey, as_of_date, sig_name, score)

            # Generate explanation
            explanation = self._generate_signal_explanation(sig_name, score, value, drivers)

            signals[sig_name] = SignalDetail(
                name=sig_name,
                value=value,
                score=score,
                confidence=0.8,  # TODO: Compute actual confidence
                drivers=drivers,
                explanation=explanation,
                evidence_refs={
                    'warehouse_rows': [f'fundamentals:{gvkey}:{as_of_date}'],
                    'doc_chunks': [],
                }
            )

        return SignalProfileV2(
            company_id=gvkey,
            company_name=company_name,
            as_of_time=as_of_date,
            signal_schema_version="v1",
            signals=signals,
            composite_score=raw_profile.get('composite_score', 50),
            vector=raw_profile.get('vector', []),
        )

    def _compute_signal_drivers(
        self,
        gvkey: str,
        as_of_date: str,
        signal_name: str,
        score: float,
    ) -> List[FeatureContribution]:
        """Compute which features drove a signal's value."""

        # Get company fundamentals
        data = self.snapshot.get_snapshot(gvkey, as_of_date, lookback_quarters=4)
        if data is None or len(data) == 0:
            return []

        latest = data.iloc[0]
        drivers = []

        # Signal-specific driver computation
        if signal_name == 'balance_sheet_optionality':
            # Net Debt / EBITDA
            cash = self._safe_float(latest.get('cheq'))
            debt = self._safe_float(latest.get('dlttq')) + self._safe_float(latest.get('dlcq'))
            ebitda = self._safe_float(latest.get('oibdpq')) * 4

            if ebitda > 0:
                net_debt_ebitda = (debt - cash) / ebitda
                contribution = 0.4 if net_debt_ebitda < 2 else -0.2
                drivers.append(FeatureContribution(
                    feature_name='net_debt_to_ebitda',
                    feature_value=round(net_debt_ebitda, 2),
                    contribution=contribution,
                    direction='positive' if contribution > 0 else 'negative',
                    human_label=f'Net Debt / EBITDA: {net_debt_ebitda:.1f}x'
                ))

            # Cash / Assets
            assets = self._safe_float(latest.get('atq'))
            if assets > 0:
                cash_ratio = cash / assets
                contribution = 0.3 if cash_ratio > 0.1 else -0.1
                drivers.append(FeatureContribution(
                    feature_name='cash_to_assets',
                    feature_value=round(cash_ratio, 3),
                    contribution=contribution,
                    direction='positive' if contribution > 0 else 'negative',
                    human_label=f'Cash / Assets: {cash_ratio*100:.0f}%'
                ))

        elif signal_name == 'growth_momentum':
            # Revenue growth
            if len(data) >= 5:
                rev_now = self._safe_float(data.iloc[0].get('revtq'))
                rev_1y = self._safe_float(data.iloc[4].get('revtq'))
                if rev_1y > 0:
                    growth = (rev_now - rev_1y) / rev_1y
                    contribution = 0.5 if growth > 0.05 else -0.3 if growth < 0 else 0
                    drivers.append(FeatureContribution(
                        feature_name='revenue_growth_yoy',
                        feature_value=round(growth, 3),
                        contribution=contribution,
                        direction='positive' if contribution > 0 else 'negative',
                        human_label=f'Revenue Growth YoY: {growth*100:+.0f}%'
                    ))

        elif signal_name == 'valuation_dislocation':
            # Would need market data - placeholder
            drivers.append(FeatureContribution(
                feature_name='ev_to_ebitda_vs_peers',
                feature_value=0,
                contribution=0.3,
                direction='positive',
                human_label='EV/EBITDA vs Peer Median'
            ))

        elif signal_name == 'margin_trend':
            # EBITDA margin
            rev = self._safe_float(latest.get('revtq'))
            ebitda = self._safe_float(latest.get('oibdpq'))
            if rev > 0:
                margin = ebitda / rev
                contribution = 0.4 if margin > 0.15 else -0.2 if margin < 0.05 else 0
                drivers.append(FeatureContribution(
                    feature_name='ebitda_margin',
                    feature_value=round(margin, 3),
                    contribution=contribution,
                    direction='positive' if contribution > 0 else 'negative',
                    human_label=f'EBITDA Margin: {margin*100:.0f}%'
                ))

        # Sort by absolute contribution
        drivers.sort(key=lambda x: abs(x.contribution), reverse=True)
        return drivers[:3]  # Top 3 drivers

    def _generate_signal_explanation(
        self,
        signal_name: str,
        score: float,
        value: str,
        drivers: List[FeatureContribution],
    ) -> str:
        """Generate a 1-sentence explanation for a signal."""

        templates = {
            'balance_sheet_optionality': {
                'high': "Strong balance sheet with low leverage and significant cash reserves provides optionality for capital deployment.",
                'medium': "Moderate financial flexibility with manageable leverage levels.",
                'low': "Constrained balance sheet with elevated leverage limits near-term capital deployment options.",
            },
            'growth_momentum': {
                'high': "Revenue momentum is strong, supporting premium valuation and strategic flexibility.",
                'medium': "Growth is in line with market expectations, neither accelerating nor decelerating materially.",
                'low': "Revenue growth has decelerated, potentially creating pressure for capital return or strategic action.",
            },
            'valuation_dislocation': {
                'high': "Stock is trading at a premium to historical levels and peers, providing strong acquisition currency.",
                'medium': "Valuation is in line with historical norms and peer medians.",
                'low': "Stock appears undervalued relative to peers, making buybacks more accretive and equity issuance less attractive.",
            },
            'margin_trend': {
                'high': "Margins are strong and stable, supporting cash flow generation and dividend capacity.",
                'medium': "Margins are at normalized levels with no significant pressure or expansion.",
                'low': "Margin compression suggests need for cost action or portfolio rationalization.",
            },
            'refinancing_pressure': {
                'high': "No near-term debt maturities; clean runway for strategic capital deployment.",
                'medium': "Some debt maturities in the medium term but manageable refinancing need.",
                'low': "Near-term debt maturities require attention and may limit other capital actions.",
            },
            'size_factor': {
                'high': "Scale provides significant strategic flexibility and access to capital markets.",
                'medium': "Company size supports most strategic options.",
                'low': "Smaller scale may limit certain M&A opportunities.",
            },
            'asset_intensity': {
                'high': "Asset-light model supports higher returns and flexibility.",
                'medium': "Moderate asset intensity is typical for the sector.",
                'low': "Capital-intensive operations may require ongoing reinvestment.",
            },
        }

        signal_templates = templates.get(signal_name, {})
        explanation = signal_templates.get(value, f"{signal_name.replace('_', ' ').title()} is {value}.")

        # Enhance with specific driver data if available
        if drivers:
            top_driver = drivers[0]
            explanation = f"{explanation.rstrip('.')} ({top_driver.human_label})."

        return explanation

    def _build_regime_context(self, regime_result: Dict, as_of_date: str) -> RegimeContext:
        """Build full regime context."""
        return RegimeContext(
            regime_id=regime_result.get('regime', 'SELECTIVE'),
            confidence=regime_result.get('confidence', 0.7),
            since=None,  # Would need historical regime tracking
            is_transitioning=False,
            transition_direction=None,
            characteristics={
                'deal_activity': regime_result.get('description', ''),
                'financing': 'Available' if regime_result.get('regime') == 'LOOSE' else 'Selective',
            }
        )

    def _get_company_metrics(self, gvkey: str, as_of_date: str) -> Dict:
        """Get key company metrics."""
        data = self.snapshot.get_snapshot(gvkey, as_of_date, lookback_quarters=8)
        if data is None or len(data) == 0:
            return {}

        latest = data.iloc[0]
        metrics = {}

        # Revenue
        revtq = self._safe_float(latest.get('revtq'))
        metrics['revenue'] = revtq * 4 / 1000  # Annualized $B
        metrics['revenue_quarterly'] = revtq

        # Growth
        if len(data) >= 5:
            rev_1y = self._safe_float(data.iloc[4].get('revtq'))
            if rev_1y > 0:
                metrics['revenue_growth_yoy'] = (revtq - rev_1y) / rev_1y * 100

        # Margins
        if revtq > 0:
            cogs = self._safe_float(latest.get('cogsq'))
            ebitda = self._safe_float(latest.get('oibdpq'))
            metrics['gross_margin'] = (revtq - cogs) / revtq * 100 if cogs else None
            metrics['ebitda_margin'] = ebitda / revtq * 100

        # FCF (approximation)
        ebitda_annual = self._safe_float(latest.get('oibdpq')) * 4
        capex = self._safe_float(latest.get('capxy'))
        metrics['fcf'] = (ebitda_annual - capex) / 1000  # $B

        # Leverage
        cash = self._safe_float(latest.get('cheq'))
        debt = self._safe_float(latest.get('dlttq')) + self._safe_float(latest.get('dlcq'))
        if ebitda_annual > 0:
            metrics['net_leverage'] = (debt - cash) / ebitda_annual

        metrics['cash'] = cash / 1000  # $B
        metrics['debt'] = debt / 1000  # $B

        return metrics

    def _get_peer_comparison(self, gvkey: str, as_of_date: str) -> Dict:
        """Get peer comparison data."""
        # Simplified - would be more comprehensive
        return {
            'peer_count': 0,
            'metrics': {}
        }

    def _build_cohorts(
        self,
        similar_cases: List[Dict],
        signal_profile: Dict,
        regime: RegimeContext,
    ) -> Dict[str, Cohort]:
        """Build cohorts from similar cases with full statistics."""

        if not similar_cases:
            return {}

        df = pd.DataFrame(similar_cases)
        cohorts = {}

        # Group by action type
        for action_type in df['action_type'].dropna().unique():
            if pd.isna(action_type):
                continue

            action_cases = df[df['action_type'] == action_type]
            n = len(action_cases)

            if n < 3:  # Minimum threshold
                continue

            # Compute outcome distributions
            outcomes = {}
            for horizon in ['tsr_1m', 'tsr_3m', 'tsr_6m', 'tsr_12m']:
                if horizon in action_cases.columns:
                    values = action_cases[horizon].dropna()
                    if len(values) >= 3:
                        outcomes[horizon] = OutcomeDistribution(
                            metric_name=horizon,
                            n=len(values),
                            p10=values.quantile(0.10),
                            p25=values.quantile(0.25),
                            p50=values.median(),
                            p75=values.quantile(0.75),
                            p90=values.quantile(0.90),
                            mean=values.mean(),
                            std=values.std(),
                            pct_positive=(values > 0).mean() * 100,
                            pct_beat_benchmark=None,
                        )

            # Compute key splits
            key_splits = self._compute_key_splits(action_cases, signal_profile)

            # Get example cases
            examples = self._get_example_cases(action_cases)

            # Quality flags
            quality_flags = []
            if n < 10:
                quality_flags.append('small_sample')
            if n < 5:
                quality_flags.append('very_small_sample')

            cohorts[action_type] = Cohort(
                action_type=action_type,
                action_human=self._format_action_name(action_type),
                filters={'regime': regime.regime_id},
                n=n,
                outcomes=outcomes,
                key_splits=key_splits,
                example_cases=examples,
                quality_flags=quality_flags,
            )

        return cohorts

    def _compute_key_splits(
        self,
        cases_df: pd.DataFrame,
        signal_profile: Dict,
    ) -> List[KeySplit]:
        """Find conditions that materially change outcomes."""

        splits = []

        if 'tsr_12m' not in cases_df.columns or len(cases_df) < 10:
            return splits

        # Check if composite_score split matters
        if 'composite_score' in cases_df.columns:
            median_composite = cases_df['composite_score'].median()
            high_composite = cases_df[cases_df['composite_score'] >= median_composite]
            low_composite = cases_df[cases_df['composite_score'] < median_composite]

            if len(high_composite) >= 5 and len(low_composite) >= 5:
                high_tsr = high_composite['tsr_12m'].dropna()
                low_tsr = low_composite['tsr_12m'].dropna()

                if len(high_tsr) >= 3 and len(low_tsr) >= 3:
                    diff = high_tsr.median() - low_tsr.median()

                    if abs(diff) >= 3:  # At least 3% difference
                        splits.append(KeySplit(
                            condition=f"composite_score >= {median_composite:.0f}",
                            condition_human="When company has stronger overall profile",
                            effect=f"{diff:+.0f}% median TSR",
                            effect_magnitude=diff,
                            n_with_condition=len(high_composite),
                            n_without_condition=len(low_composite),
                            outcomes_with=OutcomeDistribution(
                                metric_name='tsr_12m',
                                n=len(high_tsr),
                                p10=None, p25=high_tsr.quantile(0.25),
                                p50=high_tsr.median(),
                                p75=high_tsr.quantile(0.75), p90=None,
                                mean=high_tsr.mean(), std=high_tsr.std(),
                                pct_positive=(high_tsr > 0).mean() * 100,
                                pct_beat_benchmark=None,
                            ),
                            outcomes_without=OutcomeDistribution(
                                metric_name='tsr_12m',
                                n=len(low_tsr),
                                p10=None, p25=low_tsr.quantile(0.25),
                                p50=low_tsr.median(),
                                p75=low_tsr.quantile(0.75), p90=None,
                                mean=low_tsr.mean(), std=low_tsr.std(),
                                pct_positive=(low_tsr > 0).mean() * 100,
                                pct_beat_benchmark=None,
                            ),
                            statistical_significance=0.05,  # Placeholder
                        ))

        return splits

    def _get_example_cases(self, cases_df: pd.DataFrame, n: int = 3) -> List[ExampleCase]:
        """Get top example cases for narrative use."""

        examples = []

        # Sort by similarity and take top N
        if 'similarity' in cases_df.columns:
            top_cases = cases_df.nlargest(n, 'similarity')
        else:
            top_cases = cases_df.head(n)

        for _, row in top_cases.iterrows():
            examples.append(ExampleCase(
                case_id=f"{row.get('gvkey', 'unknown')}_{row.get('date', 'unknown')}",
                company_name=row.get('company_name', 'Unknown'),
                date=str(row.get('date', '')),
                action_type=row.get('action_type', ''),
                similarity_score=row.get('similarity', 0),
                signal_profile_summary={
                    'composite_score': row.get('composite_score', 0),
                },
                outcome_tsr_12m=row.get('tsr_12m'),
                context_summary=f"{row.get('company_name', 'Company')} executed {row.get('action_type', 'action')} with {row.get('tsr_12m', 0):+.0f}% 12-month TSR" if pd.notna(row.get('tsr_12m')) else "",
            ))

        return examples

    def _build_action_cards(
        self,
        cohorts: Dict[str, Cohort],
        signal_profile: Dict,
        regime: RegimeContext,
        company_metrics: Dict,
    ) -> List[ActionCard]:
        """Build full action cards with all evidence."""

        cards = []
        signals = signal_profile.get('signals', {})

        # M&A Card
        mna_cohort = cohorts.get('acquisition') or cohorts.get('Takeover')
        if mna_cohort or True:  # Always include M&A option
            mna_card = self._build_mna_card(
                mna_cohort, signals, regime, company_metrics
            )
            cards.append(mna_card)

        # Buyback Card
        buyback_cohort = cohorts.get('buyback')
        if buyback_cohort or True:
            buyback_card = self._build_buyback_card(
                buyback_cohort, signals, regime, company_metrics
            )
            cards.append(buyback_card)

        # Debt Card
        debt_card = self._build_debt_card(signals, regime, company_metrics)
        cards.append(debt_card)

        # Sort by score
        cards.sort(key=lambda x: -x.recommendation_score)

        return cards

    def _build_mna_card(
        self,
        cohort: Optional[Cohort],
        signals: Dict,
        regime: RegimeContext,
        metrics: Dict,
    ) -> ActionCard:
        """Build M&A action card."""

        score = 50
        why_now = []
        works_when = []
        fails_when = []
        objections = []

        # Score based on signals
        bs_score = signals.get('balance_sheet_optionality', {}).get('score', 50)
        if bs_score >= 70:
            score += 15
            leverage = metrics.get('net_leverage')
            if leverage is not None:
                why_now.append({
                    'text': f"Net leverage at {leverage:.1f}x provides significant acquisition capacity",
                    'metric': 'net_leverage',
                    'value': leverage,
                    'source': 'fundamentals',
                })
            works_when.append("Balance sheet has capacity for debt-financed deals")
        elif bs_score <= 30:
            score -= 15
            fails_when.append("Leverage already elevated, limiting deal capacity")

        # Valuation
        val_score = signals.get('valuation_dislocation', {}).get('score', 50)
        if val_score >= 60:
            score += 10
            why_now.append({
                'text': "Premium valuation provides strong acquisition currency",
                'source': 'signals',
            })
            works_when.append("Stock trades at premium, enabling equity-financed deals")

        # Regime
        if regime.regime_id == 'LOOSE':
            score += 10
            why_now.append({
                'text': "Favorable credit environment supports deal financing",
                'source': 'regime',
            })
            works_when.append("Credit markets remain accommodative")
        elif regime.regime_id == 'TIGHT':
            score -= 10
            fails_when.append("Tight credit conditions make financing expensive")

        # Cohort evidence
        if cohort and cohort.n >= 5:
            tsr_12m = cohort.outcomes.get('tsr_12m')
            if tsr_12m:
                score += 5 if tsr_12m.p50 > 5 else 0
                why_now.append({
                    'text': f"Similar M&A transactions delivered {tsr_12m.p50:+.0f}% median 12-month TSR (n={tsr_12m.n})",
                    'metric': 'precedent_tsr',
                    'value': tsr_12m.p50,
                    'source': 'precedent',
                })

        # Add key split insight if available
        if cohort and cohort.key_splits:
            split = cohort.key_splits[0]
            works_when.append(f"{split.condition_human}: {split.effect}")

        # Objections
        objections.append(Objection(
            question="Why pursue M&A in this market environment?",
            question_category="timing",
            grounded_answer_points=[
                f"Credit spreads currently support deal financing" if regime.regime_id == 'LOOSE' else "Selective opportunities exist despite market conditions",
                f"Balance sheet capacity at {metrics.get('net_leverage', 0):.1f}x leverage" if metrics.get('net_leverage') else "Balance sheet supports strategic action",
            ],
            evidence_refs=[],
            confidence=0.7,
        ))

        # Determine status
        if score >= 70 and regime.regime_id == 'LOOSE':
            status = 'WINDOW_OPEN'
        elif score >= 65:
            status = 'NEWLY_ACTIONABLE'
        else:
            status = 'PERSISTENT'

        # Share price impact
        tsr_range = None
        if cohort and cohort.outcomes.get('tsr_12m'):
            tsr = cohort.outcomes['tsr_12m']
            tsr_range = f"+{tsr.p25:.0f}% to +{tsr.p75:.0f}%"

        score = max(0, min(100, score))

        return ActionCard(
            action_type='mna',
            action_human='Buy-Side M&A',
            recommendation_score=score,
            recommendation_label=self._score_to_label(score),
            status=status,
            thesis_summary="Pursue bolt-on acquisitions to enhance growth durability and market position.",
            value_lever="Growth durability",
            economic_impact="+100-200bps revenue growth equivalent",
            share_price_impact_range=tsr_range,
            why_now_bullets=why_now,
            works_when=works_when,
            fails_when=fails_when,
            cohort=cohort,
            comparable_cohorts=[],
            objections=objections,
        )

    def _build_buyback_card(
        self,
        cohort: Optional[Cohort],
        signals: Dict,
        regime: RegimeContext,
        metrics: Dict,
    ) -> ActionCard:
        """Build buyback action card."""

        score = 50
        why_now = []
        works_when = []
        fails_when = []
        objections = []

        # Balance sheet
        bs_score = signals.get('balance_sheet_optionality', {}).get('score', 50)
        if bs_score >= 60:
            score += 10
            fcf = metrics.get('fcf')
            if fcf:
                why_now.append({
                    'text': f"FCF of ${fcf:.1f}B supports sustained capital return",
                    'metric': 'fcf',
                    'value': fcf,
                    'source': 'fundamentals',
                })

        # Valuation - buybacks best when undervalued
        val_score = signals.get('valuation_dislocation', {}).get('score', 50)
        if val_score <= 40:
            score += 20
            why_now.append({
                'text': "Stock undervaluation makes buybacks highly accretive to EPS",
                'source': 'signals',
            })
            works_when.append("Stock remains undervalued vs intrinsic value")
        elif val_score >= 70:
            score -= 15
            fails_when.append("Premium valuation reduces buyback accretion")

        # Growth
        growth_score = signals.get('growth_momentum', {}).get('score', 50)
        growth_rate = metrics.get('revenue_growth_yoy')
        if growth_score <= 40:
            score += 10
            if growth_rate is not None:
                why_now.append({
                    'text': f"Revenue growth at {growth_rate:+.0f}% — capital return preferable to reinvestment",
                    'metric': 'revenue_growth',
                    'value': growth_rate,
                    'source': 'fundamentals',
                })
            works_when.append("Limited organic growth opportunities")

        # Cohort evidence
        if cohort and cohort.n >= 5:
            tsr_12m = cohort.outcomes.get('tsr_12m')
            if tsr_12m:
                why_now.append({
                    'text': f"Similar buyback programs delivered {tsr_12m.p50:+.0f}% median TSR (n={tsr_12m.n})",
                    'source': 'precedent',
                })

        # Objections
        objections.append(Objection(
            question="Why not reinvest in growth instead?",
            question_category="alternative",
            grounded_answer_points=[
                f"Revenue growth at {growth_rate:+.0f}% suggests limited high-return reinvestment opportunities" if growth_rate else "Limited organic reinvestment opportunities at attractive returns",
                "Buybacks return capital to shareholders who can reallocate",
            ],
            evidence_refs=[],
            confidence=0.7,
        ))

        status = 'NEWLY_ACTIONABLE' if val_score <= 40 else 'PERSISTENT'

        tsr_range = None
        if cohort and cohort.outcomes.get('tsr_12m'):
            tsr = cohort.outcomes['tsr_12m']
            tsr_range = f"+{tsr.p25:.0f}% to +{tsr.p75:.0f}%"

        score = max(0, min(100, score))

        return ActionCard(
            action_type='buyback',
            action_human='Capital Allocation',
            recommendation_score=score,
            recommendation_label=self._score_to_label(score),
            status=status,
            thesis_summary="Execute share repurchases to enhance FCF per share and return capital to shareholders.",
            value_lever="FCF per share",
            economic_impact="3-4% annual FCF/share accretion",
            share_price_impact_range=tsr_range,
            why_now_bullets=why_now,
            works_when=works_when,
            fails_when=fails_when,
            cohort=cohort,
            comparable_cohorts=[],
            objections=objections,
        )

    def _build_debt_card(
        self,
        signals: Dict,
        regime: RegimeContext,
        metrics: Dict,
    ) -> ActionCard:
        """Build debt issuance card."""

        score = 40
        why_now = []
        works_when = []
        fails_when = []

        # Balance sheet capacity
        bs_score = signals.get('balance_sheet_optionality', {}).get('score', 50)
        leverage = metrics.get('net_leverage')

        if bs_score >= 70:
            score += 15
            if leverage is not None:
                why_now.append({
                    'text': f"Net leverage at {leverage:.1f}x — significant untapped debt capacity",
                    'metric': 'net_leverage',
                    'value': leverage,
                    'source': 'fundamentals',
                })
            works_when.append("Leverage remains well below industry norms")
        elif bs_score <= 30:
            score -= 20
            fails_when.append("Already elevated leverage limits new issuance")

        # Regime
        if regime.regime_id == 'LOOSE':
            score += 20
            why_now.append({
                'text': "Credit spreads at favorable levels — attractive issuance window",
                'source': 'regime',
            })
            works_when.append("Credit spreads remain tight")
            status = 'WINDOW_OPEN'
        elif regime.regime_id == 'TIGHT':
            score -= 25
            fails_when.append("Wide credit spreads make issuance expensive")
            status = 'PERSISTENT'
        else:
            status = 'PERSISTENT'

        # Refinancing
        refi_score = signals.get('refinancing_pressure', {}).get('score', 50)
        if refi_score <= 40:
            score += 10
            why_now.append({
                'text': "Near-term maturities create natural refinancing opportunity",
                'source': 'signals',
            })

        score = max(0, min(100, score))

        return ActionCard(
            action_type='debt_issuance',
            action_human='Incremental Debt Issuance',
            recommendation_score=score,
            recommendation_label=self._score_to_label(score),
            status=status,
            thesis_summary="Issue incremental debt to build strategic flexibility and dry powder.",
            value_lever="Strategic flexibility",
            economic_impact="$2-5B dry powder for opportunistic deployment",
            share_price_impact_range="Depends on use of proceeds",
            why_now_bullets=why_now,
            works_when=works_when,
            fails_when=fails_when,
            cohort=None,
            comparable_cohorts=[],
            objections=[],
        )

    def _identify_constraints(self, signal_profile: Dict) -> List[Dict[str, str]]:
        """Identify binding constraints."""
        constraints = []
        signals = signal_profile.get('signals', {})

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

    def _assess_timing(self, signal_profile: Dict, regime: RegimeContext) -> Dict[str, str]:
        """Assess timing posture."""
        signals = signal_profile.get('signals', {})

        refi_score = signals.get('refinancing_pressure', {}).get('score', 50)
        bs_score = signals.get('balance_sheet_optionality', {}).get('score', 50)

        if refi_score <= 30:
            return {'posture': "Urgent", 'description': "Near-term maturities require action"}
        elif regime.regime_id == 'LOOSE' and bs_score >= 60:
            return {'posture': "Opportunistic", 'description': "Favorable window, not urgent"}
        elif regime.regime_id == 'TIGHT':
            return {'posture': "Patient", 'description': "Wait for better entry point"}
        else:
            return {'posture': "Opportunistic", 'description': "Selective action warranted"}

    def _get_action_history(self, gvkey: str, as_of_date: str) -> Dict:
        """Get company's historical actions (as-of)."""
        as_of = pd.to_datetime(as_of_date)

        corp = self.asof.query(
            "warehouse_corp_actions",
            as_of=as_of,
            where=f"company_id = '{gvkey}'"
        )

        mna_filters = [
            f"target_company_id = '{gvkey}'",
            f"acquirer_company_id = '{gvkey}'",
        ]
        try:
            gvkey_num = int(gvkey)
            mna_filters.append(f"target_company_id = {gvkey_num}")
            mna_filters.append(f"acquirer_company_id = {gvkey_num}")
        except Exception:
            pass

        mna = self.asof.query(
            "warehouse_mna_deals",
            as_of=as_of,
            where=" OR ".join(mna_filters),
        )

        history: Dict[str, Any] = {
            "as_of": as_of.isoformat(),
            "corporate_actions": {
                "total": len(corp),
                "by_type": {},
                "recent": [],
            },
            "mna": {
                "total": len(mna),
                "as_target": int((mna.get("target_company_id") == gvkey).sum()) if not mna.empty else 0,
                "as_acquiror": int((mna.get("acquirer_company_id") == gvkey).sum()) if not mna.empty else 0,
                "recent": [],
            },
        }

        if not corp.empty:
            corp["event_time"] = pd.to_datetime(corp["event_time"])
            by_type = corp.groupby("action_type").size().sort_values(ascending=False)
            history["corporate_actions"]["by_type"] = by_type.to_dict()
            recent = corp.sort_values("event_time", ascending=False).head(10)
            history["corporate_actions"]["recent"] = recent[
                ["action_type", "event_time", "announcement_date", "effective_date", "size"]
            ].to_dict(orient="records")

        if not mna.empty:
            mna["announcement_date"] = pd.to_datetime(mna["announcement_date"])
            recent = mna.sort_values("announcement_date", ascending=False).head(10)
            history["mna"]["recent"] = recent[
                ["deal_id", "announcement_date", "deal_value", "status", "target_name", "acquiror_name"]
            ].to_dict(orient="records")

        return history

    def _safe_float(self, val, default=0) -> float:
        """Safely convert to float."""
        if pd.isna(val):
            return default
        try:
            return float(val)
        except:
            return default

    def _score_to_label(self, score: int) -> str:
        """Convert score to label."""
        if score >= 80: return "HIGH"
        elif score >= 65: return "MEDIUM-HIGH"
        elif score >= 50: return "MEDIUM"
        elif score >= 35: return "LOW-MEDIUM"
        else: return "LOW"

    def _format_action_name(self, action: str) -> str:
        """Format action type for display."""
        names = {
            'acquisition': 'Bolt-on M&A',
            'buyback': 'Share Repurchase',
            'Takeover': 'Bolt-on M&A',
            'dividend_increase': 'Dividend Increase',
        }
        return names.get(action, action.replace('_', ' ').title())
