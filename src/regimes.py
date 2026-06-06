"""
Market Regime Classifier (Data-Driven)
======================================
Classifies the current market environment into one of 3 regimes
based on observable market data, NOT hardcoded dates.

Data sources:
1. Realized volatility from stock returns (CRSP)
2. M&A loan volume from DealScan (credit availability proxy)

V1 Regimes:
1. LOOSE - Low volatility, high deal volume, easy credit
2. SELECTIVE - Normal conditions
3. TIGHT - High volatility, low deal volume, credit contraction
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Union
from datetime import datetime, timedelta

DATA_DIR = Path(__file__).parent.parent / 'data'


class RegimeClassifier:
    """
    Data-driven regime classifier using market observables.

    Methodology:
    1. Compute trailing realized volatility from stock returns
    2. Compute trailing M&A deal volume from DealScan
    3. Combine into a regime score
    4. Classify based on percentile thresholds
    """

    def __init__(
        self,
        prices_path: Optional[Path] = None,
        dealscan_path: Optional[Path] = None,
    ):
        """Initialize with market data."""

        # Load price data for volatility calculation
        if prices_path is None:
            prices_path = DATA_DIR / 'prices_monthly.parquet'

        self.monthly_vol = None
        if prices_path.exists():
            self._load_market_volatility(prices_path)

        # Load DealScan for deal volume
        if dealscan_path is None:
            dealscan_path = DATA_DIR / 'dealscan_ma_facilities.parquet'

        self.deal_volume = None
        if dealscan_path.exists():
            self._load_deal_volume(dealscan_path)

        # Pre-compute regime history
        self.regime_history = None
        self._compute_regime_history()

    def _load_market_volatility(self, prices_path: Path):
        """
        Compute monthly realized volatility from stock returns.
        Uses cross-sectional dispersion as a market-wide uncertainty measure.
        """
        print("Loading price data for volatility calculation...")
        prices = pd.read_parquet(prices_path)
        prices['date'] = pd.to_datetime(prices['date'])

        # Compute monthly cross-sectional volatility
        prices['month'] = prices['date'].dt.to_period('M')

        monthly_stats = prices.groupby('month')['ret'].agg(['std', 'mean', 'count'])
        monthly_stats.columns = ['vol', 'mean_ret', 'n_stocks']
        monthly_stats['vol_annualized'] = monthly_stats['vol'] * np.sqrt(12) * 100

        self.monthly_vol = monthly_stats.reset_index()
        self.monthly_vol['month'] = self.monthly_vol['month'].dt.to_timestamp()

        print(f"Computed volatility for {len(self.monthly_vol)} months")
        print(f"Vol range: {self.monthly_vol['vol_annualized'].min():.1f}% to {self.monthly_vol['vol_annualized'].max():.1f}%")

    def _load_deal_volume(self, dealscan_path: Path):
        """
        Compute monthly M&A deal volume from DealScan.
        High deal volume = loose credit, low volume = tight credit.
        """
        print("Loading DealScan for deal volume...")
        deals = pd.read_parquet(dealscan_path)
        deals['facilitystartdate'] = pd.to_datetime(deals['facilitystartdate'])
        deals['month'] = deals['facilitystartdate'].dt.to_period('M')

        # Aggregate by month
        monthly_deals = deals.groupby('month').agg({
            'facilityid': 'count',
            'facilityamt': 'sum'
        }).rename(columns={
            'facilityid': 'deal_count',
            'facilityamt': 'deal_volume'
        })

        # Convert to billions
        monthly_deals['deal_volume_bn'] = monthly_deals['deal_volume'] / 1e9

        self.deal_volume = monthly_deals.reset_index()
        self.deal_volume['month'] = self.deal_volume['month'].dt.to_timestamp()

        print(f"Computed deal volume for {len(self.deal_volume)} months")

    def _compute_regime_history(self):
        """
        Compute regime classification for all historical months.

        Uses a composite score based on:
        - Volatility percentile (high vol = tight)
        - Deal volume percentile (low volume = tight)
        """
        if self.monthly_vol is None or self.deal_volume is None:
            print("Warning: Insufficient data for data-driven regime computation")
            return

        # Merge volatility and deal volume
        merged = pd.merge(
            self.monthly_vol[['month', 'vol_annualized']],
            self.deal_volume[['month', 'deal_count', 'deal_volume_bn']],
            on='month',
            how='inner'
        )

        if len(merged) == 0:
            print("Warning: No overlapping data between volatility and deals")
            return

        # Compute rolling percentiles (36-month lookback window)
        window = 36

        merged['vol_pct'] = merged['vol_annualized'].rolling(
            window=window, min_periods=12
        ).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])

        merged['deal_pct'] = merged['deal_count'].rolling(
            window=window, min_periods=12
        ).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])

        # Composite regime score:
        # High vol (bad) + low deals (bad) = TIGHT (score near 0)
        # Low vol (good) + high deals (good) = LOOSE (score near 1)
        merged['regime_score'] = (1 - merged['vol_pct']) * 0.5 + merged['deal_pct'] * 0.5

        # Classify based on score thresholds
        def classify(score):
            if pd.isna(score):
                return 'SELECTIVE'
            elif score >= 0.6:
                return 'LOOSE'
            elif score <= 0.4:
                return 'TIGHT'
            else:
                return 'SELECTIVE'

        merged['regime'] = merged['regime_score'].apply(classify)

        self.regime_history = merged

        # Print summary
        print(f"\nRegime distribution (data-driven):")
        print(merged['regime'].value_counts())

    def classify_regime(
        self,
        as_of_date: Union[str, datetime],
    ) -> Dict:
        """
        Classify the market regime for a given date.

        Parameters
        ----------
        as_of_date : str or datetime
            The date to classify

        Returns
        -------
        dict with regime classification and underlying indicators
        """
        if isinstance(as_of_date, str):
            as_of_date = pd.to_datetime(as_of_date)

        # Find the month
        query_month = as_of_date.replace(day=1)

        if self.regime_history is None or len(self.regime_history) == 0:
            return self._fallback_classification(as_of_date)

        # Look up in history (find most recent data at or before query date)
        match = self.regime_history[
            self.regime_history['month'] <= query_month
        ].sort_values('month', ascending=False)

        if len(match) == 0:
            return self._fallback_classification(as_of_date)

        row = match.iloc[0]

        return {
            'regime': row['regime'],
            'regime_score': round(float(row['regime_score']), 3) if pd.notna(row['regime_score']) else None,
            'as_of_date': as_of_date,
            'data_date': row['month'],
            'confidence': 'high' if pd.notna(row['regime_score']) else 'low',
            'method': 'data_driven',
            'description': self._get_regime_description(row['regime'], row),
            'indicators': {
                'volatility_annualized': round(float(row['vol_annualized']), 2),
                'volatility_percentile': round(float(row['vol_pct']), 2) if pd.notna(row['vol_pct']) else None,
                'deal_count': int(row['deal_count']),
                'deal_volume_bn': round(float(row['deal_volume_bn']), 2),
                'deal_percentile': round(float(row['deal_pct']), 2) if pd.notna(row['deal_pct']) else None,
            }
        }

    def _get_regime_description(self, regime: str, row: pd.Series) -> str:
        """Generate a description based on the data."""
        vol = row['vol_annualized']
        deals = row['deal_count']

        if regime == 'LOOSE':
            return f"Low volatility ({vol:.0f}%), high deal activity ({deals} deals/month)"
        elif regime == 'TIGHT':
            return f"High volatility ({vol:.0f}%), low deal activity ({deals} deals/month)"
        else:
            return f"Normal conditions (vol: {vol:.0f}%, deals: {deals}/month)"

    def _fallback_classification(self, as_of_date: datetime) -> Dict:
        """Fallback when we don't have data."""
        return {
            'regime': 'SELECTIVE',
            'regime_score': 0.5,
            'as_of_date': as_of_date,
            'confidence': 'low',
            'method': 'fallback',
            'description': 'No data available for this date',
            'indicators': {},
        }

    def get_regime_characteristics(self, regime: str) -> Dict:
        """
        Get the expected characteristics of a regime.
        """
        characteristics = {
            'LOOSE': {
                'description': 'Low volatility, high deal activity, easy credit',
                'typical_ev_ebitda_range': (8, 14),
                'deal_activity': 'High',
                'financing_availability': 'Abundant',
                'seller_power': 'High',
                'expected_premium_range': (25, 45),
                'typical_leverage': (4.5, 6.5),
            },
            'SELECTIVE': {
                'description': 'Normal volatility, moderate deal activity',
                'typical_ev_ebitda_range': (6, 10),
                'deal_activity': 'Moderate',
                'financing_availability': 'Available for quality',
                'seller_power': 'Balanced',
                'expected_premium_range': (20, 35),
                'typical_leverage': (3.5, 5.0),
            },
            'TIGHT': {
                'description': 'High volatility, low deal activity, credit contraction',
                'typical_ev_ebitda_range': (4, 8),
                'deal_activity': 'Low (strategic/distressed)',
                'financing_availability': 'Limited',
                'seller_power': 'Low',
                'expected_premium_range': (10, 25),
                'typical_leverage': (2.5, 4.0),
            },
        }
        return characteristics.get(regime, characteristics['SELECTIVE'])

    def get_regime_adjusted_expectations(
        self,
        base_profile: Dict,
        regime: str,
    ) -> Dict:
        """
        Adjust outcome expectations based on current regime.
        """
        chars = self.get_regime_characteristics(regime)

        ev_ebitda_low, ev_ebitda_high = chars['typical_ev_ebitda_range']
        premium_low, premium_high = chars['expected_premium_range']

        # Adjust based on company quality
        quality_factor = base_profile.get('composite_score', 50) / 50

        return {
            'regime': regime,
            'expected_ev_ebitda': {
                'low': ev_ebitda_low * (0.8 + 0.4 * quality_factor),
                'base': (ev_ebitda_low + ev_ebitda_high) / 2 * quality_factor,
                'high': ev_ebitda_high * (0.6 + 0.4 * quality_factor),
            },
            'expected_premium_pct': {
                'low': premium_low,
                'base': (premium_low + premium_high) / 2,
                'high': premium_high,
            },
            'deal_likelihood': chars['deal_activity'],
            'financing_outlook': chars['financing_availability'],
        }

    def get_regime_timeline(
        self,
        start_date: str = '2010-01-01',
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get regime classifications over a time period.
        """
        if self.regime_history is None or len(self.regime_history) == 0:
            return pd.DataFrame()

        df = self.regime_history.copy()
        df = df[df['month'] >= pd.to_datetime(start_date)]

        if end_date:
            df = df[df['month'] <= pd.to_datetime(end_date)]

        return df[['month', 'regime', 'regime_score', 'vol_annualized', 'deal_count']]


def demo():
    """Demonstrate the data-driven regime classifier."""
    print("="*70)
    print("DATA-DRIVEN REGIME CLASSIFIER")
    print("="*70)

    classifier = RegimeClassifier()

    # Test different dates
    test_dates = [
        '2010-06-01',
        '2012-06-01',
        '2015-06-01',
        '2016-02-01',  # China scare
        '2018-12-01',  # Q4 selloff
        '2019-06-01',
        '2020-03-15',  # COVID
        '2020-09-01',  # Post-COVID
    ]

    print("\n" + "="*70)
    print("REGIME CLASSIFICATIONS (DATA-DRIVEN)")
    print("="*70)

    for date in test_dates:
        result = classifier.classify_regime(date)
        print(f"\n{date}: {result['regime']}")
        print(f"  Score: {result.get('regime_score', 'N/A')}")
        print(f"  {result.get('description', '')}")
        if result['indicators']:
            ind = result['indicators']
            print(f"  Vol: {ind.get('volatility_annualized', 'N/A')}% (pct: {ind.get('volatility_percentile', 'N/A')})")
            print(f"  Deals: {ind.get('deal_count', 'N/A')}/month (pct: {ind.get('deal_percentile', 'N/A')})")

    # Show transitions
    print("\n" + "="*70)
    print("REGIME TRANSITIONS")
    print("="*70)

    timeline = classifier.get_regime_timeline()
    if len(timeline) > 0:
        timeline['prev_regime'] = timeline['regime'].shift(1)
        transitions = timeline[timeline['regime'] != timeline['prev_regime']].dropna()
        print(transitions[['month', 'regime', 'regime_score']].head(20).to_string(index=False))


if __name__ == "__main__":
    demo()
