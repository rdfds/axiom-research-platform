"""
Signal Engine
=============
Computes the 5-7 interpretable signals that form the "state profile"
for V1's analog retrieval system.

V1 Signals:
1. Balance Sheet Optionality - Can this company act? (cash, debt capacity)
2. Growth Momentum - Is the business growing or shrinking?
3. Valuation Dislocation - Is it cheap/expensive vs history/peers?
4. Margin Trend - Are margins expanding or compressing?
5. Refinancing Pressure - Is there near-term debt to address?
6. Size Factor - Absolute scale matters for M&A
7. Asset Intensity - Capital structure and asset base

All signals are designed to be:
- Interpretable (a banker can explain them)
- Point-in-time (using rdq, not datadate)
- Deterministic (no ML, just math)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Union
from datetime import datetime

from .snapshot import AsOfSnapshotBuilder, DATA_DIR
from .asof_store import AsOfWarehouse


def safe_float(value, default=0):
    """Safely convert a value to float, handling pandas NA/NaN."""
    if pd.isna(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class SignalEngine:
    """
    Computes state profile signals for companies.

    Each signal is normalized to a 0-100 scale where:
    - Higher = more favorable for M&A activity
    - 50 = neutral/average
    """

    def __init__(self, snapshot_builder: Optional[AsOfSnapshotBuilder] = None):
        """Initialize with a snapshot builder."""
        if snapshot_builder is None:
            snapshot_builder = AsOfSnapshotBuilder()
        self.snapshot = snapshot_builder
        self.asof = AsOfWarehouse()
        self.link_table = None
        link_path = DATA_DIR / "wrds" / "crsp" / "ccmxpf_lnkhist.parquet"
        if link_path.exists():
            self.link_table = pd.read_parquet(link_path)
            self.link_table["linkdt"] = pd.to_datetime(self.link_table["linkdt"], errors="coerce")
            self.link_table["linkenddt"] = pd.to_datetime(self.link_table["linkenddt"], errors="coerce")
            self.link_table["linkenddt"] = self.link_table["linkenddt"].fillna(pd.Timestamp("2099-12-31"))
            self.link_table["gvkey"] = self.link_table["gvkey"].astype(str)

    def _map_gvkey_to_permno(self, gvkey: str, as_of_date: Union[str, datetime]) -> Optional[int]:
        if self.link_table is None:
            return None
        as_of = pd.to_datetime(as_of_date)
        gvkey_str = str(gvkey)
        links = self.link_table[self.link_table["gvkey"] == gvkey_str].copy()
        if links.empty:
            return None
        links = links[(links["linkdt"] <= as_of) & (links["linkenddt"] >= as_of)]
        if links.empty:
            return None
        links["rank"] = 0
        links.loc[links["linkprim"] != "P", "rank"] += 1
        links.loc[~links["linktype"].isin(["LC", "LU", "LD", "LN"]), "rank"] += 1
        links = links.sort_values(["rank", "linkdt"], ascending=[True, False])
        permno = links.iloc[0].get("lpermno")
        if pd.isna(permno):
            return None
        try:
            return int(permno)
        except Exception:
            return None

    def _get_price_history(self, permno: int, as_of_date: Union[str, datetime]) -> pd.DataFrame:
        as_of = pd.to_datetime(as_of_date)
        df = self.asof.query_prices_entity(permno, as_of)
        if df.empty:
            return df
        df["event_time"] = pd.to_datetime(df["event_time"])
        df = df[df["event_time"] <= as_of].sort_values("event_time")
        return df

    def _get_latest_price(self, permno: int, as_of_date: Union[str, datetime]) -> Optional[float]:
        history = self._get_price_history(permno, as_of_date)
        if history.empty:
            return None
        last = history.iloc[-1]
        price = last.get("adjusted_close")
        if price is None or pd.isna(price):
            price = last.get("close")
        return float(price) if price is not None and not pd.isna(price) else None

    def compute_balance_sheet_optionality(
        self,
        gvkey: str,
        as_of_date: Union[str, datetime]
    ) -> Optional[Dict]:
        """
        Signal 1: Balance Sheet Optionality

        Measures ability to act on opportunities:
        - Net cash position (cash - debt)
        - Leverage headroom vs industry norms
        - Undrawn revolver capacity (if available)

        High score = strong balance sheet, capacity for M&A
        Low score = constrained, limited optionality
        """
        data = self.snapshot.get_snapshot(gvkey, as_of_date, lookback_quarters=4)
        if data is None or len(data) == 0:
            return None

        latest = data.iloc[0]

        # Core metrics (use safe_float to handle NA values)
        cash = safe_float(latest.get('cheq'), 0)
        total_debt = safe_float(latest.get('dlttq'), 0) + safe_float(latest.get('dlcq'), 0)
        total_assets = safe_float(latest.get('atq'), 1)
        ebitda_ttm = safe_float(data['oibdpq'].sum()) if 'oibdpq' in data.columns else None

        # Compute sub-signals
        net_cash = cash - total_debt
        cash_to_assets = cash / total_assets if total_assets > 0 else 0

        # Leverage ratio (Net Debt / EBITDA)
        if ebitda_ttm and ebitda_ttm > 0:
            leverage = (total_debt - cash) / ebitda_ttm
        else:
            leverage = None

        # Score components (each 0-100)
        # Cash ratio score: 20%+ cash/assets = 100, 0% = 0
        cash_score = min(100, cash_to_assets * 500)

        # Leverage score: <1x = 100, >5x = 0
        if leverage is not None:
            leverage_score = max(0, min(100, (5 - leverage) / 4 * 100))
        else:
            leverage_score = 50  # neutral if unknown

        # Composite score
        composite = 0.5 * cash_score + 0.5 * leverage_score

        return {
            'signal': 'balance_sheet_optionality',
            'score': round(composite, 1),
            'components': {
                'cash': cash,
                'total_debt': total_debt,
                'net_cash': net_cash,
                'cash_to_assets': round(cash_to_assets, 3),
                'leverage_ratio': round(leverage, 2) if leverage else None,
                'cash_score': round(cash_score, 1),
                'leverage_score': round(leverage_score, 1),
            }
        }

    def compute_growth_momentum(
        self,
        gvkey: str,
        as_of_date: Union[str, datetime]
    ) -> Optional[Dict]:
        """
        Signal 2: Growth Momentum

        Measures business trajectory:
        - Revenue growth (YoY)
        - Sequential revenue growth
        - EBITDA growth

        High score = strong growth, attractive target
        Low score = declining business
        """
        data = self.snapshot.get_snapshot(gvkey, as_of_date, lookback_quarters=8)
        if data is None or len(data) < 5:
            return None

        data = data.sort_values('datadate')

        # YoY revenue growth (compare Q vs Q-4)
        if len(data) >= 5:
            current_rev = safe_float(data.iloc[-1]['revtq'])
            prior_rev = safe_float(data.iloc[-5]['revtq']) if len(data) >= 5 else safe_float(data.iloc[0]['revtq'])

            if prior_rev > 0:
                yoy_growth = (current_rev - prior_rev) / prior_rev
            else:
                yoy_growth = None
        else:
            yoy_growth = None

        # Sequential growth (QoQ)
        if len(data) >= 2:
            current_rev = safe_float(data.iloc[-1]['revtq'])
            prior_rev = safe_float(data.iloc[-2]['revtq'])

            if prior_rev > 0:
                qoq_growth = (current_rev - prior_rev) / prior_rev
            else:
                qoq_growth = None
        else:
            qoq_growth = None

        # TTM EBITDA growth
        if len(data) >= 8 and 'oibdpq' in data.columns:
            current_ebitda = safe_float(data.iloc[-4:]['oibdpq'].sum())
            prior_ebitda = safe_float(data.iloc[-8:-4]['oibdpq'].sum())

            if prior_ebitda > 0:
                ebitda_growth = (current_ebitda - prior_ebitda) / prior_ebitda
            else:
                ebitda_growth = None
        else:
            ebitda_growth = None

        # Score components
        # YoY growth: +30% = 100, -30% = 0, 0% = 50
        if yoy_growth is not None:
            yoy_score = min(100, max(0, (yoy_growth + 0.3) / 0.6 * 100))
        else:
            yoy_score = 50

        if qoq_growth is not None:
            qoq_score = min(100, max(0, (qoq_growth + 0.1) / 0.2 * 100))
        else:
            qoq_score = 50

        if ebitda_growth is not None:
            ebitda_score = min(100, max(0, (ebitda_growth + 0.3) / 0.6 * 100))
        else:
            ebitda_score = 50

        # Weighted composite
        composite = 0.5 * yoy_score + 0.2 * qoq_score + 0.3 * ebitda_score

        return {
            'signal': 'growth_momentum',
            'score': round(composite, 1),
            'components': {
                'yoy_revenue_growth': round(yoy_growth, 3) if yoy_growth else None,
                'qoq_revenue_growth': round(qoq_growth, 3) if qoq_growth else None,
                'yoy_ebitda_growth': round(ebitda_growth, 3) if ebitda_growth else None,
                'yoy_score': round(yoy_score, 1),
                'qoq_score': round(qoq_score, 1),
                'ebitda_score': round(ebitda_score, 1),
            }
        }

    def compute_margin_trend(
        self,
        gvkey: str,
        as_of_date: Union[str, datetime]
    ) -> Optional[Dict]:
        """
        Signal 3: Margin Trend

        Measures profitability trajectory:
        - Current EBITDA margin
        - Margin expansion/compression vs prior year
        - Gross margin trends

        High score = expanding margins, efficiency gains
        Low score = compressing margins
        """
        data = self.snapshot.get_snapshot(gvkey, as_of_date, lookback_quarters=8)
        if data is None or len(data) < 4:
            return None

        data = data.sort_values('datadate')

        # TTM metrics
        current_rev = safe_float(data.iloc[-4:]['revtq'].sum())
        current_ebitda = safe_float(data.iloc[-4:]['oibdpq'].sum()) if 'oibdpq' in data.columns else None

        if current_rev > 0 and current_ebitda:
            current_margin = current_ebitda / current_rev
        else:
            current_margin = None

        # Prior year margin
        if len(data) >= 8:
            prior_rev = safe_float(data.iloc[-8:-4]['revtq'].sum())
            prior_ebitda = safe_float(data.iloc[-8:-4]['oibdpq'].sum()) if 'oibdpq' in data.columns else None

            if prior_rev > 0 and prior_ebitda:
                prior_margin = prior_ebitda / prior_rev
            else:
                prior_margin = None
        else:
            prior_margin = None

        # Margin change
        if current_margin is not None and prior_margin is not None:
            margin_change = current_margin - prior_margin
        else:
            margin_change = None

        # Score components
        # Absolute margin: 25% = 100, 0% = 0
        if current_margin is not None:
            margin_score = min(100, max(0, current_margin * 400))
        else:
            margin_score = 50

        # Margin trend: +5pp = 100, -5pp = 0
        if margin_change is not None:
            trend_score = min(100, max(0, (margin_change + 0.05) / 0.10 * 100))
        else:
            trend_score = 50

        composite = 0.6 * margin_score + 0.4 * trend_score

        return {
            'signal': 'margin_trend',
            'score': round(composite, 1),
            'components': {
                'current_ebitda_margin': round(current_margin, 3) if current_margin else None,
                'prior_ebitda_margin': round(prior_margin, 3) if prior_margin else None,
                'margin_change_pp': round(margin_change * 100, 2) if margin_change else None,
                'margin_score': round(margin_score, 1),
                'trend_score': round(trend_score, 1),
            }
        }

    def compute_valuation_dislocation(
        self,
        gvkey: str,
        as_of_date: Union[str, datetime]
    ) -> Optional[Dict]:
        """
        Signal 4: Valuation Dislocation

        Measures relative value:
        - Price vs historical range (warehouse prices)
        - EV/EBITDA multiple (if available)
        - P/E ratio (if applicable)

        High score = trading at a premium (expensive)
        Low score = trading at a discount (cheap)
        """
        data = self.snapshot.get_snapshot(gvkey, as_of_date, lookback_quarters=8)
        if data is None or len(data) == 0:
            return None

        latest = data.iloc[0]

        # Map gvkey -> permno for price lookup
        permno = self._map_gvkey_to_permno(gvkey, as_of_date)
        price_history = self._get_price_history(permno, as_of_date) if permno else pd.DataFrame()

        price_score = None
        price_percentile = None
        price_current = None
        price_min = None
        price_max = None

        if not price_history.empty:
            price_history = price_history.copy()
            price_history["event_time"] = pd.to_datetime(price_history["event_time"])
            price_history = price_history[price_history["event_time"] <= pd.to_datetime(as_of_date)]
            price_history["price"] = price_history["adjusted_close"].fillna(price_history["close"])
            price_history = price_history.dropna(subset=["price"])

            # Use 36-month lookback if available
            lookback_start = pd.to_datetime(as_of_date) - pd.DateOffset(years=3)
            window = price_history[price_history["event_time"] >= lookback_start]
            if len(window) < 6:
                window = price_history

            if len(window) >= 6:
                price_current = float(window.iloc[-1]["price"])
                price_min = float(window["price"].min())
                price_max = float(window["price"].max())
                price_percentile = float((window["price"] <= price_current).mean())
                price_score = price_percentile * 100

        # Calculate EV using price * shares_out (if possible)
        shares_out = safe_float(latest.get('cshoq'))
        market_cap = price_current * shares_out if price_current and shares_out else None
        total_debt = safe_float(latest.get('dlttq')) + safe_float(latest.get('dlcq'))
        cash = safe_float(latest.get('cheq'))

        if market_cap and market_cap > 0:
            ev = market_cap + total_debt - cash
        else:
            ev = None

        # TTM EBITDA
        ebitda_ttm = safe_float(data['oibdpq'].sum()) if 'oibdpq' in data.columns else None

        # EV/EBITDA multiple
        if ev and ebitda_ttm and ebitda_ttm > 0:
            ev_ebitda = ev / ebitda_ttm
        else:
            ev_ebitda = None

        # P/E ratio
        eps_ttm = safe_float(data['epspxq'].sum()) if 'epspxq' in data.columns else None
        if price_current and eps_ttm and eps_ttm > 0:
            pe_ratio = price_current / eps_ttm
        else:
            pe_ratio = None

        # Scores: higher multiple => higher score (more expensive)
        if ev_ebitda is not None:
            ev_score = min(100, max(0, (ev_ebitda - 5) / 10 * 100))
        else:
            ev_score = None

        if pe_ratio is not None and pe_ratio > 0:
            pe_score = min(100, max(0, (pe_ratio - 10) / 20 * 100))
        else:
            pe_score = None

        scores = [s for s in [price_score, ev_score, pe_score] if s is not None]
        if not scores:
            return None
        composite = sum(scores) / len(scores)

        return {
            'signal': 'valuation_dislocation',
            'score': round(composite, 1),
            'components': {
                'market_cap': market_cap,
                'enterprise_value': round(ev, 1) if ev else None,
                'ev_ebitda': round(ev_ebitda, 2) if ev_ebitda else None,
                'pe_ratio': round(pe_ratio, 2) if pe_ratio else None,
                'price_current': price_current,
                'price_min_3y': price_min,
                'price_max_3y': price_max,
                'price_percentile_3y': round(price_percentile, 3) if price_percentile is not None else None,
                'price_score': round(price_score, 1) if price_score is not None else None,
                'ev_score': round(ev_score, 1) if ev_score is not None else None,
                'pe_score': round(pe_score, 1) if pe_score is not None else None,
            }
        }

    def compute_refinancing_pressure(
        self,
        gvkey: str,
        as_of_date: Union[str, datetime]
    ) -> Optional[Dict]:
        """
        Signal 5: Refinancing Pressure

        Measures near-term debt obligations:
        - Current portion of long-term debt
        - Interest coverage ratio
        - Debt/EBITDA

        High score = minimal pressure (good position)
        Low score = significant refinancing needs
        """
        data = self.snapshot.get_snapshot(gvkey, as_of_date, lookback_quarters=4)
        if data is None or len(data) == 0:
            return None

        latest = data.iloc[0]

        # Debt metrics
        current_debt = safe_float(latest.get('dlcq'))  # Debt due within 1 year
        long_term_debt = safe_float(latest.get('dlttq'))
        total_debt = current_debt + long_term_debt
        cash = safe_float(latest.get('cheq'))

        # TTM metrics
        ebitda_ttm = safe_float(data['oibdpq'].sum()) if 'oibdpq' in data.columns else None
        interest_ttm = safe_float(data['xintq'].sum()) if 'xintq' in data.columns else None

        # Current debt ratio (how much is due soon)
        if total_debt > 0:
            current_debt_ratio = current_debt / total_debt
        else:
            current_debt_ratio = 0

        # Interest coverage
        if interest_ttm and interest_ttm > 0 and ebitda_ttm:
            interest_coverage = ebitda_ttm / interest_ttm
        else:
            interest_coverage = None

        # Net debt / EBITDA
        if ebitda_ttm and ebitda_ttm > 0:
            net_leverage = (total_debt - cash) / ebitda_ttm
        else:
            net_leverage = None

        # Scores (higher = less pressure = better)
        # Current debt ratio: 0% = 100, 50% = 0
        current_score = max(0, (0.5 - current_debt_ratio) / 0.5 * 100)

        # Interest coverage: 10x+ = 100, 1x = 0
        if interest_coverage is not None:
            coverage_score = min(100, max(0, (interest_coverage - 1) / 9 * 100))
        else:
            coverage_score = 50

        # Net leverage: <1x = 100, >5x = 0
        if net_leverage is not None:
            leverage_score = max(0, min(100, (5 - net_leverage) / 4 * 100))
        else:
            leverage_score = 50

        composite = 0.3 * current_score + 0.4 * coverage_score + 0.3 * leverage_score

        return {
            'signal': 'refinancing_pressure',
            'score': round(composite, 1),
            'components': {
                'current_debt': current_debt,
                'long_term_debt': long_term_debt,
                'total_debt': total_debt,
                'current_debt_ratio': round(current_debt_ratio, 3),
                'interest_coverage': round(interest_coverage, 2) if interest_coverage else None,
                'net_leverage': round(net_leverage, 2) if net_leverage else None,
                'current_score': round(current_score, 1),
                'coverage_score': round(coverage_score, 1),
                'leverage_score': round(leverage_score, 1),
            }
        }

    def compute_size_factor(
        self,
        gvkey: str,
        as_of_date: Union[str, datetime]
    ) -> Optional[Dict]:
        """
        Signal 6: Size Factor

        Absolute scale of the company:
        - Total assets
        - Revenue run rate
        - Market cap

        Provides context for M&A (bolt-on vs transformational)
        """
        data = self.snapshot.get_snapshot(gvkey, as_of_date, lookback_quarters=4)
        if data is None or len(data) == 0:
            return None

        latest = data.iloc[0]

        total_assets = safe_float(latest.get('atq'))
        revenue_ttm = safe_float(data['revtq'].sum())
        permno = self._map_gvkey_to_permno(gvkey, as_of_date)
        price_current = self._get_latest_price(permno, as_of_date) if permno else None
        shares_out = safe_float(latest.get('cshoq'))
        market_cap = price_current * shares_out if price_current and shares_out else None

        # Size classification (log scale)
        # <$500M = small, $500M-2B = mid, $2B-10B = large, >$10B = mega
        if total_assets > 0:
            size_log = np.log10(total_assets)
            # Map: 2.7 (500M) = 25, 3.3 (2B) = 50, 4.0 (10B) = 75, 4.7 (50B) = 100
            size_score = min(100, max(0, (size_log - 2.0) / 2.7 * 100))
        else:
            size_score = 0

        # Size bucket
        if total_assets < 500:
            size_bucket = 'Small (<$500M)'
        elif total_assets < 2000:
            size_bucket = 'Mid ($500M-$2B)'
        elif total_assets < 10000:
            size_bucket = 'Large ($2B-$10B)'
        else:
            size_bucket = 'Mega (>$10B)'

        return {
            'signal': 'size_factor',
            'score': round(size_score, 1),
            'components': {
                'total_assets': total_assets,
                'revenue_ttm': revenue_ttm,
                'market_cap': market_cap,
                'size_bucket': size_bucket,
            }
        }

    def compute_asset_intensity(
        self,
        gvkey: str,
        as_of_date: Union[str, datetime]
    ) -> Optional[Dict]:
        """
        Signal 7: Asset Intensity

        Capital structure and asset base:
        - PP&E / Total Assets (capital intensity)
        - Revenue / Assets (asset turnover)
        - Capex intensity

        Informs synergy potential and integration complexity.
        """
        data = self.snapshot.get_snapshot(gvkey, as_of_date, lookback_quarters=4)
        if data is None or len(data) == 0:
            return None

        latest = data.iloc[0]

        total_assets = safe_float(latest.get('atq'), 1)
        ppe = safe_float(latest.get('ppentq'))
        revenue_ttm = safe_float(data['revtq'].sum())

        # Ratios
        ppe_ratio = ppe / total_assets if total_assets > 0 else 0
        asset_turnover = revenue_ttm / total_assets if total_assets > 0 else 0

        # Capex intensity (if available)
        capex_ttm = safe_float(data['capxy'].sum()) if 'capxy' in data.columns else None
        if capex_ttm and revenue_ttm > 0:
            capex_intensity = abs(capex_ttm) / revenue_ttm
        else:
            capex_intensity = None

        # Classification
        if ppe_ratio > 0.4:
            asset_type = 'Asset Heavy'
        elif ppe_ratio > 0.2:
            asset_type = 'Moderate'
        else:
            asset_type = 'Asset Light'

        # Score: Asset-light generally better for M&A integration
        intensity_score = max(0, min(100, (0.5 - ppe_ratio) / 0.5 * 100))
        turnover_score = min(100, asset_turnover * 50)  # Higher turnover = better

        composite = 0.6 * intensity_score + 0.4 * turnover_score

        return {
            'signal': 'asset_intensity',
            'score': round(composite, 1),
            'components': {
                'ppe': ppe,
                'total_assets': total_assets,
                'ppe_ratio': round(ppe_ratio, 3),
                'asset_turnover': round(asset_turnover, 3),
                'capex_intensity': round(capex_intensity, 3) if capex_intensity else None,
                'asset_type': asset_type,
            }
        }

    def compute_state_profile(
        self,
        gvkey: str,
        as_of_date: Union[str, datetime]
    ) -> Optional[Dict]:
        """
        Compute the complete state profile (all signals) for a company.

        Returns
        -------
        dict with:
        - 'gvkey': company identifier
        - 'as_of_date': query date
        - 'signals': dict of signal_name -> {score, components}
        - 'composite_score': weighted average of all signals
        - 'vector': list of scores for similarity calculations
        """
        signals = {}

        # Compute all signals
        signal_funcs = [
            ('balance_sheet_optionality', self.compute_balance_sheet_optionality),
            ('growth_momentum', self.compute_growth_momentum),
            ('margin_trend', self.compute_margin_trend),
            ('valuation_dislocation', self.compute_valuation_dislocation),
            ('refinancing_pressure', self.compute_refinancing_pressure),
            ('size_factor', self.compute_size_factor),
            ('asset_intensity', self.compute_asset_intensity),
        ]

        for name, func in signal_funcs:
            result = func(gvkey, as_of_date)
            if result:
                signals[name] = result

        if len(signals) == 0:
            return None

        # Create vector and composite
        vector = [signals[name]['score'] for name in signals]
        composite = np.mean(vector)

        return {
            'gvkey': gvkey,
            'as_of_date': as_of_date,
            'signals': signals,
            'composite_score': round(composite, 1),
            'vector': vector,
            'signal_names': list(signals.keys()),
        }


def demo():
    """Demonstrate the signal engine."""
    engine = SignalEngine()

    print("\n" + "="*70)
    print("DEMO: Signal Engine")
    print("="*70)

    # Get a sample company
    as_of = '2023-06-30'
    universe = engine.snapshot.get_universe_snapshot(as_of, min_assets=1000, min_revenue=200)

    if len(universe) > 0:
        sample = universe.iloc[0]
        gvkey = sample['gvkey']
        name = sample['conm']

        print(f"\nComputing state profile for {name} ({gvkey}) as of {as_of}")
        print("-" * 70)

        profile = engine.compute_state_profile(gvkey, as_of)

        if profile:
            print(f"\nComposite Score: {profile['composite_score']}/100")
            print(f"Signal Vector: {profile['vector']}")
            print("\nIndividual Signals:")

            for signal_name, signal_data in profile['signals'].items():
                print(f"\n  {signal_name}: {signal_data['score']}/100")
                for key, value in signal_data['components'].items():
                    if isinstance(value, float):
                        print(f"    {key}: {value:,.3f}")
                    else:
                        print(f"    {key}: {value}")


if __name__ == "__main__":
    demo()
