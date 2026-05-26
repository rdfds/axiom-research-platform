"""
Outcome Calculator
==================
Computes Total Shareholder Return (TSR) for corporate actions.

TSR = (Price_end - Price_start + Dividends) / Price_start

This answers "how did it turn out?" for each action.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta

from .snapshot import DATA_DIR


class OutcomeCalculator:
    """
    Calculates TSR outcomes for corporate actions.

    Measures returns at multiple horizons:
    - 1 month (immediate reaction)
    - 3 months (short-term)
    - 6 months (medium-term)
    - 12 months (full year)
    """

    def __init__(self, prices_path: Optional[Path] = None):
        """Load price data."""
        if prices_path is None:
            prices_path = DATA_DIR / 'prices_monthly.parquet'

        if prices_path.exists():
            print(f"Loading prices from {prices_path}...")
            self.prices = pd.read_parquet(prices_path)
            self.prices['date'] = pd.to_datetime(self.prices['date'])

            # Create lookup structure: gvkey -> sorted price series
            self.price_lookup = {}
            for gvkey, group in self.prices.groupby('gvkey'):
                self.price_lookup[gvkey] = group.sort_values('date')[['date', 'prc', 'ret']].copy()

            print(f"Loaded prices for {len(self.price_lookup):,} companies")
        else:
            print(f"Warning: No prices file at {prices_path}")
            self.prices = None
            self.price_lookup = {}

    def get_price_at_date(
        self,
        gvkey: str,
        target_date: datetime,
        lookback_days: int = 45,
    ) -> Optional[float]:
        """
        Get price closest to target date.

        Uses monthly data, so finds closest month-end price.
        """
        if gvkey not in self.price_lookup:
            return None

        prices = self.price_lookup[gvkey]

        # Find closest price within lookback window
        target = pd.to_datetime(target_date)
        start = target - pd.Timedelta(days=lookback_days)
        end = target + pd.Timedelta(days=lookback_days)

        mask = (prices['date'] >= start) & (prices['date'] <= end)
        nearby = prices[mask]

        if len(nearby) == 0:
            return None

        # Get closest to target
        nearby = nearby.copy()
        nearby['diff'] = abs(nearby['date'] - target)
        closest = nearby.loc[nearby['diff'].idxmin()]

        return float(closest['prc']) if pd.notna(closest['prc']) else None

    def compute_tsr(
        self,
        gvkey: str,
        action_date: datetime,
        horizon_months: int = 12,
    ) :
        """
        Compute TSR from action date over specified horizon.

        Uses monthly returns (ret column) which are split/dividend adjusted,
        then compounds them over the horizon period.

        Returns
        -------
        dict with:
        - 'tsr': total return as decimal (0.15 = 15%)
        - 'tsr_pct': total return as percentage
        - 'tsr_annualized': annualized return
        - 'n_months': number of months with data
        """
        if gvkey not in self.price_lookup:
            return None

        action_date = pd.to_datetime(action_date)
        end_date = action_date + pd.DateOffset(months=horizon_months)

        prices = self.price_lookup[gvkey]

        # Get returns in the horizon window
        # Start from the month AFTER the action (first full month of holding)
        mask = (prices['date'] > action_date) & (prices['date'] <= end_date)
        period_returns = prices[mask]['ret'].dropna()

        if len(period_returns) == 0:
            return None

        # Compound the monthly returns: (1+r1) * (1+r2) * ... - 1
        cumulative = (1 + period_returns).prod() - 1

        # Annualize
        n_months = len(period_returns)
        if n_months > 0 and n_months != 12:
            tsr_annualized = (1 + cumulative) ** (12 / n_months) - 1
        else:
            tsr_annualized = cumulative

        return {
            'tsr': round(cumulative, 4),
            'tsr_pct': round(cumulative * 100, 1),
            'tsr_annualized': round(tsr_annualized, 4),
            'n_months': n_months,
            'horizon_months': horizon_months,
        }

    def compute_multi_horizon_tsr(
        self,
        gvkey: str,
        action_date: datetime,
    ) -> Optional[Dict]:
        """
        Compute TSR at multiple horizons.

        Returns dict with tsr_1m, tsr_3m, tsr_6m, tsr_12m
        """
        results = {}

        for months in [1, 3, 6, 12]:
            tsr = self.compute_tsr(gvkey, action_date, months)
            if tsr:
                results[f'tsr_{months}m'] = tsr['tsr_pct']
            else:
                results[f'tsr_{months}m'] = None

        if len([v for v in results.values() if v is not None]) == 0:
            return None

        return results

    def compute_relative_tsr(
        self,
        gvkey: str,
        action_date: datetime,
        horizon_months: int = 12,
        benchmark: str = 'market',
    ) -> Optional[Dict]:
        """
        Compute TSR relative to market/benchmark.

        Uses median return of all stocks over same period as benchmark.
        """
        company_tsr = self.compute_tsr(gvkey, action_date, horizon_months)

        if company_tsr is None:
            return None

        # Compute market return over same period using compounded returns
        action_date = pd.to_datetime(action_date)
        end_date = action_date + pd.DateOffset(months=horizon_months)

        market_returns = []
        for other_gvkey, prices in self.price_lookup.items():
            mask = (prices['date'] > action_date) & (prices['date'] <= end_date)
            period_returns = prices[mask]['ret'].dropna()

            if len(period_returns) >= horizon_months - 1:  # Require most months
                cumulative = (1 + period_returns).prod() - 1
                if -0.99 < cumulative < 10:  # Filter extreme outliers
                    market_returns.append(cumulative)

        if len(market_returns) < 50:
            return None

        market_tsr = np.median(market_returns)

        excess_return = company_tsr['tsr'] - market_tsr

        return {
            'company_tsr': company_tsr['tsr_pct'],
            'market_tsr': round(market_tsr * 100, 1),
            'excess_return': round(excess_return * 100, 1),
            'horizon_months': horizon_months,
        }


