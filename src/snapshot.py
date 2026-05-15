"""
As-Of Snapshot Builder
======================
The most critical component for V1. This module provides point-in-time
snapshots of company fundamentals, avoiding lookahead bias by using
the report date (rdq) rather than the fiscal period end date (datadate).

Key principle: On any given date, we only know what had been FILED,
not what the company's actual financials were.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union
from datetime import datetime, timedelta

DATA_DIR = Path(__file__).parent.parent / 'data'

try:
    from .warehouse_pivots import fetch_financials_asof
except Exception:  # pragma: no cover
    fetch_financials_asof = None


class AsOfSnapshotBuilder:
    """
    Builds point-in-time snapshots of company fundamentals.

    The key insight: Financial data becomes "known" on the FILING date (rdq),
    not the fiscal period end date (datadate). A company's Q4 2023 results
    ending Dec 31 might not be filed until Feb 15, 2024.

    This class ensures we never use information that wasn't available
    at the query date.
    """

    def __init__(self, fundamentals_path: Optional[Path] = None, use_warehouse: bool = True):
        """Load fundamentals data."""
        self.use_warehouse = use_warehouse and fetch_financials_asof is not None
        self.fundamentals_path = fundamentals_path or (DATA_DIR / 'fundamentals_quarterly.parquet')

        if not self.use_warehouse:
            print(f"Loading fundamentals from {self.fundamentals_path}...")
            self.fundamentals = pd.read_parquet(self.fundamentals_path)

            # Ensure date columns are datetime
            self.fundamentals['datadate'] = pd.to_datetime(self.fundamentals['datadate'])
            self.fundamentals['rdq'] = pd.to_datetime(self.fundamentals['rdq'])

            # For records missing rdq, estimate as datadate + 45 days (conservative)
            missing_rdq = self.fundamentals['rdq'].isna()
            self.fundamentals.loc[missing_rdq, 'rdq'] = (
                self.fundamentals.loc[missing_rdq, 'datadate'] + pd.Timedelta(days=45)
            )

            # Sort for efficient lookups
            self.fundamentals = self.fundamentals.sort_values(['gvkey', 'rdq'])

            print(f"Loaded {len(self.fundamentals):,} quarterly records")
            print(f"Companies: {self.fundamentals['gvkey'].nunique():,}")
            print(f"Date range: {self.fundamentals['rdq'].min().date()} to {self.fundamentals['rdq'].max().date()}")
        else:
            self.fundamentals = None

    def get_snapshot(
        self,
        gvkey: str,
        as_of_date: Union[str, datetime],
        lookback_quarters: int = 4
    ) -> Optional[pd.DataFrame]:
        """
        Get the most recent fundamentals snapshot for a company as of a given date.

        Parameters
        ----------
        gvkey : str
            Compustat company identifier
        as_of_date : str or datetime
            The date to query (we only use data filed before this date)
        lookback_quarters : int
            Number of recent quarters to include (for trend calculations)

        Returns
        -------
        DataFrame with the most recent quarters available as of the query date,
        or None if no data available.
        """
        if isinstance(as_of_date, str):
            as_of_date = pd.to_datetime(as_of_date)

        if self.use_warehouse:
            data = fetch_financials_asof(as_of_date, company_id=gvkey)
            if data.empty:
                return None
            data = data.sort_values("event_time", ascending=False)
            data = data.head(lookback_quarters)
            data = data.rename(columns={"company_id": "gvkey", "event_time": "datadate"})
            data["rdq"] = as_of_date
            return data

        # Filter to this company and data filed before as_of_date
        company_data = self.fundamentals[
            (self.fundamentals['gvkey'] == gvkey) &
            (self.fundamentals['rdq'] <= as_of_date)
        ].copy()

        if len(company_data) == 0:
            return None

        # Get the most recent quarters
        company_data = company_data.sort_values('datadate', ascending=False)
        return company_data.head(lookback_quarters)

    def get_latest_snapshot(
        self,
        gvkey: str,
        as_of_date: Union[str, datetime]
    ) -> Optional[pd.Series]:
        """
        Get just the single most recent quarter for a company.

        Returns
        -------
        Series with the most recent quarterly data, or None if unavailable.
        """
        snapshot = self.get_snapshot(gvkey, as_of_date, lookback_quarters=1)
        if snapshot is None or len(snapshot) == 0:
            return None
        return snapshot.iloc[0]

    def get_universe_snapshot(
        self,
        as_of_date: Union[str, datetime],
        min_assets: float = 100,  # Minimum total assets in $M
        min_revenue: float = 50,  # Minimum revenue in $M
    ) -> pd.DataFrame:
        """
        Get the latest snapshot for all companies meeting criteria as of a date.

        This is the main method for building analog universes.

        Parameters
        ----------
        as_of_date : str or datetime
            Query date
        min_assets : float
            Minimum total assets (ATQ) in millions
        min_revenue : float
            Minimum quarterly revenue (REVTQ) in millions

        Returns
        -------
        DataFrame with one row per company (their most recent filing as of the date)
        """
        if isinstance(as_of_date, str):
            as_of_date = pd.to_datetime(as_of_date)

        if self.use_warehouse:
            data = fetch_financials_asof(as_of_date)
            if data.empty:
                return pd.DataFrame()
            data = data.rename(columns={"company_id": "gvkey", "event_time": "datadate"})
            data["rdq"] = as_of_date
            latest = data.sort_values("datadate", ascending=False).groupby("gvkey").head(1)
        else:
            # Filter to data filed before as_of_date
            available = self.fundamentals[
                self.fundamentals['rdq'] <= as_of_date
            ].copy()

            # Get most recent quarter for each company
            latest_idx = available.groupby('gvkey')['datadate'].idxmax()
            latest = available.loc[latest_idx].copy()

        # Apply size filters
        if min_assets > 0 and 'atq' in latest.columns:
            latest = latest[latest['atq'] >= min_assets]
        if min_revenue > 0 and 'revtq' in latest.columns:
            latest = latest[latest['revtq'] >= min_revenue]

        # Filter out stale data (more than 6 months old)
        max_staleness = as_of_date - pd.Timedelta(days=180)
        if 'rdq' in latest.columns:
            latest = latest[latest['rdq'] >= max_staleness]

        print(f"Universe as of {as_of_date.date()}: {len(latest):,} companies")
        return latest.reset_index(drop=True)

    def get_trailing_metrics(
        self,
        gvkey: str,
        as_of_date: Union[str, datetime],
        quarters: int = 4
    ) -> Optional[dict]:
        """
        Calculate trailing twelve month (TTM) metrics for a company.

        Parameters
        ----------
        gvkey : str
            Company identifier
        as_of_date : str or datetime
            Query date
        quarters : int
            Number of quarters to sum (4 for TTM)

        Returns
        -------
        dict with TTM metrics, or None if insufficient data
        """
        snapshot = self.get_snapshot(gvkey, as_of_date, lookback_quarters=quarters)

        if snapshot is None or len(snapshot) < quarters:
            return None

        # Sum flow items (income statement, cash flow)
        ttm = {
            'gvkey': gvkey,
            'as_of_date': as_of_date,
            'quarters_used': len(snapshot),
            'latest_datadate': snapshot['datadate'].max(),
            'latest_rdq': snapshot['rdq'].max(),

            # TTM Income Statement
            'revenue_ttm': snapshot['revtq'].sum(),
            'cogs_ttm': snapshot['cogsq'].sum() if 'cogsq' in snapshot else None,
            'ebitda_ttm': snapshot['oibdpq'].sum() if 'oibdpq' in snapshot else None,
            'ebit_ttm': snapshot['oiadpq'].sum() if 'oiadpq' in snapshot else None,
            'net_income_ttm': snapshot['niq'].sum() if 'niq' in snapshot else None,
            'interest_expense_ttm': snapshot['xintq'].sum() if 'xintq' in snapshot else None,

            # Latest Balance Sheet (point-in-time, not summed)
            'total_assets': snapshot.iloc[0]['atq'],
            'total_liabilities': snapshot.iloc[0]['ltq'],
            'total_debt': (
                (snapshot.iloc[0]['dlttq'] or 0) +
                (snapshot.iloc[0]['dlcq'] or 0)
            ),
            'cash': snapshot.iloc[0]['cheq'],
            'equity': snapshot.iloc[0]['ceqq'],

            # Market data (if available)
            'market_cap': snapshot.iloc[0].get('mkvaltq'),
            'stock_price': snapshot.iloc[0].get('prccq'),
            'shares_out': snapshot.iloc[0].get('cshoq'),
        }

        # Calculate derived metrics
        if ttm['revenue_ttm'] and ttm['revenue_ttm'] > 0:
            if ttm['ebitda_ttm']:
                ttm['ebitda_margin'] = ttm['ebitda_ttm'] / ttm['revenue_ttm']

        if ttm['ebitda_ttm'] and ttm['ebitda_ttm'] > 0:
            ttm['net_debt'] = ttm['total_debt'] - (ttm['cash'] or 0)
            ttm['leverage_ratio'] = ttm['net_debt'] / ttm['ebitda_ttm']

        return ttm


def demo():
    """Demonstrate the snapshot builder."""
    builder = AsOfSnapshotBuilder()

    print("\n" + "="*70)
    print("DEMO: As-Of Snapshot Builder")
    print("="*70)

    # Get universe as of a specific date
    as_of = '2023-06-30'
    universe = builder.get_universe_snapshot(as_of, min_assets=500, min_revenue=100)

    print(f"\nTop 10 companies by assets as of {as_of}:")
    print(universe.nlargest(10, 'atq')[['gvkey', 'conm', 'tic', 'atq', 'revtq', 'rdq']])

    # Get detailed snapshot for one company
    if len(universe) > 0:
        sample_gvkey = universe.iloc[0]['gvkey']
        sample_name = universe.iloc[0]['conm']

        print(f"\n{'='*70}")
        print(f"Detailed TTM snapshot for {sample_name} ({sample_gvkey})")
        print("="*70)

        ttm = builder.get_trailing_metrics(sample_gvkey, as_of)
        if ttm:
            for key, value in ttm.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:,.2f}")
                else:
                    print(f"  {key}: {value}")


if __name__ == "__main__":
    demo()
