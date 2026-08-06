"""
DealScan to Compustat Linker
============================
Links M&A financing events from DealScan to Compustat fundamentals
so we can compute state profiles at the time of each deal.

Matching strategy:
1. Ticker matching (exact + fuzzy)
2. Company name matching (fuzzy)
3. SIC code validation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
import re

DATA_DIR = Path(__file__).parent.parent / 'data'


class DealLinker:
    """
    Links DealScan M&A facilities to Compustat fundamentals.
    """

    def __init__(
        self,
        dealscan_path: Optional[Path] = None,
        fundamentals_path: Optional[Path] = None,
    ):
        """Load data for linking."""

        # Load DealScan M&A facilities
        if dealscan_path is None:
            dealscan_path = DATA_DIR / 'dealscan_ma_facilities.parquet'

        print(f"Loading DealScan from {dealscan_path}...")
        self.deals = pd.read_parquet(dealscan_path)
        self.deals['facilitystartdate'] = pd.to_datetime(self.deals['facilitystartdate'])
        print(f"Loaded {len(self.deals):,} M&A facilities")

        # Load Compustat fundamentals
        if fundamentals_path is None:
            fundamentals_path = DATA_DIR / 'fundamentals_quarterly.parquet'

        print(f"Loading fundamentals from {fundamentals_path}...")
        self.fundamentals = pd.read_parquet(fundamentals_path)
        self.fundamentals['datadate'] = pd.to_datetime(self.fundamentals['datadate'])
        self.fundamentals['rdq'] = pd.to_datetime(self.fundamentals['rdq'])
        print(f"Loaded {len(self.fundamentals):,} quarterly records")

        # Build ticker lookup from Compustat
        self._build_ticker_lookup()

        # Linked deals cache
        self.linked_deals = None

    def _build_ticker_lookup(self):
        """
        Build a lookup table from ticker -> gvkey.
        Handle multiple gvkeys per ticker by taking the most recent.
        """
        # Get unique ticker-gvkey pairs with latest date
        ticker_gvkey = self.fundamentals.groupby('tic').agg({
            'gvkey': 'first',
            'conm': 'first',
            'datadate': 'max'
        }).reset_index()

        # Clean tickers
        ticker_gvkey['tic_clean'] = ticker_gvkey['tic'].astype(str).str.upper().str.strip()
        ticker_gvkey = ticker_gvkey[ticker_gvkey['tic_clean'].notna() & (ticker_gvkey['tic_clean'] != '')]

        self.ticker_to_gvkey = dict(zip(ticker_gvkey['tic_clean'], ticker_gvkey['gvkey']))
        self.ticker_to_name = dict(zip(ticker_gvkey['tic_clean'], ticker_gvkey['conm']))

        print(f"Built ticker lookup: {len(self.ticker_to_gvkey):,} unique tickers")

    def _clean_ticker(self, ticker: str) -> Optional[str]:
        """Clean and normalize a ticker symbol."""
        if pd.isna(ticker) or ticker is None:
            return None

        # Convert to string and uppercase
        ticker = str(ticker).upper().strip()

        # Remove common suffixes for international stocks
        ticker = re.sub(r'\.\d+$', '', ticker)  # Remove .1, .2, etc.
        ticker = re.sub(r'\.[A-Z]{2}$', '', ticker)  # Remove .SZ, .HK, etc.

        # Remove spaces and special characters
        ticker = re.sub(r'[^A-Z0-9]', '', ticker)

        if len(ticker) == 0:
            return None

        return ticker

    def link_deals(self) -> pd.DataFrame:
        """
        Link DealScan M&A facilities to Compustat gvkeys.

        Returns
        -------
        DataFrame with linked deals (those that matched to Compustat)
        """
        print("\n" + "="*70)
        print("LINKING DEALSCAN TO COMPUSTAT")
        print("="*70)

        # Clean DealScan tickers
        self.deals['ticker_clean'] = self.deals['ticker'].apply(self._clean_ticker)

        # Match by ticker
        matched_gvkey = []
        matched_name = []

        for ticker in self.deals['ticker_clean']:
            if ticker and ticker in self.ticker_to_gvkey:
                matched_gvkey.append(self.ticker_to_gvkey[ticker])
                matched_name.append(self.ticker_to_name.get(ticker))
            else:
                matched_gvkey.append(None)
                matched_name.append(None)

        self.deals['gvkey'] = matched_gvkey
        self.deals['compustat_name'] = matched_name

        # Summary
        n_matched = self.deals['gvkey'].notna().sum()
        n_total = len(self.deals)
        match_rate = n_matched / n_total * 100

        print(f"\nMatching results:")
        print(f"  Total deals: {n_total:,}")
        print(f"  With ticker: {self.deals['ticker_clean'].notna().sum():,}")
        print(f"  Matched to Compustat: {n_matched:,} ({match_rate:.1f}%)")

        # Filter to linked deals
        self.linked_deals = self.deals[self.deals['gvkey'].notna()].copy()

        # Show sample
        print("\nSample linked deals:")
        sample = self.linked_deals[['borrower_name', 'ticker_clean', 'gvkey', 'compustat_name',
                                     'facilitystartdate', 'facilityamt', 'primarypurpose']].head(10)
        print(sample.to_string(index=False))

        # By deal type
        print("\nLinked deals by purpose:")
        print(self.linked_deals['primarypurpose'].value_counts())

        return self.linked_deals

    def get_deal_with_fundamentals(
        self,
        facility_id: int,
    ) -> Optional[Dict]:
        """
        Get a deal with its fundamentals at deal time.

        Returns
        -------
        dict with deal info and fundamentals snapshot
        """
        if self.linked_deals is None:
            self.link_deals()

        # Find the deal
        deal = self.linked_deals[self.linked_deals['facilityid'] == facility_id]
        if len(deal) == 0:
            return None

        deal = deal.iloc[0]
        gvkey = deal['gvkey']
        deal_date = deal['facilitystartdate']

        # Get fundamentals as of deal date
        fundamentals = self.fundamentals[
            (self.fundamentals['gvkey'] == gvkey) &
            (self.fundamentals['rdq'] <= deal_date)
        ].sort_values('datadate', ascending=False)

        if len(fundamentals) == 0:
            return {
                'deal': deal.to_dict(),
                'fundamentals': None,
                'quarters_available': 0,
            }

        return {
            'deal': deal.to_dict(),
            'fundamentals': fundamentals.head(4),  # Last 4 quarters
            'quarters_available': len(fundamentals.head(4)),
        }

    def build_deal_profiles_batch(
        self,
        signal_engine,
        max_deals: Optional[int] = None,
        deal_types: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Compute state profiles for all linked deals.

        Parameters
        ----------
        signal_engine : SignalEngine
            The signal engine to use for computing profiles
        max_deals : int, optional
            Maximum number of deals to process
        deal_types : list, optional
            Filter to specific deal types (e.g., ['LBO', 'Takeover'])

        Returns
        -------
        DataFrame with deal info and state profile vectors
        """
        if self.linked_deals is None:
            self.link_deals()

        deals_to_process = self.linked_deals.copy()

        # Filter by deal type if specified
        if deal_types:
            deals_to_process = deals_to_process[
                deals_to_process['primarypurpose'].isin(deal_types)
            ]

        # Limit if specified
        if max_deals:
            deals_to_process = deals_to_process.head(max_deals)

        print(f"\nComputing state profiles for {len(deals_to_process):,} deals...")

        results = []
        success_count = 0
        fail_count = 0

        for idx, deal in deals_to_process.iterrows():
            gvkey = deal['gvkey']
            deal_date = deal['facilitystartdate']

            # Compute state profile at deal time
            try:
                profile = signal_engine.compute_state_profile(gvkey, deal_date)

                if profile:
                    result = {
                        'facilityid': deal['facilityid'],
                        'gvkey': gvkey,
                        'borrower_name': deal['borrower_name'],
                        'deal_date': deal_date,
                        'deal_type': deal['primarypurpose'],
                        'facility_amount': deal['facilityamt'],
                        'composite_score': profile['composite_score'],
                        'signal_vector': profile['vector'],
                    }

                    # Add individual signals
                    for sig_name, sig_data in profile['signals'].items():
                        result[f'signal_{sig_name}'] = sig_data['score']

                    results.append(result)
                    success_count += 1
                else:
                    fail_count += 1

            except Exception as e:
                fail_count += 1
                continue

            # Progress
            if (success_count + fail_count) % 500 == 0:
                print(f"  Processed {success_count + fail_count:,} deals, {success_count:,} successful")

        print(f"\nCompleted: {success_count:,} profiles computed, {fail_count:,} failed")

        if results:
            profiles_df = pd.DataFrame(results)
            return profiles_df

        return pd.DataFrame()

    def save_linked_deals(self, output_path: Optional[Path] = None):
        """Save linked deals to parquet."""
        if self.linked_deals is None:
            self.link_deals()

        if output_path is None:
            output_path = DATA_DIR / 'dealscan_linked.parquet'

        self.linked_deals.to_parquet(output_path, index=False)
        print(f"Saved {len(self.linked_deals):,} linked deals to {output_path}")


def demo():
    """Demonstrate the deal linker."""
    print("="*70)
    print("DEALSCAN-COMPUSTAT LINKER DEMO")
    print("="*70)

    linker = DealLinker()

    # Link deals
    linked = linker.link_deals()

    # Save linked deals
    linker.save_linked_deals()

    # Compute profiles for a sample
    print("\n" + "="*70)
    print("COMPUTING STATE PROFILES FOR SAMPLE DEALS")
    print("="*70)

    from .signals import SignalEngine

    engine = SignalEngine()

    # Compute profiles for LBO and Takeover deals
    profiles = linker.build_deal_profiles_batch(
        engine,
        max_deals=100,
        deal_types=['LBO', 'Takeover']
    )

    if len(profiles) > 0:
        print(f"\nComputed {len(profiles)} profiles")
        print("\nSample profiles:")
        print(profiles[['borrower_name', 'deal_date', 'deal_type', 'composite_score']].head(10))

        # Save profiles
        output_path = DATA_DIR / 'deal_profiles_sample.parquet'
        profiles.to_parquet(output_path, index=False)
        print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    demo()
