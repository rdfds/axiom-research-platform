"""
Analog Retrieval System
=======================
Finds historical M&A transactions with similar state profiles
to provide empirical outcome distributions.

Key concept: "Companies in similar states faced similar decisions.
What happened to them?"

V1 uses DealScan M&A financing as the primary transaction source,
linked to Compustat for fundamentals-based state profiles.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Union, Tuple
from datetime import datetime
from scipy.spatial.distance import cosine, euclidean

from .snapshot import AsOfSnapshotBuilder, DATA_DIR
from .signals import SignalEngine


class AnalogRetriever:
    """
    Retrieves historical analog cases for a given company state profile.

    The system:
    1. Takes a query company's state profile (7 signals)
    2. Searches the DealScan M&A financing database (linked to Compustat)
    3. Uses pre-computed state profiles for deals at time of financing
    4. Returns similar cases with their deal characteristics
    """

    def __init__(
        self,
        signal_engine: Optional[SignalEngine] = None,
        use_dealscan: bool = True,
        prices_path: Optional[Path] = None,
    ):
        """Initialize the analog retrieval system."""
        if signal_engine is None:
            signal_engine = SignalEngine()
        self.signals = signal_engine

        # Load deal profiles (pre-computed from DealScan-Compustat linked deals)
        self.deal_profiles = None
        self.linked_deals = None

        if use_dealscan:
            self._load_dealscan_data()
        else:
            self._load_ciq_data()

        # Load prices for outcome calculation
        if prices_path is None:
            prices_path = DATA_DIR / 'prices_monthly.parquet'

        if prices_path.exists():
            print(f"Loading prices from {prices_path}...")
            self.prices = pd.read_parquet(prices_path)
            self.prices['date'] = pd.to_datetime(self.prices['date'])
            print(f"Loaded {len(self.prices):,} price records")
        else:
            print(f"Warning: No prices file at {prices_path}")
            self.prices = None

    def _load_dealscan_data(self):
        """Load DealScan linked deals and pre-computed profiles."""
        # Try to load pre-computed profiles first
        profiles_path = DATA_DIR / 'deal_profiles_all.parquet'
        if profiles_path.exists():
            print(f"Loading pre-computed deal profiles from {profiles_path}...")
            self.deal_profiles = pd.read_parquet(profiles_path)
            print(f"Loaded {len(self.deal_profiles):,} deal profiles")
        else:
            # Fall back to sample profiles
            sample_path = DATA_DIR / 'deal_profiles_sample.parquet'
            if sample_path.exists():
                print(f"Loading sample deal profiles from {sample_path}...")
                self.deal_profiles = pd.read_parquet(sample_path)
                print(f"Loaded {len(self.deal_profiles):,} deal profiles (sample)")

        # Load linked deals for metadata
        linked_path = DATA_DIR / 'dealscan_linked.parquet'
        if linked_path.exists():
            print(f"Loading linked deals from {linked_path}...")
            self.linked_deals = pd.read_parquet(linked_path)
            self.linked_deals['facilitystartdate'] = pd.to_datetime(
                self.linked_deals['facilitystartdate']
            )
            print(f"Loaded {len(self.linked_deals):,} linked deals")

        # For backward compatibility
        self.transactions = self.linked_deals

    def _load_ciq_data(self):
        """Load Capital IQ transaction data (legacy fallback)."""
        transactions_path = DATA_DIR / 'ciqsamp_transactions_wrds_transactions.parquet'

        if transactions_path.exists():
            print(f"Loading CIQ transactions from {transactions_path}...")
            self.transactions = pd.read_parquet(transactions_path)
            print(f"Loaded {len(self.transactions):,} transactions")
            self._prepare_ciq_transactions()
        else:
            print(f"Warning: No transactions file at {transactions_path}")
            self.transactions = None

    def _prepare_ciq_transactions(self):
        """Clean and prepare Capital IQ transaction data (legacy)."""
        if self.transactions is None:
            return

        # Identify key columns (Capital IQ naming varies)
        cols = self.transactions.columns.tolist()

        # Try to find date column
        date_cols = [c for c in cols if 'date' in c.lower() and 'announce' in c.lower()]
        if date_cols:
            self.transactions['deal_date'] = pd.to_datetime(
                self.transactions[date_cols[0]], errors='coerce'
            )
        elif 'transactionannouncedate' in cols:
            self.transactions['deal_date'] = pd.to_datetime(
                self.transactions['transactionannouncedate'], errors='coerce'
            )

        # Try to find deal value
        value_cols = [c for c in cols if 'value' in c.lower() or 'size' in c.lower()]
        if 'transactionvalue' in cols:
            self.transactions['deal_value'] = pd.to_numeric(
                self.transactions['transactionvalue'], errors='coerce'
            )
        elif value_cols:
            self.transactions['deal_value'] = pd.to_numeric(
                self.transactions[value_cols[0]], errors='coerce'
            )

        # Filter to completed M&A deals
        if 'transactionstatus' in self.transactions.columns:
            # Keep completed deals
            completed_mask = self.transactions['transactionstatus'].str.lower().str.contains(
                'complet|closed|effect', na=False
            )
            self.transactions = self.transactions[completed_mask].copy()

        # Drop rows without dates
        if 'deal_date' in self.transactions.columns:
            self.transactions = self.transactions.dropna(subset=['deal_date'])

        print(f"Prepared {len(self.transactions):,} transactions for analysis")

        # Show transaction types
        if 'transactiontype' in self.transactions.columns:
            print("\nTransaction types:")
            print(self.transactions['transactiontype'].value_counts().head(10))

    def get_deal_universe(
        self,
        deal_type: Optional[str] = None,
        min_value: float = 0,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get the universe of deals for analog matching.

        Parameters
        ----------
        deal_type : str, optional
            Filter by transaction type (e.g., 'M&A', 'Merger', 'Acquisition')
        min_value : float
            Minimum deal value in millions
        start_date, end_date : str
            Date range for deals

        Returns
        -------
        DataFrame of deals
        """
        if self.transactions is None:
            return pd.DataFrame()

        deals = self.transactions.copy()

        if deal_type and 'transactiontype' in deals.columns:
            deals = deals[
                deals['transactiontype'].str.lower().str.contains(deal_type.lower(), na=False)
            ]

        if min_value > 0 and 'deal_value' in deals.columns:
            deals = deals[deals['deal_value'] >= min_value]

        if start_date and 'deal_date' in deals.columns:
            deals = deals[deals['deal_date'] >= pd.to_datetime(start_date)]

        if end_date and 'deal_date' in deals.columns:
            deals = deals[deals['deal_date'] <= pd.to_datetime(end_date)]

        return deals

    def _compute_similarity(
        self,
        query_vector: List[float],
        candidate_vector: List[float],
        method: str = 'cosine'
    ) -> float:
        """
        Compute similarity between two state vectors.

        Returns
        -------
        float: Similarity score (0-1, higher = more similar)
        """
        if len(query_vector) != len(candidate_vector):
            return 0.0

        query = np.array(query_vector)
        candidate = np.array(candidate_vector)

        if method == 'cosine':
            # Cosine similarity (1 - cosine distance)
            similarity = 1 - cosine(query, candidate)
        elif method == 'euclidean':
            # Normalized euclidean (max distance ~141 for 7 signals 0-100)
            max_distance = np.sqrt(len(query) * 100**2)
            similarity = 1 - (euclidean(query, candidate) / max_distance)
        else:
            raise ValueError(f"Unknown method: {method}")

        return max(0, min(1, similarity))

    def find_analogs(
        self,
        query_profile: Dict,
        n_analogs: int = 10,
        min_similarity: float = 0.7,
        same_industry: bool = False,
        size_range: Tuple[float, float] = (0.3, 3.0),  # 0.3x to 3x size
        deal_types: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Find historical analog transactions for a query company.

        Uses pre-computed deal profiles from DealScan-Compustat linked data
        for fast retrieval. Falls back to on-the-fly computation if needed.

        Parameters
        ----------
        query_profile : dict
            State profile from SignalEngine.compute_state_profile()
        n_analogs : int
            Maximum number of analogs to return
        min_similarity : float
            Minimum similarity threshold (0-1)
        same_industry : bool
            Require same industry (SIC code prefix)
        size_range : tuple
            Acceptable size range as multiples of query size
        deal_types : list, optional
            Filter to specific deal types (e.g., ['LBO', 'Takeover'])

        Returns
        -------
        List of analog dicts with:
        - deal info (company, date, value, type)
        - state profile at deal time
        - similarity score
        - outcome metrics
        """
        query_vector = query_profile['vector']
        query_gvkey = query_profile['gvkey']
        query_date = query_profile['as_of_date']

        if isinstance(query_date, str):
            query_date = pd.to_datetime(query_date)

        analogs = []

        # PREFERRED: Use pre-computed DealScan profiles
        if self.deal_profiles is not None and len(self.deal_profiles) > 0:
            analogs = self._find_analogs_from_profiles(
                query_vector, query_date, min_similarity, deal_types
            )

        # FALLBACK: On-the-fly computation (slower)
        if len(analogs) == 0 and self.linked_deals is not None:
            print("Computing profiles on-the-fly (slower)...")
            analogs = self._find_analogs_compute_profiles(
                query_vector, query_date, min_similarity, deal_types
            )

        # Legacy CIQ fallback
        if len(analogs) == 0 and self.transactions is not None:
            analogs = self._find_analogs_legacy(
                query_vector, query_date, min_similarity
            )

        # Sort by similarity and return top N
        analogs.sort(key=lambda x: x['similarity_score'], reverse=True)
        return analogs[:n_analogs]

    def _find_analogs_from_profiles(
        self,
        query_vector: List[float],
        query_date: datetime,
        min_similarity: float,
        deal_types: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Find analogs using pre-computed deal profiles (fast)."""
        analogs = []

        # Filter profiles by date
        profiles = self.deal_profiles.copy()
        if 'deal_date' in profiles.columns:
            profiles['deal_date'] = pd.to_datetime(profiles['deal_date'])
            profiles = profiles[profiles['deal_date'] < query_date]

        # Filter by deal type
        if deal_types and 'deal_type' in profiles.columns:
            profiles = profiles[profiles['deal_type'].isin(deal_types)]

        if len(profiles) == 0:
            return []

        for idx, deal in profiles.iterrows():
            try:
                # Get signal vector from stored profile
                if 'signal_vector' in deal and deal['signal_vector'] is not None:
                    # Handle stored as list/array
                    deal_vector = deal['signal_vector']
                    if isinstance(deal_vector, str):
                        import ast
                        deal_vector = ast.literal_eval(deal_vector)
                else:
                    # Reconstruct from individual signals
                    signal_cols = [c for c in profiles.columns if c.startswith('signal_')]
                    if len(signal_cols) >= 5:
                        deal_vector = [deal[c] for c in signal_cols if pd.notna(deal[c])]
                    else:
                        continue

                if len(deal_vector) != len(query_vector):
                    continue

                # Compute similarity
                similarity = self._compute_similarity(query_vector, deal_vector)

                if similarity < min_similarity:
                    continue

                # Get deal metadata from linked deals if available
                deal_date = deal.get('deal_date')
                borrower_name = deal.get('borrower_name', 'Unknown')
                facility_amount = deal.get('facility_amount')
                deal_type = deal.get('deal_type', 'M&A')

                analog = {
                    'deal_id': deal.get('facilityid', idx),
                    'target_gvkey': deal.get('gvkey'),
                    'target_name': borrower_name,
                    'acquirer_name': 'N/A (Financing)',  # DealScan is financing, not acquirer
                    'deal_date': pd.to_datetime(deal_date) if deal_date else None,
                    'deal_value': facility_amount,
                    'deal_type': deal_type,
                    'similarity_score': round(similarity, 3),
                    'composite_score': deal.get('composite_score'),
                    'target_vector': list(deal_vector),
                    'query_vector': query_vector,
                }

                analogs.append(analog)

            except Exception as e:
                continue

        return analogs

    def _find_analogs_compute_profiles(
        self,
        query_vector: List[float],
        query_date: datetime,
        min_similarity: float,
        deal_types: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Compute profiles on-the-fly for linked deals (slower but complete)."""
        analogs = []

        if self.linked_deals is None:
            return []

        deals = self.linked_deals.copy()
        deals = deals[deals['facilitystartdate'] < query_date]

        if deal_types:
            deals = deals[deals['primarypurpose'].isin(deal_types)]

        # Limit for performance
        deals = deals.head(200)

        for idx, deal in deals.iterrows():
            try:
                gvkey = deal.get('gvkey')
                if pd.isna(gvkey):
                    continue

                deal_date = deal.get('facilitystartdate')
                if pd.isna(deal_date):
                    continue

                # Compute profile at deal time
                profile = self.signals.compute_state_profile(str(gvkey), deal_date)

                if profile is None:
                    continue

                # Compute similarity
                similarity = self._compute_similarity(query_vector, profile['vector'])

                if similarity < min_similarity:
                    continue

                analog = {
                    'deal_id': deal.get('facilityid', idx),
                    'target_gvkey': gvkey,
                    'target_name': deal.get('borrower_name', 'Unknown'),
                    'acquirer_name': 'N/A (Financing)',
                    'deal_date': deal_date,
                    'deal_value': deal.get('facilityamt'),
                    'deal_type': deal.get('primarypurpose', 'M&A'),
                    'similarity_score': round(similarity, 3),
                    'composite_score': profile['composite_score'],
                    'target_profile': profile,
                    'target_vector': profile['vector'],
                    'query_vector': query_vector,
                }

                analogs.append(analog)

            except Exception as e:
                continue

        return analogs

    def _find_analogs_legacy(
        self,
        query_vector: List[float],
        query_date: datetime,
        min_similarity: float,
    ) -> List[Dict]:
        """Legacy CIQ transaction search (for backward compatibility)."""
        analogs = []

        if self.transactions is None:
            return []

        deals = self.get_deal_universe(end_date=query_date.strftime('%Y-%m-%d'))

        if len(deals) == 0:
            return []

        sample_deals = deals.head(100)

        for idx, deal in sample_deals.iterrows():
            try:
                target_gvkey = deal.get('tgtcompanyid') or deal.get('targetgvkey')
                if pd.isna(target_gvkey):
                    continue

                deal_date = deal.get('deal_date')
                if pd.isna(deal_date):
                    continue

                profile_date = deal_date - pd.Timedelta(days=1)
                target_profile = self.signals.compute_state_profile(
                    str(target_gvkey), profile_date
                )

                if target_profile is None:
                    continue

                similarity = self._compute_similarity(
                    query_vector, target_profile['vector']
                )

                if similarity < min_similarity:
                    continue

                analog = {
                    'deal_id': idx,
                    'target_gvkey': target_gvkey,
                    'target_name': deal.get('tgtcompanyname', 'Unknown'),
                    'acquirer_name': deal.get('acqcompanyname', 'Unknown'),
                    'deal_date': deal_date,
                    'deal_value': deal.get('deal_value'),
                    'deal_type': deal.get('transactiontype'),
                    'similarity_score': round(similarity, 3),
                    'target_profile': target_profile,
                    'query_vector': query_vector,
                    'target_vector': target_profile['vector'],
                }

                analogs.append(analog)

            except Exception as e:
                continue

        return analogs

    def compute_outcome_distribution(
        self,
        analogs: List[Dict],
        horizon_months: int = 12,
    ) -> Dict:
        """
        Compute outcome distribution from analog cases.

        For each analog, calculate what happened after the deal:
        - TSR (Total Shareholder Return) for acquirer
        - Deal premium (if available)
        - Integration success indicators

        Returns
        -------
        dict with:
        - 'n_analogs': number of cases
        - 'outcomes': list of individual outcomes
        - 'distribution': {p25, p50, p75, mean, std}
        """
        if not analogs:
            return {'n_analogs': 0, 'outcomes': [], 'distribution': None}

        if self.prices is None:
            print("Warning: No price data for outcome calculation")
            return {'n_analogs': len(analogs), 'outcomes': [], 'distribution': None}

        outcomes = []

        for analog in analogs:
            deal_date = analog['deal_date']

            # For now, use deal value as a proxy outcome metric
            # In production, we'd calculate actual TSR from price data
            if analog.get('deal_value'):
                outcomes.append({
                    'deal_id': analog['deal_id'],
                    'target_name': analog['target_name'],
                    'deal_value': analog['deal_value'],
                    'similarity': analog['similarity_score'],
                })

        if not outcomes:
            return {'n_analogs': len(analogs), 'outcomes': [], 'distribution': None}

        # Compute distribution of deal values
        values = [o['deal_value'] for o in outcomes if o['deal_value']]

        if values:
            distribution = {
                'p25': np.percentile(values, 25),
                'p50': np.percentile(values, 50),
                'p75': np.percentile(values, 75),
                'mean': np.mean(values),
                'std': np.std(values),
                'min': min(values),
                'max': max(values),
            }
        else:
            distribution = None

        return {
            'n_analogs': len(analogs),
            'outcomes': outcomes,
            'distribution': distribution,
        }

    def generate_analog_report(
        self,
        query_gvkey: str,
        query_date: str,
        n_analogs: int = 5,
    ) -> str:
        """
        Generate a formatted report of analogs for a company.

        Returns
        -------
        str: Formatted markdown report
        """
        # Compute query profile
        profile = self.signals.compute_state_profile(query_gvkey, query_date)

        if profile is None:
            return f"Error: Could not compute state profile for {query_gvkey}"

        # Find analogs
        analogs = self.find_analogs(profile, n_analogs=n_analogs, min_similarity=0.5)

        # Build report
        report = []
        report.append(f"# Analog Analysis Report")
        report.append(f"**Query Date:** {query_date}")
        report.append(f"**Company:** {query_gvkey}")
        report.append("")

        report.append("## Query State Profile")
        report.append(f"**Composite Score:** {profile['composite_score']}/100")
        report.append("")
        report.append("| Signal | Score |")
        report.append("|--------|-------|")
        for name, data in profile['signals'].items():
            report.append(f"| {name.replace('_', ' ').title()} | {data['score']} |")
        report.append("")

        report.append(f"## Top {len(analogs)} Historical Analogs")
        report.append("")

        if not analogs:
            report.append("*No suitable analogs found in the transaction database.*")
        else:
            for i, analog in enumerate(analogs, 1):
                report.append(f"### {i}. {analog['target_name']}")
                report.append(f"- **Deal Date:** {analog['deal_date'].strftime('%Y-%m-%d') if analog['deal_date'] else 'N/A'}")
                report.append(f"- **Acquirer:** {analog['acquirer_name']}")
                report.append(f"- **Deal Value:** ${analog['deal_value']:,.0f}M" if analog['deal_value'] else "- **Deal Value:** N/A")
                report.append(f"- **Deal Type:** {analog['deal_type']}")
                report.append(f"- **Similarity Score:** {analog['similarity_score']:.1%}")
                report.append("")

        # Outcome distribution
        outcomes = self.compute_outcome_distribution(analogs)
        if outcomes['distribution']:
            report.append("## Outcome Distribution")
            dist = outcomes['distribution']
            report.append(f"Based on {outcomes['n_analogs']} analog cases:")
            report.append(f"- **P25:** ${dist['p25']:,.0f}M")
            report.append(f"- **Median:** ${dist['p50']:,.0f}M")
            report.append(f"- **P75:** ${dist['p75']:,.0f}M")
            report.append(f"- **Mean:** ${dist['mean']:,.0f}M")

        return "\n".join(report)


def demo():
    """Demonstrate the analog retrieval system."""
    print("="*70)
    print("DEMO: Analog Retrieval System")
    print("="*70)

    retriever = AnalogRetriever()

    # Get a sample company
    as_of = '2023-06-30'
    universe = retriever.signals.snapshot.get_universe_snapshot(
        as_of, min_assets=1000, min_revenue=200
    )

    if len(universe) > 0:
        sample = universe.iloc[0]
        gvkey = sample['gvkey']
        name = sample['conm']

        print(f"\nSearching analogs for {name} ({gvkey}) as of {as_of}")

        # Generate report
        report = retriever.generate_analog_report(gvkey, as_of, n_analogs=5)
        print(report)


if __name__ == "__main__":
    demo()
