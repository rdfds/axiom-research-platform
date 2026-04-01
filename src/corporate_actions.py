"""
Corporate Actions Database
==========================
Tracks ALL corporate actions, not just M&A:
- Dividends (initiate, increase, cut, suspend)
- Buybacks (repurchases)
- Acquisitions (from DealScan + CIQ)
- Divestitures
- Equity offerings
- Debt actions (from DealScan)

Links actions to company state profiles so we can answer:
"Companies in this state → What did they do → What happened?"
"""

import pandas as pd
from typing import Optional, Dict, List
import re

from .snapshot import DATA_DIR


class CorporateActionsDB:
    """
    Unified database of corporate actions with timing and outcomes.
    """

    # Action type constants
    ACTION_DIVIDEND_INITIATE = 'dividend_initiate'
    ACTION_DIVIDEND_INCREASE = 'dividend_increase'
    ACTION_DIVIDEND_CUT = 'dividend_cut'
    ACTION_DIVIDEND_SUSPEND = 'dividend_suspend'
    ACTION_BUYBACK = 'buyback'
    ACTION_ACQUISITION = 'acquisition'
    ACTION_DIVESTITURE = 'divestiture'
    ACTION_EQUITY_OFFERING = 'equity_offering'
    ACTION_DEBT_REFINANCE = 'debt_refinance'
    ACTION_DEBT_PAYDOWN = 'debt_paydown'
    ACTION_SPINOFF = 'spinoff'
    ACTION_STATUS_QUO = 'status_quo'

    def __init__(self):
        """Load all available corporate action data sources."""
        self.actions = None
        self.keydev = None
        self.dealscan = None
        self.fundamentals = None

        self._load_data()
        self._build_unified_actions()

    def _load_data(self):
        """Load data from various sources."""
        # CIQ Key Developments (news/announcements)
        keydev_path = DATA_DIR / 'ciqsamp_keydev_ciqkeydev.parquet'
        if keydev_path.exists():
            print("Loading CIQ Key Developments...")
            self.keydev = pd.read_parquet(keydev_path)
            self.keydev['announceddate'] = pd.to_datetime(self.keydev['announceddate'])
            print(f"  Loaded {len(self.keydev):,} events")

        # DealScan M&A facilities
        dealscan_path = DATA_DIR / 'dealscan_linked.parquet'
        if dealscan_path.exists():
            print("Loading DealScan linked deals...")
            self.dealscan = pd.read_parquet(dealscan_path)
            self.dealscan['facilitystartdate'] = pd.to_datetime(
                self.dealscan['facilitystartdate']
            )
            print(f"  Loaded {len(self.dealscan):,} deals")

        # Compustat fundamentals (for inferring buybacks from share changes)
        fund_path = DATA_DIR / 'fundamentals_quarterly.parquet'
        if fund_path.exists():
            print("Loading Compustat fundamentals...")
            self.fundamentals = pd.read_parquet(fund_path)
            self.fundamentals['datadate'] = pd.to_datetime(self.fundamentals['datadate'])
            print(f"  Loaded {len(self.fundamentals):,} quarters")

    def _classify_keydev_action(self, headline: str) -> Optional[str]:
        """Classify a CIQ headline into an action type."""
        if pd.isna(headline):
            return None

        headline = headline.lower()

        # Dividend actions
        if 'dividend' in headline:
            if any(w in headline for w in ['increase', 'raise', 'hike', 'boost']):
                return self.ACTION_DIVIDEND_INCREASE
            elif any(w in headline for w in ['cut', 'reduce', 'lower', 'decrease']):
                return self.ACTION_DIVIDEND_CUT
            elif any(w in headline for w in ['suspend', 'eliminate', 'omit']):
                return self.ACTION_DIVIDEND_SUSPEND
            elif any(w in headline for w in ['initiate', 'begin', 'start', 'declare']):
                return self.ACTION_DIVIDEND_INITIATE
            else:
                # Regular dividend payment - still track it
                return self.ACTION_DIVIDEND_INITIATE

        # Buybacks
        if any(w in headline for w in ['buyback', 'repurchase', 'share repurchase']):
            return self.ACTION_BUYBACK

        # Acquisitions
        if any(w in headline for w in ['acqui', 'merger', 'takeover', 'purchase of', 'to buy']):
            return self.ACTION_ACQUISITION

        # Divestitures
        if any(w in headline for w in ['divest', 'sell', 'dispose', 'asset sale']):
            return self.ACTION_DIVESTITURE

        # Spinoffs
        if 'spin' in headline and any(w in headline for w in ['off', 'out']):
            return self.ACTION_SPINOFF

        # Equity offerings
        if any(w in headline for w in ['offering', 'equity raise', 'ipo', 'secondary', 'stock sale']):
            return self.ACTION_EQUITY_OFFERING

        return None

    def _extract_company_from_headline(self, headline: str) :
        """Extract company name from headline."""
        if pd.isna(headline):
            return None

        # Most headlines start with company name followed by action
        # e.g., "Microsoft Corporation is considering acquisitions"
        # e.g., "Apple Inc., $ 0.22, Cash Dividend"

        # Try to extract up to first comma or "is" or "to" or "declares"
        patterns = [
            r'^([^,]+?),',  # Up to first comma
            r'^(.+?)\s+is\s+',  # Before "is"
            r'^(.+?)\s+to\s+',  # Before "to"
            r'^(.+?)\s+declares\s+',  # Before "declares"
            r'^(.+?)\s+announces\s+',  # Before "announces"
        ]

        for pattern in patterns:
            match = re.match(pattern, headline, re.IGNORECASE)
            if match:
                company = match.group(1).strip()
                # Clean up
                company = re.sub(r'\s+(Inc\.|Corp\.|Corporation|Company|Co\.).*$', '', company, flags=re.IGNORECASE)
                if len(company) > 3:
                    return company

        return None

    def _build_unified_actions(self):
        """Build unified actions database from all sources."""
        print("\nBuilding unified corporate actions database...")

        actions = []

        # 1. Process CIQ Key Developments
        if self.keydev is not None:
            print("  Processing CIQ Key Developments...")
            for idx, row in self.keydev.iterrows():
                action_type = self._classify_keydev_action(row['headline'])
                if action_type:
                    company = self._extract_company_from_headline(row['headline'])
                    actions.append({
                        'source': 'ciq_keydev',
                        'action_type': action_type,
                        'company_name': company,
                        'date': row['announceddate'],
                        'headline': row['headline'],
                        'details': row.get('situation'),
                    })

        # 2. Process DealScan M&A
        if self.dealscan is not None:
            print("  Processing DealScan M&A facilities...")
            for idx, row in self.dealscan.iterrows():
                purpose = row.get('primarypurpose', '')

                if purpose in ['LBO', 'Takeover', 'Acquis. line', 'SBO']:
                    action_type = self.ACTION_ACQUISITION
                elif purpose in ['Recap.', 'Dividend Recap']:
                    action_type = self.ACTION_DEBT_REFINANCE
                else:
                    action_type = self.ACTION_DEBT_REFINANCE

                actions.append({
                    'source': 'dealscan',
                    'action_type': action_type,
                    'company_name': row.get('borrower_name'),
                    'gvkey': row.get('gvkey'),
                    'ticker': row.get('ticker_clean'),
                    'date': row['facilitystartdate'],
                    'deal_value': row.get('facilityamt'),
                    'deal_type': purpose,
                })

        # 3. Infer buybacks from share count changes (Compustat)
        if self.fundamentals is not None:
            print("  Inferring buybacks from share count changes...")
            buybacks = self._infer_buybacks()
            actions.extend(buybacks)

        # Convert to DataFrame
        self.actions = pd.DataFrame(actions)

        if len(self.actions) > 0:
            self.actions['date'] = pd.to_datetime(self.actions['date'])
            self.actions = self.actions.sort_values('date', ascending=False)

        print(f"\n  Total unified actions: {len(self.actions):,}")

        # Summary by type
        print("\n  Actions by type:")
        print(self.actions['action_type'].value_counts())

    def _infer_buybacks(self) -> List[Dict]:
        """Infer buyback activity from share count decreases."""
        buybacks = []

        if self.fundamentals is None:
            return buybacks

        # Group by company
        fund = self.fundamentals.sort_values(['gvkey', 'datadate'])

        for gvkey, group in fund.groupby('gvkey'):
            if len(group) < 2:
                continue

            # Calculate share count changes
            group = group.copy()
            group['shares_prev'] = group['cshoq'].shift(1)
            group['share_change'] = (group['cshoq'] - group['shares_prev']) / group['shares_prev']

            # Significant share decrease (>2% in a quarter) indicates buyback
            buyback_quarters = group[
                (group['share_change'] < -0.02) &
                (group['shares_prev'] > 0)
            ]

            for idx, row in buyback_quarters.iterrows():
                buybacks.append({
                    'source': 'compustat_inferred',
                    'action_type': self.ACTION_BUYBACK,
                    'company_name': row.get('conm'),
                    'gvkey': gvkey,
                    'ticker': row.get('tic'),
                    'date': row['datadate'],
                    'share_change_pct': round(row['share_change'] * 100, 1),
                    'shares_before': row['shares_prev'],
                    'shares_after': row['cshoq'],
                })

        print(f"    Found {len(buybacks):,} inferred buyback events")
        return buybacks

    def get_actions_for_company(
        self,
        gvkey: Optional[str] = None,
        ticker: Optional[str] = None,
        company_name: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get all actions for a specific company."""
        if self.actions is None or len(self.actions) == 0:
            return pd.DataFrame()

        mask = pd.Series([True] * len(self.actions))

        if gvkey:
            mask &= self.actions['gvkey'] == gvkey
        if ticker:
            mask &= self.actions['ticker'].str.upper() == ticker.upper()
        if company_name:
            mask &= self.actions['company_name'].str.contains(company_name, case=False, na=False)
        if start_date:
            mask &= self.actions['date'] >= pd.to_datetime(start_date)
        if end_date:
            mask &= self.actions['date'] <= pd.to_datetime(end_date)

        return self.actions[mask].copy()

    def get_action_distribution(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get distribution of action types over time."""
        if self.actions is None:
            return pd.DataFrame()

        actions = self.actions.copy()

        if start_date:
            actions = actions[actions['date'] >= pd.to_datetime(start_date)]
        if end_date:
            actions = actions[actions['date'] <= pd.to_datetime(end_date)]

        return actions['action_type'].value_counts()

    def find_similar_action_outcomes(
        self,
        query_profile: Dict,
        signal_engine,
        min_similarity: float = 0.7,
        lookback_years: int = 10,
    ) -> Dict:
        """
        Find what actions companies in similar states took and their outcomes.

        This is the core function that answers:
        "Companies like this → What did they do → How did it turn out?"

        Returns
        -------
        dict with:
        - 'action_distribution': frequency of each action type
        - 'outcomes_by_action': TSR outcomes grouped by action
        - 'analogs': detailed list of similar cases
        """
        from scipy.spatial.distance import cosine

        query_vector = query_profile['vector']
        query_date = pd.to_datetime(query_profile['as_of_date'])

        # Get actions in lookback period
        cutoff_date = query_date - pd.Timedelta(days=lookback_years * 365)

        actions_in_window = self.actions[
            (self.actions['date'] >= cutoff_date) &
            (self.actions['date'] < query_date) &
            (self.actions['gvkey'].notna())
        ].copy()

        if len(actions_in_window) == 0:
            return {
                'action_distribution': {},
                'outcomes_by_action': {},
                'analogs': [],
                'n_similar': 0,
            }

        # Compute state profiles for each action and find similar ones
        similar_actions = []

        for idx, action in actions_in_window.iterrows():
            try:
                gvkey = action['gvkey']
                action_date = action['date']

                # Compute state profile at time of action
                profile = signal_engine.compute_state_profile(str(gvkey), action_date)

                if profile is None:
                    continue

                # Compute similarity
                similarity = 1 - cosine(query_vector, profile['vector'])

                if similarity >= min_similarity:
                    similar_actions.append({
                        'action_type': action['action_type'],
                        'company_name': action['company_name'],
                        'gvkey': gvkey,
                        'date': action_date,
                        'similarity': round(similarity, 3),
                        'composite_score': profile['composite_score'],
                        'deal_value': action.get('deal_value'),
                    })

            except Exception:
                continue

        if len(similar_actions) == 0:
            return {
                'action_distribution': {},
                'outcomes_by_action': {},
                'analogs': [],
                'n_similar': 0,
            }

        # Compute action distribution
        similar_df = pd.DataFrame(similar_actions)
        action_dist = similar_df['action_type'].value_counts()
        action_dist_pct = (action_dist / len(similar_df) * 100).round(1)

        return {
            'action_distribution': action_dist_pct.to_dict(),
            'outcomes_by_action': {},  # TODO: Add TSR calculation
            'analogs': similar_actions[:20],  # Top 20
            'n_similar': len(similar_actions),
        }


