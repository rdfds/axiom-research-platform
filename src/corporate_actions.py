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

    def _extract_company_from_headline(self, headline: str) -> Optional[str]:
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


class ActionAnalyzer:
    """
    Analyzes what actions companies in similar states took.

    This is the "what happened to companies like this?" engine.
    """

    def __init__(self, actions_db: Optional[CorporateActionsDB] = None):
        """Initialize with actions database."""
        if actions_db is None:
            actions_db = CorporateActionsDB()
        self.actions = actions_db

        # Load pre-computed profiles for fast lookup
        # Priority order:
        # 1. clean_action_profiles (new clean data with certain sources)
        # 2. action_profiles_with_outcomes (old inferred data)
        # 3. deal_profiles_with_outcomes (M&A only)
        clean_profiles_path = DATA_DIR / 'clean_action_profiles.parquet'
        action_profiles_path = DATA_DIR / 'action_profiles_with_outcomes.parquet'
        deal_profiles_path = DATA_DIR / 'deal_profiles_with_outcomes.parquet'

        self.deal_profiles = None
        self.has_outcomes = False

        # Try clean profiles first (preferred - certain data)
        if clean_profiles_path.exists():
            self.deal_profiles = pd.read_parquet(clean_profiles_path)
            self.has_outcomes = True
            print(f"Loaded {len(self.deal_profiles):,} CLEAN action profiles")
            print("  Sources: Compustat buybacks, CRSP acquisitions/bankruptcies, CRSP dividends")

        # Fall back to old action profiles
        elif action_profiles_path.exists():
            action_profiles = pd.read_parquet(action_profiles_path)
            print(f"Loaded {len(action_profiles):,} action profiles (legacy)")

            # Also load deal profiles for M&A-specific data
            if deal_profiles_path.exists():
                deal_profiles = pd.read_parquet(deal_profiles_path)
                print(f"Loaded {len(deal_profiles):,} deal profiles")

                # Combine them
                self.deal_profiles = pd.concat(
                    [action_profiles, deal_profiles], ignore_index=True
                )
                # Remove duplicates (same gvkey + date)
                if 'action_date' in self.deal_profiles.columns:
                    self.deal_profiles = self.deal_profiles.drop_duplicates(
                        subset=['gvkey', 'action_date'], keep='first'
                    )
            else:
                self.deal_profiles = action_profiles

            self.has_outcomes = True
            print(f"Total profiles available: {len(self.deal_profiles):,}")

        elif deal_profiles_path.exists():
            self.deal_profiles = pd.read_parquet(deal_profiles_path)
            self.has_outcomes = True
            print(f"Loaded {len(self.deal_profiles):,} deal profiles with TSR outcomes")
        else:
            # Try non-outcome versions
            for path in [
                DATA_DIR / 'action_profiles_all.parquet',
                DATA_DIR / 'deal_profiles_all.parquet'
            ]:
                if path.exists():
                    self.deal_profiles = pd.read_parquet(path)
                    print(f"Loaded {len(self.deal_profiles):,} profiles (no outcomes)")
                    break

    def analyze_similar_states(
        self,
        query_profile: Dict,
        min_similarity: float = 0.85,
        max_results: int = 100,
        sector_filter: Optional[str] = None,
        weight_actions: bool = True,
    ) -> Dict:
        """
        Find companies in similar states and what actions they took.

        Parameters
        ----------
        query_profile : dict
            State profile from SignalEngine
        min_similarity : float
            Minimum cosine similarity threshold
        max_results : int
            Max results to return
        sector_filter : str, optional
            2-digit SIC code to filter by sector (e.g., '35' for industrial machinery)
        weight_actions : bool
            If True, apply weighting to correct for data imbalance

        Returns
        -------
        dict with:
        - 'action_distribution': {action_type: frequency %}
        - 'similar_cases': list of similar company-action pairs
        - 'n_similar': total similar cases found
        """
        query_vector = query_profile['vector']

        # Use pre-computed profiles for speed
        if self.deal_profiles is not None and len(self.deal_profiles) > 0:
            similar = self._find_similar_from_profiles(
                query_vector, min_similarity, max_results * 3,  # Get more, then filter
                sector_filter=sector_filter
            )
        else:
            similar = []

        if len(similar) == 0:
            return {
                'action_distribution': {},
                'similar_cases': [],
                'n_similar': 0,
            }

        # Compute action distribution with optional weighting
        actions = [s['action_type'] for s in similar]
        action_counts = pd.Series(actions).value_counts()

        if weight_actions:
            # Weight to correct for data imbalance in clean action profiles
            # Data distribution: dividends 70%, buybacks 25%, others ~5%
            # Target: more balanced view of capital allocation decisions
            #
            # Weights based on actual corporate action frequency in practice:
            # - Dividends are very frequent but routine (downweight)
            # - Buybacks are common but tracked (slight downweight)
            # - M&A/acquisitions are rare but important (upweight)
            # - Bankruptcies/distress are rare but critical (upweight)
            # - Special actions (splits, special divs) are rare (upweight)
            action_weights = {
                # Dividend actions - very high volume, downweight
                'dividend_increase': 0.3,
                'dividend_cut': 0.5,       # More significant, less downweight
                'dividend_initiate': 0.4,
                'dividend_suspend': 0.6,   # Rare and significant
                'dividend_special': 0.8,   # Rare
                'dividend_irregular': 0.8,
                'dividend_liquidating': 1.0,
                # Buybacks - high volume, moderate downweight
                'buyback': 0.4,
                # Acquisitions/M&A - rare, upweight
                'acquisition': 3.0,
                'acquired': 3.0,           # Target side
                'acquisition_merger': 3.0,
                'acquisition_tender': 3.0,
                'acquisition_lbo': 3.0,
                'Takeover': 3.0,
                'Acquis. line': 3.0,
                'LBO': 3.0,
                # Distress - rare but critical
                'bankruptcy': 4.0,
                'bankruptcy_chapter': 4.0,
                'distress_other': 3.0,
                # Other capital actions
                'stock_split': 2.0,
                'reverse_split': 2.5,      # Often signals distress
                'spinoff': 3.0,
                'return_of_capital': 2.0,
                'going_private': 3.0,
                # Debt actions
                'debt_refinance': 1.5,
            }

            weighted_counts = action_counts.copy().astype(float)
            for action in weighted_counts.index:
                weight = action_weights.get(action, 1.0)
                weighted_counts[action] = weighted_counts[action] * weight

            action_pct = (weighted_counts / weighted_counts.sum() * 100).round(1)
        else:
            action_pct = (action_counts / len(similar) * 100).round(1)

        # Limit results after computing distribution
        similar = similar[:max_results]

        return {
            'action_distribution': action_pct.to_dict(),
            'similar_cases': similar,
            'n_similar': len(similar),
            'weighted': weight_actions,
        }

    def _find_similar_from_profiles(
        self,
        query_vector: List,
        min_similarity: float,
        max_results: int,
        sector_filter: Optional[str] = None,
    ) -> List[Dict]:
        """
        Find similar cases from pre-computed profiles.

        Parameters
        ----------
        query_vector : list
            Signal vector to match against
        min_similarity : float
            Minimum cosine similarity threshold
        max_results : int
            Max results to return
        sector_filter : str, optional
            2-digit SIC code to filter by sector. If provided, only returns
            companies in the same broad industry sector.
        """
        from scipy.spatial.distance import cosine

        # Build sector lookup if needed
        sector_lookup = None
        if sector_filter:
            sector_lookup = self._build_sector_lookup()

        similar = []

        for idx, row in self.deal_profiles.iterrows():
            try:
                gvkey = row.get('gvkey')

                # Apply sector filter if specified
                if sector_filter and sector_lookup is not None:
                    company_sic = sector_lookup.get(str(gvkey))
                    if company_sic is None:
                        continue
                    # Match on 2-digit SIC (broad industry sector)
                    if str(company_sic)[:2] != str(sector_filter)[:2]:
                        continue

                # Get deal vector
                deal_vector = row['signal_vector']
                if isinstance(deal_vector, str):
                    import ast
                    deal_vector = ast.literal_eval(deal_vector)

                if len(deal_vector) != len(query_vector):
                    continue

                # Compute similarity
                similarity = 1 - cosine(query_vector, deal_vector)

                if similarity >= min_similarity:
                    # Handle different column naming conventions
                    company_name = row.get('company_name') or row.get('borrower_name') or 'Unknown'
                    action_date = row.get('action_date') or row.get('deal_date')
                    action_type = row.get('action_type') or row.get('deal_type') or 'acquisition'
                    deal_value = row.get('deal_value') or row.get('facility_amount')

                    result_entry = {
                        'company_name': company_name,
                        'gvkey': gvkey,
                        'date': action_date,
                        'action_type': action_type,
                        'deal_value': deal_value,
                        'similarity': round(similarity, 3),
                        'composite_score': row.get('composite_score'),
                        'tsr_1m': row.get('tsr_1m'),
                        'tsr_3m': row.get('tsr_3m'),
                        'tsr_6m': row.get('tsr_6m'),
                        'tsr_12m': row.get('tsr_12m'),
                    }

                    # Add sector info if available
                    if sector_lookup and gvkey:
                        result_entry['sic'] = sector_lookup.get(str(gvkey))

                    similar.append(result_entry)

            except Exception:
                continue

        # Sort by similarity
        similar.sort(key=lambda x: x['similarity'], reverse=True)
        return similar[:max_results]

    def _build_sector_lookup(self) -> Dict[str, str]:
        """
        Build gvkey -> SIC code lookup from fundamentals.

        Returns dict mapping gvkey to 4-digit SIC code.
        """
        if hasattr(self, '_sector_cache'):
            return self._sector_cache

        fund_path = DATA_DIR / 'fundamentals_quarterly.parquet'
        if not fund_path.exists():
            return {}

        # Load fundamentals and extract SIC codes
        fund = pd.read_parquet(fund_path)

        # Get most recent SIC for each company
        # sic is the Compustat column for Standard Industrial Classification
        if 'sic' not in fund.columns:
            return {}

        fund = fund.dropna(subset=['sic'])
        fund = fund.sort_values('datadate', ascending=False)
        fund = fund.drop_duplicates('gvkey', keep='first')

        self._sector_cache = fund.set_index('gvkey')['sic'].astype(str).to_dict()
        return self._sector_cache

    def get_company_sector(self, gvkey: str) -> Optional[str]:
        """
        Get the 2-digit SIC sector code for a company.

        Returns None if not found.
        """
        sector_lookup = self._build_sector_lookup()
        sic = sector_lookup.get(str(gvkey))
        if sic:
            return str(sic)[:2]
        return None

    def get_sector_name(self, sic_2digit: str) -> str:
        """Convert 2-digit SIC to readable sector name."""
        sic_names = {
            '01': 'Agriculture',
            '10': 'Mining',
            '13': 'Oil & Gas',
            '14': 'Mining (Non-metallic)',
            '15': 'Construction',
            '20': 'Food Products',
            '21': 'Tobacco',
            '22': 'Textiles',
            '23': 'Apparel',
            '24': 'Lumber & Wood',
            '25': 'Furniture',
            '26': 'Paper',
            '27': 'Printing & Publishing',
            '28': 'Chemicals',
            '29': 'Petroleum Refining',
            '30': 'Rubber & Plastics',
            '31': 'Leather',
            '32': 'Stone, Clay, Glass',
            '33': 'Primary Metals',
            '34': 'Fabricated Metals',
            '35': 'Industrial Machinery',
            '36': 'Electronics',
            '37': 'Transportation Equipment',
            '38': 'Instruments',
            '39': 'Misc. Manufacturing',
            '40': 'Railroads',
            '42': 'Trucking',
            '44': 'Water Transportation',
            '45': 'Air Transportation',
            '47': 'Transportation Services',
            '48': 'Communications',
            '49': 'Utilities',
            '50': 'Wholesale - Durables',
            '51': 'Wholesale - Nondurables',
            '52': 'Building Materials Retail',
            '53': 'General Merchandise',
            '54': 'Food Stores',
            '55': 'Auto Dealers',
            '56': 'Apparel Stores',
            '57': 'Furniture Stores',
            '58': 'Eating Places',
            '59': 'Misc. Retail',
            '60': 'Banks',
            '61': 'Credit Institutions',
            '62': 'Securities',
            '63': 'Insurance',
            '64': 'Insurance Agents',
            '65': 'Real Estate',
            '67': 'Holding Companies',
            '70': 'Hotels',
            '72': 'Personal Services',
            '73': 'Business Services',
            '75': 'Auto Repair',
            '78': 'Motion Pictures',
            '79': 'Amusement',
            '80': 'Health Services',
            '81': 'Legal Services',
            '82': 'Educational Services',
            '83': 'Social Services',
            '87': 'Engineering & Accounting',
            '99': 'Non-classifiable',
        }
        return sic_names.get(sic_2digit, f'SIC {sic_2digit}')

    def generate_action_report(
        self,
        query_profile: Dict,
        min_similarity: float = 0.85,
    ) -> str:
        """
        Generate a readable report of what similar companies did.

        This is the output that would go to a banker.
        """
        result = self.analyze_similar_states(query_profile, min_similarity)

        report = []
        report.append("=" * 60)
        report.append("HISTORICAL ACTION ANALYSIS")
        report.append("=" * 60)
        report.append("\nQuery Company State:")
        report.append(f"  Composite Score: {query_profile['composite_score']}/100")
        report.append(f"  As of: {query_profile['as_of_date']}")

        if result['n_similar'] == 0:
            report.append("\n⚠️ No similar historical cases found.")
            report.append("   Try lowering the similarity threshold.")
            return "\n".join(report)

        report.append(f"\n📊 Found {result['n_similar']} companies in similar states")
        report.append(f"   (Similarity threshold: {min_similarity:.0%})")

        report.append("\n" + "-" * 60)
        report.append("WHAT THEY DID AND HOW IT TURNED OUT:")
        report.append("-" * 60)

        # Group by action and compute outcome stats
        similar_df = pd.DataFrame(result['similar_cases'])

        if 'tsr_12m' in similar_df.columns:
            outcome_stats = similar_df.groupby('action_type').agg({
                'similarity': 'count',
                'tsr_12m': 'median'
            }).rename(columns={'similarity': 'count', 'tsr_12m': 'median_tsr'})

            for action, pct in sorted(
                result['action_distribution'].items(),
                key=lambda x: -x[1]
            ):
                bar = "█" * int(pct / 5)
                if action in outcome_stats.index:
                    tsr = outcome_stats.loc[action, 'median_tsr']
                    if pd.notna(tsr):
                        tsr_str = f"{tsr:+.1f}% 12M TSR"
                    else:
                        tsr_str = ""
                else:
                    tsr_str = ""
                report.append(f"  {action:20} {pct:5.1f}%  {bar}  {tsr_str}")
        else:
            for action, pct in sorted(
                result['action_distribution'].items(),
                key=lambda x: -x[1]
            ):
                bar = "█" * int(pct / 5)
                report.append(f"  {action:20} {pct:5.1f}%  {bar}")

        report.append("\n" + "-" * 60)
        report.append("TOP SIMILAR CASES:")
        report.append("-" * 60)

        for i, case in enumerate(result['similar_cases'][:10], 1):
            date_str = case['date'].strftime('%Y-%m-%d') if case['date'] else 'N/A'
            value_str = f"${case['deal_value']:,.0f}M" if case.get('deal_value') else ''

            # TSR outcome
            tsr = case.get('tsr_12m')
            tsr_str = f"→ {tsr:+.1f}%" if pd.notna(tsr) else ""

            report.append(
                f"  {i}. {case['company_name'][:25]:25} "
                f"{date_str}  {case['action_type']:12} {tsr_str}"
            )
            report.append(f"     Similarity: {case['similarity']:.1%}  {value_str}")

        return "\n".join(report)


def demo():
    """Demonstrate the corporate actions analysis."""
    print("=" * 70)
    print("CORPORATE ACTIONS ANALYSIS DEMO")
    print("=" * 70)

    # Load the actions database
    db = CorporateActionsDB()

    # Summary
    print("\n📊 Total Actions by Type:")
    dist = db.get_action_distribution()
    total = dist.sum()
    for action, count in dist.items():
        pct = count / total * 100
        print(f"  {action:25} {count:6,}  ({pct:.1f}%)")

    # Test the analyzer with a sample company
    print("\n" + "=" * 70)
    print("ANALYZING SIMILAR STATES")
    print("=" * 70)

    from .signals import SignalEngine

    engine = SignalEngine()

    # Get a sample company profile
    sample = engine.snapshot.get_universe_snapshot('2023-06-30', min_assets=1000, min_revenue=200)
    if len(sample) > 0:
        gvkey = sample.iloc[5]['gvkey']
        company = sample.iloc[5]['conm']

        print(f"\nQuery: {company} as of 2023-06-30")

        profile = engine.compute_state_profile(gvkey, '2023-06-30')

        if profile:
            analyzer = ActionAnalyzer(db)
            report = analyzer.generate_action_report(profile, min_similarity=0.90)
            print(report)


if __name__ == "__main__":
    demo()
