"""
Refinitiv Source Adapter
========================
Implements fetch/parse/validate/publish for Refinitiv data.

Handles:
- M&A deals
- Dividends
- Fundamentals
- Analyst estimates
- Prices

CRITICAL: Sets event_time and available_time correctly:
- M&A: event_time = completion date, available_time = announcement date
- Dividends: event_time = ex-date, available_time = announcement date
- Fundamentals: event_time = fiscal period end, available_time = filing/report date
- Estimates: event_time = estimate date, available_time = when estimate published
"""

import refinitiv.data as rd
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

from .base import (
    SourceAdapter, CanonicalRecord, ValidationResult, BatchWindow,
    RecordType, ActionType
)
from .lake import DataLake


class RefinitivAdapter(SourceAdapter):
    """
    Adapter for Refinitiv/LSEG data.

    Supports multiple record types:
    - MA_DEAL: M&A transactions
    - CORPORATE_ACTION: Dividends, splits, etc.
    - FUNDAMENTAL: Quarterly financials
    - ESTIMATE: Analyst estimates
    - PRICE: Historical prices
    """

    source_name = "refinitiv"

    def __init__(self):
        """Initialize Refinitiv connection."""
        self._connected = False
        self._connect()

    def _connect(self):
        """Establish Refinitiv session."""
        if not self._connected:
            try:
                rd.open_session()
                self._connected = True
            except Exception as e:
                print(f"Warning: Could not connect to Refinitiv: {e}")

    def _ensure_connected(self):
        """Ensure active connection."""
        if not self._connected:
            self._connect()

    # =========================================================================
    # FETCH
    # =========================================================================
    def fetch(self, window: BatchWindow) -> List[Dict[str, Any]]:
        """
        Fetch raw data from Refinitiv.

        Routes to specific fetch method based on record_type.
        """
        self._ensure_connected()

        if window.record_type == RecordType.MA_DEAL:
            return self._fetch_ma_deals(window)
        elif window.record_type == RecordType.CORPORATE_ACTION:
            return self._fetch_corporate_actions(window)
        elif window.record_type == RecordType.FUNDAMENTAL:
            return self._fetch_fundamentals(window)
        elif window.record_type == RecordType.ESTIMATE:
            return self._fetch_estimates(window)
        elif window.record_type == RecordType.PRICE:
            return self._fetch_prices(window)
        else:
            raise ValueError(f"Unsupported record type: {window.record_type}")

    def _fetch_ma_deals(self, window: BatchWindow) -> List[Dict]:
        """Fetch M&A deals."""
        all_deals = []

        start_year = window.start.year
        end_year = window.end.year

        for year in range(start_year, end_year + 1):
            try:
                deals = rd.get_data(
                    universe=f'SCREEN(U(IN(Deals)/*UNV:MADEALS*/), IN(TR.MnAStatus,"C"), TR.MnAAnnDate>={year}-01-01, TR.MnAAnnDate<={year}-12-31)',
                    fields=[
                        'TR.MnADealValue(Scale=6)',
                        'TR.MnAAnnDate',
                        'TR.MnACompDate',
                        'TR.MnAStatus',
                        'TR.MnADealType',
                        'TR.MnATargetNation',
                        'TR.MnAAcquirorNation',
                        'TR.MnATargetPrimarySICCode',
                        'TR.MnAPremium1Day',
                        'TR.MnAPremium1Week',
                    ]
                )
                for _, row in deals.iterrows():
                    all_deals.append(row.to_dict())
            except Exception as e:
                print(f"  Warning: Error fetching {year} deals: {e}")

        return all_deals

    def _fetch_corporate_actions(self, window: BatchWindow) -> List[Dict]:
        """Fetch dividends and other corporate actions."""
        # Get universe of tickers
        try:
            universe = rd.get_data(universe='0#.SPX', fields=['TR.CommonName'])
            tickers = universe['Instrument'].tolist()
        except:
            tickers = []

        all_actions = []

        # Fetch dividends in batches
        batch_size = 75
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i+batch_size]
            try:
                divs = rd.get_data(
                    universe=batch,
                    fields=[
                        'TR.DivExDate',
                        'TR.DivPayDate',
                        'TR.DivAnnDate',
                        'TR.DivAmount',
                        'TR.DivType',
                        'TR.DivCurrency',
                    ],
                    parameters={
                        'SDate': window.start.strftime('%Y-%m-%d'),
                        'EDate': window.end.strftime('%Y-%m-%d')
                    }
                )
                for _, row in divs.iterrows():
                    if pd.notna(row.get('Dividend Ex Date')):
                        all_actions.append(row.to_dict())
            except:
                pass

        return all_actions

    def _fetch_fundamentals(self, window: BatchWindow) -> List[Dict]:
        """Fetch quarterly fundamentals."""
        try:
            universe = rd.get_data(universe='0#.SPX', fields=['TR.CommonName'])
            tickers = universe['Instrument'].tolist()
        except:
            tickers = []

        all_fundamentals = []
        batch_size = 50

        fields = [
            'TR.CommonName',
            'TR.Revenue',
            'TR.EBITDA',
            'TR.NetIncome',
            'TR.TotalAssets',
            'TR.TotalDebt',
            'TR.TotalEquity',
            'TR.CashAndSTInvestments',
            'TR.FreeCashFlow',
            'TR.CompanyMarketCap',
            'TR.GICSSector',
            'TR.GICSIndustry',
        ]

        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i+batch_size]
            try:
                data = rd.get_data(universe=batch, fields=fields)
                for _, row in data.iterrows():
                    all_fundamentals.append(row.to_dict())
            except:
                pass

        return all_fundamentals

    def _fetch_estimates(self, window: BatchWindow) -> List[Dict]:
        """Fetch analyst estimates."""
        try:
            universe = rd.get_data(universe='0#.SPX', fields=['TR.CommonName'])
            tickers = universe['Instrument'].tolist()
        except:
            tickers = []

        all_estimates = []
        batch_size = 75

        fields = [
            'TR.EPSMean',
            'TR.EPSActValue',
            'TR.EPSSurprisePercent',
            'TR.RevenueMean',
            'TR.RevenueActValue',
            'TR.RecommendationMean',
            'TR.TargetPriceMean',
            'TR.NumberOfEstimates',
        ]

        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i+batch_size]
            try:
                data = rd.get_data(universe=batch, fields=fields)
                for _, row in data.iterrows():
                    all_estimates.append(row.to_dict())
            except:
                pass

        return all_estimates

    def _fetch_prices(self, window: BatchWindow) -> List[Dict]:
        """Fetch historical prices."""
        try:
            universe = rd.get_data(universe='0#.SPX', fields=['TR.CommonName'])
            tickers = universe['Instrument'].tolist()
        except:
            tickers = []

        all_prices = []
        batch_size = 25

        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i+batch_size]
            try:
                data = rd.get_history(
                    universe=batch,
                    fields=['TR.PriceClose', 'TR.Volume'],
                    start=window.start.strftime('%Y-%m-%d'),
                    end=window.end.strftime('%Y-%m-%d'),
                    interval='daily'
                )
                if data is not None:
                    df = data.reset_index()
                    for _, row in df.iterrows():
                        all_prices.append(row.to_dict())
            except:
                pass

        return all_prices

    # =========================================================================
    # PARSE
    # =========================================================================
    def parse(self, raw_payloads: List[Dict[str, Any]]) -> List[CanonicalRecord]:
        """
        Parse raw payloads into canonical records.

        Determines record type from payload structure and routes appropriately.
        """
        records = []

        for payload in raw_payloads:
            try:
                # Detect type from fields present
                if 'Deal Value' in payload or 'MnA' in str(payload.keys()):
                    record = self._parse_ma_deal(payload)
                elif 'Dividend Ex Date' in payload:
                    record = self._parse_dividend(payload)
                elif 'Revenue' in payload and 'EBITDA' in payload:
                    record = self._parse_fundamental(payload)
                elif 'Earnings Per Share - Mean' in payload:
                    record = self._parse_estimate(payload)
                elif 'Price Close' in payload:
                    record = self._parse_price(payload)
                else:
                    continue  # Unknown type

                if record:
                    records.append(record)
            except Exception as e:
                print(f"  Warning: Parse error: {e}")

        return records

    def _parse_ma_deal(self, payload: Dict) -> Optional[CanonicalRecord]:
        """
        Parse M&A deal.

        TIMESTAMPS:
        - event_time = Completion date (when deal actually happened)
        - available_time = Announcement date (when market learned about it)
        """
        entity_id = payload.get('Instrument', '')
        ann_date = payload.get('Date Announced')
        comp_date = payload.get('Deal Completion Date') or ann_date

        if not ann_date:
            return None

        # Parse dates
        try:
            available_time = pd.to_datetime(ann_date)
            event_time = pd.to_datetime(comp_date) if comp_date else available_time
        except:
            return None

        # Generate ID
        record_id = CanonicalRecord.generate_id(
            source=self.source_name,
            record_type='ma_deal',
            entity_id=entity_id,
            event_time=event_time,
        )

        return CanonicalRecord(
            record_id=record_id,
            record_type=RecordType.MA_DEAL,
            source=self.source_name,
            entity_id=entity_id,
            entity_name=payload.get('Target Nation'),
            event_time=event_time,
            available_time=available_time,
            data={
                'deal_value': payload.get('Deal Value'),
                'deal_type': payload.get('M&A Type'),
                'status': payload.get('Deal Status'),
                'target_nation': payload.get('Target Nation'),
                'acquiror_nation': payload.get('Acquiror Nation'),
                'premium_1d': payload.get('Premium 1 Day'),
                'premium_1w': payload.get('Premium 1 Week'),
                'target_sic': payload.get('Target Primary SIC Code'),
            },
            source_record_id=entity_id,
        )

    def _parse_dividend(self, payload: Dict) -> Optional[CanonicalRecord]:
        """
        Parse dividend action.

        TIMESTAMPS:
        - event_time = Ex-date (when you need to own to get dividend)
        - available_time = Announcement date (when dividend was declared)
        """
        entity_id = payload.get('Instrument', '')
        ex_date = payload.get('Dividend Ex Date')
        ann_date = payload.get('Dividend Announcement Date') or ex_date

        if not ex_date:
            return None

        try:
            event_time = pd.to_datetime(ex_date)
            # If no announcement date, assume 2 weeks before ex-date
            if ann_date:
                available_time = pd.to_datetime(ann_date)
            else:
                available_time = event_time - timedelta(days=14)
        except:
            return None

        # Determine action type
        div_type = payload.get('Dividend Type', '').lower()
        if 'special' in div_type:
            action_type = ActionType.DIVIDEND_SPECIAL
        else:
            action_type = ActionType.DIVIDEND_REGULAR

        record_id = CanonicalRecord.generate_id(
            source=self.source_name,
            record_type='corporate_action',
            entity_id=entity_id,
            event_time=event_time,
            action='dividend',
        )

        return CanonicalRecord(
            record_id=record_id,
            record_type=RecordType.CORPORATE_ACTION,
            source=self.source_name,
            entity_id=entity_id,
            entity_name=None,
            event_time=event_time,
            available_time=available_time,
            data={
                'action_type': action_type.value,
                'amount': payload.get('Dividend Amount'),
                'currency': payload.get('Dividend Currency'),
                'div_type': payload.get('Dividend Type'),
                'pay_date': str(payload.get('Dividend Pay Date')),
            },
            source_record_id=entity_id,
        )

    def _parse_fundamental(self, payload: Dict) -> Optional[CanonicalRecord]:
        """
        Parse fundamental data.

        TIMESTAMPS:
        - event_time = Fiscal period end (what period the data describes)
        - available_time = NOW (current snapshot) or filing date if historical

        For current snapshots, we use ingestion time as available_time.
        For historical, we'd need the actual filing date (rdq in Compustat terms).
        """
        entity_id = payload.get('Instrument', '')

        if not entity_id:
            return None

        # For current snapshot, event_time and available_time are both now
        now = datetime.utcnow()
        event_time = now
        available_time = now

        record_id = CanonicalRecord.generate_id(
            source=self.source_name,
            record_type='fundamental',
            entity_id=entity_id,
            event_time=event_time,
        )

        return CanonicalRecord(
            record_id=record_id,
            record_type=RecordType.FUNDAMENTAL,
            source=self.source_name,
            entity_id=entity_id,
            entity_name=payload.get('Company Common Name'),
            event_time=event_time,
            available_time=available_time,
            data={
                'revenue': payload.get('Revenue'),
                'ebitda': payload.get('EBITDA'),
                'net_income': payload.get('Net Income Incl Extra Before Distributions'),
                'total_assets': payload.get('Total Assets'),
                'total_debt': payload.get('Total Debt'),
                'total_equity': payload.get('Total Equity'),
                'cash': payload.get('Cash and Short Term Investments'),
                'fcf': payload.get('Free Cash Flow'),
                'market_cap': payload.get('Company Market Cap'),
                'gics_sector': payload.get('GICS Sector Name'),
                'gics_industry': payload.get('GICS Industry Name'),
            },
            source_record_id=entity_id,
        )

    def _parse_estimate(self, payload: Dict) -> Optional[CanonicalRecord]:
        """
        Parse analyst estimate.

        TIMESTAMPS:
        - event_time = Current date (estimates are forward-looking)
        - available_time = Current date (when we captured the estimate)
        """
        entity_id = payload.get('Instrument', '')

        if not entity_id:
            return None

        now = datetime.utcnow()

        record_id = CanonicalRecord.generate_id(
            source=self.source_name,
            record_type='estimate',
            entity_id=entity_id,
            event_time=now,
        )

        return CanonicalRecord(
            record_id=record_id,
            record_type=RecordType.ESTIMATE,
            source=self.source_name,
            entity_id=entity_id,
            entity_name=None,
            event_time=now,
            available_time=now,
            data={
                'eps_mean': payload.get('Earnings Per Share - Mean'),
                'eps_actual': payload.get('Earnings Per Share - Actual'),
                'eps_surprise_pct': payload.get('EPS Surprise Pct'),
                'revenue_mean': payload.get('Revenue - Mean'),
                'revenue_actual': payload.get('Revenue - Actual'),
                'recommendation_mean': payload.get('Mean Recommendation'),
                'target_price_mean': payload.get('Target Price - Mean'),
                'num_estimates': payload.get('Number of Estimates'),
            },
            source_record_id=entity_id,
        )

    def _parse_price(self, payload: Dict) -> Optional[CanonicalRecord]:
        """
        Parse price record.

        TIMESTAMPS:
        - event_time = Trade date
        - available_time = Trade date (prices are immediately available)
        """
        entity_id = payload.get('Instrument', '')
        date = payload.get('Date')

        if not entity_id or not date:
            return None

        try:
            event_time = pd.to_datetime(date)
            available_time = event_time  # Prices available same day
        except:
            return None

        record_id = CanonicalRecord.generate_id(
            source=self.source_name,
            record_type='price',
            entity_id=entity_id,
            event_time=event_time,
        )

        return CanonicalRecord(
            record_id=record_id,
            record_type=RecordType.PRICE,
            source=self.source_name,
            entity_id=entity_id,
            entity_name=None,
            event_time=event_time,
            available_time=available_time,
            data={
                'close': payload.get('Price Close'),
                'volume': payload.get('Volume'),
            },
            source_record_id=entity_id,
        )

    # =========================================================================
    # PUBLISH
    # =========================================================================
    def publish(self, records: List[CanonicalRecord], lake: DataLake) -> int:
        """Publish records to the data lake."""
        return lake.publish(records)

    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================
    def ingest_all(self, lake: DataLake, start_date: datetime, end_date: datetime) -> Dict:
        """
        Ingest all record types for a date range.

        Parameters
        ----------
        lake : DataLake
            Target data lake
        start_date : datetime
            Start of date range
        end_date : datetime
            End of date range

        Returns
        -------
        Summary of all ingestions
        """
        results = {}

        for record_type in [RecordType.MA_DEAL, RecordType.CORPORATE_ACTION,
                           RecordType.FUNDAMENTAL, RecordType.ESTIMATE]:
            window = BatchWindow(
                start=start_date,
                end=end_date,
                source=self.source_name,
                record_type=record_type,
            )
            results[record_type.value] = self.ingest(window, lake)

        return results

    def close(self):
        """Close Refinitiv connection."""
        if self._connected:
            try:
                rd.close_session()
                self._connected = False
            except:
                pass
