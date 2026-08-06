"""
Real-Time Market Data via Refinitiv
====================================
Provides current prices, quotes, and market data.
"""

import refinitiv.data as rd
import pandas as pd
from typing import List, Dict, Optional, Union
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class MarketDataProvider:
    """
    Real-time and historical market data from Refinitiv.

    Usage:
        mdp = MarketDataProvider()

        # Real-time quote
        quote = mdp.get_quote('AAPL.O')

        # Multiple quotes
        quotes = mdp.get_quotes(['AAPL.O', 'MSFT.O', 'GOOGL.O'])

        # Historical prices
        history = mdp.get_price_history('AAPL.O', days=30)
    """

    def __init__(self):
        """Initialize and connect to Refinitiv."""
        self._connected = False
        self._connect()

    def _connect(self):
        """Establish connection to Refinitiv."""
        if not self._connected:
            try:
                rd.open_session()
                self._connected = True
            except Exception as e:
                print(f"Warning: Could not connect to Refinitiv: {e}")

    def _ensure_connected(self):
        """Ensure we have an active connection."""
        if not self._connected:
            self._connect()

    def get_quote(self, ticker: str) -> Dict:
        """
        Get real-time quote for a single ticker.

        Parameters
        ----------
        ticker : str
            Refinitiv ticker (e.g., 'AAPL.O' for Apple on NASDAQ)

        Returns
        -------
        dict with price, change, volume, etc.
        """
        self._ensure_connected()

        try:
            df = rd.get_data(
                [ticker],
                [
                    'TR.PriceClose',      # Last close
                    'TR.PriceOpen',       # Today's open
                    'TR.PriceHigh',       # Today's high
                    'TR.PriceLow',        # Today's low
                    'TR.Volume',          # Volume
                    'TR.PricePctChg1D',   # 1-day % change
                    'TR.CompanyMarketCap', # Market cap
                    'TR.52WeekHigh',      # 52-week high
                    'TR.52WeekLow',       # 52-week low
                ]
            )

            if len(df) > 0:
                row = df.iloc[0]
                return {
                    'ticker': ticker,
                    'price': row.get('Price Close'),
                    'open': row.get('Price Open'),
                    'high': row.get('Price High'),
                    'low': row.get('Price Low'),
                    'volume': row.get('Volume'),
                    'change_pct': row.get('Percent Change - 1 Day'),
                    'market_cap': row.get('Company Market Cap'),
                    'high_52w': row.get('52 Week High'),
                    'low_52w': row.get('52 Week Low'),
                    'timestamp': datetime.now().isoformat(),
                }
        except Exception as e:
            return {'ticker': ticker, 'error': str(e)}

        return {'ticker': ticker, 'error': 'No data'}

    def get_quotes(self, tickers: List[str]) -> pd.DataFrame:
        """
        Get real-time quotes for multiple tickers.

        Parameters
        ----------
        tickers : list
            List of Refinitiv tickers

        Returns
        -------
        DataFrame with quotes for all tickers
        """
        self._ensure_connected()

        try:
            df = rd.get_data(
                tickers,
                [
                    'TR.CommonName',
                    'TR.PriceClose',
                    'TR.PricePctChg1D',
                    'TR.Volume',
                    'TR.CompanyMarketCap',
                    'TR.PriceHigh',
                    'TR.PriceLow',
                ]
            )

            # Rename columns for clarity
            df = df.rename(columns={
                'Instrument': 'ticker',
                'Company Common Name': 'name',
                'Price Close': 'price',
                'Percent Change - 1 Day': 'change_1d',
                'Volume': 'volume',
                'Company Market Cap': 'market_cap',
                'Price High': 'high',
                'Price Low': 'low',
            })

            return df

        except Exception as e:
            print(f"Error getting quotes: {e}")
            return pd.DataFrame()

    def get_intraday_quote(self, ticker: str) -> Dict:
        """
        Get real-time intraday quote (most recent trade).

        Uses real-time fields for current market data.
        """
        self._ensure_connected()

        try:
            df = rd.get_data(
                [ticker],
                [
                    'TRDPRC_1',   # Last trade price
                    'HIGH_1',     # Today's high
                    'LOW_1',      # Today's low
                    'ACVOL_1',    # Accumulated volume
                    'BID',        # Current bid
                    'ASK',        # Current ask
                    'BIDSIZE',    # Bid size
                    'ASKSIZE',    # Ask size
                ]
            )

            if len(df) > 0:
                row = df.iloc[0]
                return {
                    'ticker': ticker,
                    'last': row.get('TRDPRC_1'),
                    'high': row.get('HIGH_1'),
                    'low': row.get('LOW_1'),
                    'volume': row.get('ACVOL_1'),
                    'bid': row.get('BID'),
                    'ask': row.get('ASK'),
                    'bid_size': row.get('BIDSIZE'),
                    'ask_size': row.get('ASKSIZE'),
                    'spread': (row.get('ASK') or 0) - (row.get('BID') or 0),
                    'timestamp': datetime.now().isoformat(),
                }
        except Exception as e:
            return {'ticker': ticker, 'error': str(e)}

        return {'ticker': ticker, 'error': 'No data'}

    def get_price_history(
        self,
        ticker: str,
        days: int = 30,
        interval: str = 'daily'
    ) -> pd.DataFrame:
        """
        Get historical price data.

        Parameters
        ----------
        ticker : str
            Refinitiv ticker
        days : int
            Number of days of history
        interval : str
            'daily', 'weekly', or 'monthly'

        Returns
        -------
        DataFrame with OHLCV data
        """
        self._ensure_connected()

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        try:
            df = rd.get_history(
                universe=[ticker],
                fields=['TR.PriceOpen', 'TR.PriceHigh', 'TR.PriceLow', 'TR.PriceClose', 'TR.Volume'],
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval=interval
            )

            if df is not None and len(df) > 0:
                df = df.reset_index()
                df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
                return df

        except Exception as e:
            print(f"Error getting history: {e}")

        return pd.DataFrame()

    def get_returns(
        self,
        ticker: str,
        periods: List[str] = ['1D', '1W', '1M', '3M', '6M', '1Y']
    ) -> Dict:
        """
        Get total returns for various periods.

        Parameters
        ----------
        ticker : str
            Refinitiv ticker
        periods : list
            Return periods to fetch

        Returns
        -------
        dict with return for each period
        """
        self._ensure_connected()

        field_map = {
            '1D': 'TR.TotalReturn1D',
            '1W': 'TR.TotalReturn1Wk',
            '1M': 'TR.TotalReturn1Mo',
            '3M': 'TR.TotalReturn3Mo',
            '6M': 'TR.TotalReturn6Mo',
            '1Y': 'TR.TotalReturn1Yr',
            'YTD': 'TR.TotalReturnYTD',
        }

        fields = [field_map[p] for p in periods if p in field_map]

        try:
            df = rd.get_data([ticker], fields)

            if len(df) > 0:
                row = df.iloc[0]
                return {
                    'ticker': ticker,
                    **{p: row.get(field_map[p].replace('TR.', '').replace('TotalReturn', ''))
                       for p in periods if p in field_map}
                }
        except Exception as e:
            return {'ticker': ticker, 'error': str(e)}

        return {'ticker': ticker}

    def get_index_level(self, index: str = '.SPX') -> Dict:
        """
        Get current level of a market index.

        Common indices:
        - .SPX = S&P 500
        - .DJI = Dow Jones
        - .IXIC = NASDAQ Composite
        - .RUT = Russell 2000
        - .VIX = VIX
        """
        self._ensure_connected()

        try:
            df = rd.get_data(
                [index],
                ['TR.IndexCalculationPrice', 'TR.IndexPctChg1D', 'TR.IndexHigh', 'TR.IndexLow']
            )

            if len(df) > 0:
                row = df.iloc[0]
                return {
                    'index': index,
                    'level': row.iloc[1] if len(row) > 1 else None,
                    'change_pct': row.iloc[2] if len(row) > 2 else None,
                    'timestamp': datetime.now().isoformat(),
                }
        except Exception as e:
            return {'index': index, 'error': str(e)}

        return {'index': index}

    def close(self):
        """Close the Refinitiv connection."""
        if self._connected:
            try:
                rd.close_session()
                self._connected = False
            except:
                pass


# Convenience functions
_provider = None

def get_provider() -> MarketDataProvider:
    """Get or create the global market data provider."""
    global _provider
    if _provider is None:
        _provider = MarketDataProvider()
    return _provider

def get_quote(ticker: str) -> Dict:
    """Get real-time quote for a ticker."""
    return get_provider().get_quote(ticker)

def get_quotes(tickers: List[str]) -> pd.DataFrame:
    """Get real-time quotes for multiple tickers."""
    return get_provider().get_quotes(tickers)

def get_price_history(ticker: str, days: int = 30) -> pd.DataFrame:
    """Get historical prices."""
    return get_provider().get_price_history(ticker, days)


# Demo
if __name__ == '__main__':
    print("Market Data Provider Demo")
    print("=" * 60)

    mdp = MarketDataProvider()

    # Single quote
    print("\n1. Single Quote (AAPL):")
    quote = mdp.get_quote('AAPL.O')
    for k, v in quote.items():
        print(f"   {k}: {v}")

    # Multiple quotes
    print("\n2. Multiple Quotes:")
    quotes = mdp.get_quotes(['AAPL.O', 'MSFT.O', 'GOOGL.O', 'AMZN.O'])
    print(quotes[['ticker', 'name', 'price', 'change_1d', 'volume']].to_string())

    # Intraday
    print("\n3. Intraday Quote (real-time):")
    intraday = mdp.get_intraday_quote('AAPL.O')
    for k, v in intraday.items():
        print(f"   {k}: {v}")

    # History
    print("\n4. Price History (30 days):")
    history = mdp.get_price_history('AAPL.O', days=30)
    print(history.tail(10).to_string())

    mdp.close()
    print("\nDone!")
