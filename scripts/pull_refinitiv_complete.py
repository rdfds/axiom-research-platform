"""
Pull COMPLETE Corporate Actions & Data from Refinitiv
=====================================================
All companies, all action types, 5 years of history.

Run with: nohup python -u scripts/pull_refinitiv_complete.py > /tmp/refinitiv_complete.log 2>&1 &
"""

import argparse
import refinitiv.data as rd
import pandas as pd
from pathlib import Path
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).parent.parent / 'data' / 'refinitiv'
DATA_DIR.mkdir(exist_ok=True)

START_DATE = '2020-01-01'
END_DATE = '2025-12-31'

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def save_parquet(df, name):
    path = DATA_DIR / f'{name}.parquet'
    df.to_parquet(path, index=False)
    log(f"  ✓ Saved {len(df):,} rows to {name}.parquet")
    return path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pull Refinitiv datasets (full or ECM-only)."
    )
    parser.add_argument(
        "--ecm-only",
        action="store_true",
        help="Only pull equity offerings (ECM).",
    )
    return parser.parse_args()


def get_full_universe():
    """Get all US equities we can access."""
    log("Building full universe of US companies...")

    all_tickers = set()

    # S&P 500
    try:
        df = rd.get_data(universe='0#.SPX', fields=['TR.CommonName'])
        all_tickers.update(df['Instrument'].tolist())
        log(f"  S&P 500: {len(df)} companies")
    except Exception as e:
        log(f"  S&P 500 error: {e}")

    # S&P 400 Mid Cap
    try:
        df = rd.get_data(universe='0#.MID', fields=['TR.CommonName'])
        all_tickers.update(df['Instrument'].tolist())
        log(f"  S&P 400: {len(df)} companies")
    except Exception as e:
        log(f"  S&P 400 error: {e}")

    # S&P 600 Small Cap
    try:
        df = rd.get_data(universe='0#.SP600', fields=['TR.CommonName'])
        all_tickers.update(df['Instrument'].tolist())
        log(f"  S&P 600: {len(df)} companies")
    except Exception as e:
        log(f"  S&P 600 error: {e}")

    # Russell 1000
    try:
        df = rd.get_data(universe='0#.RUI', fields=['TR.CommonName'])
        all_tickers.update(df['Instrument'].tolist())
        log(f"  Russell 1000: {len(df)} companies")
    except Exception as e:
        log(f"  Russell 1000 error: {e}")

    # Russell 2000
    try:
        df = rd.get_data(universe='0#.RUT', fields=['TR.CommonName'])
        all_tickers.update(df['Instrument'].tolist())
        log(f"  Russell 2000: {len(df)} companies")
    except Exception as e:
        log(f"  Russell 2000 error: {e}")

    # Russell 3000
    try:
        df = rd.get_data(universe='0#.RUA', fields=['TR.CommonName'])
        all_tickers.update(df['Instrument'].tolist())
        log(f"  Russell 3000: {len(df)} companies")
    except Exception as e:
        log(f"  Russell 3000 error: {e}")

    # NASDAQ 100
    try:
        df = rd.get_data(universe='0#.NDX', fields=['TR.CommonName'])
        all_tickers.update(df['Instrument'].tolist())
        log(f"  NASDAQ 100: {len(df)} companies")
    except Exception as e:
        log(f"  NASDAQ 100 error: {e}")

    tickers = list(all_tickers)
    log(f"  TOTAL UNIVERSE: {len(tickers)} unique companies")

    # Save universe for reference
    pd.DataFrame({'ticker': tickers}).to_parquet(DATA_DIR / 'universe.parquet')

    return tickers


# ============================================================================
# 1. M&A DEALS - Complete with all fields
# ============================================================================
def pull_ma_deals():
    log("=" * 70)
    log("1. M&A DEALS (All completed deals)")
    log("=" * 70)

    all_deals = []

    for year in range(2020, 2026):
        log(f"  Pulling {year}...")
        try:
            deals = rd.get_data(
                universe=f'SCREEN(U(IN(Deals)/*UNV:MADEALS*/), IN(TR.MnAStatus,"C","P"), TR.MnAAnnDate>={year}-01-01, TR.MnAAnnDate<={year}-12-31, TR.MnATargetNation=="United States")',
                fields=[
                    'TR.MnASDCDealNo',
                    'TR.MnATarget',
                    'TR.MnATargetTicker',
                    'TR.MnAAcquiror',
                    'TR.MnAAcquirorTicker',
                    'TR.MnADealValue(Scale=6)',
                    'TR.MnAAnnDate',
                    'TR.MnACompDate',
                    'TR.MnAStatus',
                    'TR.MnADealType',
                    'TR.MnATargetPrimarySICCode',
                    'TR.MnAAcquirorPrimarySICCode',
                    'TR.MnATargetGICSSubIndustry',
                    'TR.MnAPaymentType',
                    'TR.MnAPctCash',
                    'TR.MnAPctStock',
                    'TR.MnAPremium1Day',
                    'TR.MnAPremium1Week',
                    'TR.MnAPremium4Week',
                    'TR.MnATransactionNature',
                    'TR.MnADealSynopsis',
                ]
            )
            deals['year'] = year
            all_deals.append(deals)
            log(f"    Found {len(deals):,} deals")
            time.sleep(1)
        except Exception as e:
            log(f"    Error: {e}")

    if all_deals:
        combined = pd.concat(all_deals, ignore_index=True)
        save_parquet(combined, 'ma_deals_us_complete')
        return combined
    return pd.DataFrame()


# ============================================================================
# 2. DIVIDENDS - All types
# ============================================================================
