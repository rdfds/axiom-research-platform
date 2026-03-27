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
def pull_dividends(tickers):
    log("=" * 70)
    log("2. DIVIDENDS (All types)")
    log("=" * 70)

    all_data = []
    batch_size = 75

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        pct = (i + batch_size) / len(tickers) * 100
        log(f"  Batch {i//batch_size + 1}/{len(tickers)//batch_size + 1} ({pct:.0f}%)...")

        try:
            data = rd.get_data(
                universe=batch,
                fields=[
                    'TR.DivExDate',
                    'TR.DivPayDate',
                    'TR.DivRecordDate',
                    'TR.DivAnnDate',
                    'TR.DivAmount',
                    'TR.DivType',
                    'TR.DivCurrency',
                    'TR.DivFrequency',
                ],
                parameters={'SDate': START_DATE, 'EDate': END_DATE}
            )
            if len(data) > 0:
                all_data.append(data)
        except Exception as e:
            log(f"    Error: {e}")

        time.sleep(0.3)

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        combined = combined.dropna(subset=['Dividend Ex Date'])
        save_parquet(combined, 'dividends_complete')

        # Analyze dividend actions
        log("  Categorizing dividend changes...")
        combined['ex_date'] = pd.to_datetime(combined['Dividend Ex Date'])
        combined = combined.sort_values(['Instrument', 'ex_date'])
        combined['prev_amount'] = combined.groupby('Instrument')['Dividend Amount'].shift(1)

        def classify(row):
            if pd.isna(row.get('prev_amount')) or pd.isna(row.get('Dividend Amount')):
                return 'regular'
            if row['prev_amount'] == 0:
                return 'initiation'
            pct = (row['Dividend Amount'] - row['prev_amount']) / row['prev_amount']
            if pct > 0.01:
                return 'increase'
            elif pct < -0.01:
                return 'decrease'
            return 'unchanged'

        combined['action_type'] = combined.apply(classify, axis=1)
        save_parquet(combined, 'dividends_with_actions')

        log(f"  Action types: {combined['action_type'].value_counts().to_dict()}")
        return combined
    return pd.DataFrame()


# ============================================================================
# 3. STOCK SPLITS
# ============================================================================
def pull_splits(tickers):
    log("=" * 70)
    log("3. STOCK SPLITS")
    log("=" * 70)

    all_data = []
    batch_size = 100

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        pct = (i + batch_size) / len(tickers) * 100
        if i % 500 == 0:
            log(f"  Progress: {pct:.0f}%...")

        try:
            data = rd.get_data(
                universe=batch,
                fields=[
                    'TR.CAEffectiveDate',
                    'TR.CAAdjustmentFactor',
                    'TR.CAAdjustmentType',
                    'TR.CAExDate',
                    'TR.CAAnnouncementDate',
                ],
                parameters={'CAType': 'SSP', 'SDate': START_DATE, 'EDate': END_DATE}
            )
            if len(data) > 0 and 'CA Effective Date' in data.columns:
                data = data.dropna(subset=['CA Effective Date'])
                if len(data) > 0:
                    all_data.append(data)
        except:
            pass

        time.sleep(0.2)

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        save_parquet(combined, 'stock_splits')
        return combined
    return pd.DataFrame()


# ============================================================================
# 4. SHARE BUYBACKS / REPURCHASES
# ============================================================================
def pull_buybacks(tickers):
    log("=" * 70)
    log("4. SHARE BUYBACKS")
    log("=" * 70)

    # Method 1: From deals database
    log("  Pulling from deals database...")
    try:
        buyback_deals = rd.get_data(
            universe='SCREEN(U(IN(Deals)/*UNV:MADEALS*/), IN(TR.MnADealType,"Repurchases Deal","Self Tender Or Recapitalization Deal"), TR.MnAAnnDate>=2020-01-01, TR.MnATargetNation=="United States")',
            fields=[
                'TR.MnATarget',
                'TR.MnATargetTicker',
                'TR.MnADealValue(Scale=6)',
                'TR.MnAAnnDate',
                'TR.MnACompDate',
                'TR.MnADealType',
                'TR.MnATargetPrimarySICCode',
            ]
        )
        save_parquet(buyback_deals, 'buybacks_deals')
        log(f"    Found {len(buyback_deals):,} buyback deals")
    except Exception as e:
        log(f"    Deals error: {e}")
        buyback_deals = pd.DataFrame()

    # Method 2: Quarterly repurchase data from fundamentals
    log("  Pulling quarterly repurchase amounts...")
    all_data = []
    batch_size = 50

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        pct = (i + batch_size) / len(tickers) * 100
        if i % 500 == 0:
            log(f"  Progress: {pct:.0f}%...")

        try:
            data = rd.get_data(
                universe=batch,
                fields=[
                    'TR.RepurchaseOfCommonPreferredStock',
                    'TR.CommonSharesOutstanding',
                    'TR.TreasurySharesNumber',
                ],
                parameters={'SDate': START_DATE, 'EDate': END_DATE, 'Period': 'FQ0', 'Frq': 'FQ'}
            )
            if len(data) > 0:
                all_data.append(data)
        except:
            pass

        time.sleep(0.2)

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        save_parquet(combined, 'buybacks_quarterly')
        log(f"    Found {len(combined):,} quarterly records")
        return combined
    return buyback_deals


# ============================================================================
# 5. SPINOFFS
# ============================================================================
def pull_spinoffs():
    log("=" * 70)
    log("5. SPINOFFS")
    log("=" * 70)

    try:
        spinoffs = rd.get_data(
            universe='SCREEN(U(IN(Deals)/*UNV:MADEALS*/), TR.MnADealType=="Spinoff Deal", TR.MnAAnnDate>=2020-01-01, TR.MnATargetNation=="United States")',
            fields=[
                'TR.MnATarget',
                'TR.MnATargetTicker',
                'TR.MnAAcquiror',
                'TR.MnADealValue(Scale=6)',
                'TR.MnAAnnDate',
                'TR.MnACompDate',
                'TR.MnAStatus',
            ]
        )
        save_parquet(spinoffs, 'spinoffs')
        log(f"  Found {len(spinoffs):,} spinoffs")
        return spinoffs
    except Exception as e:
        log(f"  Error: {e}")
        return pd.DataFrame()


# ============================================================================
# 6. DIVESTITURES
# ============================================================================
def pull_divestitures():
    log("=" * 70)
    log("6. DIVESTITURES")
    log("=" * 70)

    try:
        divest = rd.get_data(
            universe='SCREEN(U(IN(Deals)/*UNV:MADEALS*/), TR.MnADealType=="Divestiture Deal", TR.MnAAnnDate>=2020-01-01, TR.MnATargetNation=="United States")',
            fields=[
                'TR.MnATarget',
                'TR.MnATargetTicker',
                'TR.MnAAcquiror',
                'TR.MnADealValue(Scale=6)',
                'TR.MnAAnnDate',
                'TR.MnACompDate',
                'TR.MnAStatus',
            ]
        )
        save_parquet(divest, 'divestitures')
        log(f"  Found {len(divest):,} divestitures")
        return divest
    except Exception as e:
        log(f"  Error: {e}")
        return pd.DataFrame()


# ============================================================================
# 7. DEBT ISSUANCE
# ============================================================================
def pull_debt():
    log("=" * 70)
    log("7. DEBT ISSUANCE")
    log("=" * 70)

    all_data = []

    for year in range(2020, 2026):
        log(f"  Pulling {year}...")
        try:
            data = rd.get_data(
                universe=f'SCREEN(U(IN(Deals)/*UNV:CorpBonds*/), TR.FIIssueDate>={year}-01-01, TR.FIIssueDate<={year}-12-31, TR.FIIssuerNation=="United States")',
                fields=[
                    'TR.FIIssuerName',
                    'TR.FIIssuerTicker',
                    'TR.FIPrincipalAmount(Scale=6)',
                    'TR.FIIssueDate',
                    'TR.FIMaturityDate',
                    'TR.FICoupon',
                    'TR.FIInstrumentType',
                    'TR.FIMoodyRating',
                    'TR.FISPRating',
                    'TR.FIIssuerPrimarySICCode',
                ]
            )
            all_data.append(data)
            log(f"    Found {len(data):,} issuances")
            time.sleep(1)
        except Exception as e:
            log(f"    Error: {e}")

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        save_parquet(combined, 'debt_issuance')
        return combined
    return pd.DataFrame()


# ============================================================================
# 8. EQUITY OFFERINGS (IPO + Secondary)
# ============================================================================
def pull_equity_offerings():
    log("=" * 70)
    log("8. EQUITY OFFERINGS")
    log("=" * 70)

    all_data = []

    for year in range(2020, 2026):
        log(f"  Pulling {year}...")
        base_universe = (
            f"SCREEN(U(IN(Deals)/*UNV:EQDeals*/), "
            f"TR.EQOfferDate>={year}-01-01, "
            f"TR.EQOfferDate<={year}-12-31"
            f")"
        )
        country_filter = (
            f"SCREEN(U(IN(Deals)/*UNV:EQDeals*/), "
            f"TR.EQOfferDate>={year}-01-01, "
            f"TR.EQOfferDate<={year}-12-31, "
            f"TR.EQIssuerNation=\"United States\""
            f")"
        )
        fields = [
            'TR.EQIssuerName',
            'TR.EQIssuerTicker',
            # Avoid Scale= param to prevent formula parsing errors.
            'TR.EQOfferAmount',
            'TR.EQOfferDate',
            'TR.EQOfferPrice',
            'TR.EQOfferType',
            'TR.EQOfferMethod',
            'TR.EQIssuerPrimarySICCode',
            'TR.EQIssuerNation',
        ]
        try:
            data = rd.get_data(universe=country_filter, fields=fields)
        except Exception as e:
            log(f"    Country filter error: {e}")
            log("    Retrying without country filter (will filter locally)...")
            try:
                data = rd.get_data(universe=base_universe, fields=fields)
            except Exception as e2:
                log(f"    Error: {e2}")
                continue

        # Local US filter (if needed)
        if "TR.EQIssuerNation" in data.columns:
            data = data[data["TR.EQIssuerNation"].astype(str).str.contains("United States", case=False, na=False)]

        all_data.append(data)
        log(f"    Found {len(data):,} offerings")
        time.sleep(1)

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        save_parquet(combined, 'equity_offerings')
        return combined
    return pd.DataFrame()


# ============================================================================
# 9. FUNDAMENTALS (Latest + Historical)
# ============================================================================
def pull_fundamentals(tickers):
    log("=" * 70)
    log("9. FUNDAMENTALS (Current snapshot)")
    log("=" * 70)

    all_data = []
    batch_size = 50

    fields = [
        'TR.CommonName',
        'TR.CompanyName',
        'TR.Revenue',
        'TR.RevenueGrowthPct',
        'TR.EBITDA',
        'TR.EBITDAMargin',
        'TR.NetIncome',
        'TR.NetProfitMargin',
        'TR.TotalAssets',
        'TR.TotalDebt',
        'TR.TotalEquity',
        'TR.CashAndSTInvestments',
        'TR.FreeCashFlow',
        'TR.CapitalExpenditures',
        'TR.NetDebtToEBITDA',
        'TR.TotalDebtToTotalEquity',
        'TR.CurrentRatio',
        'TR.QuickRatio',
        'TR.ReturnOnEquity',
        'TR.ReturnOnAssets',
        'TR.CompanyMarketCap',
        'TR.EV',
        'TR.EVToEBITDA',
        'TR.PriceToBookValuePerShare',
        'TR.PERatio',
        'TR.DividendYield',
        'TR.GICSSector',
        'TR.GICSIndustryGroup',
        'TR.GICSIndustry',
        'TR.GICSSubIndustry',
        'TR.TRBCEconomicSector',
        'TR.TRBCBusinessSector',
        'TR.OrganizationStatusCode',
    ]

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        pct = (i + batch_size) / len(tickers) * 100
        if i % 250 == 0:
            log(f"  Progress: {pct:.0f}%...")

        try:
            data = rd.get_data(universe=batch, fields=fields)
            if len(data) > 0:
                all_data.append(data)
        except Exception as e:
            if i % 500 == 0:
                log(f"    Error at {i}: {e}")

        time.sleep(0.2)

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        save_parquet(combined, 'fundamentals_current')
        log(f"  Total: {len(combined):,} companies")
        return combined
    return pd.DataFrame()


# ============================================================================
# 10. QUARTERLY FUNDAMENTALS (Historical)
# ============================================================================
def pull_quarterly_fundamentals(tickers):
    log("=" * 70)
    log("10. QUARTERLY FUNDAMENTALS (Historical)")
    log("=" * 70)

    all_data = []
    batch_size = 30  # Smaller for historical

    fields = [
        'TR.Revenue',
        'TR.EBITDA',
        'TR.NetIncome',
        'TR.TotalAssets',
        'TR.TotalDebt',
        'TR.CashAndSTInvestments',
        'TR.FreeCashFlow',
        'TR.CommonSharesOutstanding',
    ]

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        pct = (i + batch_size) / len(tickers) * 100
        if i % 300 == 0:
            log(f"  Progress: {pct:.0f}%...")

        try:
            data = rd.get_data(
                universe=batch,
                fields=fields,
                parameters={'SDate': START_DATE, 'EDate': END_DATE, 'Period': 'FQ0', 'Frq': 'FQ'}
            )
            if len(data) > 0:
                all_data.append(data)
        except:
            pass

        time.sleep(0.3)

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        save_parquet(combined, 'fundamentals_quarterly')
        log(f"  Total: {len(combined):,} quarterly records")
        return combined
    return pd.DataFrame()


# ============================================================================
# 11. ANALYST ESTIMATES
# ============================================================================
def pull_estimates(tickers):
    log("=" * 70)
    log("11. ANALYST ESTIMATES (I/B/E/S)")
    log("=" * 70)

    all_data = []
    batch_size = 75

    fields = [
        'TR.EPSMean',
        'TR.EPSHigh',
        'TR.EPSLow',
        'TR.EPSMedian',
        'TR.EPSActValue',
        'TR.EPSSurprisePercent',
        'TR.RevenueMean',
        'TR.RevenueHigh',
        'TR.RevenueLow',
        'TR.RevenueActValue',
        'TR.EBITDAMean',
        'TR.NumberOfEstimates',
        'TR.RecommendationMean',
        'TR.NumOfRecommendations',
        'TR.NumOfBuys',
        'TR.NumOfHolds',
        'TR.NumOfSells',
        'TR.TargetPriceMean',
        'TR.TargetPriceHigh',
        'TR.TargetPriceLow',
    ]

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        pct = (i + batch_size) / len(tickers) * 100
        if i % 375 == 0:
            log(f"  Progress: {pct:.0f}%...")

        try:
            data = rd.get_data(universe=batch, fields=fields)
            if len(data) > 0:
                all_data.append(data)
        except:
            pass

        time.sleep(0.2)

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        save_parquet(combined, 'analyst_estimates')
        log(f"  Total: {len(combined):,} estimate records")
        return combined
    return pd.DataFrame()


# ============================================================================
# 12. HISTORICAL PRICES (Monthly for TSR calculation)
# ============================================================================
def pull_prices(tickers):
    log("=" * 70)
    log("12. HISTORICAL PRICES (Monthly)")
    log("=" * 70)

    all_data = []
    batch_size = 25  # Small batches for history

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        pct = (i + batch_size) / len(tickers) * 100
        if i % 250 == 0:
            log(f"  Progress: {pct:.0f}%...")

        try:
            data = rd.get_history(
                universe=batch,
                fields=['TR.PriceClose', 'TR.Volume', 'TR.TotalReturn1Mo'],
                start=START_DATE,
                end=END_DATE,
                interval='monthly'
            )
            if data is not None and len(data) > 0:
                df = data.reset_index()
                all_data.append(df)
        except:
            pass

        time.sleep(0.3)

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        save_parquet(combined, 'prices_monthly')
        log(f"  Total: {len(combined):,} price records")
        return combined
    return pd.DataFrame()


# ============================================================================
# 13. INSIDER TRANSACTIONS
# ============================================================================
def pull_insider_transactions(tickers):
    log("=" * 70)
    log("13. INSIDER TRANSACTIONS")
    log("=" * 70)

    all_data = []
    batch_size = 50

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        pct = (i + batch_size) / len(tickers) * 100
        if i % 500 == 0:
            log(f"  Progress: {pct:.0f}%...")

        try:
            data = rd.get_data(
                universe=batch,
                fields=[
                    'TR.InsiderFilingDate',
                    'TR.InsiderTransactionType',
                    'TR.InsiderShares',
                    'TR.InsiderValue',
                    'TR.InsiderName',
                    'TR.InsiderTitle',
                ],
                parameters={'SDate': START_DATE, 'EDate': END_DATE}
            )
            if len(data) > 0:
                all_data.append(data)
        except:
            pass

        time.sleep(0.2)

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        combined = combined.dropna(subset=['Insider Filing Date'])
        save_parquet(combined, 'insider_transactions')
        log(f"  Total: {len(combined):,} insider transactions")
        return combined
    return pd.DataFrame()


# ============================================================================
# MAIN
# ============================================================================
def main():
    args = parse_args()
    print("=" * 70)
    print("REFINITIV COMPLETE DATA PULL")
    print("=" * 70)
    print(f"Started: {datetime.now()}")
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Output: {DATA_DIR}")
    print()

    # Connect
    log("Connecting to Refinitiv...")
    rd.open_session()
    log("Connected!")

    if args.ecm_only:
        log("ECM-only mode: pulling equity offerings only.")
        pull_equity_offerings()
        rd.close_session()
        print()
        print("=" * 70)
        print("ECM SUMMARY")
        print("=" * 70)
        out = DATA_DIR / "equity_offerings.parquet"
        if out.exists():
            size_mb = out.stat().st_size / 1024 / 1024
            df = pd.read_parquet(out)
            print(f"  equity_offerings.parquet: {len(df):,} rows ({size_mb:.1f} MB)")
        else:
            print("  equity_offerings.parquet not found (no data or pull failed).")
        print()
        print("DONE!")
        return

    # Get full universe
    tickers = get_full_universe()

    # Pull all data types
    results = {}

    # Corporate Actions from Deals Database
    results['ma'] = pull_ma_deals()
    results['spinoffs'] = pull_spinoffs()
    results['divestitures'] = pull_divestitures()
    results['debt'] = pull_debt()
    results['equity'] = pull_equity_offerings()

    # Company-level data
    results['dividends'] = pull_dividends(tickers)
    results['splits'] = pull_splits(tickers)
    results['buybacks'] = pull_buybacks(tickers)
    results['fundamentals'] = pull_fundamentals(tickers)
    results['quarterly'] = pull_quarterly_fundamentals(tickers)
    results['estimates'] = pull_estimates(tickers)
    results['prices'] = pull_prices(tickers)
    results['insider'] = pull_insider_transactions(tickers)

    # Summary
    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Completed: {datetime.now()}")
    print()

    total_rows = 0
    for f in sorted(DATA_DIR.glob('*.parquet')):
        df = pd.read_parquet(f)
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.name}: {len(df):,} rows ({size_mb:.1f} MB)")
        total_rows += len(df)

    print()
    print(f"TOTAL: {total_rows:,} rows across all datasets")

    rd.close_session()
    print()
    print("DONE!")


if __name__ == '__main__':
    main()
