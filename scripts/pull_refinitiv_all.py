"""
Pull All Corporate Actions & Data from Refinitiv
================================================
Comprehensive data pull for Axiom V1.
Requires Refinitiv Workspace/Eikon terminal to be running.

Run with: python -u scripts/pull_refinitiv_all.py
"""

import os
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
MNA_START_YEAR = int(os.getenv("MNA_START_YEAR", "2000"))
MNA_END_YEAR = int(os.getenv("MNA_END_YEAR", str(datetime.now().year)))
ONLY_MNA = os.getenv("ONLY_MNA", "0") == "1"
MNA_PROBE_FIELDS = os.getenv("MNA_PROBE_FIELDS", "1") == "1"
MNA_PROBE_MAX_YEARS = int(os.getenv("MNA_PROBE_MAX_YEARS", "6"))
MNA_SAVE_BY_YEAR = os.getenv("MNA_SAVE_BY_YEAR", "1") == "1"
MNA_SKIP_EXISTING = os.getenv("MNA_SKIP_EXISTING", "1") == "1"
MNA_FALLBACK_MONTHLY = os.getenv("MNA_FALLBACK_MONTHLY", "1") == "1"
MNA_MONTH_SKIP_EXISTING = os.getenv("MNA_MONTH_SKIP_EXISTING", "1") == "1"

MNA_YEARLY_DIR = DATA_DIR / "mna_yearly"
MNA_YEARLY_DIR.mkdir(parents=True, exist_ok=True)

MNA_FIELD_BASE = [
    "TR.MnADealValue(Scale=6)",
    "TR.MnAAnnDate",
    "TR.MnACompDate",
    "TR.MnAStatus",
    "TR.MnADealType",
    "TR.MnATargetNation",
    "TR.MnAAcquirorNation",
]

# Candidate fields to probe; only working ones will be used.
MNA_FIELD_CANDIDATES = [
    # Deal identifiers
    "TR.MnASDCDealNo",
    "TR.MnADealNo",

    # Core dates/status
    "TR.MnAAnnDate",
    "TR.MnACompDate",
    "TR.MnAStatus",
    "TR.MnADealType",

    # Value and terms
    "TR.MnADealValue(Scale=6)",
    "TR.MnADealValue",
    "TR.MnAPaymentType",
    "TR.MnAPctCash",
    "TR.MnAPctStock",
    "TR.MnAPremium1Day",
    "TR.MnAPremium1Week",
    "TR.MnAPremium4Week",
    "TR.MnATransactionNature",
    "TR.MnADealSynopsis",

    # Target identifiers
    "TR.MnATarget",
    "TR.MnATargetName",
    "TR.MnATargetTicker",
    "TR.MnATargetRIC",
    "TR.MnATargetPermID",
    "TR.MnATargetCUSIP",
    "TR.MnATargetISIN",

    # Acquiror identifiers
    "TR.MnAAcquiror",
    "TR.MnAAcquirorName",
    "TR.MnAAcquirorTicker",
    "TR.MnAAcquirorRIC",
    "TR.MnAAcquirorPermID",
    "TR.MnAAcquirorCUSIP",
    "TR.MnAAcquirorISIN",

    # Classification
    "TR.MnATargetNation",
    "TR.MnAAcquirorNation",
    "TR.MnATargetPrimarySICCode",
    "TR.MnAAcquirorPrimarySICCode",
    "TR.MnATargetGICSSubIndustry",
]

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def save_parquet(df, name):
    """Save dataframe to parquet with logging."""
    path = DATA_DIR / f'{name}.parquet'
    df.to_parquet(path, index=False)
    log(f"  Saved {len(df):,} rows to {path.name}")
    return path


def ensure_session() -> bool:
    """Verify Refinitiv Desktop/Workspace session is actually usable."""
    try:
        _ = rd.get_data(universe='0#.SPX', fields=['TR.CommonName'])
        return True
    except Exception as e:
        log(f"Refinitiv session check failed: {e}")
        log("Make sure Refinitiv Workspace/Desktop is running and you are logged in.")
        log("Then rerun this script in the same terminal.")
        return False


def _find_ma_probe_universe() -> str:
    current_year = datetime.now().year
    for offset in range(MNA_PROBE_MAX_YEARS):
        year = current_year - offset
        universe = (
            "SCREEN(U(IN(Deals)/*UNV:MADEALS*/), "
            "IN(TR.MnAStatus,\"C\",\"P\"), "
            f"TR.MnAAnnDate>={year}-01-01, "
            f"TR.MnAAnnDate<={year}-12-31)"
        )
        try:
            df = rd.get_data(universe=universe, fields=["TR.MnAAnnDate"])
            if df is not None and len(df) > 0:
                return universe
        except Exception as e:
            log(f"  Probe universe {year} failed: {e}")
    return ""


def _probe_ma_fields(universe: str) -> list:
    if not universe:
        return []
    log("  Probing M&A fields...")
    working = []
    for field in MNA_FIELD_CANDIDATES:
        try:
            _ = rd.get_data(universe=universe, fields=[field])
            working.append(field)
        except Exception as e:
            log(f"    Field not available: {field} ({e})")
    if not working:
        return []
    # Prefer scaled deal value if both present
    if "TR.MnADealValue(Scale=6)" in working and "TR.MnADealValue" in working:
        working = [f for f in working if f != "TR.MnADealValue"]
    log(f"  Working M&A fields: {working}")
    return working


def _ma_universe_for_range(start_date: str, end_date: str) -> str:
    return (
        "SCREEN(U(IN(Deals)/*UNV:MADEALS*/), "
        "IN(TR.MnAStatus,\"C\"), "
        f"TR.MnAAnnDate>={start_date}, "
        f"TR.MnAAnnDate<={end_date})"
    )


def _pull_ma_range(start_date: str, end_date: str, fields: list, label: str) -> pd.DataFrame:
    try:
        deals = rd.get_data(
            universe=_ma_universe_for_range(start_date, end_date),
            fields=fields,
        )
        return deals
    except Exception as e:
        log(f"    Error {label}: {e}")
        return pd.DataFrame()


def _month_ranges(year: int):
    ranges = []
    for month in range(1, 13):
        start = pd.Timestamp(year=year, month=month, day=1)
        end = (start + pd.offsets.MonthEnd(0)).to_pydatetime()
        ranges.append((start.date().isoformat(), end.date().isoformat(), month))
    return ranges

# ============================================================================
# 1. M&A DEALS
# ============================================================================
def pull_ma_deals():
    log("="*60)
    log("1. PULLING M&A DEALS")
    log("="*60)

    fields = list(MNA_FIELD_BASE)
    if MNA_PROBE_FIELDS:
        probe_universe = _find_ma_probe_universe()
        probed = _probe_ma_fields(probe_universe)
        if probed:
            fields = probed
        else:
            log("  M&A field probe failed or empty; falling back to base fields.")
    else:
        log("  Skipping M&A field probe (MNA_PROBE_FIELDS=0).")

    all_deals = []

    # Pull completed deals by year to avoid timeout
    for year in range(MNA_START_YEAR, MNA_END_YEAR + 1):
        log(f"  Pulling {year} completed deals...")
        year_path = MNA_YEARLY_DIR / f"ma_deals_{year}.parquet"
        if MNA_SAVE_BY_YEAR and MNA_SKIP_EXISTING and year_path.exists():
            log(f"    Skipping {year} (already saved).")
            continue
        try:
            deals = _pull_ma_range(
                f"{year}-01-01",
                f"{year}-12-31",
                fields,
                str(year),
            )
            if deals is None or len(deals) == 0:
                raise RuntimeError("empty")
            deals["year"] = year
            if MNA_SAVE_BY_YEAR:
                deals.to_parquet(year_path, index=False)
                log(f"    Found {len(deals):,} deals in {year} (saved {year_path.name})")
            else:
                all_deals.append(deals)
                log(f"    Found {len(deals):,} deals in {year}")
            time.sleep(1)  # Rate limit
        except Exception:
            if not MNA_FALLBACK_MONTHLY:
                log(f"    Year {year} failed; skipping (MNA_FALLBACK_MONTHLY=0).")
                continue
            log(f"    Year {year} failed; falling back to monthly pulls.")
            month_frames = []
            for start, end, month in _month_ranges(year):
                month_path = MNA_YEARLY_DIR / f"ma_deals_{year}_{month:02d}.parquet"
                if MNA_MONTH_SKIP_EXISTING and month_path.exists():
                    log(f"      Skipping {year}-{month:02d} (already saved).")
                    continue
                deals = _pull_ma_range(start, end, fields, f"{year}-{month:02d}")
                if deals is None or len(deals) == 0:
                    continue
                deals["year"] = year
                deals.to_parquet(month_path, index=False)
                month_frames.append(deals)
                log(f"      Found {len(deals):,} deals in {year}-{month:02d} (saved {month_path.name})")
                time.sleep(1)
            # Build year file from monthly parts if any exist
            month_files = sorted(MNA_YEARLY_DIR.glob(f"ma_deals_{year}_??.parquet"))
            if month_files:
                combined_year = pd.concat((pd.read_parquet(p) for p in month_files), ignore_index=True)
                combined_year.to_parquet(year_path, index=False)
                log(f"    Built yearly file from months -> {year_path.name} ({len(combined_year):,} rows)")

    if MNA_SAVE_BY_YEAR:
        year_files = []
        for y in range(MNA_START_YEAR, MNA_END_YEAR + 1):
            year_path = MNA_YEARLY_DIR / f"ma_deals_{y}.parquet"
            if not year_path.exists():
                month_files = sorted(MNA_YEARLY_DIR.glob(f"ma_deals_{y}_??.parquet"))
                if month_files:
                    combined_year = pd.concat((pd.read_parquet(p) for p in month_files), ignore_index=True)
                    combined_year.to_parquet(year_path, index=False)
            if year_path.exists():
                year_files.append(year_path)
        if year_files:
            combined = pd.concat((pd.read_parquet(p) for p in year_files), ignore_index=True)
            save_parquet(combined, 'ma_deals_all')
            if 'Target Nation' in combined.columns:
                us_deals = combined[combined['Target Nation'] == 'United States']
                save_parquet(us_deals, 'ma_deals_us')
                log(f"  Total: {len(combined):,} deals, {len(us_deals):,} US deals")
            else:
                log(f"  Total: {len(combined):,} deals (Target Nation not available)")
            return combined
        return pd.DataFrame()

    if all_deals:
        combined = pd.concat(all_deals, ignore_index=True)
        save_parquet(combined, 'ma_deals_all')

        # Filter to US deals
        if 'Target Nation' in combined.columns:
            us_deals = combined[combined['Target Nation'] == 'United States']
            save_parquet(us_deals, 'ma_deals_us')
            log(f"  Total: {len(combined):,} deals, {len(us_deals):,} US deals")
        else:
            log(f"  Total: {len(combined):,} deals")
        return combined
    return pd.DataFrame()

# ============================================================================
# 2. DIVIDEND ACTIONS
# ============================================================================
def pull_dividends():
    log("="*60)
    log("2. PULLING DIVIDEND ACTIONS")
    log("="*60)

    # Get S&P 1500 + additional companies
    log("  Getting universe of dividend-paying companies...")

    # Get companies from index
    try:
        sp500 = rd.get_data(
            universe='0#.SPX',
            fields=['TR.CommonName', 'TR.TRBCEconomicSector']
        )
        log(f"  Found {len(sp500)} S&P 500 companies")
    except:
        sp500 = pd.DataFrame()

    try:
        sp400 = rd.get_data(
            universe='0#.MID',
            fields=['TR.CommonName']
        )
        log(f"  Found {len(sp400)} S&P 400 companies")
    except:
        sp400 = pd.DataFrame()

    try:
        sp600 = rd.get_data(
            universe='0#.SML',
            fields=['TR.CommonName']
        )
        log(f"  Found {len(sp600)} S&P 600 companies")
    except:
        sp600 = pd.DataFrame()

    # Combine tickers
    all_tickers = []
    for df in [sp500, sp400, sp600]:
        if len(df) > 0 and 'Instrument' in df.columns:
            all_tickers.extend(df['Instrument'].tolist())

    all_tickers = list(set(all_tickers))
    log(f"  Total universe: {len(all_tickers)} companies")

    # Pull dividends in batches
    all_divs = []
    batch_size = 100

    for i in range(0, len(all_tickers), batch_size):
        batch = all_tickers[i:i+batch_size]
        log(f"  Pulling dividends batch {i//batch_size + 1}/{len(all_tickers)//batch_size + 1}...")

        try:
            divs = rd.get_data(
                universe=batch,
                fields=[
                    'TR.DivExDate',
                    'TR.DivPayDate',
                    'TR.DivRecordDate',
                    'TR.DivAmount',
                    'TR.DivType',
                    'TR.DivCurrency',
                    'TR.DivYield',
                    'TR.DivFrequency'
                ],
                parameters={'SDate': START_DATE, 'EDate': END_DATE}
            )
            if len(divs) > 0:
                all_divs.append(divs)
        except Exception as e:
            log(f"    Batch error: {e}")

        time.sleep(0.5)  # Rate limit

    if all_divs:
        combined = pd.concat(all_divs, ignore_index=True)
        # Remove rows with no dividend data
        combined = combined.dropna(subset=['Dividend Ex Date'])
        save_parquet(combined, 'dividends_all')

        # Categorize dividend actions
        log("  Categorizing dividend actions...")
        combined['Ex Date'] = pd.to_datetime(combined['Dividend Ex Date'])
        combined = combined.sort_values(['Instrument', 'Ex Date'])

        # Calculate dividend changes
        combined['prev_amount'] = combined.groupby('Instrument')['Dividend Amount'].shift(1)
        combined['pct_change'] = (combined['Dividend Amount'] - combined['prev_amount']) / combined['prev_amount']

        # Classify actions
        def classify_div_action(row):
            if pd.isna(row['prev_amount']):
                return 'initiation'
            elif row['pct_change'] > 0.01:
                return 'increase'
            elif row['pct_change'] < -0.01:
                return 'decrease'
            else:
                return 'unchanged'

        combined['action_type'] = combined.apply(classify_div_action, axis=1)
        save_parquet(combined, 'dividends_with_actions')

        log(f"  Total: {len(combined):,} dividend records")
        log(f"  Action breakdown: {combined['action_type'].value_counts().to_dict()}")
        return combined
    return pd.DataFrame()

# ============================================================================
# 3. SHARE BUYBACKS
# ============================================================================
def pull_buybacks():
    log("="*60)
    log("3. PULLING BUYBACK DATA")
    log("="*60)

    # Get companies with buyback activity via screening
    log("  Searching for companies with buyback activity...")

    try:
        # Screen for companies that have repurchased shares
        buybacks = rd.get_data(
            universe='SCREEN(U(IN(Deals)/*UNV:MADEALS*/), TR.MnADealType=="Self Tender Or Recapitalization Deal" OR TR.MnADealType=="Repurchases Deal", TR.MnAAnnDate>=2020-01-01)',
            fields=[
                'TR.MnADealValue(Scale=6)',
                'TR.MnAAnnDate',
                'TR.MnACompDate',
                'TR.MnADealType',
                'TR.MnATargetNation',
                'TR.MnATargetPrimarySICCode'
            ]
        )
        log(f"  Found {len(buybacks):,} buyback/repurchase deals")
        save_parquet(buybacks, 'buybacks_deals')
    except Exception as e:
        log(f"  Error: {e}")
        buybacks = pd.DataFrame()

    # Also get share repurchase from fundamentals
    log("  Pulling quarterly share repurchases from fundamentals...")
    try:
        # Get S&P 500 quarterly repurchase data
        sp500 = rd.get_data(universe='0#.SPX', fields=['TR.CommonName'])
        tickers = sp500['Instrument'].tolist()[:200]  # Start with 200

        repurchases = rd.get_data(
            universe=tickers,
            fields=[
                'TR.SharesRepurchased',
                'TR.RepurchaseOfCommonPreferredStock',
                'TR.CommonSharesOutstanding'
            ],
            parameters={'SDate': START_DATE, 'EDate': END_DATE, 'Period': 'FQ0', 'Frq': 'FQ'}
        )
        if len(repurchases) > 0:
            save_parquet(repurchases, 'share_repurchases_quarterly')
            log(f"  Found {len(repurchases):,} quarterly repurchase records")
    except Exception as e:
        log(f"  Quarterly repurchase error: {e}")

    return buybacks

# ============================================================================
# 4. STOCK SPLITS
# ============================================================================
def pull_splits():
    log("="*60)
    log("4. PULLING STOCK SPLITS")
    log("="*60)

    try:
        # Get S&P 1500 tickers
        sp500 = rd.get_data(universe='0#.SPX', fields=['TR.CommonName'])
        sp400 = rd.get_data(universe='0#.MID', fields=['TR.CommonName'])
        sp600 = rd.get_data(universe='0#.SML', fields=['TR.CommonName'])

        all_tickers = []
        for df in [sp500, sp400, sp600]:
            if 'Instrument' in df.columns:
                all_tickers.extend(df['Instrument'].tolist())
        all_tickers = list(set(all_tickers))

        all_splits = []
        batch_size = 200

        for i in range(0, len(all_tickers), batch_size):
            batch = all_tickers[i:i+batch_size]
            log(f"  Pulling splits batch {i//batch_size + 1}...")

            try:
                splits = rd.get_data(
                    universe=batch,
                    fields=[
                        'TR.CAEffectiveDate',
                        'TR.CAAdjustmentFactor',
                        'TR.CAAdjustmentType',
                        'TR.CAExDate'
                    ],
                    parameters={'CAType': 'SSP', 'SDate': START_DATE, 'EDate': END_DATE}  # SSP = Stock Split
                )
                if len(splits) > 0:
                    all_splits.append(splits)
            except Exception as e:
                log(f"    Batch error: {e}")

            time.sleep(0.5)

        if all_splits:
            combined = pd.concat(all_splits, ignore_index=True)
            combined = combined.dropna(subset=['CA Effective Date'])
            save_parquet(combined, 'stock_splits')
            log(f"  Total: {len(combined):,} stock splits")
            return combined
    except Exception as e:
        log(f"  Error: {e}")

    return pd.DataFrame()

# ============================================================================
# 5. SPINOFFS & DIVESTITURES
# ============================================================================
def pull_spinoffs():
    log("="*60)
    log("5. PULLING SPINOFFS & DIVESTITURES")
    log("="*60)

    try:
        # Spinoffs from M&A database
        spinoffs = rd.get_data(
            universe='SCREEN(U(IN(Deals)/*UNV:MADEALS*/), TR.MnADealType=="Spinoff Deal", TR.MnAAnnDate>=2020-01-01)',
            fields=[
                'TR.MnADealValue(Scale=6)',
                'TR.MnAAnnDate',
                'TR.MnACompDate',
                'TR.MnATargetNation',
                'TR.MnAStatus'
            ]
        )
        save_parquet(spinoffs, 'spinoffs')
        log(f"  Found {len(spinoffs):,} spinoff deals")
    except Exception as e:
        log(f"  Spinoff error: {e}")
        spinoffs = pd.DataFrame()

    try:
        # Divestitures
        divestitures = rd.get_data(
            universe='SCREEN(U(IN(Deals)/*UNV:MADEALS*/), TR.MnADealType=="Divestiture Deal", TR.MnAAnnDate>=2020-01-01)',
            fields=[
                'TR.MnADealValue(Scale=6)',
                'TR.MnAAnnDate',
                'TR.MnACompDate',
                'TR.MnATargetNation',
                'TR.MnAStatus'
            ]
        )
        save_parquet(divestitures, 'divestitures')
        log(f"  Found {len(divestitures):,} divestiture deals")
    except Exception as e:
        log(f"  Divestiture error: {e}")
        divestitures = pd.DataFrame()

    return spinoffs, divestitures

# ============================================================================
# 6. DEBT ISSUANCE
# ============================================================================
def pull_debt_issuance():
    log("="*60)
    log("6. PULLING DEBT ISSUANCE")
    log("="*60)

    try:
        debt = rd.get_data(
            universe='SCREEN(U(IN(Deals)/*UNV:FIDeals*/), TR.FIIssueDate>=2020-01-01)',
            fields=[
                'TR.FIPrincipalAmount(Scale=6)',
                'TR.FIIssueDate',
                'TR.FIMaturityDate',
                'TR.FICoupon',
                'TR.FIIssuerName',
                'TR.FIIssuerNation',
                'TR.FIInstrumentType',
                'TR.FIMoodyRating',
                'TR.FISPRating'
            ]
        )
        save_parquet(debt, 'debt_issuance')
        log(f"  Found {len(debt):,} debt issuances")
        return debt
    except Exception as e:
        log(f"  Error: {e}")
        return pd.DataFrame()

# ============================================================================
# 7. EQUITY OFFERINGS (IPO, Secondary)
# ============================================================================
def pull_equity_offerings():
    log("="*60)
    log("7. PULLING EQUITY OFFERINGS")
    log("="*60)

    try:
        # IPOs
        ipos = rd.get_data(
            universe='SCREEN(U(IN(Deals)/*UNV:EQDeals*/), TR.EQOfferType=="IPO", TR.EQOfferDate>=2020-01-01)',
            fields=[
                'TR.EQOfferAmount(Scale=6)',
                'TR.EQOfferDate',
                'TR.EQOfferPrice',
                'TR.EQIssuerName',
                'TR.EQIssuerNation',
                'TR.EQOfferType'
            ]
        )
        save_parquet(ipos, 'ipos')
        log(f"  Found {len(ipos):,} IPOs")
    except Exception as e:
        log(f"  IPO error: {e}")
        ipos = pd.DataFrame()

    try:
        # Secondary offerings
        secondary = rd.get_data(
            universe='SCREEN(U(IN(Deals)/*UNV:EQDeals*/), TR.EQOfferType=="Follow-on", TR.EQOfferDate>=2020-01-01)',
            fields=[
                'TR.EQOfferAmount(Scale=6)',
                'TR.EQOfferDate',
                'TR.EQOfferPrice',
                'TR.EQIssuerName',
                'TR.EQIssuerNation'
            ]
        )
        save_parquet(secondary, 'secondary_offerings')
        log(f"  Found {len(secondary):,} secondary offerings")
    except Exception as e:
        log(f"  Secondary error: {e}")
        secondary = pd.DataFrame()

    return ipos, secondary

# ============================================================================
# 8. FUNDAMENTALS (for state profiles)
# ============================================================================
def pull_fundamentals():
    log("="*60)
    log("8. PULLING FUNDAMENTALS")
    log("="*60)

    # Get S&P 1500
    log("  Getting company universe...")
    sp500 = rd.get_data(universe='0#.SPX', fields=['TR.CommonName'])
    sp400 = rd.get_data(universe='0#.MID', fields=['TR.CommonName'])
    sp600 = rd.get_data(universe='0#.SML', fields=['TR.CommonName'])

    all_tickers = []
    for df in [sp500, sp400, sp600]:
        if 'Instrument' in df.columns:
            all_tickers.extend(df['Instrument'].tolist())
    all_tickers = list(set(all_tickers))
    log(f"  Universe: {len(all_tickers)} companies")

    all_fundamentals = []
    batch_size = 50

    fields = [
        'TR.Revenue',
        'TR.RevenueGrowthRate',
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
        'TR.ReturnOnEquity',
        'TR.ReturnOnAssets',
        'TR.CompanyMarketCap',
        'TR.EV',
        'TR.PriceToBookValuePerShare',
        'TR.EVToEBITDA',
        'TR.TRBCEconomicSector',
        'TR.TRBCBusinessSector',
        'TR.GICSSector'
    ]

    for i in range(0, len(all_tickers), batch_size):
        batch = all_tickers[i:i+batch_size]
        log(f"  Pulling fundamentals batch {i//batch_size + 1}/{len(all_tickers)//batch_size + 1}...")

        try:
            fund = rd.get_data(
                universe=batch,
                fields=fields
            )
            if len(fund) > 0:
                all_fundamentals.append(fund)
        except Exception as e:
            log(f"    Batch error: {e}")

        time.sleep(0.5)

    if all_fundamentals:
        combined = pd.concat(all_fundamentals, ignore_index=True)
        save_parquet(combined, 'fundamentals_current')
        log(f"  Total: {len(combined):,} company fundamentals")
        return combined
    return pd.DataFrame()

# ============================================================================
# 9. ANALYST ESTIMATES
# ============================================================================
def pull_estimates():
    log("="*60)
    log("9. PULLING ANALYST ESTIMATES")
    log("="*60)

    # Get S&P 500 tickers
    sp500 = rd.get_data(universe='0#.SPX', fields=['TR.CommonName'])
    tickers = sp500['Instrument'].tolist()

    all_estimates = []
    batch_size = 100

    fields = [
        'TR.EPSMean',
        'TR.EPSHigh',
        'TR.EPSLow',
        'TR.EPSMedian',
        'TR.EPSActValue',
        'TR.EPSSurprisePercent',
        'TR.RevenueMean',
        'TR.RevenueActValue',
        'TR.EBITDAMean',
        'TR.NumberOfEstimates',
        'TR.RecommendationMean',
        'TR.NumOfRecommendations',
        'TR.TargetPriceMean'
    ]

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        log(f"  Pulling estimates batch {i//batch_size + 1}/{len(tickers)//batch_size + 1}...")

        try:
            est = rd.get_data(
                universe=batch,
                fields=fields
            )
            if len(est) > 0:
                all_estimates.append(est)
        except Exception as e:
            log(f"    Batch error: {e}")

        time.sleep(0.5)

    if all_estimates:
        combined = pd.concat(all_estimates, ignore_index=True)
        save_parquet(combined, 'analyst_estimates')
        log(f"  Total: {len(combined):,} estimate records")
        return combined
    return pd.DataFrame()

# ============================================================================
# 10. HISTORICAL PRICES (for TSR)
# ============================================================================
def pull_prices():
    log("="*60)
    log("10. PULLING HISTORICAL PRICES")
    log("="*60)

    # Get S&P 1500
    sp500 = rd.get_data(universe='0#.SPX', fields=['TR.CommonName'])
    sp400 = rd.get_data(universe='0#.MID', fields=['TR.CommonName'])
    sp600 = rd.get_data(universe='0#.SML', fields=['TR.CommonName'])

    all_tickers = []
    for df in [sp500, sp400, sp600]:
        if 'Instrument' in df.columns:
            all_tickers.extend(df['Instrument'].tolist())
    all_tickers = list(set(all_tickers))
    log(f"  Universe: {len(all_tickers)} companies")

    all_prices = []
    batch_size = 50

    for i in range(0, len(all_tickers), batch_size):
        batch = all_tickers[i:i+batch_size]
        log(f"  Pulling prices batch {i//batch_size + 1}/{len(all_tickers)//batch_size + 1}...")

        try:
            prices = rd.get_history(
                universe=batch,
                fields=['TR.CLOSEPRICE', 'TR.VOLUME', 'TR.TOTRETURN'],
                start=START_DATE,
                end=END_DATE,
                interval='monthly'
            )
            if prices is not None and len(prices) > 0:
                # Reset index to make it saveable
                prices_df = prices.reset_index()
                all_prices.append(prices_df)
        except Exception as e:
            log(f"    Batch error: {e}")

        time.sleep(0.5)

    if all_prices:
        combined = pd.concat(all_prices, ignore_index=True)
        save_parquet(combined, 'prices_monthly')
        log(f"  Total: {len(combined):,} price records")
        return combined
    return pd.DataFrame()


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("="*70)
    print("REFINITIV COMPREHENSIVE DATA PULL")
    print("="*70)
    print(f"Started at: {datetime.now()}")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print(f"Output directory: {DATA_DIR}")
    print()

    # Connect to Refinitiv
    log("Connecting to Refinitiv...")
    try:
        rd.open_session()
    except Exception as e:
        log(f"Failed to open Refinitiv session: {e}")
        return
    if not ensure_session():
        rd.close_session()
        return
    log("Connected!")

    # Pull everything
    results = {}

    try:
        results['ma_deals'] = pull_ma_deals()
    except Exception as e:
        log(f"M&A FAILED: {e}")

    if ONLY_MNA:
        rd.close_session()
        log("ONLY_MNA=1 set; skipping remaining pulls.")
        return

    try:
        results['dividends'] = pull_dividends()
    except Exception as e:
        log(f"DIVIDENDS FAILED: {e}")

    try:
        results['buybacks'] = pull_buybacks()
    except Exception as e:
        log(f"BUYBACKS FAILED: {e}")

    try:
        results['splits'] = pull_splits()
    except Exception as e:
        log(f"SPLITS FAILED: {e}")

    try:
        results['spinoffs'] = pull_spinoffs()
    except Exception as e:
        log(f"SPINOFFS FAILED: {e}")

    try:
        results['debt'] = pull_debt_issuance()
    except Exception as e:
        log(f"DEBT FAILED: {e}")

    try:
        results['equity'] = pull_equity_offerings()
    except Exception as e:
        log(f"EQUITY FAILED: {e}")

    try:
        results['fundamentals'] = pull_fundamentals()
    except Exception as e:
        log(f"FUNDAMENTALS FAILED: {e}")

    try:
        results['estimates'] = pull_estimates()
    except Exception as e:
        log(f"ESTIMATES FAILED: {e}")

    try:
        results['prices'] = pull_prices()
    except Exception as e:
        log(f"PRICES FAILED: {e}")

    # Summary
    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Completed at: {datetime.now()}")
    print(f"Files saved to: {DATA_DIR}")
    print()

    # List output files
    for f in sorted(DATA_DIR.glob('*.parquet')):
        size_mb = f.stat().st_size / 1024 / 1024
        df = pd.read_parquet(f)
        print(f"  {f.name}: {len(df):,} rows ({size_mb:.1f} MB)")

    rd.close_session()
    print()
    print("DONE!")

if __name__ == '__main__':
    main()
