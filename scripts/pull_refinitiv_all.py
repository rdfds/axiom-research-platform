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
