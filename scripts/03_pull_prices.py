"""
Pull Historical Stock Prices from CRSP
======================================
This script pulls daily stock prices for calculating TSR outcomes.

Usage:
  python 03_pull_prices.py USERNAME monthly   # Monthly prices (recommended for V1)
  python 03_pull_prices.py USERNAME daily     # Daily prices (large)

Output: data/prices_monthly.parquet or data/prices_daily.parquet
"""

import sys
import wrds
import pandas as pd
from pathlib import Path

# Configuration
START_DATE = '2010-01-01'
OUTPUT_DIR = Path(__file__).parent.parent / 'data'


def get_crsp_prices(db):
    """
    Pull daily stock prices from CRSP.

    Key fields for TSR calculation:
    - ret: Daily return (includes dividends)
    - prc: Price (negative means bid/ask average)
    - shrout: Shares outstanding
    - cfacpr: Cumulative factor to adjust price
    - cfacshr: Cumulative factor to adjust shares
    """

    query = """
    SELECT
        a.permno,
        a.permco,
        a.date,
        a.prc,
        a.ret,
        a.retx,
        a.shrout,
        a.vol,
        a.cfacpr,
        a.cfacshr,
        b.gvkey,
        b.linkprim

    FROM crsp.dsf a
    LEFT JOIN crsp.ccmxpf_lnkhist b
        ON a.permno = b.lpermno
        AND a.date >= b.linkdt
        AND (a.date <= b.linkenddt OR b.linkenddt IS NULL)
        AND b.linktype IN ('LU', 'LC')
        AND b.linkprim IN ('P', 'C')

    WHERE a.date >= %(start_date)s

    ORDER BY a.permno, a.date
    """

    print("Executing query (this may take a while)...")
    df = db.raw_sql(query, params={'start_date': START_DATE})
    print(f"Retrieved {len(df):,} rows")

    return df


def get_monthly_prices(db):
    """
    Alternative: Pull monthly prices (smaller dataset).
    Good for initial testing.
    """

    query = """
    SELECT
        a.permno,
        a.permco,
        a.date,
        a.prc,
        a.ret,
        a.retx,
        a.shrout,
        a.cfacpr,
        a.cfacshr,
        b.gvkey,
        b.linkprim

    FROM crsp.msf a
    LEFT JOIN crsp.ccmxpf_lnkhist b
        ON a.permno = b.lpermno
        AND a.date >= b.linkdt
        AND (a.date <= b.linkenddt OR b.linkenddt IS NULL)
        AND b.linktype IN ('LU', 'LC')
        AND b.linkprim IN ('P', 'C')

    WHERE a.date >= %(start_date)s

    ORDER BY a.permno, a.date
    """

    print("Executing query...")
    df = db.raw_sql(query, params={'start_date': START_DATE})
    print(f"Retrieved {len(df):,} rows")

    return df


def validate_data(df):
    """Basic validation checks."""
    print("\n--- Data Validation ---")

    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    n_permnos = df['permno'].nunique()
    print(f"Unique securities (permno): {n_permnos:,}")

    # Check gvkey linkage (how many have Compustat link)
    linked = df['gvkey'].notna().sum()
    linked_pct = linked / len(df) * 100
    print(f"Linked to Compustat: {linked_pct:.1f}%")

    # Check return coverage
    ret_coverage = (1 - df['ret'].isna().sum() / len(df)) * 100
    print(f"Return coverage: {ret_coverage:.1f}%")

    return True


def main():
    # Parse command line args
    if len(sys.argv) < 2:
        print("Usage: python 03_pull_prices.py USERNAME [monthly|daily]")
        print("  Default is monthly if not specified")
        sys.exit(1)

    WRDS_USERNAME = sys.argv[1]
    frequency = sys.argv[2] if len(sys.argv) > 2 else 'monthly'

    print("Connecting to WRDS...")
    db = wrds.Connection(wrds_username=WRDS_USERNAME)
    print("Connected!\n")

    OUTPUT_DIR.mkdir(exist_ok=True)

    if frequency == 'daily':
        print("="*60)
        print(f"PULLING DAILY PRICES FROM {START_DATE}")
        print("="*60)
        df = get_crsp_prices(db)
        output_path = OUTPUT_DIR / 'prices_daily.parquet'
    else:
        print("="*60)
        print(f"PULLING MONTHLY PRICES FROM {START_DATE}")
        print("="*60)
        df = get_monthly_prices(db)
        output_path = OUTPUT_DIR / 'prices_monthly.parquet'

    # Validate
    validate_data(df)

    # Save
    print(f"\nSaving to {output_path}...")
    df.to_parquet(output_path, index=False)
    print(f"Saved! File size: {output_path.stat().st_size / 1e6:.1f} MB")

    db.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
