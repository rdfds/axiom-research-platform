"""
Explore Corporate Actions Data in WRDS
======================================
We want to track ALL corporate actions, not just M&A:
- Buybacks
- Dividends (initiate, increase, cut)
- Equity issuances
- Spin-offs
- Divestitures
- Debt actions (already have from DealScan)

This script explores what's available in WRDS.

Usage:
  python 08_explore_corporate_actions.py USERNAME
"""

import sys
import wrds
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / 'data'


def explore_crsp_distributions(db):
    """
    CRSP has distribution data - dividends, splits, spin-offs.
    """
    print("\n" + "="*70)
    print("EXPLORING CRSP DISTRIBUTION DATA")
    print("="*70)

    # Check what distribution types exist
    query = """
    SELECT
        distcd,
        COUNT(*) as count
    FROM crsp.dsedist
    WHERE exdt >= '2010-01-01'
    GROUP BY distcd
    ORDER BY count DESC
    LIMIT 30
    """

    try:
        df = db.raw_sql(query)
        print("\nDistribution codes (distcd) in CRSP:")
        print(df)

        # CRSP distcd meanings:
        # 1xxx = Cash dividends
        # 2xxx = Stock dividends
        # 3xxx = Liquidating dividends
        # 4xxx = Stock splits
        # 5xxx = Rights and warrants
        # 6xxx = Spin-offs

        print("""
Distribution Code Guide:
  1xxx = Cash dividends (1232 = regular cash div)
  2xxx = Stock dividends
  3xxx = Liquidating dividends
  4xxx = Stock splits
  5xxx = Rights and warrants
  6xxx = Spin-offs
        """)

    except Exception as e:
        print(f"Error querying CRSP distributions: {e}")
        return None

    # Get sample of actual distribution data
    query = """
    SELECT
        permno,
        exdt as ex_date,
        paydt as pay_date,
        distcd as dist_code,
        divamt as div_amount,
        facpr as factor_price,
        facshr as factor_share
    FROM crsp.dsedist
    WHERE exdt >= '2015-01-01'
      AND exdt <= '2023-12-31'
    ORDER BY exdt DESC
    LIMIT 1000
    """

    try:
        df = db.raw_sql(query)
        print(f"\nSample CRSP distribution data: {len(df):,} records")
        print(df.head(20))
        return df
    except Exception as e:
        print(f"Error: {e}")
        return None


def explore_compustat_buybacks(db):
    """
    Compustat tracks treasury stock - can infer buyback activity.
    """
    print("\n" + "="*70)
    print("EXPLORING COMPUSTAT BUYBACK DATA (Treasury Stock)")
    print("="*70)

    # Treasury stock changes indicate buyback activity
    query = """
    SELECT
        gvkey,
        datadate,
        tic,
        conm,
        tstkq as treasury_stock,
        cshoq as shares_outstanding,
        prccq as stock_price,
        mkvaltq as market_cap,
        -- Cash flow items related to buybacks
        prstkcy as purchase_common_stock,  -- YTD purchase of common stock
        sstky as sale_stock  -- YTD sale of stock
    FROM comp.fundq
    WHERE datadate >= '2015-01-01'
      AND tstkq IS NOT NULL
      AND tstkq > 0
    ORDER BY datadate DESC
    LIMIT 500
    """

    try:
        df = db.raw_sql(query)
        print(f"\nCompustat buyback-related data: {len(df):,} records")
        print("\nColumns available:")
        print(df.columns.tolist())
        print(df.head(20))
        return df
    except Exception as e:
        print(f"Error: {e}")

        # Try alternative query
        print("\nTrying alternative query...")
        query = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'comp'
          AND table_name = 'fundq'
          AND column_name LIKE '%stk%'
        """
        try:
            cols = db.raw_sql(query)
            print("Stock-related columns in comp.fundq:")
            print(cols)
        except:
            pass

        return None


def explore_compustat_dividends(db):
    """
    Compustat dividend fields.
    """
    print("\n" + "="*70)
    print("EXPLORING COMPUSTAT DIVIDEND DATA")
    print("="*70)

    query = """
    SELECT
        gvkey,
        datadate,
        tic,
        conm,
        dvpq as dividends_preferred,
        dvy as dividends_common_annual,
        dvpspq as div_per_share_preferred,
        cshoq as shares_out
    FROM comp.fundq
    WHERE datadate >= '2015-01-01'
      AND (dvpq > 0 OR dvy > 0)
    ORDER BY datadate DESC
    LIMIT 500
    """

    try:
        df = db.raw_sql(query)
        print(f"\nCompustat dividend data: {len(df):,} records")
        print(df.head(20))
        return df
    except Exception as e:
        print(f"Error: {e}")
        return None


def explore_ciq_transactions(db):
    """
    Capital IQ has various transaction types.
    """
    print("\n" + "="*70)
    print("EXPLORING CAPITAL IQ TRANSACTION TYPES")
    print("="*70)

    # First check what tables we have access to
    try:
        tables = db.list_tables(library='ciqsamp')
        print(f"\nCIQ sample tables available: {tables}")
    except Exception as e:
        print(f"Cannot list CIQ tables: {e}")

    # Check transaction types
    query = """
    SELECT
        transactiontype,
        COUNT(*) as count
    FROM ciqsamp_transactions.wrds_transactions
    GROUP BY transactiontype
    ORDER BY count DESC
    """

    try:
        df = db.raw_sql(query)
        print("\nTransaction types in CIQ sample:")
        print(df)
        return df
    except Exception as e:
        print(f"Error querying CIQ transactions: {e}")
        return None


def explore_keydev(db):
    """
    Capital IQ Key Developments - news events including corporate actions.
    """
    print("\n" + "="*70)
    print("EXPLORING CAPITAL IQ KEY DEVELOPMENTS")
    print("="*70)

    # Check what key development types exist
    query = """
    SELECT
        keydevtypename,
        COUNT(*) as count
    FROM ciqsamp.ciqkeydev
    GROUP BY keydevtypename
    ORDER BY count DESC
    LIMIT 50
    """

    try:
        df = db.raw_sql(query)
        print("\nKey Development types in CIQ:")
        print(df)

        # This is potentially very useful - keydev might have:
        # - Buyback announcements
        # - Dividend changes
        # - Spin-off announcements
        # - Divestiture announcements

        return df
    except Exception as e:
        print(f"Error querying CIQ keydev: {e}")

        # Try alternative
        try:
            query = """
            SELECT * FROM ciqsamp.ciqkeydev LIMIT 5
            """
            df = db.raw_sql(query)
            print("\nSample keydev data:")
            print(df)
            print("\nColumns:", df.columns.tolist())
        except Exception as e2:
            print(f"Alternative also failed: {e2}")

        return None


def explore_tfn_insider(db):
    """
    Thomson Financial insider data - may have buyback signals.
    """
    print("\n" + "="*70)
    print("EXPLORING THOMSON FINANCIAL DATA")
    print("="*70)

    try:
        tables = db.list_tables(library='tfn')
        print(f"\nTFN tables available: {tables[:20]}...")  # Limit output
    except Exception as e:
        print(f"Cannot list TFN tables: {e}")


def explore_ibes(db):
    """
    IBES has analyst data - can be useful for expectations.
    """
    print("\n" + "="*70)
    print("EXPLORING IBES (Analyst Expectations)")
    print("="*70)

    try:
        tables = db.list_tables(library='ibes')
        print(f"\nIBES tables available: {tables[:20]}...")
    except Exception as e:
        print(f"Cannot list IBES tables: {e}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python 08_explore_corporate_actions.py USERNAME")
        sys.exit(1)

    WRDS_USERNAME = sys.argv[1]

    print("Connecting to WRDS...")
    db = wrds.Connection(wrds_username=WRDS_USERNAME)
    print("Connected!\n")

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Explore each data source
    crsp_dist = explore_crsp_distributions(db)
    buyback_data = explore_compustat_buybacks(db)
    div_data = explore_compustat_dividends(db)
    ciq_types = explore_ciq_transactions(db)
    keydev = explore_keydev(db)
    explore_tfn_insider(db)
    explore_ibes(db)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Corporate Actions Data Availability")
    print("="*70)
    print("""
┌─────────────────────┬──────────────┬─────────────────────────────────┐
│ Corporate Action    │ Available?   │ Source                          │
├─────────────────────┼──────────────┼─────────────────────────────────┤
│ M&A / Acquisitions  │ ✓ YES        │ DealScan, CIQ transactions      │
│ Debt Issuance       │ ✓ YES        │ DealScan (facility table)       │
│ Cash Dividends      │ ✓ YES        │ CRSP dsedist (distcd 1xxx)      │
│ Dividend Changes    │ ~ PARTIAL    │ Infer from CRSP payment changes │
│ Buybacks            │ ~ PARTIAL    │ Compustat treasury stock changes│
│ Spin-offs           │ ~ PARTIAL    │ CRSP dsedist (distcd 6xxx)      │
│ Equity Offerings    │ ? CHECK      │ May be in CIQ keydev            │
│ Divestitures        │ ? CHECK      │ May be in CIQ keydev            │
└─────────────────────┴──────────────┴─────────────────────────────────┘

NEXT STEPS:
1. Pull CRSP dividend history (dsedist with distcd 1xxx)
2. Build buyback detection from Compustat treasury stock changes
3. Extract spin-offs from CRSP (distcd 6xxx)
4. Explore CIQ keydev for announcements
5. Link all actions to state profiles for outcome analysis
    """)

    db.close()


if __name__ == "__main__":
    main()
