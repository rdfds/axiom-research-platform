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


