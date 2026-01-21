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


