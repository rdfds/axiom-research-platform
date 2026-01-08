"""
Pull ALL Available Corporate Actions from WRDS
==============================================
Comprehensive pull of every corporate action we can track with certainty.

Sources:
- Compustat: Buybacks (prstkcy)
- CRSP Distributions: Dividends, splits, spin-offs, special distributions
- CRSP Delistings: Acquisitions, bankruptcies, going private
- FISD: Bond issuances
- DealScan: Debt refinancing (already have)
"""

import psycopg2
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / 'data'


def get_connection():
    return psycopg2.connect(
        host='wrds-pgdata.wharton.upenn.edu',
        port=9737,
        database='wrds',
        user='rvarian1'
    )


def pull_stock_splits():
    """Pull stock splits from CRSP distributions."""
    print("\n" + "=" * 60)
    print("STOCK SPLITS (CRSP)")
    print("=" * 60)

    conn = get_connection()

    # Distribution codes 5xxx are splits
    # 5523 = 2:1, 5532 = 3:2, 5542 = 3:1, etc.
    query = """
    SELECT
        d.permno,
        d.exdt as action_date,
        d.distcd,
        d.facpr as split_factor,
        n.comnam as company_name,
        n.ticker,
        n.siccd as sic
    FROM crsp.msedist d
    LEFT JOIN crsp.msenames n
        ON d.permno = n.permno
        AND d.exdt BETWEEN n.namedt AND n.nameendt
    WHERE d.exdt >= '2010-01-01'
      AND d.distcd BETWEEN 5500 AND 5599  -- Forward splits
    ORDER BY d.exdt DESC
    """

    df = pd.read_sql(query, conn)
    df['action_type'] = 'stock_split'
    df['action_date'] = pd.to_datetime(df['action_date'])
    print(f"  Retrieved {len(df):,} stock splits")

    # Save
    df.to_parquet(DATA_DIR / 'stock_splits.parquet')
    conn.close()
    return df


def pull_reverse_splits():
    """Pull reverse splits from CRSP."""
    print("\n" + "=" * 60)
    print("REVERSE SPLITS (CRSP)")
    print("=" * 60)

    conn = get_connection()

    # Reverse splits have facpr < 1 (or codes 56xx)
    query = """
    SELECT
        d.permno,
        d.exdt as action_date,
        d.distcd,
        d.facpr as split_factor,
        n.comnam as company_name,
        n.ticker,
        n.siccd as sic
    FROM crsp.msedist d
    LEFT JOIN crsp.msenames n
        ON d.permno = n.permno
        AND d.exdt BETWEEN n.namedt AND n.nameendt
    WHERE d.exdt >= '2010-01-01'
      AND (d.distcd BETWEEN 5600 AND 5699 OR d.facpr < 1)
    ORDER BY d.exdt DESC
    """

    df = pd.read_sql(query, conn)
    df['action_type'] = 'reverse_split'
    df['action_date'] = pd.to_datetime(df['action_date'])
    print(f"  Retrieved {len(df):,} reverse splits")

    df.to_parquet(DATA_DIR / 'reverse_splits.parquet')
    conn.close()
    return df


