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


def pull_special_dividends():
    """Pull special/irregular dividends from CRSP."""
    print("\n" + "=" * 60)
    print("SPECIAL DIVIDENDS (CRSP)")
    print("=" * 60)

    conn = get_connection()

    # 1272 = special dividend, 1262 = irregular, 1292 = liquidating
    query = """
    SELECT
        d.permno,
        d.exdt as action_date,
        d.distcd,
        d.divamt as dividend_amount,
        n.comnam as company_name,
        n.ticker,
        n.siccd as sic
    FROM crsp.msedist d
    LEFT JOIN crsp.msenames n
        ON d.permno = n.permno
        AND d.exdt BETWEEN n.namedt AND n.nameendt
    WHERE d.exdt >= '2010-01-01'
      AND d.distcd IN (1272, 1262, 1292, 1273, 1263)  -- Special, irregular, liquidating
    ORDER BY d.exdt DESC
    """

    df = pd.read_sql(query, conn)

    def classify_dividend(code):
        if code in [1272, 1273]:
            return 'dividend_special'
        elif code in [1262, 1263]:
            return 'dividend_irregular'
        elif code == 1292:
            return 'dividend_liquidating'
        return 'dividend_other'

    df['action_type'] = df['distcd'].apply(classify_dividend)
    df['action_date'] = pd.to_datetime(df['action_date'])
    print(f"  Retrieved {len(df):,} special dividends")

    df.to_parquet(DATA_DIR / 'special_dividends.parquet')
    conn.close()
    return df


def pull_spinoffs():
    """Pull spin-offs from CRSP distributions."""
    print("\n" + "=" * 60)
    print("SPIN-OFFS (CRSP)")
    print("=" * 60)

    conn = get_connection()

    # 4112 = spin-off, 41xx range
    query = """
    SELECT
        d.permno,
        d.exdt as action_date,
        d.distcd,
        n.comnam as company_name,
        n.ticker,
        n.siccd as sic
    FROM crsp.msedist d
    LEFT JOIN crsp.msenames n
        ON d.permno = n.permno
        AND d.exdt BETWEEN n.namedt AND n.nameendt
    WHERE d.exdt >= '2010-01-01'
      AND d.distcd BETWEEN 4100 AND 4199
    ORDER BY d.exdt DESC
    """

    df = pd.read_sql(query, conn)
    df['action_type'] = 'spinoff'
    df['action_date'] = pd.to_datetime(df['action_date'])
    print(f"  Retrieved {len(df):,} spin-offs")

    df.to_parquet(DATA_DIR / 'spinoffs.parquet')
    conn.close()
    return df


def pull_rights_offerings():
    """Pull rights offerings from CRSP."""
    print("\n" + "=" * 60)
    print("RIGHTS OFFERINGS (CRSP)")
    print("=" * 60)

    conn = get_connection()

    # 4122 = rights distribution, 41xx-42xx range
    query = """
    SELECT
        d.permno,
        d.exdt as action_date,
        d.distcd,
        n.comnam as company_name,
        n.ticker,
        n.siccd as sic
    FROM crsp.msedist d
    LEFT JOIN crsp.msenames n
        ON d.permno = n.permno
        AND d.exdt BETWEEN n.namedt AND n.nameendt
    WHERE d.exdt >= '2010-01-01'
      AND d.distcd BETWEEN 4120 AND 4199
    ORDER BY d.exdt DESC
    """

    df = pd.read_sql(query, conn)
    df['action_type'] = 'rights_offering'
    df['action_date'] = pd.to_datetime(df['action_date'])
    print(f"  Retrieved {len(df):,} rights offerings")

    df.to_parquet(DATA_DIR / 'rights_offerings.parquet')
    conn.close()
    return df


def pull_return_of_capital():
    """Pull return of capital distributions."""
    print("\n" + "=" * 60)
    print("RETURN OF CAPITAL (CRSP)")
    print("=" * 60)

    conn = get_connection()

    # 45xx = return of capital
    query = """
    SELECT
        d.permno,
        d.exdt as action_date,
        d.distcd,
        d.divamt as amount,
        n.comnam as company_name,
        n.ticker,
        n.siccd as sic
    FROM crsp.msedist d
    LEFT JOIN crsp.msenames n
        ON d.permno = n.permno
        AND d.exdt BETWEEN n.namedt AND n.nameendt
    WHERE d.exdt >= '2010-01-01'
      AND d.distcd BETWEEN 4500 AND 4599
    ORDER BY d.exdt DESC
    """

    df = pd.read_sql(query, conn)
    df['action_type'] = 'return_of_capital'
    df['action_date'] = pd.to_datetime(df['action_date'])
    print(f"  Retrieved {len(df):,} return of capital")

    df.to_parquet(DATA_DIR / 'return_of_capital.parquet')
    conn.close()
    return df


def pull_going_private():
    """Pull going private transactions from CRSP delistings."""
    print("\n" + "=" * 60)
    print("GOING PRIVATE (CRSP DELISTINGS)")
    print("=" * 60)

    conn = get_connection()

    # 251 = went private
    query = """
    SELECT
        d.permno,
        d.dlstdt as action_date,
        d.dlstcd as delist_code,
        n.comnam as company_name,
        n.ticker,
        n.siccd as sic
    FROM crsp.msedelist d
    LEFT JOIN crsp.msenames n
        ON d.permno = n.permno
        AND d.dlstdt BETWEEN n.namedt AND n.nameendt
    WHERE d.dlstdt >= '2010-01-01'
      AND d.dlstcd = 251
    ORDER BY d.dlstdt DESC
    """

    df = pd.read_sql(query, conn)
    df['action_type'] = 'going_private'
    df['action_date'] = pd.to_datetime(df['action_date'])
    print(f"  Retrieved {len(df):,} going private transactions")

    df.to_parquet(DATA_DIR / 'going_private.parquet')
    conn.close()
    return df


def pull_bond_issuances():
    """Pull corporate bond issuances from FISD."""
    print("\n" + "=" * 60)
    print("BOND ISSUANCES (FISD)")
    print("=" * 60)

    conn = get_connection()

    # Check what tables exist in FISD
    try:
        query = """
        SELECT
            issue_id,
            issuer_id,
            offering_date as action_date,
            offering_amt as deal_value,
            maturity,
            coupon,
            security_level,
            sic_code as sic
        FROM fisd.fisd_issue
        WHERE offering_date >= '2010-01-01'
          AND offering_amt > 100  -- $100M+ issuances
        ORDER BY offering_date DESC
        LIMIT 50000
        """

        df = pd.read_sql(query, conn)
        df['action_type'] = 'bond_issuance'
        df['action_date'] = pd.to_datetime(df['action_date'])
        print(f"  Retrieved {len(df):,} bond issuances")

        df.to_parquet(DATA_DIR / 'bond_issuances.parquet')
    except Exception as e:
        print(f"  Error accessing FISD: {e}")
        df = pd.DataFrame()

    conn.close()
    return df


