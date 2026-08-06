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


def pull_ticker_changes():
    """Pull ticker symbol changes from CRSP."""
    print("\n" + "=" * 60)
    print("TICKER CHANGES (CRSP)")
    print("=" * 60)

    conn = get_connection()

    query = """
    WITH ticker_hist AS (
        SELECT
            permno,
            ticker,
            comnam,
            namedt,
            nameendt,
            LAG(ticker) OVER (PARTITION BY permno ORDER BY namedt) as prev_ticker
        FROM crsp.msenames
        WHERE namedt >= '2010-01-01'
    )
    SELECT
        permno,
        namedt as action_date,
        prev_ticker,
        ticker as new_ticker,
        comnam as company_name
    FROM ticker_hist
    WHERE prev_ticker IS NOT NULL
      AND prev_ticker != ticker
    ORDER BY namedt DESC
    """

    df = pd.read_sql(query, conn)
    df['action_type'] = 'ticker_change'
    df['action_date'] = pd.to_datetime(df['action_date'])
    print(f"  Retrieved {len(df):,} ticker changes")

    df.to_parquet(DATA_DIR / 'ticker_changes.parquet')
    conn.close()
    return df


def link_all_to_compustat(actions_list):
    """Link all CRSP-based actions to Compustat gvkey."""
    print("\n" + "=" * 60)
    print("LINKING ALL TO COMPUSTAT")
    print("=" * 60)

    conn = get_connection()

    # Get the linking table
    query = """
    SELECT lpermno as permno, gvkey, linkdt, linkenddt
    FROM crsp.ccmxpf_lnkhist
    WHERE linktype IN ('LU', 'LC', 'LS')
      AND linkprim IN ('P', 'C')
    """

    link = pd.read_sql(query, conn)
    link['linkdt'] = pd.to_datetime(link['linkdt'])
    link['linkenddt'] = pd.to_datetime(link['linkenddt'].fillna('2099-12-31'))

    def add_gvkey(df):
        if 'permno' not in df.columns or len(df) == 0:
            return df
        merged = df.merge(link, on='permno', how='left')
        # Filter to valid link dates
        if 'action_date' in merged.columns:
            merged = merged[
                (merged['action_date'] >= merged['linkdt']) &
                (merged['action_date'] <= merged['linkenddt'])
            ]
        merged = merged.drop_duplicates('permno', keep='first')
        return merged

    linked_count = 0
    total_count = 0

    for name, df in actions_list:
        if len(df) > 0 and 'permno' in df.columns:
            linked = add_gvkey(df)
            linked_count += linked['gvkey'].notna().sum()
            total_count += len(df)
            # Save linked version
            linked.to_parquet(DATA_DIR / f'{name}_linked.parquet')

    print(f"  Linked {linked_count:,} / {total_count:,} actions to Compustat")
    conn.close()


def main():
    print("=" * 70)
    print("PULLING ALL CORPORATE ACTIONS FROM WRDS")
    print("=" * 70)

    all_actions = []

    # 1. Stock splits
    splits = pull_stock_splits()
    all_actions.append(('stock_splits', splits))

    # 2. Reverse splits
    reverse = pull_reverse_splits()
    all_actions.append(('reverse_splits', reverse))

    # 3. Special dividends
    special_div = pull_special_dividends()
    all_actions.append(('special_dividends', special_div))

    # 4. Spin-offs
    spinoffs = pull_spinoffs()
    all_actions.append(('spinoffs', spinoffs))

    # 5. Rights offerings
    rights = pull_rights_offerings()
    all_actions.append(('rights_offerings', rights))

    # 6. Return of capital
    roc = pull_return_of_capital()
    all_actions.append(('return_of_capital', roc))

    # 7. Going private
    going_priv = pull_going_private()
    all_actions.append(('going_private', going_priv))

    # 8. Bond issuances
    bonds = pull_bond_issuances()
    all_actions.append(('bond_issuances', bonds))

    # 9. Ticker changes
    tickers = pull_ticker_changes()
    all_actions.append(('ticker_changes', tickers))

    # Link to Compustat
    link_all_to_compustat(all_actions)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY - ALL CORPORATE ACTIONS")
    print("=" * 70)

    total = 0
    for name, df in all_actions:
        count = len(df)
        total += count
        print(f"  {name:25} {count:>8,}")

    print("-" * 70)
    print(f"  {'TOTAL NEW ACTIONS':25} {total:>8,}")

    print("\n  Plus existing data:")
    print(f"    Buybacks (Compustat):     53,025")
    print(f"    Acquisitions (CRSP):       3,726")
    print(f"    Bankruptcies (CRSP):       2,014")
    print(f"    Dividends (CRSP):        151,457")


if __name__ == "__main__":
    main()
