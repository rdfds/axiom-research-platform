"""
Pull M&A Deal Data from Capital IQ Sample Tables
================================================
This script pulls M&A transaction data from the accessible CIQ sample tables.

Usage:
  python 05_pull_ma_deals.py USERNAME

Output: data/ma_deals.parquet
"""

import sys
import wrds
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / 'data'


def test_table_access(db, library, table):
    """Test if we can actually query a table."""
    try:
        df = db.raw_sql(f"SELECT * FROM {library}.{table} LIMIT 1")
        return True, len(df.columns)
    except Exception as e:
        return False, str(e)[:50]


def find_accessible_tables(db):
    """Find which CIQ tables we can actually access."""

    print("="*70)
    print("FINDING ACCESSIBLE CIQ/TRANSACTION TABLES")
    print("="*70)

    libraries_to_check = [
        'ciqsamp',
        'ciqsamp_transactions',
        'ciqsamp_keydev',
    ]

    accessible = []

    for lib in libraries_to_check:
        print(f"\nChecking {lib}...")
        try:
            tables = db.list_tables(library=lib)
            if hasattr(tables, 'values'):
                tables = list(tables.values.flatten())

            for table in tables:
                success, info = test_table_access(db, lib, table)
                status = "✓" if success else "✗"
                print(f"  {status} {lib}.{table}: {info if not success else f'{info} columns'}")
                if success:
                    accessible.append((lib, table))
        except Exception as e:
            print(f"  Could not list tables: {e}")

    return accessible


def pull_accessible_transaction_data(db, accessible_tables):
    """Pull data from accessible transaction-related tables."""

    results = {}

    # Priority tables for M&A data
    priority_tables = [
        ('ciqsamp_transactions', 'wrds_transactions'),
        ('ciqsamp_transactions', 'wrds_consideration_financials'),
        ('ciqsamp', 'wrds_transactions'),
        ('ciqsamp_keydev', 'ciqkeydev'),
    ]

    for lib, table in priority_tables:
        if (lib, table) in accessible_tables:
            print(f"\n{'='*70}")
            print(f"PULLING {lib}.{table}")
            print("="*70)

            try:
                df = db.raw_sql(f"SELECT * FROM {lib}.{table}")
                print(f"Retrieved {len(df):,} rows, {len(df.columns)} columns")
                print(f"Columns: {list(df.columns)}")

                # Show sample
                print("\nSample:")
                print(df.head(3))

                results[f"{lib}.{table}"] = df

            except Exception as e:
                print(f"Error: {e}")

    return results


def pull_compustat_acquisitions(db):
    """
    Pull acquisition data from Compustat (AQC field).
    This is a backup - shows which companies made acquisitions and rough amounts.
    """

    print("\n" + "="*70)
    print("PULLING COMPUSTAT ACQUISITION DATA (Backup)")
    print("="*70)

    query = """
    SELECT
        gvkey,
        datadate,
        conm,
        tic,
        aqc,      -- Acquisitions ($ amount spent)
        sale,     -- Sales for context
        at,       -- Total assets for context
        sic       -- Industry
    FROM comp.funda
    WHERE aqc IS NOT NULL
      AND aqc > 0
      AND datadate >= '2010-01-01'
      AND indfmt = 'INDL'
      AND datafmt = 'STD'
      AND popsrc = 'D'
      AND consol = 'C'
    ORDER BY datadate, aqc DESC
    """

    try:
        df = db.raw_sql(query)
        print(f"Retrieved {len(df):,} company-years with acquisitions")

        # Summary stats
        print(f"\nUnique acquirers: {df['gvkey'].nunique():,}")
        print(f"Date range: {df['datadate'].min()} to {df['datadate'].max()}")
        print(f"Total acquisition value: ${df['aqc'].sum():,.0f}M")

        return df
    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python 05_pull_ma_deals.py USERNAME")
        sys.exit(1)

    WRDS_USERNAME = sys.argv[1]

    print("Connecting to WRDS...")
    db = wrds.Connection(wrds_username=WRDS_USERNAME)
    print("Connected!\n")

    # Find what we can actually access
    accessible = find_accessible_tables(db)

    print(f"\n\nAccessible tables: {len(accessible)}")

    # Pull data from accessible tables
    results = {}
    if accessible:
        results = pull_accessible_transaction_data(db, accessible)

    # Always pull Compustat acquisition data as backup
    aqc_df = pull_compustat_acquisitions(db)
    if aqc_df is not None:
        results['comp.funda_acquisitions'] = aqc_df

    # Save data
    OUTPUT_DIR.mkdir(exist_ok=True)

    for name, df in results.items():
        safe_name = name.replace('.', '_')
        output_path = OUTPUT_DIR / f'{safe_name}.parquet'
        df.to_parquet(output_path, index=False)
        print(f"\nSaved {name} to {output_path}")
        print(f"  Rows: {len(df):,}")

        # CSV sample
        df.head(500).to_csv(OUTPUT_DIR / f'{safe_name}_sample.csv', index=False)

    db.close()

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"""
Data saved to {OUTPUT_DIR}

For V1 M&A analog retrieval, you have:

1. Compustat AQC data (comp_funda_acquisitions.parquet)
   - Shows which companies made acquisitions and amounts
   - Can link to fundamentals for "state at time of acquisition"
   - Limited: no target company info, no deal terms

2. Any accessible CIQ sample tables above

For richer M&A data, options:
   - LSEG Workspace (via JHU library website)
   - SDC Platinum (terminals at Carey Business School)
   - Manual curation from SEC 8-K filings
    """)


if __name__ == "__main__":
    main()
