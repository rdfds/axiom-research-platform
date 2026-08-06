"""
Pull Recent CIQ Key Developments from WRDS
==========================================
The sample data is mostly pre-2010. This script pulls recent key developments
(2010-2024) to get dividends, equity offerings, divestitures, etc.

Usage:
  python scripts/12_pull_ciq_keydev_recent.py

Output: data/ciq_keydev_recent.parquet
"""

import sys
sys.path.insert(0, '.')

import os
import wrds
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / 'data'
CIQ_START_DATE = os.getenv('CIQ_START_DATE', '2010-01-01')
CIQ_END_DATE = os.getenv('CIQ_END_DATE', '2024-12-31')
CRSP_DIV_START = os.getenv('CRSP_DIV_START', CIQ_START_DATE)
CRSP_DIV_END = os.getenv('CRSP_DIV_END', CIQ_END_DATE)


def main():
    print("=" * 70)
    print("PULLING RECENT CIQ KEY DEVELOPMENTS FROM WRDS")
    print("=" * 70)

    # Connect to WRDS
    print("\nConnecting to WRDS...")
    db = wrds.Connection()
    print("Connected!")

    # First, explore what tables are available in CIQ
    print("\nExploring CIQ tables...")

    try:
        # List CIQ libraries
        libs = db.list_libraries()
        ciq_libs = [l for l in libs if 'ciq' in l.lower()]
        print(f"CIQ-related libraries: {ciq_libs}")
    except Exception as e:
        print(f"Error listing libraries: {e}")

    # Try to query the full CIQ keydev table (not sample)
    print("\nAttempting to query ciq.ciqkeydev...")

    try:
        # First check if we have access
        query = """
        SELECT COUNT(*) as cnt
        FROM ciq.ciqkeydev
        WHERE announceddate >= %(start_date)s
        """
        result = db.raw_sql(query, params={'start_date': CIQ_START_DATE})
        print(f"Records available (2010+): {result['cnt'].iloc[0]:,}")

    except Exception as e:
        print(f"No access to ciq.ciqkeydev: {e}")
        print("\nTrying ciq.wrds_keydev...")

        try:
            query = """
            SELECT COUNT(*) as cnt
            FROM ciq.wrds_keydev
            WHERE announceddate >= %(start_date)s
            """
            result = db.raw_sql(query, params={'start_date': CIQ_START_DATE})
            print(f"Records in ciq.wrds_keydev (2010+): {result['cnt'].iloc[0]:,}")

        except Exception as e2:
            print(f"No access to ciq.wrds_keydev: {e2}")

    # Try different CIQ tables
    tables_to_try = [
        ('ciq', 'ciqkeydev'),
        ('ciq', 'wrds_keydev'),
        ('ciq_keydev', 'keydev'),
        ('ciq_common', 'ciqkeydev'),
    ]

    keydev_table = None
    for lib, table in tables_to_try:
        try:
            query = f"SELECT * FROM {lib}.{table} LIMIT 1"
            test = db.raw_sql(query)
            print(f"\n✓ Found accessible table: {lib}.{table}")
            print(f"  Columns: {test.columns.tolist()}")
            keydev_table = f"{lib}.{table}"
            break
        except Exception as e:
            continue

    if keydev_table is None:
        print("\nNo direct CIQ keydev access. Trying alternative approach...")

        # Try to get key developments from CIQ company events
        try:
            query = """
            SELECT *
            FROM ciq.ciqevent
            WHERE eventdate >= %(start_date)s
            LIMIT 100
            """
            events = db.raw_sql(query, params={'start_date': CIQ_START_DATE})
            print(f"Found ciq.ciqevent: {len(events)} sample rows")
            print(events.head())
        except Exception as e:
            print(f"No ciq.ciqevent: {e}")

        # Try CRSP for dividend data instead
        print("\n" + "=" * 70)
        print("PULLING DIVIDEND DATA FROM CRSP")
        print("=" * 70)

        try:
            # CRSP has dividend distribution data
            query = """
            SELECT
                a.permno,
                a.permco,
                b.gvkey,
                a.exdt as ex_date,
                a.paydt as pay_date,
                a.divamt as div_amount,
                a.distcd as dist_code
            FROM crsp.dsedist a
            LEFT JOIN crsp.ccmxpf_linktable b
                ON a.permno = b.lpermno
                AND a.exdt >= b.linkdt
                AND (a.exdt <= b.linkenddt OR b.linkenddt IS NULL)
            WHERE a.exdt >= %(start_date)s
              AND a.exdt <= %(end_date)s
              AND a.distcd BETWEEN 1000 AND 1999  -- Cash dividends
              AND b.gvkey IS NOT NULL
            ORDER BY a.exdt DESC
            """

            print("Pulling CRSP dividend data...")
            dividends = db.raw_sql(query, params={'start_date': CRSP_DIV_START, 'end_date': CRSP_DIV_END})
            print(f"Retrieved {len(dividends):,} dividend records")

            if len(dividends) > 0:
                print(f"\nDate range: {dividends['ex_date'].min()} to {dividends['ex_date'].max()}")
                print(f"Unique companies: {dividends['gvkey'].nunique():,}")

                # Save
                output_path = OUTPUT_DIR / 'crsp_dividends.parquet'
                dividends.to_parquet(output_path, index=False)
                print(f"\n✅ Saved {len(dividends):,} dividend records to {output_path}")

                # Show sample
                print("\nSample data:")
                print(dividends.head(20))

        except Exception as e:
            print(f"Error pulling CRSP dividends: {e}")

        # Also try to get equity issuances from SDC or other sources
        print("\n" + "=" * 70)
        print("CHECKING FOR EQUITY ISSUANCE DATA")
        print("=" * 70)

        try:
            # Check SDC for equity offerings
            tables = db.list_tables(library='sdc')
            print(f"SDC tables: {tables[:20]}...")

            # Try global new issues
            query = """
            SELECT *
            FROM sdc.globalnewissues
            WHERE issuedate >= %(start_date)s
            LIMIT 100
            """
            sdc_sample = db.raw_sql(query, params={'start_date': CIQ_START_DATE})
            print(f"\nSDC global new issues sample: {len(sdc_sample)} rows")
            print(sdc_sample.columns.tolist())

        except Exception as e:
            print(f"SDC not accessible: {e}")

    else:
        # We have access to keydev - pull recent data
        print(f"\nPulling recent key developments from {keydev_table}...")

        query = f"""
        SELECT *
        FROM {keydev_table}
        WHERE announceddate >= %(start_date)s
          AND announceddate <= %(end_date)s
        ORDER BY announceddate DESC
        """

        try:
            keydev = db.raw_sql(query, params={'start_date': CIQ_START_DATE, 'end_date': CIQ_END_DATE})
            print(f"Retrieved {len(keydev):,} key developments")

            if len(keydev) > 0:
                output_path = OUTPUT_DIR / 'ciq_keydev_recent.parquet'
                keydev.to_parquet(output_path, index=False)
                print(f"\n✅ Saved to {output_path}")

        except Exception as e:
            print(f"Error: {e}")

    db.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
