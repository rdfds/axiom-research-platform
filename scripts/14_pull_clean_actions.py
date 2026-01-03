"""
Pull Clean Corporate Actions Data
=================================
Uses CERTAIN data sources (not inferred):
1. Compustat prstkcy - actual buyback amounts
2. CRSP delistings - acquisitions, bankruptcies
3. CRSP dividends - already processed

This gives us reliable corporate actions data.
"""

import os
import psycopg2
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / 'data'
START_DATE = os.getenv('ACTIONS_START_DATE', '2000-01-01')


def pull_buybacks():
    """
    Pull actual buyback data from Compustat.

    Uses prstkcy field = Purchase of Common/Preferred Stock (YTD, $ millions)
    This is actual cash spent on buybacks, not inferred from share count.
    """
    print("=" * 60)
    print("PULLING COMPUSTAT BUYBACK DATA")
    print("=" * 60)

    conn = psycopg2.connect(
        host='wrds-pgdata.wharton.upenn.edu',
        port=9737,
        database='wrds',
        user='rvarian1'
    )

    # Pull quarterly buyback data
    # prstkcy = Purchase of Common and Preferred Stock (YTD)
    # We want quarters where there was meaningful buyback activity
    # SIC is in comp.company, not fundq - join to get it
    query = """
    SELECT
        f.gvkey,
        f.datadate,
        f.conm as company_name,
        f.tic as ticker,
        f.prstkcy as buyback_amount_ytd,
        f.cshoq as shares_outstanding,
        f.atq as total_assets,
        c.sic
    FROM comp.fundq f
    LEFT JOIN comp.company c ON f.gvkey = c.gvkey
    WHERE f.datadate >= %(start_date)s
      AND f.prstkcy > 0
      AND f.datafmt = 'STD'
      AND f.indfmt = 'INDL'
      AND f.consol = 'C'
      AND f.popsrc = 'D'
    ORDER BY f.gvkey, f.datadate
    """

    print("Pulling buyback data from Compustat...")
    df = pd.read_sql(query, conn, params={'start_date': START_DATE})
    print(f"  Retrieved {len(df):,} quarters with buyback activity")

    # Convert YTD to quarterly amounts
    df = df.sort_values(['gvkey', 'datadate'])
    df['datadate'] = pd.to_datetime(df['datadate'])
    df['quarter'] = df['datadate'].dt.quarter

    # For Q1, quarterly = YTD. For Q2-Q4, quarterly = YTD - prev YTD
    df['prev_ytd'] = df.groupby(['gvkey', df['datadate'].dt.year])['buyback_amount_ytd'].shift(1)
    df['buyback_amount_qtr'] = df.apply(
        lambda r: r['buyback_amount_ytd'] if r['quarter'] == 1
                  else (r['buyback_amount_ytd'] - r['prev_ytd'] if pd.notna(r['prev_ytd']) else r['buyback_amount_ytd']),
        axis=1
    )

    # Filter to meaningful buybacks (>$10M quarterly)
    significant = df[df['buyback_amount_qtr'] > 10].copy()
    print(f"  Significant buybacks (>$10M/qtr): {len(significant):,}")

    # Create action records
    buybacks = significant[['gvkey', 'datadate', 'company_name', 'ticker',
                            'buyback_amount_qtr', 'shares_outstanding', 'sic']].copy()
    buybacks['action_type'] = 'buyback'
    buybacks['action_date'] = buybacks['datadate']
    buybacks['deal_value'] = buybacks['buyback_amount_qtr']
    buybacks['source'] = 'compustat_prstkcy'

    # Save
    output_path = DATA_DIR / 'buybacks_clean.parquet'
    buybacks.to_parquet(output_path)
    print(f"  Saved to {output_path}")

    conn.close()
    return buybacks


def pull_acquisitions():
    """
    Pull acquisition data from CRSP delistings.

    Delisting codes 200-299 indicate company was acquired/merged.
    This is definitive - the company ceased trading due to acquisition.
    """
    print("\n" + "=" * 60)
    print("PULLING CRSP ACQUISITION DATA (DELISTINGS)")
    print("=" * 60)

    conn = psycopg2.connect(
        host='wrds-pgdata.wharton.upenn.edu',
        port=9737,
        database='wrds',
        user='rvarian1'
    )

    # Pull delistings that are acquisitions
    query = """
    SELECT
        d.permno,
        d.permco,
        d.dlstdt as action_date,
        d.dlstcd as delist_code,
        d.dlamt as deal_amount,
        d.dlret as delist_return,
        n.comnam as company_name,
        n.ticker,
        n.siccd as sic
    FROM crsp.msedelist d
    LEFT JOIN crsp.msenames n
        ON d.permno = n.permno
        AND d.dlstdt BETWEEN n.namedt AND n.nameendt
    WHERE d.dlstdt >= %(start_date)s
      AND d.dlstcd BETWEEN 200 AND 299  -- Acquisition codes
    ORDER BY d.dlstdt DESC
    """

    print("Pulling acquisition delistings from CRSP...")
    df = pd.read_sql(query, conn, params={'start_date': START_DATE})
    print(f"  Retrieved {len(df):,} acquisitions")

    # Classify acquisition type
    def classify_acquisition(code):
        if code in [231, 241]:
            return 'acquisition_tender'
        elif code in [232, 242]:
            return 'acquisition_exchange'
        elif code == 233:
            return 'acquisition_lbo'
        elif code == 244:
            return 'acquisition_reverse'
        else:
            return 'acquisition_merger'

    df['action_type'] = df['delist_code'].apply(classify_acquisition)
    df['source'] = 'crsp_delist'
    df['action_date'] = pd.to_datetime(df['action_date'])

    # Save
    output_path = DATA_DIR / 'acquisitions_clean.parquet'
    df.to_parquet(output_path)
    print(f"  Saved to {output_path}")

    # Summary by type
    print("\n  Acquisitions by type:")
    print(df['action_type'].value_counts().to_string())

    conn.close()
    return df


