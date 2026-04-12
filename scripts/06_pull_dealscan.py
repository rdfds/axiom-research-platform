"""
Pull DealScan Loan Data for M&A and Refinancing Analysis
=========================================================
DealScan contains leveraged loan facilities - useful for:
- Acquisition financing (identifies M&A activity)
- LBO financing
- Refinancing events
- Debt restructuring

Usage:
  python 06_pull_dealscan.py USERNAME

Output: data/dealscan_*.parquet
"""

import sys
import wrds
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / 'data'


def pull_dealscan_facilities(db):
    """
    Pull loan facilities with deal purposes related to M&A/LBO.
    """

    print("="*70)
    print("PULLING DEALSCAN FACILITY DATA")
    print("="*70)

    # Main facility query with package info (using correct column names)
    query = """
    SELECT
        f.facilityid,
        f.packageid,
        f.facilitystartdate,
        f.facilityenddate,
        f.facilityamt,
        f.currency,
        f.primarypurpose,
        f.secondarypurpose,
        f.loantype,
        f.maturity,
        f.secured,
        f.seniority,
        f.company as facility_company,
        f.targetcompany,

        -- Package (deal) info
        p.dealactivedate,
        p.company as borrower_name,
        p.dealamount,
        p.dealpurpose,
        p.dealstatus,
        p.salesatclose,

        -- Company info
        c.ticker,
        c.primarysiccode,
        c.country,
        c.sales,
        c.publicprivate

    FROM dealscan.facility f
    LEFT JOIN dealscan.package p ON f.packageid = p.packageid
    LEFT JOIN dealscan.company c ON f.borrowercompanyid = c.companyid

    WHERE f.facilitystartdate >= '2005-01-01'
      AND f.currency = 'United States Dollars'

    ORDER BY f.facilitystartdate DESC
    """

    print("Executing query...")
    df = db.raw_sql(query)
    print(f"Retrieved {len(df):,} facilities")

    return df


def pull_ma_related_facilities(db):
    """
    Pull facilities specifically related to M&A activity.
    """

    print("\n" + "="*70)
    print("FILTERING TO M&A/LBO RELATED FACILITIES")
    print("="*70)

    # Use exact purpose values from the database
    query = """
    SELECT
        f.facilityid,
        f.packageid,
        f.facilitystartdate,
        f.facilityamt,
        f.primarypurpose,
        f.secondarypurpose,
        f.loantype,
        f.maturity,
        f.targetcompany,

        p.dealactivedate,
        p.company as borrower_name,
        p.dealamount,
        p.dealpurpose,

        c.ticker,
        c.primarysiccode,
        c.sales

    FROM dealscan.facility f
    LEFT JOIN dealscan.package p ON f.packageid = p.packageid
    LEFT JOIN dealscan.company c ON f.borrowercompanyid = c.companyid

    WHERE f.facilitystartdate >= '2005-01-01'
      AND f.currency = 'United States Dollars'
      AND (
          -- M&A related primary purposes
          f.primarypurpose IN ('LBO', 'Takeover', 'Acquis. line', 'SBO', 'Recap.', 'Dividend Recap')
          -- Or M&A related deal purposes
          OR p.dealpurpose IN ('LBO', 'Takeover', 'Acquis. line', 'SBO', 'Recap.', 'Dividend Recap')
      )

    ORDER BY f.facilitystartdate DESC
    """

    print("Executing query...")
    df = db.raw_sql(query)
    print(f"Retrieved {len(df):,} M&A/LBO related facilities")

    # Show purpose breakdown
    if len(df) > 0:
        print("\nFacility primary purposes:")
        print(df['primarypurpose'].value_counts().head(15))

        print("\nDeal purposes:")
        print(df['dealpurpose'].value_counts().head(15))

    return df


def pull_borrower_link(db):
    """
    Pull the linking table between DealScan and Compustat.
    This allows us to connect loans to our fundamentals data.
    """

    print("\n" + "="*70)
    print("PULLING DEALSCAN-COMPUSTAT LINK TABLE")
    print("="*70)

    # Check if link table exists
    try:
        query = """
        SELECT *
        FROM dealscan.lpc_loanconnector_company_id_map
        """
        df = db.raw_sql(query)
        print(f"Retrieved {len(df):,} company links")
        return df
    except Exception as e:
        print(f"Link table not available: {e}")

        # Try alternative linking via ticker
        print("\nWill use ticker matching as fallback...")
        return None


def analyze_data(facilities_df, ma_df):
    """Basic analysis of the data."""

    print("\n" + "="*70)
    print("DATA SUMMARY")
    print("="*70)

    print(f"\nAll facilities: {len(facilities_df):,}")
    print(f"M&A/LBO facilities: {len(ma_df):,}")

    if len(ma_df) > 0:
        print(f"\nDate range: {ma_df['facilitystartdate'].min()} to {ma_df['facilitystartdate'].max()}")
        print(f"Unique borrowers: {ma_df['borrower_name'].nunique():,}")

        # Deal sizes
        ma_df['facilityamt'] = pd.to_numeric(ma_df['facilityamt'], errors='coerce')
        print(f"\nFacility amounts ($ millions):")
        print(f"  Mean: ${ma_df['facilityamt'].mean():,.0f}M")
        print(f"  Median: ${ma_df['facilityamt'].median():,.0f}M")
        print(f"  Max: ${ma_df['facilityamt'].max():,.0f}M")

        # By year
        ma_df['year'] = pd.to_datetime(ma_df['facilitystartdate']).dt.year
        yearly = ma_df.groupby('year').agg({
            'facilityid': 'count',
            'facilityamt': 'sum'
        }).rename(columns={'facilityid': 'count', 'facilityamt': 'total_amt'})
        print(f"\nM&A/LBO facilities by year:")
        print(yearly.tail(10))


