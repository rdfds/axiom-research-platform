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


