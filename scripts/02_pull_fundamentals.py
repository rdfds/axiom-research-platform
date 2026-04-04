"""
Pull Historical Fundamentals from Compustat
============================================
This script pulls quarterly fundamentals for US industrials companies
with filing dates (rdq) for proper as-of snapshot logic.

Output: data/fundamentals_quarterly.parquet
"""

import os
import sys
import wrds
import pandas as pd
from pathlib import Path

# Configuration
START_DATE = os.getenv('FUNDAMENTALS_START_DATE', '2000-01-01')
OUTPUT_DIR = Path(__file__).parent.parent / 'data'


def get_fundamentals(db):
    """
    Pull quarterly fundamentals with company info.

    Key fields:
    - gvkey: Company identifier
    - datadate: Fiscal period end date
    - rdq: Report Date of Quarterly earnings (FILING DATE - critical for as-of!)
    """

    # Pull all US companies first, filter to industrials after
    query = """
    SELECT
        -- Identifiers
        f.gvkey,
        f.datadate,
        f.rdq,
        f.fyearq,
        f.fqtr,
        f.tic,
        f.conm,

        -- Income Statement
        f.revtq,
        f.cogsq,
        f.xsgaq,
        f.oibdpq,
        f.oiadpq,
        f.niq,
        f.xintq,

        -- Balance Sheet - Assets
        f.atq,
        f.actq,
        f.cheq,
        f.rectq,
        f.invtq,
        f.ppentq,

        -- Balance Sheet - Liabilities & Equity
        f.ltq,
        f.lctq,
        f.dlcq,
        f.dlttq,
        f.ceqq,
        f.seqq,

        -- Cash Flow
        f.capxy,
        f.oancfy,

        -- Per Share
        f.epspxq,
        f.cshoq,

        -- Market data
        f.prccq,
        f.mkvaltq,

        -- SIC code from company table
        c.sic

    FROM comp.fundq f
    LEFT JOIN comp.company c ON f.gvkey = c.gvkey

    WHERE f.indfmt = 'INDL'
      AND f.datafmt = 'STD'
      AND f.popsrc = 'D'
      AND f.consol = 'C'
      AND f.datadate >= %(start_date)s
      AND f.curcdq = 'USD'

    ORDER BY f.gvkey, f.datadate
    """

    print("Executing query (this may take a few minutes)...")
    df = db.raw_sql(query, params={'start_date': START_DATE})
    print(f"Retrieved {len(df):,} rows")

    # Filter to industrials (SIC codes)
    if 'sic' in df.columns:
        # Convert SIC to string and pad
        df['sic'] = df['sic'].astype(str).str.strip()

        # Filter to manufacturing + wholesale/retail
        industrials_mask = (
            (df['sic'].str[:2].isin(['20','21','22','23','24','25','26','27','28','29',
                                      '30','31','32','33','34','35','36','37','38','39'])) |  # Manufacturing 2000-3999
            (df['sic'].str[:2].isin(['50','51','52','53','54','55','56','57','58','59']))     # Wholesale/Retail 5000-5999
        )
        df_industrials = df[industrials_mask].copy()
        print(f"Filtered to industrials: {len(df_industrials):,} rows")
        return df_industrials

    return df


