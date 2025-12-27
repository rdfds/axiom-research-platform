"""
Pull Historical Stock Prices from CRSP
======================================
This script pulls daily stock prices for calculating TSR outcomes.

Usage:
  python 03_pull_prices.py USERNAME monthly   # Monthly prices (recommended for V1)
  python 03_pull_prices.py USERNAME daily     # Daily prices (large)

Output: data/prices_monthly.parquet or data/prices_daily.parquet
"""

import sys
import wrds
import pandas as pd
from pathlib import Path

# Configuration
START_DATE = '2010-01-01'
OUTPUT_DIR = Path(__file__).parent.parent / 'data'


def get_crsp_prices(db):
    """
    Pull daily stock prices from CRSP.

    Key fields for TSR calculation:
    - ret: Daily return (includes dividends)
    - prc: Price (negative means bid/ask average)
    - shrout: Shares outstanding
    - cfacpr: Cumulative factor to adjust price
    - cfacshr: Cumulative factor to adjust shares
    """

    query = """
    SELECT
        a.permno,
        a.permco,
        a.date,
        a.prc,
        a.ret,
        a.retx,
        a.shrout,
        a.vol,
        a.cfacpr,
        a.cfacshr,
        b.gvkey,
        b.linkprim

    FROM crsp.dsf a
    LEFT JOIN crsp.ccmxpf_lnkhist b
        ON a.permno = b.lpermno
        AND a.date >= b.linkdt
        AND (a.date <= b.linkenddt OR b.linkenddt IS NULL)
        AND b.linktype IN ('LU', 'LC')
        AND b.linkprim IN ('P', 'C')

    WHERE a.date >= %(start_date)s

    ORDER BY a.permno, a.date
    """

    print("Executing query (this may take a while)...")
    df = db.raw_sql(query, params={'start_date': START_DATE})
    print(f"Retrieved {len(df):,} rows")

    return df


