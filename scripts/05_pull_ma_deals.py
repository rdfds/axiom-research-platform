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


