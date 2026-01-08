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


