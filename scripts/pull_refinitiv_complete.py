"""
Pull COMPLETE Corporate Actions & Data from Refinitiv
=====================================================
All companies, all action types, 5 years of history.

Run with: nohup python -u scripts/pull_refinitiv_complete.py > /tmp/refinitiv_complete.log 2>&1 &
"""

import argparse
import refinitiv.data as rd
import pandas as pd
from pathlib import Path
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).parent.parent / 'data' / 'refinitiv'
DATA_DIR.mkdir(exist_ok=True)

START_DATE = '2020-01-01'
END_DATE = '2025-12-31'

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def save_parquet(df, name):
    path = DATA_DIR / f'{name}.parquet'
    df.to_parquet(path, index=False)
    log(f"  ✓ Saved {len(df):,} rows to {name}.parquet")
    return path


