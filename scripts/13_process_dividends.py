"""
Process CRSP Dividend Data into Corporate Actions
==================================================
We have 336K dividend payment records. This script:
1. Identifies dividend CHANGES (initiate, increase, cut, suspend)
2. Links to Compustat gvkeys
3. Computes state profiles at time of dividend action

Usage:
  python scripts/13_process_dividends.py

Output: data/dividend_actions.parquet, data/dividend_profiles.parquet
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from src.snapshot import AsOfSnapshotBuilder, DATA_DIR
from src.signals import SignalEngine


def identify_dividend_changes(dividends: pd.DataFrame) -> pd.DataFrame:
    """
    Identify dividend changes (not just payments).

    A dividend "action" is when the dividend amount changes:
    - Initiate: First dividend payment
    - Increase: Higher than previous
    - Cut: Lower than previous
    - Suspend: No payment after regular payments
    """
    print("Identifying dividend changes...")

    # Sort by company and date
    df = dividends.copy()
    df['ex_date'] = pd.to_datetime(df['ex_date'])
    df = df.sort_values(['gvkey', 'ex_date'])

    # Filter to regular cash dividends (distcd 1232 is most common)
    # 1xxx = cash dividends
    regular_divs = df[df['dist_code'].between(1200, 1299)].copy()
    print(f"  Regular cash dividends: {len(regular_divs):,}")

    # Group by company and find changes
    actions = []

    for gvkey, group in regular_divs.groupby('gvkey'):
        group = group.sort_values('ex_date')

        if len(group) < 2:
            continue

        prev_amount = None
        prev_date = None

        for idx, row in group.iterrows():
            amount = row['div_amount']
            date = row['ex_date']

            if prev_amount is None:
                # First dividend - this is an initiation
                actions.append({
                    'gvkey': gvkey,
                    'action_date': date,
                    'action_type': 'dividend_initiate',
                    'div_amount': amount,
                    'prev_amount': None,
                    'change_pct': None,
                })
            else:
                # Check for change (avoid division by zero)
                if prev_amount > 0:
                    if amount > prev_amount * 1.02:  # >2% increase
                        change_pct = (amount - prev_amount) / prev_amount * 100
                        actions.append({
                            'gvkey': gvkey,
                            'action_date': date,
                            'action_type': 'dividend_increase',
                            'div_amount': amount,
                            'prev_amount': prev_amount,
                            'change_pct': change_pct,
                        })
                    elif amount < prev_amount * 0.98:  # >2% decrease
                        change_pct = (amount - prev_amount) / prev_amount * 100
                        actions.append({
                            'gvkey': gvkey,
                            'action_date': date,
                            'action_type': 'dividend_cut',
                            'div_amount': amount,
                            'prev_amount': prev_amount,
                            'change_pct': change_pct,
                        })
                # Note: No change = regular payment, we skip

            prev_amount = amount
            prev_date = date

    actions_df = pd.DataFrame(actions)
    print(f"  Total dividend actions: {len(actions_df):,}")

    if len(actions_df) > 0:
        print("\n  By type:")
        print(actions_df['action_type'].value_counts())

    return actions_df


