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


def main():
    print("=" * 70)
    print("PROCESSING CRSP DIVIDEND DATA")
    print("=" * 70)

    # Load dividend data
    div_path = DATA_DIR / 'crsp_dividends.parquet'
    print(f"\nLoading dividends from {div_path}...")
    dividends = pd.read_parquet(div_path)
    print(f"Loaded {len(dividends):,} dividend records")
    print(f"Unique companies (gvkey): {dividends['gvkey'].nunique():,}")

    # Identify dividend changes
    actions = identify_dividend_changes(dividends)

    if len(actions) == 0:
        print("No dividend actions found!")
        return

    # Save dividend actions
    actions_path = DATA_DIR / 'dividend_actions.parquet'
    actions.to_parquet(actions_path, index=False)
    print(f"\n✅ Saved {len(actions):,} dividend actions to {actions_path}")

    # Now compute state profiles for these actions
    print("\n" + "=" * 70)
    print("COMPUTING STATE PROFILES FOR DIVIDEND ACTIONS")
    print("=" * 70)

    # Initialize signal engine
    print("\nInitializing signal engine...")
    engine = SignalEngine()

    # Get company names from fundamentals
    fund = pd.read_parquet(DATA_DIR / 'fundamentals_quarterly.parquet')
    gvkey_to_name = fund.drop_duplicates('gvkey').set_index('gvkey')['conm'].to_dict()

    # Compute profiles
    print(f"\nComputing profiles for {len(actions):,} dividend actions...")

    results = []
    success = 0
    fail = 0

    for i, (idx, row) in enumerate(actions.iterrows()):
        gvkey = str(row['gvkey'])
        action_date = row['action_date']
        action_type = row['action_type']

        try:
            profile = engine.compute_state_profile(gvkey, action_date)

            if profile:
                result = {
                    'gvkey': gvkey,
                    'company_name': gvkey_to_name.get(gvkey, 'Unknown'),
                    'action_date': action_date,
                    'action_type': action_type,
                    'div_amount': row['div_amount'],
                    'change_pct': row.get('change_pct'),
                    'source': 'crsp_dividends',
                    'composite_score': profile['composite_score'],
                    'signal_vector': profile['vector'],
                }

                # Add individual signals
                for sig_name, sig_data in profile['signals'].items():
                    result[f'signal_{sig_name}'] = sig_data['score']

                results.append(result)
                success += 1
            else:
                fail += 1

        except Exception as e:
            fail += 1
            continue

        if (i + 1) % 1000 == 0:
            pct = (i + 1) / len(actions) * 100
            print(f"  {i + 1:,}/{len(actions):,} ({pct:.0f}%) - {success:,} profiles")

    print(f"\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Actions processed: {len(actions):,}")
    print(f"Successful profiles: {success:,} ({success/len(actions)*100:.1f}%)")
    print(f"Failed: {fail:,}")

    if results:
        profiles_df = pd.DataFrame(results)

        print(f"\n📊 Profiles by Action Type:")
        by_type = profiles_df.groupby('action_type').agg({
            'gvkey': 'count',
            'composite_score': 'mean'
        }).rename(columns={'gvkey': 'count', 'composite_score': 'avg_score'})
        print(by_type.to_string())

        # Save
        output_path = DATA_DIR / 'dividend_profiles.parquet'
        profiles_df.to_parquet(output_path, index=False)
        print(f"\n✅ Saved {len(profiles_df):,} dividend profiles to {output_path}")

        return profiles_df

    return None


if __name__ == "__main__":
    main()
