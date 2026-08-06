"""
Compute State Profiles for Linked CIQ Events
=============================================
Now that we've linked 5,550 CIQ events to Compustat, compute their
state profiles. This adds dividends, divestitures, etc. to our database.

Usage:
  python scripts/11_compute_ciq_profiles.py

Output: data/ciq_profiles_all.parquet
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import re
from pathlib import Path
from datetime import datetime

from src.snapshot import AsOfSnapshotBuilder, DATA_DIR
from src.signals import SignalEngine


def classify_action(headline):
    """Classify CIQ headline into action type."""
    if pd.isna(headline):
        return 'other'

    h = headline.lower()

    if 'dividend' in h:
        if any(w in h for w in ['increase', 'raise', 'hike']):
            return 'dividend_increase'
        elif any(w in h for w in ['cut', 'reduce', 'lower']):
            return 'dividend_cut'
        elif any(w in h for w in ['suspend', 'eliminate', 'omit']):
            return 'dividend_suspend'
        else:
            return 'dividend'
    elif any(w in h for w in ['buyback', 'repurchase']):
        return 'buyback'
    elif any(w in h for w in ['acqui', 'merger', 'takeover', 'purchase of']):
        return 'acquisition'
    elif any(w in h for w in ['divest', 'sell', 'dispose', 'asset sale']):
        return 'divestiture'
    elif any(w in h for w in ['offering', 'ipo', 'secondary']):
        return 'equity_offering'
    elif 'spin' in h and any(w in h for w in ['off', 'out']):
        return 'spinoff'

    return 'other'


def main():
    print("=" * 70)
    print("COMPUTING STATE PROFILES FOR LINKED CIQ EVENTS")
    print("=" * 70)

    # Load linked CIQ events
    ciq_path = DATA_DIR / 'ciq_linked.parquet'
    print(f"\nLoading linked CIQ events from {ciq_path}...")
    ciq = pd.read_parquet(ciq_path)
    ciq['announceddate'] = pd.to_datetime(ciq['announceddate'])
    print(f"Loaded {len(ciq):,} linked events")

    # Classify action types
    ciq['action_type'] = ciq['headline'].apply(classify_action)

    print("\nAction types:")
    print(ciq['action_type'].value_counts())

    # Filter out 'other' - we want specific corporate actions
    ciq_filtered = ciq[ciq['action_type'] != 'other'].copy()
    print(f"\nFiltered to {len(ciq_filtered):,} classifiable actions")

    # Initialize signal engine
    print("\nInitializing signal engine...")
    engine = SignalEngine()

    # Compute profiles
    print(f"\nComputing state profiles for {len(ciq_filtered):,} events...")

    results = []
    success = 0
    fail = 0

    for i, (idx, row) in enumerate(ciq_filtered.iterrows()):
        gvkey = str(row['gvkey'])
        action_date = row['announceddate']
        action_type = row['action_type']

        try:
            profile = engine.compute_state_profile(gvkey, action_date)

            if profile:
                result = {
                    'ciq_idx': row['ciq_idx'],
                    'gvkey': gvkey,
                    'company_name': row['extracted_company'],
                    'action_date': action_date,
                    'action_type': action_type,
                    'source': 'ciq_keydev',
                    'headline': row['headline'],
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

        if (i + 1) % 500 == 0:
            pct = (i + 1) / len(ciq_filtered) * 100
            print(f"  {i + 1:,}/{len(ciq_filtered):,} ({pct:.0f}%) - {success:,} profiles")

    print(f"\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Events processed: {len(ciq_filtered):,}")
    print(f"Successful profiles: {success:,} ({success/len(ciq_filtered)*100:.1f}%)")
    print(f"Failed: {fail:,}")

    if results:
        profiles_df = pd.DataFrame(results)

        print(f"\n📊 Profiles by Action Type:")
        by_type = profiles_df.groupby('action_type').agg({
            'ciq_idx': 'count',
            'composite_score': 'mean'
        }).rename(columns={'ciq_idx': 'count', 'composite_score': 'avg_score'})
        by_type = by_type.sort_values('count', ascending=False)
        print(by_type.to_string())

        # Save
        output_path = DATA_DIR / 'ciq_profiles_all.parquet'
        profiles_df.to_parquet(output_path, index=False)
        print(f"\n✅ Saved {len(profiles_df):,} CIQ profiles to {output_path}")

        return profiles_df

    return None


if __name__ == "__main__":
    main()
