"""
Compute State Profiles for ALL Corporate Actions
=================================================
Expands from 742 M&A deals to all corporate actions with gvkeys.

This creates the full analog database for:
"Companies in this state → What did they do → How did it turn out?"

Usage:
  python scripts/09_compute_all_action_profiles.py

Expected time: ~25-30 minutes
Output: data/action_profiles_all.parquet
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from src.snapshot import AsOfSnapshotBuilder, DATA_DIR
from src.signals import SignalEngine
from src.corporate_actions import CorporateActionsDB


def main():
    print("=" * 70)
    print("COMPUTING STATE PROFILES FOR ALL CORPORATE ACTIONS")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load corporate actions
    print("\n[1/4] Loading corporate actions database...")
    db = CorporateActionsDB()

    # Filter to actions with gvkeys
    actions = db.actions[db.actions['gvkey'].notna()].copy()
    print(f"Actions with gvkeys: {len(actions):,}")

    # Show breakdown
    print("\nBy action type:")
    print(actions['action_type'].value_counts())

    # Initialize signal engine
    print("\n[2/4] Initializing signal engine...")
    engine = SignalEngine()

    # Process all actions
    print(f"\n[3/4] Computing state profiles for {len(actions):,} actions...")
    print("This will take ~25-30 minutes...\n")

    results = []
    success_count = 0
    fail_count = 0
    start_time = datetime.now()

    for i, (idx, action) in enumerate(actions.iterrows()):
        gvkey = str(action['gvkey'])
        action_date = action['date']
        action_type = action['action_type']

        try:
            profile = engine.compute_state_profile(gvkey, action_date)

            if profile:
                result = {
                    'action_id': idx,
                    'gvkey': gvkey,
                    'company_name': action.get('company_name'),
                    'ticker': action.get('ticker'),
                    'action_date': action_date,
                    'action_type': action_type,
                    'source': action.get('source'),
                    'deal_value': action.get('deal_value'),
                    'composite_score': profile['composite_score'],
                    'signal_vector': profile['vector'],
                }

                # Add individual signal scores
                for sig_name, sig_data in profile['signals'].items():
                    result[f'signal_{sig_name}'] = sig_data['score']

                results.append(result)
                success_count += 1
            else:
                fail_count += 1

        except Exception as e:
            fail_count += 1
            continue

        # Progress every 500
        if (i + 1) % 500 == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            rate = (i + 1) / elapsed
            remaining = (len(actions) - i - 1) / rate / 60
            pct = (i + 1) / len(actions) * 100
            print(f"  {i + 1:,}/{len(actions):,} ({pct:.0f}%) - "
                  f"{success_count:,} profiles - "
                  f"~{remaining:.0f} min remaining")

    print(f"\n[4/4] Saving results...")

    elapsed_total = (datetime.now() - start_time).total_seconds() / 60
    print(f"\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Total actions processed: {len(actions):,}")
    print(f"Successful profiles: {success_count:,} ({success_count/len(actions)*100:.1f}%)")
    print(f"Failed: {fail_count:,}")
    print(f"Time elapsed: {elapsed_total:.1f} minutes")

    if results:
        profiles_df = pd.DataFrame(results)

        # Stats
        print(f"\n📊 Profile Statistics:")
        print(f"   Composite Score - Mean: {profiles_df['composite_score'].mean():.1f}, "
              f"Std: {profiles_df['composite_score'].std():.1f}")

        # By action type
        print(f"\n📊 Profiles by Action Type:")
        type_stats = profiles_df.groupby('action_type').agg({
            'action_id': 'count',
            'composite_score': 'mean',
        }).rename(columns={'action_id': 'count', 'composite_score': 'avg_score'})
        type_stats = type_stats.sort_values('count', ascending=False)
        print(type_stats.to_string())

        # Save
        output_path = DATA_DIR / 'action_profiles_all.parquet'
        profiles_df.to_parquet(output_path, index=False)
        print(f"\n✅ Saved {len(profiles_df):,} action profiles to {output_path}")

        print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return profiles_df

    return None


if __name__ == "__main__":
    main()
