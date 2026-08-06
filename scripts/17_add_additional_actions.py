"""
Add Additional Corporate Actions to Profile Database
=====================================================
Adds stock splits, special dividends, and other actions from script 16
to the clean action profiles database.

Run AFTER 15_compute_clean_action_profiles.py completes.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.signals import SignalEngine
from src.outcomes import OutcomeCalculator

DATA_DIR = Path(__file__).parent.parent / 'data'


def compute_profiles_for_actions(actions_df, action_type_name, engine, outcomes):
    """
    Compute state profiles and TSR outcomes for a set of actions.
    """
    print(f"\nComputing profiles for {len(actions_df):,} {action_type_name}...")

    results = []
    errors = 0

    for i, (idx, row) in enumerate(actions_df.iterrows()):
        if i % 500 == 0 and i > 0:
            print(f"  Processed {i:,} / {len(actions_df):,} ({i/len(actions_df)*100:.1f}%)")

        try:
            gvkey = str(row['gvkey'])
            action_date = pd.to_datetime(row['action_date'])

            # Skip if no gvkey
            if pd.isna(gvkey) or gvkey == 'nan':
                errors += 1
                continue

            # Compute state profile at time of action
            profile = engine.compute_state_profile(gvkey, action_date)

            if profile is None:
                errors += 1
                continue

            # Compute TSR outcomes
            tsr = outcomes.compute_tsr(gvkey, action_date)

            # Build result record
            result = {
                'gvkey': gvkey,
                'action_date': action_date,
                'action_type': row.get('action_type', action_type_name),
                'company_name': row.get('company_name', ''),
                'ticker': row.get('ticker', ''),
                'deal_value': row.get('deal_value') or row.get('amount'),
                'sic': row.get('sic'),
                'signal_vector': profile['vector'],
                'composite_score': profile['composite_score'],
            }

            # Add individual signals
            for sig_name, sig_data in profile['signals'].items():
                result[f'signal_{sig_name}'] = sig_data['score']

            # Add TSR outcomes
            if tsr:
                result['tsr_1m'] = tsr.get('tsr_1m')
                result['tsr_3m'] = tsr.get('tsr_3m')
                result['tsr_6m'] = tsr.get('tsr_6m')
                result['tsr_12m'] = tsr.get('tsr_12m')

            results.append(result)

        except Exception:
            errors += 1
            continue

    print(f"  Completed: {len(results):,} profiles, {errors:,} errors")

    if results:
        return pd.DataFrame(results)
    return pd.DataFrame()


def main():
    print("=" * 70)
    print("ADDING ADDITIONAL CORPORATE ACTIONS TO PROFILE DATABASE")
    print("=" * 70)
    print(f"Started at: {datetime.now()}")

    # Check if clean profiles exist
    clean_path = DATA_DIR / 'clean_action_profiles.parquet'
    if not clean_path.exists():
        print("\n⚠️ clean_action_profiles.parquet not found!")
        print("   Run 15_compute_clean_action_profiles.py first.")
        return

    # Load existing clean profiles
    existing = pd.read_parquet(clean_path)
    print(f"\nExisting clean profiles: {len(existing):,}")

    # Initialize engines
    print("\nInitializing engines...")
    engine = SignalEngine()
    outcomes = OutcomeCalculator()

    new_profiles = []

    # 1. Stock Splits
    splits_path = DATA_DIR / 'stock_splits_linked.parquet'
    if splits_path.exists():
        splits = pd.read_parquet(splits_path)
        splits = splits.dropna(subset=['gvkey'])
        splits = splits.head(3000)  # Sample for speed

        profiles = compute_profiles_for_actions(splits, 'stock_split', engine, outcomes)
        if len(profiles) > 0:
            new_profiles.append(profiles)
            print(f"  Added {len(profiles):,} stock split profiles")

    # 2. Special Dividends
    special_div_path = DATA_DIR / 'special_dividends_linked.parquet'
    if special_div_path.exists():
        special = pd.read_parquet(special_div_path)
        special = special.dropna(subset=['gvkey'])
        special = special.head(3000)  # Sample

        profiles = compute_profiles_for_actions(special, 'dividend_special', engine, outcomes)
        if len(profiles) > 0:
            new_profiles.append(profiles)
            print(f"  Added {len(profiles):,} special dividend profiles")

    # 3. Return of Capital
    roc_path = DATA_DIR / 'return_of_capital_linked.parquet'
    if roc_path.exists():
        roc = pd.read_parquet(roc_path)
        roc = roc.dropna(subset=['gvkey'])

        profiles = compute_profiles_for_actions(roc, 'return_of_capital', engine, outcomes)
        if len(profiles) > 0:
            new_profiles.append(profiles)
            print(f"  Added {len(profiles):,} return of capital profiles")

    # 4. Going Private
    gp_path = DATA_DIR / 'going_private_linked.parquet'
    if gp_path.exists():
        gp = pd.read_parquet(gp_path)
        gp = gp.dropna(subset=['gvkey'])

        profiles = compute_profiles_for_actions(gp, 'going_private', engine, outcomes)
        if len(profiles) > 0:
            new_profiles.append(profiles)
            print(f"  Added {len(profiles):,} going private profiles")

    # 5. Ticker Changes (name changes often accompany strategic actions)
    ticker_path = DATA_DIR / 'ticker_changes_linked.parquet'
    if ticker_path.exists():
        tickers = pd.read_parquet(ticker_path)
        tickers = tickers.dropna(subset=['gvkey'])
        tickers = tickers.head(1500)  # Sample

        profiles = compute_profiles_for_actions(tickers, 'ticker_change', engine, outcomes)
        if len(profiles) > 0:
            new_profiles.append(profiles)
            print(f"  Added {len(profiles):,} ticker change profiles")

    # Combine new profiles with existing
    if new_profiles:
        all_new = pd.concat(new_profiles, ignore_index=True)
        combined = pd.concat([existing, all_new], ignore_index=True)

        # Remove duplicates (same gvkey + date)
        combined = combined.drop_duplicates(
            subset=['gvkey', 'action_date'], keep='first'
        )

        # Save
        combined.to_parquet(clean_path)

        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Previous profiles: {len(existing):,}")
        print(f"New profiles added: {len(all_new):,}")
        print(f"Total after merge: {len(combined):,}")
        print(f"\nBy action type:")
        print(combined['action_type'].value_counts().to_string())
        print(f"\nSaved to: {clean_path}")
        print(f"Completed at: {datetime.now()}")
    else:
        print("\nNo new profiles to add!")


if __name__ == "__main__":
    main()
