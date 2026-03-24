"""
Compute State Profiles for Clean Corporate Actions
===================================================
Computes state profiles at the time of each corporate action
using the clean data sources (not inferred).

This is compute-intensive - will process in batches.
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

    Parameters
    ----------
    actions_df : DataFrame
        Must have 'gvkey' and 'action_date' columns
    action_type_name : str
        Name for logging
    engine : SignalEngine
        For computing state profiles
    outcomes : OutcomeCalculator
        For computing TSR

    Returns
    -------
    DataFrame with profiles and outcomes
    """
    print(f"\nComputing profiles for {len(actions_df):,} {action_type_name}...")

    results = []
    errors = 0

    for i, (idx, row) in enumerate(actions_df.iterrows()):
        if i % 1000 == 0 and i > 0:
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
                'deal_value': row.get('deal_value') or row.get('buyback_amount_qtr'),
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

        except Exception as e:
            errors += 1
            continue

    print(f"  Completed: {len(results):,} profiles, {errors:,} errors")

    if results:
        return pd.DataFrame(results)
    return pd.DataFrame()


def main():
    print("=" * 70)
    print("COMPUTING STATE PROFILES FOR CLEAN CORPORATE ACTIONS")
    print("=" * 70)
    print(f"Started at: {datetime.now()}")

    # Initialize engines
    print("\nInitializing engines...")
    engine = SignalEngine()
    outcomes = OutcomeCalculator()

    all_profiles = []

    # 1. Process Buybacks (largest dataset - sample first 20K for speed)
    buybacks_path = DATA_DIR / 'buybacks_clean.parquet'
    if buybacks_path.exists():
        buybacks = pd.read_parquet(buybacks_path)
        buybacks['action_date'] = buybacks['datadate']

        # Sample to keep compute manageable - take most recent and largest
        buybacks = buybacks.sort_values('buyback_amount_qtr', ascending=False)
        buybacks_sample = buybacks.head(20000)  # Top 20K by size

        profiles = compute_profiles_for_actions(
            buybacks_sample, 'buyback', engine, outcomes
        )
        if len(profiles) > 0:
            all_profiles.append(profiles)

    # 2. Process Acquisitions (all - only 3.7K)
    acq_path = DATA_DIR / 'acquisitions_linked.parquet'
    if acq_path.exists():
        acquisitions = pd.read_parquet(acq_path)
        # Note: These are TARGET companies that got acquired
        # We want the state BEFORE acquisition

        profiles = compute_profiles_for_actions(
            acquisitions, 'acquired', engine, outcomes
        )
        if len(profiles) > 0:
            all_profiles.append(profiles)

    # 3. Process Bankruptcies (all - only 2K)
    bk_path = DATA_DIR / 'bankruptcies_linked.parquet'
    if bk_path.exists():
        bankruptcies = pd.read_parquet(bk_path)

        profiles = compute_profiles_for_actions(
            bankruptcies, 'bankruptcy', engine, outcomes
        )
        if len(profiles) > 0:
            all_profiles.append(profiles)

    # 4. Process Dividends (sample - 151K is too many)
    div_path = DATA_DIR / 'dividend_actions.parquet'
    if div_path.exists():
        dividends = pd.read_parquet(div_path)

        # Sample: 5K each of increases, cuts, initiations
        div_sample = pd.concat([
            dividends[dividends['action_type'] == 'dividend_increase'].head(5000),
            dividends[dividends['action_type'] == 'dividend_cut'].head(5000),
            dividends[dividends['action_type'] == 'dividend_initiate'].head(5000),
        ])

        profiles = compute_profiles_for_actions(
            div_sample, 'dividend', engine, outcomes
        )
        if len(profiles) > 0:
            all_profiles.append(profiles)

    # Combine all profiles
    if all_profiles:
        combined = pd.concat(all_profiles, ignore_index=True)

        # Save
        output_path = DATA_DIR / 'clean_action_profiles.parquet'
        combined.to_parquet(output_path)

        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Total profiles computed: {len(combined):,}")
        print(f"\nBy action type:")
        print(combined['action_type'].value_counts().to_string())
        print(f"\nSaved to: {output_path}")
        print(f"Completed at: {datetime.now()}")
    else:
        print("No profiles computed!")


if __name__ == "__main__":
    main()
