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
                'sic': row['sic'],
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


