"""
Compute State Profiles for All Linked DealScan Deals
=====================================================
This script computes state profiles for all 1,152 DealScan deals
that were successfully linked to Compustat.

This creates the pre-computed analog database used for fast retrieval.

Usage:
  python 07_compute_all_deal_profiles.py

Output: data/deal_profiles_all.parquet
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from src.snapshot import AsOfSnapshotBuilder, DATA_DIR
from src.signals import SignalEngine


def main():
    print("=" * 70)
    print("COMPUTING STATE PROFILES FOR ALL LINKED DEALS")
    print("=" * 70)

    # Load linked deals
    linked_path = DATA_DIR / 'dealscan_linked.parquet'
    print(f"\nLoading linked deals from {linked_path}...")
    deals = pd.read_parquet(linked_path)
    deals['facilitystartdate'] = pd.to_datetime(deals['facilitystartdate'])
    print(f"Loaded {len(deals):,} linked deals")

    # Initialize signal engine
    print("\nInitializing signal engine...")
    engine = SignalEngine()

    # Process all deals
    print(f"\nComputing state profiles for {len(deals):,} deals...")
    print("This may take a few minutes...\n")

    results = []
    success_count = 0
    fail_count = 0

    for i, (idx, deal) in enumerate(deals.iterrows()):
        gvkey = deal['gvkey']
        deal_date = deal['facilitystartdate']

        try:
            profile = engine.compute_state_profile(gvkey, deal_date)

            if profile:
                result = {
                    'facilityid': deal['facilityid'],
                    'packageid': deal.get('packageid'),
                    'gvkey': gvkey,
                    'borrower_name': deal['borrower_name'],
                    'ticker': deal.get('ticker_clean'),
                    'deal_date': deal_date,
                    'deal_type': deal['primarypurpose'],
                    'facility_amount': deal['facilityamt'],
                    'target_company': deal.get('targetcompany'),
                    'sic_code': deal.get('primarysiccode'),
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

        # Progress
        if (i + 1) % 100 == 0:
            pct = (i + 1) / len(deals) * 100
            print(f"  Processed {i + 1:,}/{len(deals):,} ({pct:.0f}%) - {success_count:,} profiles computed")

    print(f"\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Total deals processed: {len(deals):,}")
    print(f"Successful profiles: {success_count:,} ({success_count/len(deals)*100:.1f}%)")
    print(f"Failed: {fail_count:,}")

    if results:
        profiles_df = pd.DataFrame(results)

        # Analyze the profiles
        print(f"\n📊 Profile Statistics:")
        print(f"   Composite Score - Mean: {profiles_df['composite_score'].mean():.1f}, "
              f"Std: {profiles_df['composite_score'].std():.1f}")

        if 'signal_balance_sheet_optionality' in profiles_df.columns:
            print(f"   Balance Sheet Optionality - Mean: {profiles_df['signal_balance_sheet_optionality'].mean():.1f}")
        if 'signal_growth_momentum' in profiles_df.columns:
            print(f"   Growth Momentum - Mean: {profiles_df['signal_growth_momentum'].mean():.1f}")
        if 'signal_valuation_dislocation' in profiles_df.columns:
            print(f"   Valuation Dislocation - Mean: {profiles_df['signal_valuation_dislocation'].mean():.1f}")

        # By deal type
        print(f"\n📊 Profiles by Deal Type:")
        type_stats = profiles_df.groupby('deal_type').agg({
            'facilityid': 'count',
            'composite_score': 'mean',
            'facility_amount': 'mean'
        }).rename(columns={'facilityid': 'count', 'composite_score': 'avg_score', 'facility_amount': 'avg_amount'})
        print(type_stats.to_string())

        # By year
        profiles_df['year'] = profiles_df['deal_date'].dt.year
        print(f"\n📊 Profiles by Year:")
        year_stats = profiles_df.groupby('year').agg({
            'facilityid': 'count',
            'composite_score': 'mean'
        }).rename(columns={'facilityid': 'count', 'composite_score': 'avg_score'})
        print(year_stats.tail(10).to_string())

        # Save
        output_path = DATA_DIR / 'deal_profiles_all.parquet'
        profiles_df.to_parquet(output_path, index=False)
        print(f"\n✅ Saved {len(profiles_df):,} deal profiles to {output_path}")

        # Also save CSV sample for inspection
        csv_path = DATA_DIR / 'deal_profiles_sample.csv'
        sample_cols = ['borrower_name', 'deal_date', 'deal_type', 'facility_amount',
                       'composite_score', 'signal_balance_sheet_optionality',
                       'signal_growth_momentum', 'signal_valuation_dislocation']
        sample_cols = [c for c in sample_cols if c in profiles_df.columns]
        profiles_df[sample_cols].head(100).to_csv(csv_path, index=False)
        print(f"   Saved sample CSV to {csv_path}")

        return profiles_df

    return None


if __name__ == "__main__":
    main()
