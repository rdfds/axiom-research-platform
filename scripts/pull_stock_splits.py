#!/usr/bin/env python
"""
Pull Stock Splits from Refinitiv
================================
Detects stock splits by analyzing changes in shares outstanding over time.
"""

import refinitiv.data as rd
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import time

def detect_splits_for_ticker(ticker: str, start_date: str = '2000-01-01') -> list:
    """Detect stock splits for a single ticker by analyzing shares outstanding changes."""
    splits = []

    try:
        # Get shares outstanding history
        df = rd.get_history(
            [ticker],
            ['TR.SharesOutstanding'],
            start=start_date,
            end=datetime.now().strftime('%Y-%m-%d')
        )

        if df is None or len(df) < 10:
            return splits

        df = df.dropna()
        if len(df) < 10:
            return splits

        df.columns = ['shares']

        # Calculate day-over-day change
        df['pct_change'] = df['shares'].pct_change()

        # Detect large positive changes (forward splits)
        # A 2:1 split = 100% increase, 4:1 = 300%, etc.
        forward_splits = df[df['pct_change'] > 0.5]  # >50% increase

        for date, row in forward_splits.iterrows():
            ratio = row['pct_change'] + 1

            # Map to common split ratios
            if 1.8 < ratio < 2.2:
                split_ratio = '2:1'
                split_factor = 2.0
            elif 2.8 < ratio < 3.2:
                split_ratio = '3:1'
                split_factor = 3.0
            elif 3.8 < ratio < 4.2:
                split_ratio = '4:1'
                split_factor = 4.0
            elif 4.8 < ratio < 5.2:
                split_ratio = '5:1'
                split_factor = 5.0
            elif 6.8 < ratio < 7.2:
                split_ratio = '7:1'
                split_factor = 7.0
            elif 9.5 < ratio < 10.5:
                split_ratio = '10:1'
                split_factor = 10.0
            elif 14.5 < ratio < 15.5:
                split_ratio = '15:1'
                split_factor = 15.0
            elif 19 < ratio < 21:
                split_ratio = '20:1'
                split_factor = 20.0
            else:
                split_ratio = f'{ratio:.2f}:1'
                split_factor = ratio

            splits.append({
                'ticker': ticker,
                'split_date': date,
                'split_type': 'forward',
                'split_ratio': split_ratio,
                'split_factor': split_factor,
                'shares_before': row['shares'] / (row['pct_change'] + 1),
                'shares_after': row['shares'],
            })

        # Detect large negative changes (reverse splits)
        # A 1:2 reverse split = 50% decrease, 1:10 = 90% decrease
        reverse_splits = df[df['pct_change'] < -0.3]  # >30% decrease

        for date, row in reverse_splits.iterrows():
            ratio = 1 / (1 + row['pct_change'])

            if 1.8 < ratio < 2.2:
                split_ratio = '1:2'
                split_factor = 0.5
            elif 2.8 < ratio < 3.2:
                split_ratio = '1:3'
                split_factor = 1/3
            elif 3.8 < ratio < 4.2:
                split_ratio = '1:4'
                split_factor = 0.25
            elif 4.8 < ratio < 5.2:
                split_ratio = '1:5'
                split_factor = 0.2
            elif 9.5 < ratio < 10.5:
                split_ratio = '1:10'
                split_factor = 0.1
            elif 19 < ratio < 21:
                split_ratio = '1:20'
                split_factor = 0.05
            else:
                split_ratio = f'1:{ratio:.2f}'
                split_factor = 1/ratio

            splits.append({
                'ticker': ticker,
                'split_date': date,
                'split_type': 'reverse',
                'split_ratio': split_ratio,
                'split_factor': split_factor,
                'shares_before': row['shares'] / (1 + row['pct_change']),
                'shares_after': row['shares'],
            })

    except Exception as e:
        pass

    return splits


def main():
    print("=" * 60)
    print("Stock Splits Detection via Shares Outstanding Analysis")
    print("=" * 60)

    # Initialize Refinitiv
    rd.open_session()

    # Load universe
    universe_path = Path('data/refinitiv/universe.parquet')
    if not universe_path.exists():
        print("Error: universe.parquet not found")
        return

    universe = pd.read_parquet(universe_path)
    tickers = universe['ticker'].tolist()
    print(f"\nAnalyzing {len(tickers)} companies for stock splits...")

    all_splits = []
    processed = 0
    errors = 0

    # Process in batches to avoid rate limits
    batch_size = 50

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        print(f"\nProcessing batch {i//batch_size + 1} ({i+1}-{min(i+batch_size, len(tickers))} of {len(tickers)})...")

        for ticker in batch:
            try:
                splits = detect_splits_for_ticker(ticker)
                if splits:
                    all_splits.extend(splits)
                    for s in splits:
                        print(f"  {s['ticker']}: {s['split_date'].date()} - {s['split_ratio']} ({s['split_type']})")
                processed += 1
            except Exception as e:
                errors += 1

        # Small delay between batches
        if i + batch_size < len(tickers):
            time.sleep(1)

    # Close Refinitiv session
    rd.close_session()

    # Save results
    if all_splits:
        df = pd.DataFrame(all_splits)
        df['split_date'] = pd.to_datetime(df['split_date'])
        df = df.sort_values('split_date')

        output_path = Path('data/refinitiv/stock_splits.parquet')
        df.to_parquet(output_path, index=False)

        print("\n" + "=" * 60)
        print("STOCK SPLITS DETECTION COMPLETE")
        print("=" * 60)
        print(f"\nTotal splits detected: {len(df)}")
        print(f"Forward splits: {len(df[df['split_type'] == 'forward'])}")
        print(f"Reverse splits: {len(df[df['split_type'] == 'reverse'])}")
        print(f"Companies processed: {processed}")
        print(f"Errors: {errors}")
        print(f"\nSaved to: {output_path}")

        # Show sample
        print("\nSample splits:")
        print(df.head(20).to_string())

        # Most common ratios
        print("\nSplit ratio distribution:")
        print(df['split_ratio'].value_counts().head(10))
    else:
        print("\nNo splits detected!")


if __name__ == '__main__':
    main()
