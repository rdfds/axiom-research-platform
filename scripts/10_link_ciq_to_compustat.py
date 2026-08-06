"""
Link CIQ Key Developments to Compustat
======================================
The 39K CIQ events (dividends, equity offerings, divestitures) only have
company names in headlines. This script links them to Compustat gvkeys
so we can compute state profiles.

Approach:
1. Extract company names from CIQ headlines
2. Build lookup table from Compustat (name -> gvkey, ticker -> gvkey)
3. Match via exact ticker, then fuzzy company name

Usage:
  python scripts/10_link_ciq_to_compustat.py

Output: data/ciq_linked.parquet
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import re
from pathlib import Path
from datetime import datetime
from difflib import SequenceMatcher

from src.snapshot import DATA_DIR


def clean_company_name(name):
    """Normalize company name for matching."""
    if pd.isna(name):
        return None

    name = str(name).upper().strip()

    # Remove common suffixes
    suffixes = [
        ' INC', ' INCORPORATED', ' CORP', ' CORPORATION', ' CO', ' COMPANY',
        ' LTD', ' LIMITED', ' LLC', ' LP', ' PLC', ' SA', ' AG', ' NV',
        ' GROUP', ' HOLDINGS', ' HOLDING', ' INTERNATIONAL', ' INTL',
        ' & CO', ' AND CO', ' THE', ', THE',
    ]
    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[:-len(suffix)]

    # Remove punctuation
    name = re.sub(r'[^\w\s]', '', name)
    name = re.sub(r'\s+', ' ', name).strip()

    return name


def extract_ticker_from_headline(headline):
    """Try to extract ticker from headline patterns."""
    if pd.isna(headline):
        return None

    # Pattern: "TICKER - Company Name" or "Company Name (TICKER)"
    patterns = [
        r'\(([A-Z]{1,5})\)',  # (AAPL)
        r'^([A-Z]{1,5})\s*[-–]\s*',  # AAPL - Apple Inc
    ]

    for pattern in patterns:
        match = re.search(pattern, headline)
        if match:
            return match.group(1)

    return None


def extract_company_from_headline(headline):
    """Extract company name from headline."""
    if pd.isna(headline):
        return None

    headline = str(headline)

    # Common patterns in CIQ headlines:
    # "Company Name, $ X.XX, Cash Dividend"
    # "Company Name is considering acquisitions"
    # "Company Name Announces..."

    patterns = [
        r'^([^,]+),\s*\$',  # Before comma and dollar sign
        r'^(.+?)\s+is\s+',  # Before "is"
        r'^(.+?)\s+to\s+',  # Before "to"
        r'^(.+?)\s+declares\s+',  # Before "declares"
        r'^(.+?)\s+announces\s+',  # Before "announces"
        r'^(.+?)\s+completes\s+',  # Before "completes"
        r'^(.+?)\s+reports\s+',  # Before "reports"
    ]

    for pattern in patterns:
        match = re.match(pattern, headline, re.IGNORECASE)
        if match:
            company = match.group(1).strip()
            # Clean up
            if len(company) > 3 and len(company) < 100:
                return company

    return None


def fuzzy_match_score(name1, name2):
    """Compute fuzzy match score between two names."""
    if not name1 or not name2:
        return 0
    return SequenceMatcher(None, name1, name2).ratio()


def main():
    print("=" * 70)
    print("LINKING CIQ KEY DEVELOPMENTS TO COMPUSTAT")
    print("=" * 70)

    # Load CIQ Key Developments
    ciq_path = DATA_DIR / 'ciqsamp_keydev_ciqkeydev.parquet'
    print(f"\nLoading CIQ Key Developments from {ciq_path}...")
    ciq = pd.read_parquet(ciq_path)
    ciq['announceddate'] = pd.to_datetime(ciq['announceddate'])
    print(f"Loaded {len(ciq):,} events")

    # Load Compustat for company name/ticker lookup
    fund_path = DATA_DIR / 'fundamentals_quarterly.parquet'
    print(f"\nLoading Compustat fundamentals from {fund_path}...")
    fund = pd.read_parquet(fund_path)
    print(f"Loaded {len(fund):,} records")

    # Build lookup tables from Compustat
    print("\nBuilding company lookup tables...")

    # Get unique company info
    companies = fund[['gvkey', 'tic', 'conm']].drop_duplicates()
    companies = companies.dropna(subset=['conm'])

    # Ticker -> gvkey lookup
    ticker_to_gvkey = {}
    for _, row in companies.iterrows():
        if pd.notna(row['tic']):
            ticker = str(row['tic']).upper().strip()
            if ticker not in ticker_to_gvkey:
                ticker_to_gvkey[ticker] = row['gvkey']

    print(f"  Ticker lookup: {len(ticker_to_gvkey):,} tickers")

    # Clean name -> gvkey lookup
    name_to_gvkey = {}
    for _, row in companies.iterrows():
        clean_name = clean_company_name(row['conm'])
        if clean_name and clean_name not in name_to_gvkey:
            name_to_gvkey[clean_name] = row['gvkey']

    print(f"  Name lookup: {len(name_to_gvkey):,} company names")

    # Also create a list for fuzzy matching
    compustat_names = list(name_to_gvkey.keys())

    # Process CIQ events
    print(f"\nLinking {len(ciq):,} CIQ events to Compustat...")

    results = []
    matched_ticker = 0
    matched_exact_name = 0
    matched_fuzzy_name = 0
    unmatched = 0

    for i, (idx, row) in enumerate(ciq.iterrows()):
        headline = row['headline']

        # Extract company info from headline
        ticker = extract_ticker_from_headline(headline)
        company_name = extract_company_from_headline(headline)
        clean_name = clean_company_name(company_name) if company_name else None

        gvkey = None
        match_method = None

        # Try ticker match first
        if ticker and ticker in ticker_to_gvkey:
            gvkey = ticker_to_gvkey[ticker]
            match_method = 'ticker'
            matched_ticker += 1

        # Try exact name match
        elif clean_name and clean_name in name_to_gvkey:
            gvkey = name_to_gvkey[clean_name]
            match_method = 'exact_name'
            matched_exact_name += 1

        # Try fuzzy name match (only for first word match + high similarity)
        elif clean_name:
            first_word = clean_name.split()[0] if clean_name else None

            if first_word and len(first_word) > 2:
                # Find candidates starting with same word
                candidates = [n for n in compustat_names if n.startswith(first_word)]

                if candidates:
                    # Find best fuzzy match
                    best_score = 0
                    best_match = None

                    for candidate in candidates[:20]:  # Limit for speed
                        score = fuzzy_match_score(clean_name, candidate)
                        if score > best_score:
                            best_score = score
                            best_match = candidate

                    if best_score >= 0.85:  # High threshold
                        gvkey = name_to_gvkey[best_match]
                        match_method = 'fuzzy_name'
                        matched_fuzzy_name += 1

        if gvkey is None:
            unmatched += 1

        results.append({
            'ciq_idx': idx,
            'headline': headline,
            'extracted_company': company_name,
            'extracted_ticker': ticker,
            'gvkey': gvkey,
            'match_method': match_method,
            'announceddate': row['announceddate'],
        })

        # Progress
        if (i + 1) % 10000 == 0:
            total_matched = matched_ticker + matched_exact_name + matched_fuzzy_name
            pct = total_matched / (i + 1) * 100
            print(f"  Processed {i + 1:,}/{len(ciq):,} - {total_matched:,} matched ({pct:.1f}%)")

    # Summary
    print(f"\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    total_matched = matched_ticker + matched_exact_name + matched_fuzzy_name
    print(f"Total CIQ events: {len(ciq):,}")
    print(f"Matched to Compustat: {total_matched:,} ({total_matched/len(ciq)*100:.1f}%)")
    print(f"  - Via ticker: {matched_ticker:,}")
    print(f"  - Via exact name: {matched_exact_name:,}")
    print(f"  - Via fuzzy name: {matched_fuzzy_name:,}")
    print(f"Unmatched: {unmatched:,}")

    # Save results
    results_df = pd.DataFrame(results)

    # Filter to matched only
    matched_df = results_df[results_df['gvkey'].notna()].copy()

    output_path = DATA_DIR / 'ciq_linked.parquet'
    matched_df.to_parquet(output_path, index=False)
    print(f"\n✅ Saved {len(matched_df):,} linked CIQ events to {output_path}")

    # Show sample
    print("\nSample matched events:")
    print(matched_df[['extracted_company', 'gvkey', 'match_method', 'announceddate']].head(20).to_string())

    return matched_df


if __name__ == "__main__":
    main()
