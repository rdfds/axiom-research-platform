#!/usr/bin/env python
"""
Ingest Refinitiv Data into Data Lake
=====================================
Converts all existing Refinitiv parquet files into canonical records
and publishes to the immutable data lake.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_plane import DataLake, CanonicalRecord, RecordType, ActionType


def ingest_ma_deals(lake: DataLake, data_dir: Path) -> int:
    """Ingest M&A deals into the lake."""
    print("\n--- Ingesting M&A Deals ---")

    # Try both US and global files
    files = [
        data_dir / 'ma_deals_us.parquet',
        data_dir / 'ma_deals_all.parquet',
    ]

    total_records = []

    for file_path in files:
        if not file_path.exists():
            continue

        df = pd.read_parquet(file_path)
        print(f"Loaded {len(df):,} from {file_path.name}")

        for _, row in df.iterrows():
            try:
                ann_date = pd.to_datetime(row.get('Date Announced'))
                if pd.isna(ann_date):
                    continue

                entity_id = str(row.get('Instrument', row.get('Target Name', 'unknown')))

                record = CanonicalRecord(
                    record_id=CanonicalRecord.generate_id(
                        'refinitiv', 'ma_deal', entity_id, ann_date.to_pydatetime(),
                        deal_value=str(row.get('Deal Value', ''))
                    ),
                    record_type=RecordType.MA_DEAL,
                    source='refinitiv',
                    entity_id=entity_id,
                    entity_name=row.get('Target Name'),
                    event_time=ann_date.to_pydatetime(),
                    available_time=ann_date.to_pydatetime(),
                    data={
                        'deal_value': row.get('Deal Value'),
                        'deal_value_usd': row.get('Deal Value (USD)'),
                        'deal_status': row.get('Deal Status'),
                        'ma_type': row.get('M&A Type'),
                        'target_name': row.get('Target Name'),
                        'target_nation': row.get('Target Nation'),
                        'target_sector': row.get('Target TRBC Economic Sector'),
                        'acquiror_name': row.get('Acquiror Name'),
                        'acquiror_nation': row.get('Acquiror Nation'),
                        'premium_1day': row.get('Premium 1 Day'),
                        'premium_1week': row.get('Premium 1 Week'),
                        'premium_4week': row.get('Premium 4 Weeks'),
                        'date_effective': str(row.get('Date Effective', '')),
                    }
                )
                total_records.append(record)
            except Exception as e:
                continue

    if total_records:
        # Dedupe by record_id
        seen = set()
        unique_records = []
        for r in total_records:
            if r.record_id not in seen:
                seen.add(r.record_id)
                unique_records.append(r)

        published = lake.publish(unique_records)
        print(f"Published {published:,} M&A deal records")
        return published

    return 0


def ingest_dividends(lake: DataLake, data_dir: Path) -> int:
    """Ingest dividend records into the lake."""
    print("\n--- Ingesting Dividends ---")

    file_path = data_dir / 'dividends_complete.parquet'
    if not file_path.exists():
        print("Dividends file not found")
        return 0

    df = pd.read_parquet(file_path)
    print(f"Loaded {len(df):,} dividend records")

    records = []
    for _, row in df.iterrows():
        try:
            # Try multiple date column naming conventions
            ex_date = pd.to_datetime(
                row.get('Dividend Ex Date') or row.get('Ex-Date') or row.get('Ex Date')
            )
            if pd.isna(ex_date):
                continue

            # Declaration date is when it became knowable (use ex_date as fallback)
            decl_date = pd.to_datetime(
                row.get('Declaration Date') or row.get('Announcement Date')
            )
            if pd.isna(decl_date):
                decl_date = ex_date  # Fallback

            entity_id = str(row.get('Instrument', 'unknown'))

            record = CanonicalRecord(
                record_id=CanonicalRecord.generate_id(
                    'refinitiv', 'corporate_action', entity_id, ex_date.to_pydatetime(),
                    action_type='dividend'
                ),
                record_type=RecordType.CORPORATE_ACTION,
                source='refinitiv',
                entity_id=entity_id,
                entity_name=row.get('Company Name'),
                event_time=ex_date.to_pydatetime(),
                available_time=decl_date.to_pydatetime(),
                data={
                    'action_type': ActionType.DIVIDEND_REGULAR.value,
                    'dividend_amount': row.get('Dividend Amount') or row.get('Gross Amount'),
                    'dividend_type': row.get('Dividend Type'),
                    'payment_date': str(row.get('Dividend Pay Date') or row.get('Payment Date', '')),
                    'record_date': str(row.get('Dividend Record Date') or row.get('Record Date', '')),
                    'currency': row.get('Currency'),
                    'frequency': row.get('Frequency'),
                }
            )
            records.append(record)
        except Exception as e:
            continue

    if records:
        published = lake.publish(records)
        print(f"Published {published:,} dividend records")
        return published

    return 0


def ingest_fundamentals(lake: DataLake, data_dir: Path) -> int:
    """Ingest fundamental data into the lake."""
    print("\n--- Ingesting Fundamentals ---")

    files = [
        data_dir / 'fundamentals_all.parquet',
        data_dir / 'fundamentals_quarterly.parquet',
    ]

    total_records = []

    for file_path in files:
        if not file_path.exists():
            continue

        df = pd.read_parquet(file_path)
        print(f"Loaded {len(df):,} from {file_path.name}")

        for _, row in df.iterrows():
            try:
                entity_id = str(row.get('Instrument', 'unknown'))

                # For current fundamentals, use today as event_time
                # For quarterly, use fiscal period end
                fiscal_end = pd.to_datetime(row.get('Fiscal Period End Date'))
                if pd.isna(fiscal_end):
                    fiscal_end = datetime.now()

                # Filing date is when it became knowable
                filing_date = pd.to_datetime(row.get('Filing Date'))
                if pd.isna(filing_date):
                    filing_date = fiscal_end

                record = CanonicalRecord(
                    record_id=CanonicalRecord.generate_id(
                        'refinitiv', 'fundamental', entity_id, fiscal_end.to_pydatetime() if hasattr(fiscal_end, 'to_pydatetime') else fiscal_end
                    ),
                    record_type=RecordType.FUNDAMENTAL,
                    source='refinitiv',
                    entity_id=entity_id,
                    entity_name=row.get('Company Common Name') or row.get('Company Name'),
                    event_time=fiscal_end.to_pydatetime() if hasattr(fiscal_end, 'to_pydatetime') else fiscal_end,
                    available_time=filing_date.to_pydatetime() if hasattr(filing_date, 'to_pydatetime') else filing_date,
                    data={
                        'revenue': row.get('Revenue'),
                        'net_income': row.get('Net Income'),
                        'ebitda': row.get('EBITDA'),
                        'total_assets': row.get('Total Assets'),
                        'total_debt': row.get('Total Debt'),
                        'market_cap': row.get('Company Market Cap'),
                        'pe_ratio': row.get('PE'),
                        'ev_ebitda': row.get('EV/EBITDA'),
                        'sector': row.get('GICS Sector') or row.get('TRBC Economic Sector'),
                        'industry': row.get('GICS Industry') or row.get('TRBC Industry'),
                    }
                )
                total_records.append(record)
            except Exception as e:
                continue

    if total_records:
        # Dedupe
        seen = set()
        unique_records = []
        for r in total_records:
            if r.record_id not in seen:
                seen.add(r.record_id)
                unique_records.append(r)

        published = lake.publish(unique_records)
        print(f"Published {published:,} fundamental records")
        return published

    return 0


def ingest_estimates(lake: DataLake, data_dir: Path) -> int:
    """Ingest analyst estimates into the lake."""
    print("\n--- Ingesting Analyst Estimates ---")

    file_path = data_dir / 'analyst_estimates_all.parquet'
    if not file_path.exists():
        print("Estimates file not found")
        return 0

    df = pd.read_parquet(file_path)
    print(f"Loaded {len(df):,} estimate records")

    records = []
    for _, row in df.iterrows():
        try:
            entity_id = str(row.get('Instrument', 'unknown'))

            # For estimates, event_time is the target period
            # available_time is when the estimate was made
            target_date = pd.to_datetime(row.get('Target Period End Date'))
            if pd.isna(target_date):
                target_date = datetime.now()

            record = CanonicalRecord(
                record_id=CanonicalRecord.generate_id(
                    'refinitiv', 'estimate', entity_id, target_date.to_pydatetime() if hasattr(target_date, 'to_pydatetime') else target_date
                ),
                record_type=RecordType.ESTIMATE,
                source='refinitiv',
                entity_id=entity_id,
                entity_name=row.get('Company Common Name'),
                event_time=target_date.to_pydatetime() if hasattr(target_date, 'to_pydatetime') else target_date,
                available_time=datetime.now(),  # Current estimates
                data={
                    'eps_mean': row.get('EPS Mean'),
                    'eps_high': row.get('EPS High'),
                    'eps_low': row.get('EPS Low'),
                    'revenue_mean': row.get('Revenue Mean'),
                    'num_analysts': row.get('Number of Analysts'),
                    'target_price_mean': row.get('Target Price Mean'),
                    'recommendation_mean': row.get('Recommendation Mean'),
                    'num_buy': row.get('Number of Buys'),
                    'num_hold': row.get('Number of Holds'),
                    'num_sell': row.get('Number of Sells'),
                }
            )
            records.append(record)
        except Exception as e:
            continue

    if records:
        published = lake.publish(records)
        print(f"Published {published:,} estimate records")
        return published

    return 0


def main():
    print("=" * 60)
    print("Refinitiv Data → Canonical Lake Ingestion")
    print("=" * 60)

    # Setup paths
    data_dir = Path('data/refinitiv')
    lake_dir = Path('data/lake')

    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return

    # Create lake
    lake = DataLake(lake_dir)
    print(f"Data Lake initialized at: {lake_dir}")

    # Ingest all data types
    total = 0
    total += ingest_ma_deals(lake, data_dir)
    total += ingest_dividends(lake, data_dir)
    total += ingest_fundamentals(lake, data_dir)
    total += ingest_estimates(lake, data_dir)

    # Print stats
    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)

    stats = lake.get_stats()
    print("\nLake Statistics:")
    for rt, count in stats['record_counts'].items():
        print(f"  {rt}: {count:,} records")
    print(f"  Total Size: {stats['total_size_mb']:.2f} MB")

    print(f"\nTotal records published: {total:,}")


if __name__ == '__main__':
    main()
