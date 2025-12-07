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


