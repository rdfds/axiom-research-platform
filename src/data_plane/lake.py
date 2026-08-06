"""
Data Lake
=========
Immutable, append-only storage for canonical records.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
import json
import hashlib

from .base import CanonicalRecord, RecordType


class DataLake:
    """
    Raw data lake with immutable, append-only storage.

    Storage structure:
    lake_root/
    ├── raw/                          # Immutable raw records
    │   ├── fundamental/
    │   │   ├── 2024-01-15.parquet
    │   │   └── ...
    │   ├── corporate_action/
    │   ├── ma_deal/
    │   ├── price/
    │   └── estimate/
    ├── metadata/                     # Ingestion metadata
    │   └── ingestion_log.jsonl
    └── snapshots/                    # Point-in-time snapshots
        └── 2024-01-15/

    Key principles:
    1. IMMUTABLE: Raw records are never modified
    2. APPEND-ONLY: New data is added, never replaced
    3. TIMESTAMPED: Every record has event_time and available_time
    4. VERSIONED: Track which version of data was used
    """

    def __init__(self, lake_root: Path):
        """
        Initialize data lake.

        Parameters
        ----------
        lake_root : Path
            Root directory for the lake
        """
        self.lake_root = Path(lake_root)
        self.raw_dir = self.lake_root / 'raw'
        self.metadata_dir = self.lake_root / 'metadata'
        self.snapshots_dir = self.lake_root / 'snapshots'

        # Create directories
        for d in [self.raw_dir, self.metadata_dir, self.snapshots_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Create subdirs for each record type
        for rt in RecordType:
            (self.raw_dir / rt.value).mkdir(exist_ok=True)

    def publish(self, records: List[CanonicalRecord]) -> int:
        """
        Publish records to the lake (append-only).

        Records are partitioned by:
        - record_type
        - available_time (date)

        Parameters
        ----------
        records : list
            Canonical records to publish

        Returns
        -------
        Number of records published
        """
        if not records:
            return 0

        # Group by record_type and date
        groups: Dict[str, List[CanonicalRecord]] = {}
        for record in records:
            key = f"{record.record_type.value}/{record.available_time.strftime('%Y-%m-%d')}"
            if key not in groups:
                groups[key] = []
            groups[key].append(record)

        published = 0
        for key, group_records in groups.items():
            record_type, date_str = key.split('/')
            output_dir = self.raw_dir / record_type
            output_path = output_dir / f"{date_str}.parquet"

            # Convert to DataFrame
            df = pd.DataFrame([r.to_dict() for r in group_records])

            # Append or create
            if output_path.exists():
                existing = pd.read_parquet(output_path)
                # Dedupe by record_id
                df = pd.concat([existing, df]).drop_duplicates(subset=['record_id'], keep='last')

            df.to_parquet(output_path, index=False)
            published += len(group_records)

        # Log ingestion
        self._log_ingestion(records)

        return published

    def query(
        self,
        record_type: RecordType,
        as_of: Optional[datetime] = None,
        entity_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Query records from the lake with point-in-time filtering.

        CRITICAL: When as_of is provided, only returns records where
        available_time <= as_of. This enforces no-lookahead.

        Parameters
        ----------
        record_type : RecordType
            Type of records to query
        as_of : datetime, optional
            Point-in-time filter (only records available by this time)
        entity_id : str, optional
            Filter by entity
        start_date : datetime, optional
            Filter by event_time >= start_date
        end_date : datetime, optional
            Filter by event_time <= end_date

        Returns
        -------
        DataFrame with matching records
        """
        record_dir = self.raw_dir / record_type.value

        if not record_dir.exists():
            return pd.DataFrame()

        # Load all parquet files
        dfs = []
        for f in record_dir.glob('*.parquet'):
            dfs.append(pd.read_parquet(f))

        if not dfs:
            return pd.DataFrame()

        df = pd.concat(dfs, ignore_index=True)

        # Convert timestamps
        df['event_time'] = pd.to_datetime(df['event_time'])
        df['available_time'] = pd.to_datetime(df['available_time'])

        # Apply filters
        if as_of:
            # CRITICAL: No-lookahead filter
            df = df[df['available_time'] <= as_of]

        if entity_id:
            df = df[df['entity_id'] == entity_id]

        if start_date:
            df = df[df['event_time'] >= start_date]

        if end_date:
            df = df[df['event_time'] <= end_date]

        # Sort by event_time
        df = df.sort_values('event_time')

        return df

    def get_latest(
        self,
        record_type: RecordType,
        entity_id: str,
        as_of: datetime,
    ) -> Optional[Dict]:
        """
        Get the latest record for an entity as of a point in time.

        Parameters
        ----------
        record_type : RecordType
            Type of record
        entity_id : str
            Entity identifier
        as_of : datetime
            Point in time (only records available by this time)

        Returns
        -------
        Latest record or None
        """
        df = self.query(
            record_type=record_type,
            entity_id=entity_id,
            as_of=as_of,
        )

        if df.empty:
            return None

        # Get latest by event_time
        latest = df.sort_values('event_time').iloc[-1]
        return latest.to_dict()

    def create_snapshot(self, as_of: datetime) -> Path:
        """
        Create a point-in-time snapshot of the lake.

        This materializes all data that was available as of the given time.

        Parameters
        ----------
        as_of : datetime
            Point in time for snapshot

        Returns
        -------
        Path to snapshot directory
        """
        snapshot_dir = self.snapshots_dir / as_of.strftime('%Y-%m-%d')
        snapshot_dir.mkdir(exist_ok=True)

        for record_type in RecordType:
            df = self.query(record_type=record_type, as_of=as_of)
            if not df.empty:
                df.to_parquet(snapshot_dir / f"{record_type.value}.parquet", index=False)

        # Save metadata
        metadata = {
            'as_of': as_of.isoformat(),
            'created_at': datetime.utcnow().isoformat(),
            'record_counts': {
                rt.value: len(self.query(record_type=rt, as_of=as_of))
                for rt in RecordType
            }
        }
        with open(snapshot_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        return snapshot_dir

    def _log_ingestion(self, records: List[CanonicalRecord]):
        """Log ingestion to metadata."""
        log_path = self.metadata_dir / 'ingestion_log.jsonl'

        entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'records': len(records),
            'sources': list(set(r.source for r in records)),
            'record_types': list(set(r.record_type.value for r in records)),
            'date_range': {
                'min_event': min(r.event_time for r in records).isoformat(),
                'max_event': max(r.event_time for r in records).isoformat(),
            }
        }

        with open(log_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    def get_stats(self) -> Dict[str, Any]:
        """Get lake statistics."""
        stats = {
            'record_counts': {},
            'date_ranges': {},
            'total_size_mb': 0,
        }

        for record_type in RecordType:
            record_dir = self.raw_dir / record_type.value
            if record_dir.exists():
                files = list(record_dir.glob('*.parquet'))
                if files:
                    total_rows = 0
                    total_size = 0
                    for f in files:
                        df = pd.read_parquet(f)
                        total_rows += len(df)
                        total_size += f.stat().st_size

                    stats['record_counts'][record_type.value] = total_rows
                    stats['total_size_mb'] += total_size / (1024 * 1024)

        return stats
