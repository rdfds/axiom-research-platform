"""
Base Classes for Data Plane
===========================
Defines the interfaces and canonical formats for all source adapters.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Generator
from enum import Enum
import pandas as pd
import hashlib
import json


class RecordType(Enum):
    """Types of canonical records."""
    FUNDAMENTAL = "fundamental"
    PRICE = "price"
    CORPORATE_ACTION = "corporate_action"
    MA_DEAL = "ma_deal"
    ESTIMATE = "estimate"
    INSIDER = "insider"


class ActionType(Enum):
    """Standardized corporate action types."""
    # M&A
    ACQUISITION = "acquisition"
    MERGER = "merger"
    DIVESTITURE = "divestiture"
    SPINOFF = "spinoff"
    LBO = "lbo"
    GOING_PRIVATE = "going_private"

    # Capital Returns
    DIVIDEND_REGULAR = "dividend_regular"
    DIVIDEND_SPECIAL = "dividend_special"
    DIVIDEND_INCREASE = "dividend_increase"
    DIVIDEND_CUT = "dividend_cut"
    DIVIDEND_INITIATE = "dividend_initiate"
    DIVIDEND_SUSPEND = "dividend_suspend"
    BUYBACK = "buyback"

    # Equity
    IPO = "ipo"
    SECONDARY_OFFERING = "secondary_offering"
    STOCK_SPLIT = "stock_split"
    REVERSE_SPLIT = "reverse_split"

    # Debt
    DEBT_ISSUANCE = "debt_issuance"
    DEBT_REFINANCE = "debt_refinance"

    # Other
    BANKRUPTCY = "bankruptcy"
    OTHER = "other"


@dataclass
class CanonicalRecord:
    """
    Standardized record format across all data sources.

    CRITICAL: Every record has two timestamps:
    - event_time: When the event/data describes (fiscal period end, action date)
    - available_time: When this information became knowable (filing date, announcement)

    This dual timestamping enforces no-lookahead in backtesting.
    """

    # Identity
    record_id: str                    # Unique ID (hash of key fields)
    record_type: RecordType           # Type of record
    source: str                       # Source system (refinitiv, wrds, etc.)

    # Entity
    entity_id: str                    # Company identifier (ticker, gvkey, etc.)
    entity_name: Optional[str]        # Company name

    # CRITICAL: Dual timestamps for no-lookahead
    event_time: datetime              # When the event/data describes
    available_time: datetime          # When it became knowable

    # Payload
    data: Dict[str, Any]              # Source-specific data fields

    # Metadata
    ingested_at: datetime = field(default_factory=datetime.utcnow)
    source_record_id: Optional[str] = None  # Original ID from source
    version: int = 1

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            'record_id': self.record_id,
            'record_type': self.record_type.value,
            'source': self.source,
            'entity_id': self.entity_id,
            'entity_name': self.entity_name,
            'event_time': self.event_time.isoformat(),
            'available_time': self.available_time.isoformat(),
            'data': self.data,
            'ingested_at': self.ingested_at.isoformat(),
            'source_record_id': self.source_record_id,
            'version': self.version,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'CanonicalRecord':
        """Create from dictionary."""
        return cls(
            record_id=d['record_id'],
            record_type=RecordType(d['record_type']),
            source=d['source'],
            entity_id=d['entity_id'],
            entity_name=d.get('entity_name'),
            event_time=datetime.fromisoformat(d['event_time']),
            available_time=datetime.fromisoformat(d['available_time']),
            data=d['data'],
            ingested_at=datetime.fromisoformat(d['ingested_at']),
            source_record_id=d.get('source_record_id'),
            version=d.get('version', 1),
        )

    @staticmethod
    def generate_id(source: str, record_type: str, entity_id: str, event_time: datetime, **kwargs) -> str:
        """Generate deterministic record ID from key fields."""
        key = f"{source}:{record_type}:{entity_id}:{event_time.isoformat()}"
        for k, v in sorted(kwargs.items()):
            key += f":{k}={v}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]


@dataclass
class ValidationResult:
    """Result of validating canonical records."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    records_in: int = 0
    records_valid: int = 0
    records_invalid: int = 0

    def add_error(self, msg: str):
        self.errors.append(msg)
        self.is_valid = False

    def add_warning(self, msg: str):
        self.warnings.append(msg)


@dataclass
class BatchWindow:
    """Time window for batch fetching."""
    start: datetime
    end: datetime
    source: str
    record_type: Optional[RecordType] = None

    def __str__(self):
        return f"{self.source}[{self.start.date()} to {self.end.date()}]"


class SourceAdapter(ABC):
    """
    Abstract base class for data source adapters.

    Each source (Refinitiv, WRDS, etc.) implements this interface:
    - fetch(): Pull raw data from source
    - parse(): Convert to canonical records
    - validate(): Check data quality
    - publish(): Write to raw lake

    The adapter is responsible for:
    1. Setting event_time correctly (what date the data describes)
    2. Setting available_time correctly (when it was knowable)
    3. Deduplicating records
    4. Handling source-specific quirks
    """

    source_name: str = "unknown"

    @abstractmethod
    def fetch(self, window: BatchWindow) -> List[Dict[str, Any]]:
        """
        Fetch raw payloads from the source.

        Parameters
        ----------
        window : BatchWindow
            Time window to fetch

        Returns
        -------
        List of raw payloads (source-specific format)
        """
        pass

    @abstractmethod
    def parse(self, raw_payloads: List[Dict[str, Any]]) -> List[CanonicalRecord]:
        """
        Parse raw payloads into canonical records.

        CRITICAL: This is where event_time and available_time must be set correctly.

        Parameters
        ----------
        raw_payloads : list
            Raw data from fetch()

        Returns
        -------
        List of CanonicalRecord objects
        """
        pass

    def validate(self, records: List[CanonicalRecord]) -> ValidationResult:
        """
        Validate canonical records.

        Default implementation checks:
        - Required fields present
        - Timestamps are valid
        - No future available_time
        - event_time <= available_time

        Override for source-specific validation.
        """
        result = ValidationResult(is_valid=True, records_in=len(records))
        now = datetime.utcnow()

        for record in records:
            is_valid = True

            # Check required fields
            if not record.entity_id:
                result.add_error(f"Record {record.record_id}: missing entity_id")
                is_valid = False

            if not record.event_time:
                result.add_error(f"Record {record.record_id}: missing event_time")
                is_valid = False

            if not record.available_time:
                result.add_error(f"Record {record.record_id}: missing available_time")
                is_valid = False

            # Check timestamp logic
            if record.available_time and record.available_time > now:
                result.add_warning(f"Record {record.record_id}: available_time in future")

            if record.event_time and record.available_time:
                if record.event_time > record.available_time:
                    result.add_warning(
                        f"Record {record.record_id}: event_time > available_time "
                        f"({record.event_time} > {record.available_time})"
                    )

            if is_valid:
                result.records_valid += 1
            else:
                result.records_invalid += 1

        return result

    @abstractmethod
    def publish(self, records: List[CanonicalRecord], lake: 'DataLake') -> int:
        """
        Publish validated records to the data lake.

        Parameters
        ----------
        records : list
            Validated canonical records
        lake : DataLake
            Target data lake

        Returns
        -------
        Number of records published
        """
        pass

    def ingest(self, window: BatchWindow, lake: 'DataLake') -> Dict[str, Any]:
        """
        Full ingestion pipeline: fetch → parse → validate → publish.

        Parameters
        ----------
        window : BatchWindow
            Time window to ingest
        lake : DataLake
            Target data lake

        Returns
        -------
        Ingestion summary with counts and any errors
        """
        summary = {
            'source': self.source_name,
            'window': str(window),
            'started_at': datetime.utcnow().isoformat(),
            'status': 'running',
        }

        try:
            # Fetch
            raw_payloads = self.fetch(window)
            summary['fetched'] = len(raw_payloads)

            # Parse
            records = self.parse(raw_payloads)
            summary['parsed'] = len(records)

            # Validate
            validation = self.validate(records)
            summary['valid'] = validation.records_valid
            summary['invalid'] = validation.records_invalid
            summary['errors'] = validation.errors[:10]  # First 10 errors
            summary['warnings'] = validation.warnings[:10]

            # Publish (only valid records)
            if validation.records_valid > 0:
                valid_records = [r for r in records if self._is_record_valid(r, validation)]
                published = self.publish(valid_records, lake)
                summary['published'] = published
            else:
                summary['published'] = 0

            summary['status'] = 'completed'

        except Exception as e:
            summary['status'] = 'failed'
            summary['error'] = str(e)

        summary['completed_at'] = datetime.utcnow().isoformat()
        return summary

    def _is_record_valid(self, record: CanonicalRecord, validation: ValidationResult) -> bool:
        """Check if a specific record is valid."""
        # Simple check - if record_id appears in any error, it's invalid
        for error in validation.errors:
            if record.record_id in error:
                return False
        return True
