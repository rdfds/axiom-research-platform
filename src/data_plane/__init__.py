"""
Data Plane
==========
Ingestion → Storage → Normalization

This module implements the data plane architecture with:
- Source adapters (fetch, parse, validate, publish)
- Dual timestamping (event_time, available_time) for no lookahead
- Raw lake storage with immutable records
- Canonical record format across all sources
"""

from .base import (
    SourceAdapter,
    CanonicalRecord,
    ValidationResult,
    RecordType,
    ActionType,
    BatchWindow,
)
from .refinitiv_adapter import RefinitivAdapter
from .lake import DataLake

__all__ = [
    'SourceAdapter',
    'CanonicalRecord',
    'ValidationResult',
    'RecordType',
    'ActionType',
    'BatchWindow',
    'RefinitivAdapter',
    'DataLake',
]
