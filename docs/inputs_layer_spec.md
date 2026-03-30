# Inputs Layer Specification

This document defines the Inputs Layer contract for the system. It is the gate between raw data ingestion and all downstream modeling.

The Inputs Layer produces:
RawDocumentStore
RawTimeSeriesStore
EventRegistry
ExtractedFactRegistry
EntityGraph
PrivateOverlayRegistry
DataIntegrityLog

All inputs must be time-indexed, reproducible as-of, traceable to a source, and non-leaky.

All objects must carry:
source_id
source_type
entity_id (or equivalent)
published_at
effective_at (if applicable)
ingested_at
confidence_score
raw_pointer

Schemas live in:
schemas/inputs_layer/

Validator script:
scripts/validate_inputs_layer.py

