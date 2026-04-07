# Data Contract (MVP)

This document defines the bitemporal, append-only data contract for the MVP.
It applies to all structured and unstructured sources and enforces as-of
semantics for downstream consumers.

## Storage Layout

- Raw immutable lake:
  - `data/lake/raw/<source_system>/ingest_date=YYYY-MM-DD/*.jsonl`
  - `data/lake/raw_manifest.parquet`
- Normalized warehouse (bitemporal):
  - `data/warehouse/*.parquet`
- Entity mapping:
  - `data/mappings/entity_id_map.parquet`

