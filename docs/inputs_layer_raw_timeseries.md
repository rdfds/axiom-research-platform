# RawTimeSeriesStore README

This dataset normalizes **prices**, **macro series**, and **estimates** into a single
point‑in‑time table with full provenance. It is the canonical time‑series input to
downstream pipelines.

## Location

`data/inputs_layer/raw_timeseries.parquet`

## Schema (Required Columns)

- `series_id`: stable series identifier
- `series_type`: `price`, `macro`, or `estimate`
- `entity_id`: entity/series identifier (issuer, instrument, or macro series)
- `date`: observation time (UTC)
- `value`: numeric value
- `published_at`: when the observation was available
- `ingested_at`: ingestion timestamp
- `confidence_score`: 0–1
- `raw_pointer`: pointer back to the source row

See full schema: `schemas/inputs_layer/raw_timeseries_store.schema.json`

