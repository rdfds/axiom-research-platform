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

## Series ID Conventions

### Prices (from `data/warehouse/warehouse_prices.parquet`)
Long‑format series:
- `price.close`
- `price.adjusted_close`
- `price.volume`
- `price.ret`
- `price.retx`

### Macro (from `data/warehouse/warehouse_macro.parquet`)
Series ID uses the instrument identifier:
- e.g., `DGS10`, `DGS2`, `VIXCLS`

### Estimates (from `data/warehouse/warehouse_estimates.parquet`)
Series ID format:
- `estimate.<metric>.<period>.consensus`
- Example: `estimate.EPS.FY1.consensus`

## Provenance Rules

Each record carries:
- `published_at`: when the data was observable (e.g., `available_time`)
- `effective_at`: usually same as observation date
- `ingested_at`: ingestion time from the warehouse
- `raw_pointer`: `path#row=<row_id>[:<metric>]`

## Notes / Assumptions

- `entity_id_type` is `entity_id` for issuer‑level series and `macro_series` for macro.
- Units:
  - returns are `pct`
  - price is `price`
  - volume is `shares`
- Frequency:
  - prices default to `D`
  - macro/estimates are left null unless explicit

## How to Rebuild

```bash
python -u scripts/build_raw_timeseries_store.py
python -u scripts/validate_inputs_layer.py --config configs/inputs_layer.json
```

