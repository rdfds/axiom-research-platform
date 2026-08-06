# ExtractedFactRegistry README

This dataset stores structured facts extracted from documents and transcripts,
with provenance and citations suitable for audit and backtesting.

## Location

`data/inputs_layer/extracted_fact_registry_enriched/`  
Hive‑partitioned by year (`year=YYYY/part_*.parquet`)

## Schema (Required Columns)

- `fact_id`
- `document_id`
- `entity_id`
- `fact_type`
- `confidence_score`
- `source_id`
- `source_type`
- `published_at`
- `ingested_at`
- `raw_pointer`

Full schema: `schemas/inputs_layer/extracted_fact_registry.schema.json`

## Source

Built from:
- `data/warehouse/warehouse_text_signals` (signal_name/value/confidence)
- Joined to `data/warehouse/warehouse_doc_chunks` for citations and speakers

Note: the pre‑enrichment raw facts live at
`data/inputs_layer/extracted_fact_registry/` if you need the unjoined base. 
