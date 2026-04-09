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

## Global Requirements (All Records)

Every normalized record must include:

- `source_system`
- `entity_id`
- `company_id` (nullable)
- `security_id` (nullable)
- `event_time`
- `available_time`
- `ingestion_time`
- `version_id`
- `raw_payload_hash`
- `upstream_version_ids` (array)
- `quality_flags` (array)

Hard rules:

- `available_time >= event_time`
- No record without both timestamps
- Append-only; never overwrite (supersede instead)
- Derived data must reference `upstream_version_ids`

## Raw Ingestion Contract (Append-Only)

Raw payloads are immutable. Each raw record must also write a row in
`data/lake/raw_manifest.parquet` with:

- `source_system`
- `raw_record_id`
- `raw_payload_hash`
- `raw_path`
- `entity_id`
- `company_id` (nullable)
- `security_id` (nullable)
- `event_time`
- `available_time`
- `ingestion_time`
- `version_id`
- `supersedes_version_id` (nullable)

Hashing:

- `raw_payload_hash = SHA256(normalized_payload_bytes)`
- `version_id = SHA256(source_system|entity_id|event_time|available_time|raw_payload_hash|schema_version)`

## Canonical Tables (Normalized Warehouse)

### A1. Financial Statements (SEC/XBRL)
`data/warehouse/warehouse_financials.parquet`

Required fields:

- `company_id`
- `fiscal_period_end`
- `fiscal_year`
- `fiscal_quarter`
- `statement_type` (income, balance_sheet, cash_flow)
- `line_item` (normalized taxonomy)
- `value`
- `currency`
- `units`
- `restatement_flag`

Notes:
- Compustat fundamentals are ingested into this table as a long-form
  canonical representation (line_item + value) to support full as-of
  routing in the MVP.

Temporal semantics:

- `event_time = fiscal_period_end`
- `available_time = filing_timestamp`

Validation:

- Balance sheet must balance
- Unit consistency enforced

Restatements:

- New records only; never overwrite

### A2. Market Prices (Equities & Indices)
`data/warehouse/warehouse_prices.parquet` (monthly)
`data/warehouse/warehouse_prices_daily/` (daily, CRSP)
`data/warehouse/warehouse_prices_daily_rdp/` (daily, Refinitiv RDP)

Required fields:

- `security_id`
- `trade_date`
- `open`
- `high`
- `low`
- `close`
- `adjusted_close`
- `volume`
- `total_return_index`

Temporal semantics:

- `event_time = trade_date`
- `available_time = market_close_time`

Special handling:

- Corporate-action-adjusted returns
- Trading halts flagged
- Missing days explicitly marked

### A3. Rates / Spreads / Volatility
`data/warehouse/warehouse_macro.parquet`

Required fields:

- `instrument_id`
- `instrument_type` (rate, spread, volatility)
- `tenor`
- `value`
- `units`

Temporal semantics:

- `event_time = observation_date`
- `available_time = publication_time`

Validation:

- Curve monotonicity checks
- Outlier detection

### A4. Consensus Estimates & Revisions
`data/warehouse/warehouse_estimates.parquet`

Required fields:

- `company_id`
- `metric`
- `period`
- `consensus_value`
- `num_estimates`
- `revision_direction`
- `revision_magnitude`

Temporal semantics:

- `event_time = estimate_period_end`
- `available_time = estimate_publish_time`

MVP note (A4-lite):

- If publish_time is unavailable, records are ingested with
  `estimated_available_time` + `estimated_period_end` quality flags.
  Forward-looking period_end is preserved in `period_end`.

Revisions:

- Tracked as deltas (no overwrite)

### A5. Corporate Actions (Non-M&A)
`data/warehouse/warehouse_corp_actions.parquet`

Required fields:

- `company_id`
- `action_type` (buyback, dividend, issuance, debt_issuance, etc.)
- `announcement_date`
- `effective_date`
- `size`
- `units`
- `funding_source`

Temporal semantics:

- `event_time = announcement_date`
- `available_time = announcement_timestamp`

Edge cases:

- Authorization vs execution flagged separately

### A6. Public M&A Deal Metadata
`data/warehouse/warehouse_mna_deals.parquet`

Required fields:

- `deal_id`
- `acquirer_company_id`
- `target_company_id`
- `announcement_date`
- `close_date`
- `deal_value`
- `consideration_type` (cash, stock, mixed)
- `deal_type` (heuristic)
- `status`

Temporal semantics:

- `event_time = announcement_date`
- `available_time = announcement_timestamp`

## Unstructured Documents

### B4. Press Releases (SEC 8-K)
`data/warehouse/warehouse_press_releases/`

- `document_id`
- `release_date`
- `headline`
- `text`
- `form_type`
- `cik`
- `accession`
- `primary_document`

### Documents (planned)
`data/warehouse/warehouse_documents.parquet`

- `document_id`
- `source_system`
- `company_id`
- `document_type`
- `event_time`
- `available_time`
- `title`
- `raw_payload_hash`
- `version_id`

### Document Chunks (planned)
`data/warehouse/warehouse_doc_chunks.parquet`

- `chunk_id`
- `document_id`
- `chunk_index`
- `text`
- `speaker`
- `speaker_role`
- `section_type`
- `token_count`
- `event_time`
- `available_time`
- `raw_payload_hash`
- `version_id`

### Text Signals (planned)
`data/warehouse/warehouse_text_signals.parquet`

- `signal_name`
- `value`
- `confidence`
- `supporting_chunk_ids`

### Extracted Signals
`data/warehouse/warehouse_extracted_signals.parquet`

- `signal_id`
- `document_id`
- `chunk_id`
- `signal_name`
- `value`
- `confidence`
- `supporting_chunk_ids` (array)
- `event_time`
- `available_time`
- `version_id`

## Entity ID Mapping

`data/mappings/entity_id_map.parquet`

- `company_id` (gvkey)
- `security_id` (permno)
- `ric`
- `permid`
- `cusip`
- `isin`
- `valid_from`
- `valid_to`
- `source_system`

## As-Of Enforcement (DuckDB Pattern)

All downstream reads must apply:

- `available_time <= as_of`
- `event_time <= as_of`
- latest record by `(natural_key, available_time, ingestion_time, version_id)`

Example:

```sql
WITH filtered AS (
  SELECT *
  FROM warehouse_financials
  WHERE available_time <= :as_of
    AND event_time     <= :as_of
),
ranked AS (
  SELECT *,
         ROW_NUMBER() OVER (
           PARTITION BY company_id, event_time, statement_type, line_item
           ORDER BY available_time DESC, ingestion_time DESC, version_id DESC
         ) AS rn
  FROM filtered
)
SELECT * FROM ranked WHERE rn = 1;
```

Natural keys by table:

- Financials: `(company_id, event_time, statement_type, line_item)`
- Prices: `(security_id, event_time)`
- Rates: `(instrument_id, event_time, tenor)`
- Estimates: `(company_id, metric, period, available_time)`
- Corp actions: `(company_id, action_type, event_time, announcement_date)`
- M&A: `(deal_id, announcement_date)`
- Documents: `(document_id, version_id)`
- Chunks: `(chunk_id, version_id)`
- Signals: `(signal_id, chunk_id, version_id)`
