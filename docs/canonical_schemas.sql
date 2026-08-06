-- Canonical Schemas (MVP)
-- Dialect: DuckDB/Postgres-friendly SQL
-- Note: LIST/TEXT array types may vary by engine (DuckDB: VARCHAR[], Postgres: TEXT[])

-- =====================================================================
-- Common bitemporal fields (all canonical tables)
--   source_system, entity_id, company_id, security_id
--   event_time, available_time, ingestion_time
--   version_id, raw_payload_hash
--   upstream_version_ids (array), quality_flags (array)
-- =====================================================================

-- A1. Financial Statements
CREATE TABLE warehouse_financials (
  source_system           VARCHAR,
  entity_id               VARCHAR,
  company_id              VARCHAR,
  security_id             VARCHAR,
  event_time              TIMESTAMP,
  available_time          TIMESTAMP,
  ingestion_time          TIMESTAMP,
  version_id              VARCHAR,
  raw_payload_hash        VARCHAR,
  upstream_version_ids    VARCHAR[],
  quality_flags           VARCHAR[],
  fiscal_period_end       TIMESTAMP,
  fiscal_year             INTEGER,
  fiscal_quarter          INTEGER,
  statement_type          VARCHAR,
  line_item               VARCHAR,
  value                   DOUBLE,
  currency                VARCHAR,
  units                   VARCHAR,
  restatement_flag        BOOLEAN
);

-- A2. Market Prices (monthly)
CREATE TABLE warehouse_prices (
  source_system           VARCHAR,
  entity_id               VARCHAR,
  company_id              VARCHAR,
  security_id             VARCHAR,
  event_time              TIMESTAMP,
  available_time          TIMESTAMP,
  ingestion_time          TIMESTAMP,
  version_id              VARCHAR,
  raw_payload_hash        VARCHAR,
  upstream_version_ids    VARCHAR[],
  quality_flags           VARCHAR[],
  trade_date              TIMESTAMP,
  open                    DOUBLE,
  high                    DOUBLE,
  low                     DOUBLE,
  close                   DOUBLE,
  adjusted_close          DOUBLE,
  volume                  DOUBLE,
  total_return_index      DOUBLE,
  ret                     DOUBLE,
  retx                    DOUBLE,
  cusip                   VARCHAR
);

-- A2. Market Prices (daily, CRSP)
-- Stored as partitioned parquet: data/warehouse/warehouse_prices_daily/year=YYYY/part_*.parquet
CREATE TABLE warehouse_prices_daily (
  source_system           VARCHAR,
  entity_id               VARCHAR,
  company_id              VARCHAR,
  security_id             VARCHAR,
  event_time              TIMESTAMP,
  available_time          TIMESTAMP,
  ingestion_time          TIMESTAMP,
  version_id              VARCHAR,
  raw_payload_hash        VARCHAR,
  upstream_version_ids    VARCHAR[],
  quality_flags           VARCHAR[],
  open                    DOUBLE,
  high                    DOUBLE,
  low                     DOUBLE,
  close                   DOUBLE,
  adjusted_close          DOUBLE,
  volume                  DOUBLE,
  total_return_index      DOUBLE,
  ret                     DOUBLE,
  retx                    DOUBLE,
  permno                  BIGINT,
  date                    TIMESTAMP
);

-- A2. Market Prices (daily, Refinitiv RDP)
-- Stored as partitioned parquet: data/warehouse/warehouse_prices_daily_rdp/year=YYYY/part_*.parquet
CREATE TABLE warehouse_prices_daily_rdp (
  source_system           VARCHAR,
  entity_id               VARCHAR,
  company_id              VARCHAR,
  security_id             VARCHAR,
  event_time              TIMESTAMP,
  available_time          TIMESTAMP,
  ingestion_time          TIMESTAMP,
  version_id              VARCHAR,
  raw_payload_hash        VARCHAR,
  upstream_version_ids    VARCHAR[],
  quality_flags           VARCHAR[],
  open                    DOUBLE,
  high                    DOUBLE,
  low                     DOUBLE,
  close                   DOUBLE,
  adjusted_close          DOUBLE,
  volume                  DOUBLE,
  total_return_index      DOUBLE,
  ret                     DOUBLE,
  ric                     VARCHAR,
  cusip8                  VARCHAR,
  permno                  BIGINT
);

-- A3. Rates / Spreads / Volatility (FRED macro)
CREATE TABLE warehouse_macro (
  source_system           VARCHAR,
  entity_id               VARCHAR,
  company_id              VARCHAR,
  security_id             VARCHAR,
  event_time              TIMESTAMP,
  available_time          TIMESTAMP,
  ingestion_time          TIMESTAMP,
  version_id              VARCHAR,
  raw_payload_hash        VARCHAR,
  upstream_version_ids    VARCHAR[],
  quality_flags           VARCHAR[],
  instrument_id           VARCHAR,
  instrument_type         VARCHAR,
  tenor                   VARCHAR,
  value                   DOUBLE,
  units                   VARCHAR
);

-- A4. Consensus Estimates & Revisions (A4-lite)
CREATE TABLE warehouse_estimates (
  source_system           VARCHAR,
  entity_id               VARCHAR,
  company_id              VARCHAR,
  security_id             VARCHAR,
  event_time              TIMESTAMP,
  available_time          TIMESTAMP,
  ingestion_time          TIMESTAMP,
  version_id              VARCHAR,
  raw_payload_hash        VARCHAR,
  upstream_version_ids    VARCHAR[],
  quality_flags           VARCHAR[],
  metric                  VARCHAR,
  period                  VARCHAR,
  consensus_value         DOUBLE,
  num_estimates           DOUBLE,
  revision_direction      VARCHAR,
  revision_magnitude      DOUBLE,
  period_end              TIMESTAMP
);

-- A5. Corporate Actions
CREATE TABLE warehouse_corp_actions (
  source_system           VARCHAR,
  entity_id               VARCHAR,
  company_id              VARCHAR,
  security_id             VARCHAR,
  event_time              TIMESTAMP,
  available_time          TIMESTAMP,
  ingestion_time          TIMESTAMP,
  version_id              VARCHAR,
  raw_payload_hash        VARCHAR,
  upstream_version_ids    VARCHAR[],
  quality_flags           VARCHAR[],
  action_type             VARCHAR,
  announcement_date       TIMESTAMP,
  effective_date          TIMESTAMP,
  size                    DOUBLE,
  units                   VARCHAR,
  funding_source          VARCHAR
);

-- A6. M&A Deals
CREATE TABLE warehouse_mna_deals (
  source_system           VARCHAR,
  entity_id               VARCHAR,
  company_id              VARCHAR,
  security_id             VARCHAR,
  event_time              TIMESTAMP,
  available_time          TIMESTAMP,
  ingestion_time          TIMESTAMP,
  version_id              VARCHAR,
  raw_payload_hash        VARCHAR,
  upstream_version_ids    VARCHAR[],
  quality_flags           VARCHAR[],
  deal_id                 VARCHAR,
  acquirer_company_id     VARCHAR,
  target_company_id       VARCHAR,
  announcement_date       TIMESTAMP,
  close_date              TIMESTAMP,
  deal_value              DOUBLE,
  consideration_type      VARCHAR,
  deal_type               VARCHAR,
  status                  VARCHAR
);

-- B4. Press Releases (SEC 8-K)
-- Stored as partitioned parquet: data/warehouse/warehouse_press_releases/year=YYYY/part_*.parquet
CREATE TABLE warehouse_press_releases (
  source_system           VARCHAR,
  entity_id               VARCHAR,
  company_id              VARCHAR,
  security_id             VARCHAR,
  event_time              TIMESTAMP,
  available_time          TIMESTAMP,
  ingestion_time          TIMESTAMP,
  version_id              VARCHAR,
  raw_payload_hash        VARCHAR,
  upstream_version_ids    VARCHAR[],
  quality_flags           VARCHAR[],
  document_id             VARCHAR,
  release_date            TIMESTAMP,
  headline                VARCHAR,
  text                    VARCHAR,
  form_type               VARCHAR,
  cik                     VARCHAR,
  accession               VARCHAR,
  primary_document        VARCHAR
);

-- B1/B2/B3. Unstructured Documents (canonical doc registry)
-- Use document_type to distinguish: earnings_call, research_report, presentation
CREATE TABLE warehouse_documents (
  source_system           VARCHAR,
  entity_id               VARCHAR,
  company_id              VARCHAR,
  security_id             VARCHAR,
  event_time              TIMESTAMP,
  available_time          TIMESTAMP,
  ingestion_time          TIMESTAMP,
  version_id              VARCHAR,
  raw_payload_hash        VARCHAR,
  upstream_version_ids    VARCHAR[],
  quality_flags           VARCHAR[],
  document_id             VARCHAR,
  document_type           VARCHAR,
  title                   VARCHAR,
  publisher               VARCHAR,
  analyst                 VARCHAR,
  rating                  VARCHAR,
  price_target            DOUBLE,
  call_date               TIMESTAMP,
  publish_date            TIMESTAMP,
  presentation_date       TIMESTAMP,
  release_date            TIMESTAMP,
  source_url              VARCHAR
);

-- B1/B2/B3. Document chunks (speaker/section_type for transcripts,
-- slide_number for presentations)
CREATE TABLE warehouse_doc_chunks (
  source_system           VARCHAR,
  entity_id               VARCHAR,
  company_id              VARCHAR,
  security_id             VARCHAR,
  event_time              TIMESTAMP,
  available_time          TIMESTAMP,
  ingestion_time          TIMESTAMP,
  version_id              VARCHAR,
  raw_payload_hash        VARCHAR,
  upstream_version_ids    VARCHAR[],
  quality_flags           VARCHAR[],
  chunk_id                VARCHAR,
  document_id             VARCHAR,
  chunk_index             INTEGER,
  slide_number            INTEGER,
  text                    VARCHAR,
  speaker                 VARCHAR,
  speaker_role            VARCHAR,
  section_type            VARCHAR,
  token_count             INTEGER
);

CREATE TABLE warehouse_text_signals (
  source_system           VARCHAR,
  entity_id               VARCHAR,
  company_id              VARCHAR,
  security_id             VARCHAR,
  event_time              TIMESTAMP,
  available_time          TIMESTAMP,
  ingestion_time          TIMESTAMP,
  version_id              VARCHAR,
  raw_payload_hash        VARCHAR,
  upstream_version_ids    VARCHAR[],
  quality_flags           VARCHAR[],
  signal_name             VARCHAR,
  value                   DOUBLE,
  confidence              DOUBLE,
  supporting_chunk_ids    VARCHAR[]
);
