# As-Of Views (DuckDB Templates)

These templates enforce bitemporal semantics (`event_time`, `available_time`)
and select the most recent record as-of a given time.

Programmatic helper: `src/asof_store.py` provides a tiny DuckDB wrapper to
query these views or raw parquet tables with as-of filtering.

**Global constraint:** `available_time <= :as_of AND event_time <= :as_of`

## Financials

```sql
CREATE OR REPLACE VIEW asof_financials AS
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

## Prices

```sql
CREATE OR REPLACE VIEW asof_prices AS
WITH filtered AS (
  SELECT *
  FROM warehouse_prices
  WHERE available_time <= :as_of
    AND event_time     <= :as_of
),
ranked AS (
  SELECT *,
         ROW_NUMBER() OVER (
           PARTITION BY security_id, event_time
           ORDER BY available_time DESC, ingestion_time DESC, version_id DESC
         ) AS rn
  FROM filtered
)
SELECT * FROM ranked WHERE rn = 1;
```

## Rates / Spreads / Volatility

```sql
CREATE OR REPLACE VIEW asof_rates AS
WITH filtered AS (
  SELECT *
  FROM warehouse_rates
  WHERE available_time <= :as_of
    AND event_time     <= :as_of
),
ranked AS (
  SELECT *,
         ROW_NUMBER() OVER (
           PARTITION BY instrument_id, event_time, tenor
           ORDER BY available_time DESC, ingestion_time DESC, version_id DESC
         ) AS rn
  FROM filtered
)
SELECT * FROM ranked WHERE rn = 1;
```

## Estimates

```sql
CREATE OR REPLACE VIEW asof_estimates AS
WITH filtered AS (
  SELECT *
  FROM warehouse_estimates
  WHERE available_time <= :as_of
    AND event_time     <= :as_of
),
ranked AS (
  SELECT *,
         ROW_NUMBER() OVER (
           PARTITION BY company_id, metric, period
           ORDER BY available_time DESC, ingestion_time DESC, version_id DESC
         ) AS rn
  FROM filtered
)
SELECT * FROM ranked WHERE rn = 1;
```

## Corporate Actions

```sql
CREATE OR REPLACE VIEW asof_corp_actions AS
WITH filtered AS (
  SELECT *
  FROM warehouse_corp_actions
  WHERE available_time <= :as_of
    AND event_time     <= :as_of
),
ranked AS (
  SELECT *,
         ROW_NUMBER() OVER (
           PARTITION BY company_id, action_type, event_time, announcement_date
           ORDER BY available_time DESC, ingestion_time DESC, version_id DESC
         ) AS rn
  FROM filtered
)
SELECT * FROM ranked WHERE rn = 1;
```

