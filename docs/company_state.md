# CompanyState README

CompanyState is a **point‑in‑time snapshot** of all available signals for each entity.
It is assembled from:

- RawTimeSeriesStore (prices, macro, estimates)
- EventRegistry (corporate actions)
- ExtractedFactRegistry (text‑derived signals)
- EntityGraph (ID resolution)

## Location

`data/company_state/company_state.parquet`

## Format (Long)

CompanyState is stored in **long format** for scalability.

Columns:
- `entity_id`
- `feature_group` (`ts`, `fact`, `event`)
- `feature_key`
- `value_num`
- `value_str`
- `value_ts`
- `published_at`
- `asof`
- `built_at`

