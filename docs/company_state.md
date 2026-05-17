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

## Example Rows

| entity_id | feature_group | feature_key | value_num | value_str | value_ts |
|---|---|---|---:|---|---|
| 001004 | ts | price.close | 102.4 | NULL | NULL |
| 001004 | fact | guidance_change | NULL | "down" | NULL |
| 001004 | event | acquisition | NULL | NULL | 2023‑08‑12 |

## Rebuild (DuckDB Fast Path)

```bash
python -u scripts/build_company_state.py \
  --asof 2024-12-31 \
  --engine duckdb \
  --format long \
  --threads 6 \
  --memory 10GB
```

## Wide Export (Optional)

Wide format is convenient for modeling, but heavier.
Use `scripts/export_company_state_wide.py`.

