# PrivateOverlayRegistry README

This dataset is a **private overlay layer** for non‑public inputs such as
covenant packages, internal projections, and board constraints.

## Location

`data/inputs_layer/private_overlay_registry.parquet`

## Schema (Required Columns)

- `overlay_id`
- `entity_id`
- `overlay_type`
- `overlay_payload`
- `version`
- `author`
- `created_at`
- `status`
- `source_id`
- `source_type`
- `published_at`
- `ingested_at`
- `confidence_score`
- `raw_pointer`

Full schema: `schemas/inputs_layer/private_overlay_registry.schema.json`

## Overlay Types (examples)

- `covenants`
- `debt_schedule`
- `internal_projection`
- `segment_kpis`
- `board_constraints`

## Example Payload (internal projection)

```json
{
  "scenario": "base",
  "horizon_years": 3,
  "revenue_growth": [0.05, 0.04, 0.03],
  "ebitda_margin": [0.22, 0.23, 0.24],
  "capex_pct_sales": [0.04, 0.04, 0.04]
}
```

## Key Rules

- Overlays **do not overwrite** public data.
- Every overlay must be versioned and removable.
- Overlays should expire or be explicitly revoked.

