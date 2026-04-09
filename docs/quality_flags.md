# Quality Flags (MVP)

All normalized records include a `quality_flags` array. Use these values to
indicate missing data, uncertainty, or validation failures. Multiple flags may
apply to a single record.

## Global Flags

- `missing_data`
- `delayed_data`
- `partial_coverage`
- `source_conflict`
- `outlier_detected`
- `unit_inconsistency`
- `restatement`
- `stale_data`
- `estimated_available_time`
- `estimated_event_time`
- `estimated_period_end`
- `estimated_company_id`
- `schema_violation`

