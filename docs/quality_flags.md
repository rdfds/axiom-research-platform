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

## Domain-Specific Flags

### Financial Statements

- `balance_sheet_unbalanced`
- `cash_flow_mismatch`
- `missing_line_item`

### Market Prices

- `halted_session`
- `missing_trade_day`
- `price_outlier`

### Rates / Spreads / Volatility

- `curve_non_monotonic`
- `jump_outlier`

### Corporate Actions

- `authorization_only`
- `execution_only`
- `open_ended`
- `missing_size`

### M&A

- `withdrawn`
- `deal_failed`
- `value_missing`

## Usage Rules

- Never allow silent failure: errors must emit a flag.
- Conflicts should prefer earliest `available_time` unless overridden.
- Missing data must propagate downstream via `quality_flags`.
