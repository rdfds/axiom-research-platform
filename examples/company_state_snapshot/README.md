# Company State Snapshot Example

This example shows the data layer that everything else depends on.

The sample file is:

```text
examples/company_state_snapshot/company_state_hd.sample.json
```

It is a compact public version of a Home Depot company-state context, reshaped into the `CompanyStateSnapshot` contract.

## What To Look At

The important part is not just the metric values. It is the metadata around every metric:

- `as_of_time`
- `computed_at`
- `confidence`
- `unit`
- `provenance`
- `fallback_used`
- `missing_reason`

Example feature:

```json
{
  "name": "capital_structure.net_leverage",
  "value": 2.1485,
  "unit": "x_ebitda",
  "confidence": 0.7933,
  "provenance": [
    {
      "source_id": "balance_sheet_and_ebitda_snapshot",
      "method": "net_debt_divided_by_ebitda_ttm"
    }
  ]
}
```

