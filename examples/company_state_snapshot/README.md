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

## Why This Matters

Most model demos quietly use whatever data is available today. That makes historical validation fragile because future information can leak into the past.

Axiom's company-state layer is designed around point-in-time semantics:

- what was known
- when it was known
- where it came from
- whether a fallback was used
- whether the value is strong enough for decision evidence

## Related Code

- `src/company_state_builder.py`
- `src/asof_store.py`
- `docs/data_contract.md`
- `schemas/company_state/snapshot.json`
- `schemas/company_state/feature_record.json`

## Sample Contract

The sample intentionally keeps only a small set of features:

- market capitalization
- enterprise value
- share price
- equity and credit window proxies
- revenue, EBITDA, and EBITDA margin
- free cash flow and FCF yield
- deployable liquidity
- net leverage

That keeps the example readable while still demonstrating the technical contract.
