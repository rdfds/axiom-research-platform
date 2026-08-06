Core Provider-Direct Metrics Status As Of March 25, 2026

Artifact
- `/tmp/consumer_industrial_snapshots_2024_12_31/company_state_snapshots_asof=2024-12-31.input_layer_v1.asofsafe_corefixed_v2.jsonl`

Scope
- 789 consumer and industrial names
- metrics covered:
- `market.market_cap_provider_direct`
- `operating.revenue_ttm_provider_direct`
- `operating.ebitda_ltm_provider_direct`
- `earnings.net_income_ttm_provider_direct`
- `liquidity.cash_and_short_term_investments_provider_direct`
- `capital_structure.total_debt_provider_direct`

Final production policy
- `market.market_cap_provider_direct`
- keep provider market cap unless PIT `price x shares` reconstruction is exact
- `operating.revenue_ttm_provider_direct`
- prefer SEC companyfacts reconstruction when available, including framed latest-FY proxy fallback
- `operating.ebitda_ltm_provider_direct`
- prefer SEC bridge only when exact; otherwise retain provider
- `earnings.net_income_ttm_provider_direct`
- prefer SEC companyfacts reconstruction when available
- `liquidity.cash_and_short_term_investments_provider_direct`
- prefer SEC companyfacts reconstruction when available, even when support is proxy due to missing/stale STI subcomponents
- `capital_structure.total_debt_provider_direct`
- prefer SEC debt stack only when exact; otherwise retain provider unless no provider value exists

Observed basis mix on the 789-name slice
- `market.market_cap_provider_direct`
- provider `789`
- `operating.revenue_ttm_provider_direct`
- SEC companyfacts `753`
- provider `36`
- `operating.ebitda_ltm_provider_direct`
- SEC companyfacts `644`
- provider `145`
- `earnings.net_income_ttm_provider_direct`
- SEC companyfacts `730`
- provider `59`
- `liquidity.cash_and_short_term_investments_provider_direct`
- SEC companyfacts `740`
- provider `49`
- `capital_structure.total_debt_provider_direct`
- SEC companyfacts `636`
- provider `153`

Observed support on the 789-name slice
- `market.market_cap_provider_direct`
- exact `789`
- `operating.revenue_ttm_provider_direct`
- exact `778`
- proxy `6`
- unsupported `5`
- `operating.ebitda_ltm_provider_direct`
- exact `789`
- `earnings.net_income_ttm_provider_direct`
- exact `789`
- `liquidity.cash_and_short_term_investments_provider_direct`
- exact `186`
- proxy `603`
- `capital_structure.total_debt_provider_direct`
- exact `785`
- proxy `1`
- unsupported `3`

Notes
- market cap stays provider-based on this slice because the current PIT price path is still proxy-only for the sampled names
- EBITDA intentionally retains provider values whenever the SEC bridge is only partial
- total debt intentionally retains provider values whenever the SEC debt stack is partial or unavailable
- cash plus short-term investments intentionally accepts SEC proxy cases because the SEC value is still closer to the filing basis than the legacy provider value for many important names

Reference artifacts
- `/tmp/consumer_industrial_snapshots_2024_12_31/core_provider_metric_selection_summary_v2.json`
- `/tmp/consumer_industrial_snapshots_2024_12_31/core_provider_metric_gap_stats_v2.json`
- `/tmp/consumer_industrial_snapshots_2024_12_31/core_metric_sample_validation_v2.json`
