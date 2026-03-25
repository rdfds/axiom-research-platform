# Live Feature Contract Status (2026-03-25)

This memo separates the features referenced by the live policy stack into:

- core live features
- source-limited live features
- unwired / contract-mismatch features

The policy references were inspected in:

- `./src/candidate_generation.py`
- `./src/mechanism_brain.py`
- `./src/recommendation_run.py`

The live builder output was checked in:

- `./src/company_state_builder.py`

Representative live snapshot used for sanity check:

- `/tmp/hd_live_snapshot_2024_12_31/company_state_snapshots_asof=2024-12-31.jsonl`

Representative source checks for `0000354950` (Home Depot):

- `event_store.parquet`: `0` company rows
- `ownership_13f_summary.parquet`: `4409` company rows
- `issuer_rating_history.parquet`: `204` company rows
- `entity_graph.parquet`: `5` related rows

## Core Live Features

These are the financially important / market-important live inputs that are already wired into the builder and used by the policy stack.

- `capital_structure.fixed_charge_coverage`
- `capital_structure.interest_coverage`
- `capital_structure.net_debt`
- `capital_structure.net_leverage`
- `capital_structure.rating_state`
- `capital_structure.total_debt`
- `liquidity.available_for_actions`
- `liquidity.cash`
- `market.credit_window_proxy`
- `market.drawdown_90d`
- `market.equity_window_proxy`
- `market.market_cap`
- `market.volatility_30d`
- `market.volatility_90d`
- `operating.ebitda_margin_ttm`
- `operating.fcf_conversion`
- `operating.revenue_cagr_3y`

## Source-Limited Live Features

These are real live features, but they only populate when the underlying source has enough coverage or disclosure.

- `capital_return.dividend_payer_flag`
- `capital_return.last_dividend_event_type`
- `capital_structure.debt_due_0_12m`
- `capital_structure.debt_due_12_24m`
- `capital_structure.debt_due_next_24m`
- `capital_structure.debt_schedule_inconsistency_flag`
- `capital_structure.debt_schedule_total`
- `capital_structure.debt_schedule_vs_total_debt`
- `capital_structure.maturity_wall_ratio_24m`
- `liquidity.minimum_cash_policy_proxy`
- `liquidity.runway_months`
- `market.fcf_yield`
- `operating.revenue_yoy_last_q`
- `strategic.action_frequency_24m`
- `strategic.last_action_type`
- `strategic.recent_actions_count_24m`

Notes:

- The event-history features above are not broken in the builder; they fail when `event_store.parquet` has no rows for the company.
- The maturity-wall / debt-due features are intentionally unsupported when the filing does not disclose enough maturity detail.

## Unwired / Contract-Mismatch Features

These are the main cleanup targets. They are referenced in the policy stack but are not cleanly provided by the live builder.

- `capital_return.buyback_capacity_proxy`
- `capital_return.share_count_trend`
- `market.conglomerate_discount_signal`
- `market.ev_ebitda_vs_peer_z`
- `market.fcf_yield_percentile_peers`
- `operating.ebitda_margin_percentile_peers`
- `operating.segment_margin_divergence`
- `ownership_governance.activist_presence_flag`
- `peer_context.relative_positioning.market_share_percentile`
- `strategic.segment_count`
- `strategic.segment_references`

## Known Alias / Namespace Mismatches

These features are close to available today, but the builder and policy use different names.

- Builder emits `ownership_governance.activist_signal`
  - Policy expects `ownership_governance.activist_presence_flag`
- Builder emits `peer_context.valuation_z`
  - Policy expects `market.ev_ebitda_vs_peer_z`
- Builder emits `peer_context.margin_percentile`
  - Policy expects `operating.ebitda_margin_percentile_peers`
- Builder emits peer valuation / leverage / margin context under `peer_context.*`
  - Policy still expects some older `market.*` / `operating.*` peer-relative aliases

## Recommended Order Of Fixes

1. Wire alias features from existing builder outputs:
   - `ownership_governance.activist_presence_flag`
   - `market.ev_ebitda_vs_peer_z`
   - `operating.ebitda_margin_percentile_peers`
2. Restore event-history coverage:
   - `capital_return.dividend_payer_flag`
   - `capital_return.last_dividend_event_type`
   - `strategic.last_action_type`
   - `strategic.recent_actions_count_24m`
   - `strategic.action_frequency_24m`
3. Add peer-relative valuation / quality features that are already partly available in `peer_context.*`
4. Decide whether to wire or remove the remaining truly hollow features:
   - `capital_return.buyback_capacity_proxy`
   - `capital_return.share_count_trend`
   - `market.conglomerate_discount_signal`
   - `operating.segment_margin_divergence`
   - `peer_context.relative_positioning.market_share_percentile`
   - `strategic.segment_count`
   - `strategic.segment_references`
