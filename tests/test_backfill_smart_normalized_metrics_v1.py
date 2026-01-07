import json
from pathlib import Path

import scripts.backfill_smart_normalized_metrics_v1 as smart_mod
from scripts.backfill_smart_normalized_metrics_v1 import (
    _build_fail_open_smart_metrics,
    _effective_cash_equivalents_value,
    _effective_liquidity_component_values,
    _effective_lease_liability_value,
    _extract_retirement_note_components_from_html,
    _effective_total_debt_baseline,
    _grouped_cash_proxy_can_complete_with_exact_marketable_securities,
    _grouped_cash_proxy_can_promote_to_exact_cash_baseline,
    _load_completed_company_ids,
    _load_companyfacts,
    _load_retirement_note_components,
    _market_availability_adjustment,
    _summarize_output_rows,
    materialize_smart_metrics_for_row,
)


def test_effective_liquidity_components_infers_zero_restricted_cash_from_grouped_reconciliation():
    resolved = _effective_liquidity_component_values(
        cash_grouped={"support_mode": "exact", "value": 120.0},
        cash_exact={"support_mode": "exact", "value": 100.0},
        restricted_cash_sec={"support_mode": "unsupported", "value": None, "missing_reason": "sec_concept_unavailable"},
        marketable_sec={"support_mode": "exact", "value": 20.0},
        restricted_cash={"support_mode": "unsupported", "value": None},
        marketable={"support_mode": "unsupported", "value": None},
    )

    assert resolved["restricted_cash_value"] == 0.0
    assert resolved["restricted_cash_inferred_zero"] is True
    assert resolved["restricted_cash_zero_reconciled"] is True


