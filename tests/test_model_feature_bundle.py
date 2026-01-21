from __future__ import annotations

import math

from src.model_feature_bundle import (
    attach_model_feature_bundle,
    build_model_feature_bundle,
    feature_view_from_snapshot,
    get_bundle_value,
)


def test_build_model_feature_bundle_surfaces_canonical_metrics(monkeypatch):
    monkeypatch.setenv("AXIOM_ENABLE_RUNTIME_FEATURE_ADAPTER", "1")
    monkeypatch.setenv(
        "AXIOM_RUNTIME_FEATURE_ADAPTER_RULES",
        "normalized_net_leverage,ust_10y_alias,credit_ig_alias",
    )
    snapshot = {
        "company_id": "00001234",
        "as_of_time": "2026-02-28T00:00:00+00:00",
        "features": {
            "market.market_cap": {"value": 1500.0, "support_mode": "exact"},
            "operating.revenue_ttm": {"value": 900.0, "support_mode": "exact"},
            "capital_structure.net_leverage_normalized": {"value": 2.25, "support_mode": "exact"},
            "macro.ust_10y_yield": {"value": 4.58, "support_mode": "exact"},
            "macro.ig_oas": {"value": 1.05, "support_mode": "exact"},
            "market.vix": {"value": 18.0, "support_mode": "exact"},
            "capital_structure.net_pension_liability": {"value": 75.0, "support_mode": "exact"},
            "capital_structure.combined_retirement_liability": {"value": 95.0, "support_mode": "exact"},
            "capital_structure.debt_like_obligations_including_retirement": {
                "value": 1_240.0,
                "support_mode": "proxy_missing_component",
            },
            "capital_structure.net_debt_including_retirement": {
                "value": 930.0,
                "support_mode": "proxy_missing_component",
            },
            "capital_structure.gross_leverage_including_retirement": {
                "value": 3.1,
                "support_mode": "proxy_missing_component",
            },
            "capital_structure.net_leverage_including_retirement": {
                "value": 2.4,
                "support_mode": "proxy_missing_component",
            },
            "capital_structure.retirement_obligation_regime": {
                "value": "combined_retirement_only",
                "support_mode": "present",
            },
        },
    }

    bundle = build_model_feature_bundle(snapshot)

    assert get_bundle_value(bundle, "capital.net_leverage") == 2.25
    assert bundle["support"]["capital.net_leverage"]["source_metric"] == "capital_structure.net_leverage_normalized"
    assert bundle["canonical"]["macro.ust_10y_yield"] == 4.58
    assert bundle["canonical"]["macro.ig_oas"] == 1.05
    assert bundle["canonical"]["capital.net_pension_liability"] == 75.0
    assert bundle["canonical"]["capital.combined_retirement_liability"] == 95.0
    assert bundle["canonical"]["capital.debt_like_obligations_including_retirement"] == 1_240.0
    assert bundle["canonical"]["capital.net_debt_including_retirement"] == 930.0
    assert bundle["canonical"]["capital.gross_leverage_including_retirement"] == 3.1
    assert bundle["canonical"]["capital.net_leverage_including_retirement"] == 2.4
    assert bundle["canonical"]["capital.retirement_obligation_regime"] == "combined_retirement_only"
    assert bundle["canonical"]["capital.retirement_regime_combined_retirement_only"] == 1.0
    assert bundle["canonical"]["capital.retirement_regime_pension_exact"] == 0.0
    assert bundle["canonical"]["capital.retirement_regime_not_surfaced"] == 0.0
    assert bundle["diagnostics"]["canonical_count"] > 0


def test_action_gated_bundle_views_only_override_capital_structure_when_context_matches(monkeypatch):
    monkeypatch.setenv("AXIOM_ENABLE_RUNTIME_FEATURE_ADAPTER", "1")
    monkeypatch.setenv(
        "AXIOM_RUNTIME_FEATURE_ADAPTER_RULES",
        "normalized_net_debt,normalized_available_liquidity",
    )
    snapshot = {
        "features": {
            "capital_structure.net_debt": {"value": 500.0, "support_mode": "exact"},
            "capital_structure.net_debt_normalized": {"value": 420.0, "support_mode": "exact"},
            "liquidity.available_for_actions": {"value": 120.0, "support_mode": "exact"},
            "liquidity.available_liquidity_normalized": {"value": 135.0, "support_mode": "exact"},
        }
    }

    generic_bundle = build_model_feature_bundle(snapshot)
    capstruct_bundle = build_model_feature_bundle(
        snapshot,
        action_id="capital_structure.refinancing",
        action_type="capital_structure",
    )

    assert generic_bundle["views"]["mechanism"]["capital_structure.net_debt"]["value"] == 500.0
    assert generic_bundle["views"]["mechanism"]["liquidity.available_for_actions"]["value"] == 120.0
    assert capstruct_bundle["views"]["mechanism"]["capital_structure.net_debt"]["value"] == 420.0
    assert capstruct_bundle["views"]["mechanism"]["liquidity.available_for_actions"]["value"] == 135.0


