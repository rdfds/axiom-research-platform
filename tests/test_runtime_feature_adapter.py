from __future__ import annotations

from src.runtime_feature_adapter import adapt_snapshot, resolve_feature_record, resolve_feature_value


def test_adapter_defaults_to_disabled(monkeypatch):
    monkeypatch.delenv("AXIOM_ENABLE_RUNTIME_FEATURE_ADAPTER", raising=False)
    features = {
        "capital_structure.net_debt": {"value": 500.0, "support_mode": "exact"},
        "capital_structure.net_debt_normalized": {"value": 420.0, "support_mode": "exact"},
    }

    assert resolve_feature_value(features, "capital_structure.net_debt") == 500.0


def test_normalized_leverage_and_capital_structure_liquidity_override_legacy(monkeypatch):
    monkeypatch.setenv("AXIOM_ENABLE_RUNTIME_FEATURE_ADAPTER", "1")
    monkeypatch.setenv(
        "AXIOM_RUNTIME_FEATURE_ADAPTER_RULES",
        "normalized_net_leverage,normalized_available_liquidity",
    )
    features = {
        "capital_structure.net_leverage": {"value": 5.0, "support_mode": "exact"},
        "capital_structure.net_leverage_normalized": {"value": 2.75, "support_mode": "exact"},
        "liquidity.available_for_actions": {"value": 10.0, "support_mode": "exact"},
        "liquidity.available_liquidity_normalized": {"value": 125.0, "support_mode": "exact"},
    }

    assert resolve_feature_value(features, "capital_structure.net_leverage") == 2.75
    assert resolve_feature_value(
        features,
        "liquidity.available_for_actions",
        action_family="capital_structure",
        action_id="capital_structure.refinancing",
    ) == 125.0


def test_enabled_adapter_defaults_to_leverage_only_profile(monkeypatch):
    monkeypatch.setenv("AXIOM_ENABLE_RUNTIME_FEATURE_ADAPTER", "1")
    monkeypatch.delenv("AXIOM_RUNTIME_FEATURE_ADAPTER_RULES", raising=False)
    monkeypatch.delenv("AXIOM_RUNTIME_FEATURE_ADAPTER_PROFILE", raising=False)
    features = {
        "capital_structure.net_leverage_normalized": {"value": 2.75, "support_mode": "exact"},
        "liquidity.available_liquidity_normalized": {"value": 125.0, "support_mode": "exact"},
    }

    assert resolve_feature_value(features, "capital_structure.net_leverage") == 2.75
    assert resolve_feature_value(features, "liquidity.available_for_actions") is None

    _, diagnostics = adapt_snapshot({"features": features})
    assert diagnostics["profile"] == "leverage_only"
    assert diagnostics["allowed_rules"] == [
        "normalized_gross_leverage",
        "normalized_net_leverage",
    ]


def test_unsupported_normalized_nodes_fall_back_to_legacy(monkeypatch):
    monkeypatch.setenv("AXIOM_ENABLE_RUNTIME_FEATURE_ADAPTER", "1")
    monkeypatch.setenv("AXIOM_RUNTIME_FEATURE_ADAPTER_RULES", "normalized_net_leverage,normalized_operating_earnings_fill")
    features = {
        "capital_structure.net_leverage": {"value": 4.25, "support_mode": "exact"},
        "capital_structure.net_leverage_normalized": {
            "value": 1.5,
            "support_mode": "unsupported",
            "applicability_status": "unsupported",
            "quality_flags": ["unsupported_metric"],
        },
        "operating.ebitda_ttm": {"value": 180.0, "support_mode": "exact"},
        "operating.operating_earnings_normalized": {
            "value": None,
            "support_mode": "unsupported",
            "applicability_status": "unsupported",
        },
    }

    assert resolve_feature_value(features, "capital_structure.net_leverage") == 4.25
    assert resolve_feature_value(features, "operating.ebitda_ttm") == 180.0


def test_proxy_normalized_nodes_do_not_override_exact_legacy(monkeypatch):
    monkeypatch.setenv("AXIOM_ENABLE_RUNTIME_FEATURE_ADAPTER", "1")
    monkeypatch.setenv("AXIOM_RUNTIME_FEATURE_ADAPTER_RULES", "normalized_net_debt,normalized_available_liquidity")
    features = {
        "capital_structure.net_debt": {"value": 500.0, "support_mode": "exact"},
        "capital_structure.net_debt_normalized": {"value": 420.0, "support_mode": "proxy_missing_component"},
        "liquidity.available_for_actions": {"value": 140.0, "support_mode": "exact"},
        "liquidity.available_liquidity_normalized": {"value": 130.0, "support_mode": "proxy_missing_component"},
    }

    assert resolve_feature_value(features, "capital_structure.net_debt") == 500.0
    assert resolve_feature_value(features, "liquidity.available_for_actions") == 140.0


def test_rule_allowlist_limits_runtime_substitutions(monkeypatch):
    monkeypatch.setenv("AXIOM_ENABLE_RUNTIME_FEATURE_ADAPTER", "1")
    monkeypatch.setenv("AXIOM_RUNTIME_FEATURE_ADAPTER_RULES", "ust_10y_alias")
    features = {
        "capital_structure.net_debt": {"value": 500.0, "support_mode": "exact"},
        "capital_structure.net_debt_normalized": {"value": 420.0, "support_mode": "exact"},
        "macro.ust_10y_yield": {"value": 4.58, "support_mode": "exact"},
        "macro.ust_2y_yield": {"value": 4.25, "support_mode": "exact"},
    }

    assert resolve_feature_value(features, "capital_structure.net_debt") == 500.0
    assert resolve_feature_value(features, "macro.rate_10y") == 4.58
    assert resolve_feature_value(features, "macro.rate_2y") is None

    adapted, diagnostics = adapt_snapshot({"features": features})
    assert adapted["features"]["capital_structure.net_debt"]["value"] == 500.0
    assert diagnostics["allowed_rules"] == ["ust_10y_alias"]
    assert diagnostics["counts_by_target"] == {"macro.rate_10y": 1}


def test_debt_liquidity_aliases_require_capital_structure_context(monkeypatch):
    monkeypatch.setenv("AXIOM_ENABLE_RUNTIME_FEATURE_ADAPTER", "1")
    monkeypatch.setenv(
        "AXIOM_RUNTIME_FEATURE_ADAPTER_RULES",
        "normalized_net_debt,normalized_available_liquidity",
    )
    features = {
        "capital_structure.net_debt": {"value": 500.0, "support_mode": "exact"},
        "capital_structure.net_debt_normalized": {"value": 420.0, "support_mode": "exact"},
        "liquidity.available_for_actions": {"value": 120.0, "support_mode": "exact"},
        "liquidity.available_liquidity_normalized": {"value": 135.0, "support_mode": "exact"},
    }

    assert resolve_feature_value(features, "capital_structure.net_debt") == 500.0
    assert resolve_feature_value(features, "liquidity.available_for_actions") == 120.0
    assert resolve_feature_value(
        features,
        "capital_structure.net_debt",
        action_family="capital_return",
        action_id="capital_return.open_market_buyback",
    ) == 500.0
    assert resolve_feature_value(
        features,
        "capital_structure.net_debt",
        action_family="capital_structure",
        action_id="capital_structure.refinancing",
    ) == 420.0
    assert resolve_feature_value(
        features,
        "liquidity.available_for_actions",
        action_family="capital_structure",
        action_id="capital_structure.refinancing",
    ) == 135.0

    adapted_no_context, diagnostics_no_context = adapt_snapshot({"features": features})
    assert adapted_no_context["features"]["capital_structure.net_debt"]["value"] == 500.0
    assert diagnostics_no_context["replacement_count"] == 0

    adapted_capstruct, diagnostics_capstruct = adapt_snapshot(
        {"features": features},
        action_family="capital_structure",
        action_id="capital_structure.refinancing",
    )
    assert adapted_capstruct["features"]["capital_structure.net_debt"]["value"] == 420.0
    assert adapted_capstruct["features"]["liquidity.available_for_actions"]["value"] == 135.0
    assert diagnostics_capstruct["action_family"] == "capital_structure"


def test_banned_total_debt_substitution_never_happens(monkeypatch):
    monkeypatch.setenv("AXIOM_ENABLE_RUNTIME_FEATURE_ADAPTER", "1")
    features = {
        "capital_structure.total_debt": {"value": 400.0, "support_mode": "exact"},
        "capital_structure.debt_like_obligations_normalized": {"value": 900.0, "support_mode": "exact"},
    }

    assert resolve_feature_value(features, "capital_structure.total_debt") == 400.0
    record = resolve_feature_record(features, "capital_structure.total_debt")
    assert isinstance(record, dict)
    assert record["value"] == 400.0


def test_macro_and_credit_aliases_resolve_from_expected_sources(monkeypatch):
    monkeypatch.setenv("AXIOM_ENABLE_RUNTIME_FEATURE_ADAPTER", "1")
    monkeypatch.setenv(
        "AXIOM_RUNTIME_FEATURE_ADAPTER_RULES",
        "ust_10y_alias,ust_2y_alias,sofr_compatibility_fallback,credit_ig_alias,credit_hy_alias,pe_ratio_compatibility_alias",
    )
    features = {
        "macro.ust_10y_yield": {"value": 4.58, "support_mode": "exact"},
        "macro.ust_2y_yield": {"value": 4.25, "support_mode": "exact"},
        "macro.curve_2s10s": {"value": 0.33, "support_mode": "exact"},
        "macro.sofr": {"value": 4.49, "support_mode": "exact"},
        "macro.sofr_or_fed_funds": {"value": 4.40, "support_mode": "exact"},
        "macro.ig_oas": {"value": 1.02, "support_mode": "exact"},
        "macro.hy_oas": {"value": 3.44, "support_mode": "exact"},
        "market.pe_ratio": {"value": 19.3, "support_mode": "exact"},
    }

    assert resolve_feature_value(features, "macro.rate_10y") == 4.58
    assert resolve_feature_value(features, "macro.rate_2y") == 4.25
    assert resolve_feature_value(features, "macro.sofr") == 4.49
    assert resolve_feature_value(features, "market.ig_oas") == 1.02
    assert resolve_feature_value(features, "market.hy_oas") == 3.44
    assert resolve_feature_value(features, "market.pe") == 19.3


def test_macro_rate_2y_can_be_synthesized_from_curve(monkeypatch):
    monkeypatch.setenv("AXIOM_ENABLE_RUNTIME_FEATURE_ADAPTER", "1")
    monkeypatch.setenv("AXIOM_RUNTIME_FEATURE_ADAPTER_RULES", "ust_10y_minus_curve_2s10s")
    features = {
        "macro.ust_10y_yield": {"value": 4.58, "support_mode": "exact"},
        "macro.curve_2s10s": {"value": 0.33, "support_mode": "exact"},
    }

    assert resolve_feature_value(features, "macro.rate_2y") == 4.25


def test_adapter_does_not_use_fed_funds_to_impersonate_sofr(monkeypatch):
    monkeypatch.setenv("AXIOM_ENABLE_RUNTIME_FEATURE_ADAPTER", "1")
    features = {
        "macro.fed_funds_effective": {"value": 4.40, "support_mode": "exact"},
    }

    assert resolve_feature_value(features, "macro.sofr") is None


def test_adapt_snapshot_emits_replacement_diagnostics(monkeypatch):
    monkeypatch.setenv("AXIOM_ENABLE_RUNTIME_FEATURE_ADAPTER", "1")
    monkeypatch.setenv("AXIOM_RUNTIME_FEATURE_ADAPTER_RULES", "normalized_net_debt,ust_10y_alias")
    snapshot = {
        "company_id": "ABC",
        "as_of_time": "2026-02-28T00:00:00+00:00",
        "features": {
            "capital_structure.net_debt": {"value": 500.0, "support_mode": "exact"},
            "capital_structure.net_debt_normalized": {"value": 420.0, "support_mode": "exact"},
            "macro.ust_10y_yield": {"value": 4.58, "support_mode": "exact"},
        },
        "provenance": {},
    }

    adapted, diagnostics = adapt_snapshot(
        snapshot,
        action_family="capital_structure",
        action_id="capital_structure.refinancing",
    )

    assert adapted["features"]["capital_structure.net_debt"]["value"] == 420.0
    assert adapted["features"]["macro.rate_10y"]["value"] == 4.58
    assert diagnostics["replacement_count"] == 2
    assert diagnostics["profile"] == "custom"
    assert diagnostics["action_family"] == "capital_structure"
    assert diagnostics["counts_by_target"]["capital_structure.net_debt"] == 1
    assert diagnostics["counts_by_target"]["macro.rate_10y"] == 1
