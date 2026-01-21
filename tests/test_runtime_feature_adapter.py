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


