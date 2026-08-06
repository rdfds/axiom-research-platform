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


def test_attach_model_feature_bundle_exposes_candidate_and_dossier_views(monkeypatch):
    monkeypatch.setenv("AXIOM_ENABLE_RUNTIME_FEATURE_ADAPTER", "1")
    monkeypatch.setenv("AXIOM_RUNTIME_FEATURE_ADAPTER_RULES", "normalized_net_leverage")
    snapshot = {
        "features": {
            "capital_structure.net_leverage_normalized": {"value": 1.75, "support_mode": "exact"},
            "market.market_cap": {"value": 1000.0, "support_mode": "exact"},
            "operating.revenue_ttm": {"value": 900.0, "support_mode": "exact"},
        }
    }

    attached = attach_model_feature_bundle(snapshot)

    candidate_view = feature_view_from_snapshot(attached, view_name="candidate_generation")
    dossier_view = feature_view_from_snapshot(attached, view_name="dossier")
    precedent_view = feature_view_from_snapshot(attached, view_name="precedent")

    assert candidate_view["capital_structure.net_leverage"]["value"] == 1.75
    assert dossier_view["capital_structure.net_leverage"]["value"] == 1.75
    assert precedent_view["state_vector_v1.size_log_revenue"]["value"] == math.log10(900.0)
    assert "_model_feature_bundle" in attached


def test_build_model_feature_bundle_derives_state_vector_v1():
    snapshot = {
        "sector": "Consumer Staples",
        "subsector": "Household Products",
        "features": {
            "taxonomy.sector": {"value": "Consumer Staples", "support_mode": "exact"},
            "taxonomy.subsector": {"value": "Household Products", "support_mode": "exact"},
            "operating.revenue_ttm_provider_direct": {"value": 83_905.0, "support_mode": "exact"},
            "operating.revenue_ttm_lag_1y": {"value": 81_385.0, "support_mode": "exact"},
            "operating.ebitda_ltm_provider_direct": {"value": 21_497.0, "support_mode": "exact"},
            "operating.ebitda_margin_ttm": {"value": 0.2562, "support_mode": "exact"},
            "capital_structure.total_debt_provider_direct": {"value": 36_153.0, "support_mode": "exact"},
            "capital_structure.net_debt_normalized": {"value": 24_906.0, "support_mode": "exact"},
            "liquidity.cash_and_short_term_investments_provider_direct": {"value": 12_156.0, "support_mode": "exact"},
            "capital_structure.current_debt_statement_direct": {"value": 10_409.0, "support_mode": "exact"},
            "capital_structure.interest_expense_statement_direct": {"value": 705.0, "support_mode": "exact"},
            "market.market_cap_provider_direct": {"value": 394_823.0, "support_mode": "exact"},
            "capital_structure.lease_liabilities_sec_exact": {"value": 909.0, "support_mode": "exact"},
            "capital_structure.combined_retirement_liability": {
                "value": 2_884.0,
                "support_mode": "proxy_missing_component",
            },
            "cash_flow.free_cash_flow_ttm": {"value": 15_871.8846, "support_mode": "exact"},
            "market.fcf_yield": {"value": 0.0402, "support_mode": "exact"},
            "market.volatility_90d": {"value": 0.1513, "support_mode": "exact"},
            "market.drawdown_90d": {"value": -0.1038, "support_mode": "exact"},
            "market.credit_window_proxy": {"value": 0.88, "support_mode": "proxy_missing_component"},
            "market.equity_window_proxy": {"value": 0.77, "support_mode": "inferred"},
            "market.credit_spread_level": {"value": 0.0112, "support_mode": "proxy_missing_component"},
            "macro.fed_funds_effective": {"value": 4.33, "support_mode": "exact"},
            "macro.hy_oas": {"value": 2.92, "support_mode": "exact"},
            "capital_structure.retirement_obligation_regime": {"value": "pension_exact", "support_mode": "present"},
        },
    }

    bundle = build_model_feature_bundle(snapshot)
    state = bundle["state_vector_v1"]

    assert math.isclose(state["values"]["state_vector_v1.size_log_revenue"], math.log10(83_905.0))
    assert math.isclose(state["values"]["state_vector_v1.profitability"], 21_497.0 / 83_905.0)
    assert math.isclose(state["values"]["state_vector_v1.growth"], (83_905.0 / 81_385.0) - 1.0)
    assert math.isclose(state["values"]["state_vector_v1.gross_obligation_burden"], (36_153.0 + 909.0 + 2_884.0) / 21_497.0)
    assert math.isclose(state["values"]["state_vector_v1.net_obligation_burden"], (24_906.0 + 2_884.0) / 21_497.0)
    assert math.isclose(state["values"]["state_vector_v1.liquidity_flexibility"], 12_156.0 / 10_409.0)
    assert math.isclose(state["values"]["state_vector_v1.interest_coverage"], 21_497.0 / 705.0)
    assert math.isclose(
        state["values"]["state_vector_v1.valuation_multiple"],
        (394_823.0 + 36_153.0 + 909.0 - 12_156.0) / 21_497.0,
    )
    assert math.isclose(state["values"]["state_vector_v1.cash_generation"], 15_871.8846 / 394_823.0)
    assert math.isclose(
        state["values"]["state_vector_v1.market_stress"],
        (0.6 * 0.1513) + (0.4 * 0.1038),
    )
    assert math.isclose(
        state["values"]["state_vector_v1.market_access"],
        ((0.4 * 0.88) + (0.4 * 0.77) + (0.2 * (1.0 - (0.0112 / 0.08)))) / 1.0,
    )
    assert state["values"]["state_vector_v1.rates_level"] == 4.33
    assert state["values"]["state_vector_v1.credit_spread"] == 2.92
    assert state["meta"]["sector"] == "Consumer Staples"
    assert state["meta"]["subsector"] == "Household Products"
    assert state["meta"]["retirement_regime"] == "pension_exact"
    assert "state_vector_v1.credit_spread" in bundle["canonical"]


def test_state_vector_v1_uses_explicit_fallback_rules_for_liquidity_and_valuation():
    snapshot = {
        "features": {
            "operating.revenue_ttm_provider_direct": {"value": 5_000.0, "support_mode": "exact"},
            "operating.ebitda_ltm_provider_direct": {"value": 500.0, "support_mode": "exact"},
            "operating.ebitda_margin_ttm": {"value": 0.10, "support_mode": "exact"},
            "liquidity.cash_and_short_term_investments_provider_direct": {"value": 400.0, "support_mode": "exact"},
            "capital_structure.current_debt_statement_direct": {"value": 100.0, "support_mode": "exact"},
            "market.market_cap_provider_direct": {"value": 600.0, "support_mode": "exact"},
            "capital_structure.total_debt_provider_direct": {"value": 1_000.0, "support_mode": "exact"},
            "market.volatility_90d": {"value": 0.22, "support_mode": "exact"},
            "market.credit_window_proxy": {"value": 0.70, "support_mode": "proxy_missing_component"},
            "macro.fed_funds_effective": {"value": 4.33, "support_mode": "exact"},
            "macro.hy_oas": {"value": 3.10, "support_mode": "exact"},
        }
    }

    bundle = build_model_feature_bundle(snapshot)
    state = bundle["state_vector_v1"]
    liquidity_record = state["records"]["state_vector_v1.liquidity_flexibility"]
    valuation_record = state["records"]["state_vector_v1.valuation_multiple"]

    assert math.isclose(state["values"]["state_vector_v1.liquidity_flexibility"], 4.0)
    assert state["support"]["state_vector_v1.liquidity_flexibility"]["support_mode"] == "proxy_missing_component"
    assert "marketable_securities_missing_assumed_zero" in liquidity_record["quality_flags"]
    assert "revolver_undrawn_missing_assumed_zero" in liquidity_record["quality_flags"]
    assert "current_debt_fallback" in liquidity_record["quality_flags"]

    assert math.isclose(state["values"]["state_vector_v1.valuation_multiple"], 2.4)
    assert state["support"]["state_vector_v1.valuation_multiple"]["support_mode"] == "proxy_missing_component"
    assert "lease_liabilities_missing_assumed_zero" in valuation_record["quality_flags"]

    assert state["support"]["state_vector_v1.market_stress"]["support_mode"] == "proxy_missing_component"
    assert "drawdown_90d_missing" in state["records"]["state_vector_v1.market_stress"]["quality_flags"]
    assert state["support"]["state_vector_v1.market_access"]["support_mode"] == "proxy_missing_component"


def test_state_vector_v1_uses_direct_current_debt_when_canonical_current_debt_is_absent():
    snapshot = {
        "features": {
            "operating.revenue_ttm_provider_direct": {"value": 5_000.0, "support_mode": "exact"},
            "operating.ebitda_ltm_provider_direct": {"value": 500.0, "support_mode": "exact"},
            "liquidity.cash_and_short_term_investments_provider_direct": {"value": 400.0, "support_mode": "exact"},
            "capital_structure.current_debt_provider_direct": {"value": 100.0, "support_mode": "exact"},
            "market.market_cap_provider_direct": {"value": 600.0, "support_mode": "exact"},
            "capital_structure.total_debt_provider_direct": {"value": 1_000.0, "support_mode": "exact"},
            "market.volatility_90d": {"value": 0.22, "support_mode": "exact"},
            "market.credit_window_proxy": {"value": 0.70, "support_mode": "proxy_missing_component"},
            "macro.fed_funds_effective": {"value": 4.33, "support_mode": "exact"},
            "macro.hy_oas": {"value": 3.10, "support_mode": "exact"},
        }
    }

    bundle = build_model_feature_bundle(snapshot)
    state = bundle["state_vector_v1"]
    liquidity_record = state["records"]["state_vector_v1.liquidity_flexibility"]

    assert math.isclose(state["values"]["state_vector_v1.liquidity_flexibility"], 4.0)
    assert liquidity_record["component_breakdown"]["near_term_debt_source_metric"] == "capital_structure.current_debt_provider_direct"
    assert "current_debt_fallback" in (liquidity_record.get("quality_flags") or [])


def test_state_vector_v1_prefers_raw_growth_and_cash_generation_when_available():
    snapshot = {
        "features": {
            "operating.revenue_ttm_provider_direct": {"value": 120.0, "support_mode": "exact"},
            "operating.revenue_ttm_lag_1y": {"value": 100.0, "support_mode": "exact"},
            "operating.ebitda_ltm_provider_direct": {"value": 30.0, "support_mode": "exact"},
            "operating.revenue_yoy_last_q": {"value": 0.05, "support_mode": "exact"},
            "cash_flow.free_cash_flow_ttm": {"value": 12.0, "support_mode": "exact"},
            "market.market_cap_provider_direct": {"value": 80.0, "support_mode": "exact"},
            "macro.fed_funds_effective": {"value": 4.0, "support_mode": "exact"},
            "macro.hy_oas": {"value": 3.0, "support_mode": "exact"},
        }
    }

    state = build_model_feature_bundle(snapshot)["state_vector_v1"]

    assert math.isclose(state["values"]["state_vector_v1.growth"], 0.2)
    assert state["records"]["state_vector_v1.growth"]["component_breakdown"]["revenue_ttm_lag_1y"] == 100.0
    assert math.isclose(state["values"]["state_vector_v1.cash_generation"], 0.15)
    assert state["support"]["state_vector_v1.cash_generation"]["support_mode"] == "exact"


def test_state_vector_v1_uses_vix_fallback_for_market_stress_when_price_history_is_missing():
    snapshot = {
        "features": {
            "operating.revenue_ttm_provider_direct": {"value": 120.0, "support_mode": "exact"},
            "operating.ebitda_ltm_provider_direct": {"value": 30.0, "support_mode": "exact"},
            "market.vix": {"value": 24.0, "support_mode": "exact"},
            "macro.fed_funds_effective": {"value": 4.0, "support_mode": "exact"},
            "macro.hy_oas": {"value": 3.0, "support_mode": "exact"},
        }
    }

    state = build_model_feature_bundle(snapshot)["state_vector_v1"]
    record = state["records"]["state_vector_v1.market_stress"]

    assert math.isclose(state["values"]["state_vector_v1.market_stress"], 24.0 / 80.0)
    assert state["support"]["state_vector_v1.market_stress"]["support_mode"] == "exact"
    assert record["fallback_used"] == "market.vix"
    assert record["component_breakdown"]["market.vix"] == 24.0
    assert "market_stress_vix_fallback" in (record.get("quality_flags") or [])
