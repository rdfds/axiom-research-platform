from __future__ import annotations

from src.causal_feature_contract import (
    build_contract_feature_map,
    canonicalize_feature_name,
    normalize_feature_value,
    resolve_mapping_value,
)


def test_contract_resolves_legacy_aliases_and_param_overrides():
    feature_source = {
        "base_market_cap": 2_500_000_000.0,
        "base_total_debt": 1_100_000_000.0,
        "base_net_debt": 750_000_000.0,
        "base_leverage": 2.5,
        "base_net_pension_liability": 120_000_000.0,
        "base_combined_retirement_liability": 150_000_000.0,
        "base_debt_like_obligations_including_retirement": 1_250_000_000.0,
        "base_net_debt_including_retirement": 900_000_000.0,
        "base_gross_leverage_including_retirement": 3.2,
        "base_net_leverage_including_retirement": 2.7,
        "base_retirement_obligation_regime": "combined_retirement_only",
        "base_available_liquidity": 350_000_000.0,
        "macro_rate_10y": 4.58,
        "macro_rate_2y": 4.25,
        "macro_sofr": 4.49,
        "macro_fed_funds_effective": 4.33,
        "macro_real_gdp_growth_yoy": 0.028,
        "macro_ig_oas": 1.05,
        "macro_hy_oas": 3.50,
        "macro_vix": 18.0,
    }

    out = build_contract_feature_map(
        feature_source,
        params={
            "size_pct_market_cap": 0.10,
            "funding_mix": {"cash": 0.25, "debt": 0.75, "equity": 0.0},
        },
        regime={"credit_regime": "tight", "vol_regime": "high"},
    )

    assert out["scale.market_cap"] == 2_500_000_000.0
    assert out["capital.total_debt"] == 1_100_000_000.0
    assert out["capital.net_debt"] == 750_000_000.0
    assert out["capital.net_leverage"] == 2.5
    assert out["capital.net_pension_liability"] == 120_000_000.0
    assert out["capital.combined_retirement_liability"] == 150_000_000.0
    assert out["capital.debt_like_obligations_including_retirement"] == 1_250_000_000.0
    assert out["capital.net_debt_including_retirement"] == 900_000_000.0
    assert out["capital.gross_leverage_including_retirement"] == 3.2
    assert out["capital.net_leverage_including_retirement"] == 2.7
    assert out["capital.retirement_regime_combined_retirement_only"] == 1.0
    assert out["capital.retirement_regime_pension_exact"] == 0.0
    assert out["capital.retirement_regime_defined_contribution_only"] == 0.0
    assert out["liquidity.available_liquidity"] == 350_000_000.0
    assert out["macro.fed_funds_effective"] == 4.33
    assert out["macro.real_gdp_growth_yoy"] == 0.028
    assert out["action.size_absolute_usd"] == 250_000_000.0
    assert out["action.funding_mix_cash"] == 0.25
    assert out["action.funding_mix_debt"] == 0.75
    assert out["regime.credit_tight"] == 1.0
    assert out["regime.vol_high"] == 1.0


def test_contract_helpers_support_canonical_and_legacy_names():
    mapping = {"macro.ust_10y_yield": {"value": 4.58}}

    assert canonicalize_feature_name("base_market_cap") == "scale.market_cap"
    assert resolve_mapping_value(mapping, "macro_rate_10y") == 4.58
    assert normalize_feature_value("base_market_cap", 2_500_000_000.0) is not None
    assert normalize_feature_value("scale.market_cap", 2_500_000_000.0) == normalize_feature_value(
        "base_market_cap",
        2_500_000_000.0,
    )
    assert normalize_feature_value(
        "base_combined_retirement_liability",
        150_000_000.0,
    ) == normalize_feature_value("capital.combined_retirement_liability", 150_000_000.0)
    assert normalize_feature_value(
        "base_net_debt_including_retirement",
        900_000_000.0,
    ) == normalize_feature_value("capital.net_debt_including_retirement", 900_000_000.0)
