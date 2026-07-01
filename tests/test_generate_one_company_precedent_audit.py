from __future__ import annotations

import importlib.util
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


MODULE_PATH = Path("./scripts/generate_one_company_precedent_audit.py")
SPEC = importlib.util.spec_from_file_location("generate_one_company_precedent_audit", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_confidence_lines_include_coverage_and_action_score() -> None:
    lines = audit._confidence_lines(
        "Learned",
        {
            "calibration_confidence": 0.61,
            "confidence_label": "medium",
            "retrieval_tier": "exact",
            "out_of_sample_flag": False,
            "exact_match_count": 42,
            "minimum_exact_support": 5,
            "top_similarity_mean": 0.88,
            "top_weighted_feature_coverage": 0.97,
            "top_critical_feature_coverage": 0.91,
            "top_action_match_score": 1.0,
        },
    )
    joined = "\n".join(lines)
    assert "top-support critical coverage" in joined
    assert "top-support action match score" in joined


def test_action_params_from_outcome_row_carries_refinancing_family_hints() -> None:
    params = audit._action_params_from_outcome_row(
        {
            "normalized_action_id": "capital_structure.refinancing",
            "raw_action_subtype": "Revolver/Line >= 1 Yr.",
            "action_size": 180_000_000.0,
        }
    )
    assert params["amount_usd"] == 180_000_000.0
    assert params["source_action_subtype"] == "Revolver/Line >= 1 Yr."
    assert params["instrument_type"] == "revolver"


def test_match_explanation_lines_call_out_closest_and_gaps() -> None:
    target_values = {key: 0.0 for key in audit._STATE_VECTOR_V1_FEATURES}
    target_values["state_vector_v1.valuation_multiple"] = 60.0
    target_values["state_vector_v1.growth"] = 0.15
    target_values["state_vector_v1.market_stress"] = 0.40
    match_row = pd.Series(
        {
            "state_vector_v1.valuation_multiple": 58.0,
            "state_vector_v1.growth": 0.12,
            "state_vector_v1.market_stress": 0.05,
        }
    )
    feature_scales = {key: 1.0 for key in audit._STATE_VECTOR_V1_FEATURES}
    lines = audit._match_explanation_lines(
        action_id="capital_return.open_market_buyback",
        target_values=target_values,
        match_row=match_row,
        feature_scales=feature_scales,
    )
    joined = "\n".join(lines)
    assert "Why it matched" in joined
    assert "Main gaps" in joined
    assert "valuation multiple" in joined
    assert "market stress" in joined


def test_synthesized_snapshot_row_from_outcome_row_preserves_core_features() -> None:
    row = audit._synthesized_snapshot_row_from_outcome_row(
        {
            "action_size": 250000000.0,
            "base_revenue_ttm": 1000.0,
            "base_revenue_ttm_lag_1y": 900.0,
            "base_ebitda_ttm": 200.0,
            "base_margin": 0.2,
            "base_fcf_margin": 0.1,
            "base_total_debt": 300.0,
            "base_net_debt": 120.0,
            "base_cash": 180.0,
            "base_available_liquidity": 260.0,
            "base_current_debt": 40.0,
            "base_interest_expense": 20.0,
            "base_market_cap": 2500.0,
            "base_ev_ebitda": 12.5,
            "base_fcf_yield": 0.04,
            "base_volatility_30d": 0.2,
            "base_volatility_90d": 0.3,
            "base_drawdown_90d": -0.2,
            "base_momentum_60d": 0.08,
            "base_credit_spread_level": 0.035,
            "base_credit_window_proxy": 0.7,
            "base_equity_window_proxy": 0.8,
            "base_revenue_growth_yoy": 0.11,
            "macro_vix": 18.0,
            "macro_fed_funds_effective": 0.0525,
            "macro_hy_oas": 0.038,
            "macro_ig_oas": 0.012,
            "macro_real_gdp_growth_yoy": 0.021,
            "macro_sofr": 0.053,
            "macro_rate_10y": 0.041,
            "macro_rate_2y": 0.045,
            "sector": "Industrials",
            "subsector": "Electrical Equipment",
        },
        company_id="0000012345",
        as_of_time="2024-08-01T00:00:00+00:00",
        outcomes_path=Path("/tmp/mock_outcomes.parquet"),
    )

    features = row["features"]
    assert row["action_params"]["amount_usd"] == 250000000.0
    assert row["action_params"]["action_size"] == 250000000.0
    assert features["market.ev_ebitda"]["value"] == 12.5
    assert features["cash_flow.free_cash_flow_ttm"]["value"] == 100.0 * 1_000_000.0
    assert features["capital_structure.current_debt_provider_direct"]["value"] == 40.0 * 1_000_000.0
    assert features["capital_structure.debt_due_next_24m"]["value"] == 40.0 * 1_000_000.0
    assert features["capital_structure.debt_due_next_24m"]["support_mode"] == "proxy_missing_component"
    assert features["capital_structure.interest_coverage"]["value"] == 10.0
    assert features["market.enterprise_value"]["value"] == 2620.0 * 1_000_000.0
    assert features["market.vix"]["value"] == 18.0
    assert features["macro.fed_funds_effective"]["value"] == 0.0525
    assert features["macro.hy_oas"]["value"] == 0.038
    assert features["taxonomy.sector"]["value"] == "Industrials"
    assert features["operating.revenue_yoy_last_q"]["value"] == 0.11


def test_synthesized_snapshot_row_uses_direct_ticker_taxonomy_fallback(monkeypatch) -> None:
    monkeypatch.setenv("PRECEDENT_DISABLE_HISTORICAL_TAXONOMY_LOOKUP", "1")
    monkeypatch.setattr(audit, "_enrich_missing_historical_taxonomy", lambda df: df)
    monkeypatch.setattr(
        audit,
        "_historical_taxonomy_for_ticker",
        lambda ticker, allow_sec_identity_heuristics=False: {
            "taxonomy.sector": "Information Technology",
            "taxonomy.subsector": "Semiconductors",
        },
    )

    row = audit._synthesized_snapshot_row_from_outcome_row(
        {
            "ticker": "FLNC",
            "action_size": 400000000.0,
            "base_revenue_ttm": 1000.0,
            "base_ebitda_ttm": 100.0,
            "base_market_cap": 2500.0,
        },
        company_id="0001868941",
        as_of_time="2024-08-01T00:00:00+00:00",
        outcomes_path=Path("/tmp/mock_outcomes.parquet"),
    )

    assert row["features"]["taxonomy.sector"]["value"] == "Information Technology"
    assert row["features"]["taxonomy.subsector"]["value"] == "Semiconductors"


