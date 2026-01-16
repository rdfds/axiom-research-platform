from pathlib import Path

import pandas as pd

from scripts.build_peer_screen_view import build_peer_screen_view


def test_build_peer_screen_view_converts_units_and_keeps_support() -> None:
    df = pd.DataFrame(
        [
            {
                "company_id": "0000001111",
                "company_name": "Test Co",
                "liquidity__available_liquidity_normalized__value": 250_000_000.0,
                "liquidity__available_liquidity_normalized__support_mode": "exact",
                "capital_structure__debt_like_obligations_normalized__value": 1_000_000_000.0,
                "capital_structure__debt_like_obligations_normalized__support_mode": "exact",
                "capital_structure__net_debt_normalized__value": 750_000_000.0,
                "capital_structure__net_debt_normalized__support_mode": "exact",
                "capital_structure__gross_leverage_normalized__value": 3.2,
                "capital_structure__gross_leverage_normalized__support_mode": "exact",
                "capital_structure__net_leverage_normalized__value": 2.5,
                "capital_structure__net_leverage_normalized__support_mode": "exact",
                "operating__revenue_yoy_last_q__value": 0.125,
                "operating__revenue_yoy_last_q__support_mode": "exact",
                "operating__revenue_cagr_3y__value": 0.081,
                "operating__revenue_cagr_3y__support_mode": "exact",
                "operating__ebitda_margin_ttm__value": 0.223,
                "operating__ebitda_margin_ttm__support_mode": "exact",
                "market__ev_ebitda__value": 8.4,
                "market__ev_ebitda__support_mode": "exact",
                "market__fcf_yield__value": 0.047,
                "market__fcf_yield__support_mode": "exact",
                "operating__fcf_conversion__value": 0.51,
                "operating__fcf_conversion__support_mode": "exact",
                "capital_structure__rating_state__rating": "BB+",
                "capital_structure__rating_state__rating_support_mode": "exact",
                "capital_structure__rating_state__score__value": 11.0,
                "capital_structure__rating_state__score__support_mode": "exact",
                "market__credit_spread_level__value": 0.0325,
                "market__credit_spread_level__support_mode": "exact",
                "market__credit_truth_tier": "cds_exact",
                "market__credit_truth_tier_rank__value": 3.0,
                "market__credit_spread_percentile_2y__value": 0.84,
                "market__credit_spread_percentile_2y__support_mode": "exact",
            }
        ]
    )

    out = build_peer_screen_view(df)
    row = out.iloc[0]
    assert row["company_id"] == "0000001111"
    assert row["available_liquidity_usd"] == 250_000_000.0
    assert row["revenue_yoy_last_q_pct"] == 12.5
    assert row["ebitda_margin_ttm_pct"] == 22.3
    assert row["fcf_yield_pct"] == 4.7
    assert row["credit_spread_bps"] == 325.0
    assert row["credit_spread_percentile_2y_pct"] == 84.0
    assert row["credit_truth_tier"] == "cds_exact"
    assert row["credit_spread_support_mode"] == "exact"
