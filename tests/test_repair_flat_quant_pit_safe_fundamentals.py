from pathlib import Path

import pandas as pd

from scripts import repair_flat_quant_pit_safe_fundamentals as pit_safe


def test_overlay_pit_safe_fundamentals_replaces_overlayed_metrics(monkeypatch, tmp_path: Path):
    df = pd.DataFrame(
        [
            {
                "company_id": "1",
                "company_name": "Example Co",
                "as_of_time": "2024-12-31T00:00:00+00:00",
                "market__enterprise_value__value": 900.0,
                "market__enterprise_value__support_mode": "exact",
                "market__market_cap_provider_direct__value": 1000.0,
                "market__market_cap_provider_direct__support_mode": "exact",
                "capital_structure__debt_like_obligations_normalized__value": 0.0,
                "capital_structure__debt_like_obligations_normalized__support_mode": "exact",
                "operating__ebitda_margin_ttm__value": 0.55,
                "operating__ebitda_margin_ttm__support_mode": "exact",
                "market__ev_ebitda__value": 3.0,
                "market__ev_ebitda__support_mode": "exact",
                "market__fcf_yield__value": 0.01,
                "market__fcf_yield__support_mode": "exact",
                "operating__fcf_conversion__value": 0.02,
                "operating__fcf_conversion__support_mode": "exact",
            }
        ]
    )

    entity_identifier_path = tmp_path / "entity_identifier.parquet"
    raw_timeseries_path = tmp_path / "raw_timeseries.parquet"
    pd.DataFrame(
        [
            {
                "entity_id": "1",
                "identifier_type": "permno",
                "identifier_value": "10001",
            }
        ]
    ).to_parquet(entity_identifier_path, index=False)
    pd.DataFrame(
        [
            {
                "entity_id": "10001",
                "trade_date": "2024-12-31",
                "series_type": "price",
                "close": 8.0,
            }
        ]
    ).to_parquet(raw_timeseries_path, index=False)

    monkeypatch.setattr(pit_safe.core, "_load_companyfacts", lambda _path: {"dummy": True})

    def fake_build(metric_name, _companyfacts, _as_of_date):
        if metric_name == "operating.revenue_ttm_provider_direct":
            return 200.0, "exact", None, None, None
        if metric_name == "operating.ebitda_ltm_provider_direct":
            return 40.0, "exact", None, None, None
        if metric_name == "liquidity.cash_and_short_term_investments_provider_direct":
            return 120.0, "exact", None, None, None
        raise AssertionError(metric_name)

    monkeypatch.setattr(pit_safe.core, "_build_sec_core_metric", fake_build)
    monkeypatch.setattr(
        pit_safe.market_macro,
        "_latest_shares_outstanding",
        lambda _companyfacts, _as_of_date: (80.0, {"end": "2024-12-31", "formula": "latest_shares"}),
    )
    monkeypatch.setattr(
        pit_safe.cashflow,
        "_repairable_fcf_inputs",
        lambda *, companyfacts, as_of_date: (20.0, 50.0, 30.0, {"formula": "ocf-capex"}),
    )
    monkeypatch.setattr(pit_safe, "_sec_cash_direct_metric", lambda _companyfacts, _as_of_date: (120.0, "exact"))
    monkeypatch.setattr(
        pit_safe,
        "_sec_debt_like_metrics",
        lambda _companyfacts, _as_of_date: {
            "debt_like": 0.0,
            "debt_like_support": "exact",
            "debt_like_fallback": "sec_total_debt_plus_0_absent_lease",
        },
    )

    repaired = pit_safe.overlay_pit_safe_fundamentals(
        df,
        tmp_path,
        entity_identifier_path=entity_identifier_path,
        raw_timeseries_path=raw_timeseries_path,
    )
    row = repaired.iloc[0]

    assert row["operating__revenue_ttm_sec__value"] == 200.0
    assert row["operating__ebitda_ttm_sec__value"] == 40.0
    assert row["operating__free_cash_flow_ttm_sec__value"] == 20.0
    assert row["liquidity__cash_and_short_term_investments_sec__value"] == 120.0
    assert row["liquidity__cash_and_short_term_investments_provider_direct__value"] == 120.0
    assert row["liquidity__cash_and_short_term_investments_provider_direct__support_mode"] == "exact"
    assert row["market__market_cap_pit_safe__value"] == 640.0
    assert row["market__market_cap_provider_direct__value"] == 640.0
    assert row["market__market_cap_provider_direct__support_mode"] == "exact"
    assert row["operating__ebitda_margin_ttm__value"] == 0.2
    assert row["operating__ebitda_margin_ttm__support_mode"] == "exact"
    assert row["market__enterprise_value__value"] == 520.0
    assert row["market__enterprise_value__support_mode"] == "exact"
    assert row["market__enterprise_value__fallback_used"] == "pit_market_cap_plus_debt_like_minus_sec_cash_sti"
    assert row["market__ev_ebitda__value"] == 13.0
    assert row["market__ev_ebitda__support_mode"] == "exact"
    assert row["market__fcf_yield__value"] == 0.03125
    assert row["market__fcf_yield__support_mode"] == "exact"
    assert row["operating__fcf_conversion__value"] == 0.5


def test_overlay_promotes_cash_only_to_exact_when_no_sti_concepts_exist(monkeypatch, tmp_path: Path):
    df = pd.DataFrame(
        [
            {
                "company_id": "2",
                "company_name": "Cash Only Co",
                "as_of_time": "2024-12-31T00:00:00+00:00",
                "capital_structure__debt_like_obligations_normalized__value": 50.0,
                "capital_structure__debt_like_obligations_normalized__support_mode": "exact",
                "liquidity__marketable_securities_sec_exact__value": None,
                "liquidity__marketable_securities_sec_exact__support_mode": "unsupported",
            }
        ]
    )

    entity_identifier_path = tmp_path / "entity_identifier.parquet"
    raw_timeseries_path = tmp_path / "raw_timeseries.parquet"
    pd.DataFrame(
        [{"entity_id": "2", "identifier_type": "permno", "identifier_value": "20002"}]
    ).to_parquet(entity_identifier_path, index=False)
    pd.DataFrame(
        [{"entity_id": "20002", "trade_date": "2024-12-31", "series_type": "price", "close": 10.0}]
    ).to_parquet(raw_timeseries_path, index=False)

    monkeypatch.setattr(pit_safe.core, "_load_companyfacts", lambda _path: {"dummy": True})

    def fake_build(metric_name, _companyfacts, _as_of_date):
        if metric_name == "operating.revenue_ttm_provider_direct":
            return 100.0, "exact", None, None, None
        if metric_name == "operating.ebitda_ltm_provider_direct":
            return 20.0, "exact", None, None, None
        if metric_name == "liquidity.cash_and_short_term_investments_provider_direct":
            return 25.0, "proxy_missing_component", "cash_or_sti_component_missing", None, None
        raise AssertionError(metric_name)

    monkeypatch.setattr(pit_safe.core, "_build_sec_core_metric", fake_build)
    monkeypatch.setattr(
        pit_safe.market_macro,
        "_latest_shares_outstanding",
        lambda _companyfacts, _as_of_date: (10.0, {"end": "2024-12-31"}),
    )
    monkeypatch.setattr(
        pit_safe.cashflow,
        "_repairable_fcf_inputs",
        lambda *, companyfacts, as_of_date: (5.0, 9.0, 4.0, {"formula": "ocf-capex"}),
    )
    monkeypatch.setattr(pit_safe, "_sec_cash_direct_metric", lambda _companyfacts, _as_of_date: (25.0, "exact"))
    monkeypatch.setattr(pit_safe, "_can_promote_cash_only_exact", lambda _companyfacts, _as_of_date: True)
    monkeypatch.setattr(
        pit_safe,
        "_sec_debt_like_metrics",
        lambda _companyfacts, _as_of_date: {
            "debt_like": 50.0,
            "debt_like_support": "exact",
            "debt_like_fallback": "sec_total_debt_plus_0_absent_lease",
        },
    )

    repaired = pit_safe.overlay_pit_safe_fundamentals(
        df,
        tmp_path,
        entity_identifier_path=entity_identifier_path,
        raw_timeseries_path=raw_timeseries_path,
    )
    row = repaired.iloc[0]

    assert row["liquidity__cash_and_short_term_investments_provider_direct__value"] == 25.0
    assert row["liquidity__cash_and_short_term_investments_provider_direct__support_mode"] == "exact"
    assert row["market__enterprise_value__value"] == 125.0
    assert row["market__enterprise_value__support_mode"] == "exact"
    assert row["market__ev_ebitda__value"] == 6.25
    assert row["market__ev_ebitda__support_mode"] == "exact"
