from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.replay_snapshot_enrichment import enrich_snapshot_with_revenue_growth_inputs


def test_enrich_snapshot_with_revenue_growth_inputs_backfills_missing_metrics(tmp_path, monkeypatch):
    companyfacts_root = tmp_path / "companyfacts"
    companyfacts_root.mkdir()
    (companyfacts_root / "CIK0000123456.json").write_text("{}")

    def _fake_builders():
        def _load(path: Path):
            return {"ok": True}

        def _build(metric_name: str, companyfacts: dict, as_of_date: str):
            if metric_name == "operating.revenue_ttm_provider_direct":
                return 125.0, "exact", None, {"formula": "ttm"}, None
            if metric_name == "operating.revenue_ttm_lag_1y":
                return 100.0, "exact", None, {"formula": "ttm_prior"}, None
            if metric_name == "liquidity.cash_and_short_term_investments_provider_direct":
                return 40.0, "exact", None, {"formula": "cash"}, None
            if metric_name == "capital_structure.total_debt_provider_direct":
                return 30.0, "exact", None, {"formula": "debt"}, None
            raise AssertionError(metric_name)

        return _load, _build

    monkeypatch.setattr("src.replay_snapshot_enrichment._sec_metric_builders", _fake_builders)

    snapshot = {
        "company_id": "0000123456",
        "as_of_time": "2024-07-24T00:00:00+00:00",
        "features": {
            "operating.revenue_yoy_last_q": {
                "name": "operating.revenue_yoy_last_q",
                "value": None,
                "support_mode": "unsupported",
            }
        },
    }

    enriched, changed, summary = enrich_snapshot_with_revenue_growth_inputs(
        snapshot,
        companyfacts_root=companyfacts_root,
    )

    assert changed is True
    assert enriched["features"]["operating.revenue_ttm_provider_direct"]["value"] == 125.0
    assert enriched["features"]["operating.revenue_ttm_lag_1y"]["value"] == 100.0
    assert enriched["features"]["liquidity.cash_and_short_term_investments_provider_direct"]["value"] == 40.0
    assert enriched["features"]["capital_structure.total_debt_provider_direct"]["value"] == 30.0
    assert enriched["features"]["operating.revenue_ttm_provider_direct"]["support_mode"] == "exact"
    assert summary["metrics"]["operating.revenue_ttm_provider_direct"]["changed"] is True


def test_enrich_snapshot_with_revenue_growth_inputs_keeps_existing_supported_metrics(tmp_path, monkeypatch):
    companyfacts_root = tmp_path / "companyfacts"
    companyfacts_root.mkdir()
    (companyfacts_root / "CIK0000123456.json").write_text("{}")

    def _fake_builders():
        def _load(path: Path):
            return {"ok": True}

        def _build(metric_name: str, companyfacts: dict, as_of_date: str):
            raise AssertionError("existing supported metrics should not be rebuilt")

        return _load, _build

    monkeypatch.setattr("src.replay_snapshot_enrichment._sec_metric_builders", _fake_builders)

    snapshot = {
        "company_id": "0000123456",
        "as_of_time": "2024-07-24T00:00:00+00:00",
        "features": {
            "operating.revenue_ttm_provider_direct": {
                "name": "operating.revenue_ttm_provider_direct",
                "value": 125.0,
                "support_mode": "exact",
            },
            "operating.revenue_ttm_lag_1y": {
                "name": "operating.revenue_ttm_lag_1y",
                "value": 100.0,
                "support_mode": "proxy_missing_component",
            },
            "liquidity.cash_and_short_term_investments_provider_direct": {
                "name": "liquidity.cash_and_short_term_investments_provider_direct",
                "value": 40.0,
                "support_mode": "exact",
            },
            "capital_structure.total_debt_provider_direct": {
                "name": "capital_structure.total_debt_provider_direct",
                "value": 30.0,
                "support_mode": "exact",
            },
        },
    }

    enriched, changed, summary = enrich_snapshot_with_revenue_growth_inputs(
        snapshot,
        companyfacts_root=companyfacts_root,
    )

    assert changed is True
    assert enriched["features"]["operating.revenue_ttm_provider_direct"] == snapshot["features"]["operating.revenue_ttm_provider_direct"]
    assert enriched["features"]["operating.revenue_ttm_lag_1y"] == snapshot["features"]["operating.revenue_ttm_lag_1y"]
    assert summary["metrics"]["liquidity.cash"]["changed"] is True


def test_enrich_snapshot_with_revenue_growth_inputs_backfills_buyback_matching_inputs_without_companyfacts():
    snapshot = {
        "company_id": "0000002488",
        "as_of_time": "2024-09-02T00:00:00+00:00",
        "features": {
            "market.market_cap": {
                "name": "market.market_cap",
                "value": 2336.0,
                "support_mode": "exact",
                "confidence": 0.22,
                "fallback_used": "price*shares",
                "component_breakdown": {"price_observation_age_days": 33.0},
            },
            "market.enterprise_value": {
                "name": "market.enterprise_value",
                "value": 2333.0,
                "support_mode": "exact",
            },
            "market.ev_ebitda": {
                "name": "market.ev_ebitda",
                "value": 454.0,
                "support_mode": "proxy_missing_component",
                "component_breakdown": {
                    "ebitda_ttm": 5.0,
                    "reference_ev_ebitda": 60.0,
                    "reference_instrument": "AMD.OQ",
                },
            },
            "capital_structure.net_debt_normalized": {
                "name": "capital_structure.net_debt_normalized",
                "value": -2.0,
                "support_mode": "proxy_missing_component",
            },
            "capital_structure.total_debt": {
                "name": "capital_structure.total_debt",
                "value": 2.5,
                "support_mode": "proxy_missing_component",
            },
            "liquidity.cash": {
                "name": "liquidity.cash",
                "value": 4.1,
                "support_mode": "exact",
            },
            "operating.fcf_conversion": {
                "name": "operating.fcf_conversion",
                "value": 0.75,
                "support_mode": "proxy_missing_component",
            },
            "capital_structure.debt_due_0_12m": {
                "name": "capital_structure.debt_due_0_12m",
                "value": 0.0,
                "support_mode": "exact",
            },
            "capital_structure.debt_due_12_24m": {
                "name": "capital_structure.debt_due_12_24m",
                "value": 0.7,
                "support_mode": "exact",
            },
        },
    }

    enriched, changed, summary = enrich_snapshot_with_revenue_growth_inputs(
        snapshot,
        companyfacts_root=None,
    )

    assert changed is True
    features = enriched["features"]
    assert features["operating.ebitda_ltm_provider_direct"]["value"] == 5.0
    assert features["capital_structure.total_debt_provider_direct"]["value"] == 2.5
    assert features["liquidity.usable_cash"]["value"] == 4.1
    assert features["liquidity.available_liquidity_normalized"]["value"] == 4.1
    assert features["liquidity.available_for_actions"]["value"] == 4.1
    assert round(features["capital_structure.net_debt"]["value"], 8) == round(-1.6, 8)
    assert round(features["capital_structure.net_leverage"]["value"], 8) == round(-1.6 / 5.0, 8)
    assert features["capital_structure.debt_due_next_24m"]["value"] == 0.7
    assert features["liquidity.cash_and_short_term_investments_provider_direct"]["value"] == 4.1
    assert features["market.market_cap_provider_direct"]["value"] == 302.0
    assert features["market.enterprise_value"]["value"] == 300.0
    assert features["market.ev_ebitda"]["value"] == 60.0
    assert features["cash_flow.free_cash_flow_ttm"]["value"] == 3.75
    assert round(features["market.fcf_yield"]["value"], 8) == round(3.75 / 302.0, 8)
    assert "market.market_cap_provider_direct" in summary["metrics"]


def test_enrich_snapshot_with_revenue_growth_inputs_backfills_liquidity_proxies_from_companyfacts(tmp_path, monkeypatch):
    companyfacts_root = tmp_path / "companyfacts"
    companyfacts_root.mkdir()
    (companyfacts_root / "CIK0000002488.json").write_text("{}")

    def _fake_builders():
        def _load(path: Path):
            return {"ok": True}

        def _build(metric_name: str, companyfacts: dict, as_of_date: str):
            if metric_name == "operating.revenue_ttm_provider_direct":
                return None, "unsupported", "missing", None, None
            if metric_name == "operating.revenue_ttm_lag_1y":
                return None, "unsupported", "missing", None, None
            if metric_name == "liquidity.cash_and_short_term_investments_provider_direct":
                return 7_500_000_000.0, "exact", None, {"formula": "cash"}, None
            if metric_name == "capital_structure.total_debt_provider_direct":
                return 1_700_000_000.0, "exact", None, {"formula": "debt"}, None
            raise AssertionError(metric_name)

        return _load, _build

    monkeypatch.setattr("src.replay_snapshot_enrichment._sec_metric_builders", _fake_builders)

    snapshot = {
        "company_id": "0000002488",
        "as_of_time": "2024-09-02T00:00:00+00:00",
        "features": {
            "capital_structure.maturity_wall_ratio_24m": {
                "name": "capital_structure.maturity_wall_ratio_24m",
                "value": 0.4,
                "support_mode": "exact",
            },
            "capital_structure.total_debt": {
                "name": "capital_structure.total_debt",
                "value": 1_700_000_000.0,
                "support_mode": "proxy_missing_component",
            },
            "operating.ebitda_ltm_provider_direct": {
                "name": "operating.ebitda_ltm_provider_direct",
                "value": 5_100_000_000.0,
                "support_mode": "proxy_missing_component",
            },
            "liquidity.revolver_undrawn": {
                "name": "liquidity.revolver_undrawn",
                "value": None,
                "support_mode": "unsupported",
            },
        },
    }

    enriched, changed, summary = enrich_snapshot_with_revenue_growth_inputs(
        snapshot,
        companyfacts_root=companyfacts_root,
    )

    assert changed is True
    features = enriched["features"]
    assert features["liquidity.cash"]["value"] == 7_500_000_000.0
    assert features["liquidity.available_for_actions"]["value"] == 7_500_000_000.0
    assert features["capital_structure.net_debt"]["value"] == -5_800_000_000.0
    assert round(features["capital_structure.net_leverage"]["value"], 8) == round(-5_800_000_000.0 / 5_100_000_000.0, 8)
    assert summary["metrics"]["liquidity.available_for_actions"]["changed"] is True


def test_enrich_snapshot_with_revenue_growth_inputs_repairs_price_history_metrics_from_crsp(monkeypatch):
    trade_dates = pd.date_range("2024-05-01", "2024-08-30", freq="B", tz="UTC")
    price_history = pd.DataFrame(
        {
            "trade_date": trade_dates,
            "price": [100.0 + (idx * 0.4) + ((idx % 5) * 0.1) for idx in range(len(trade_dates))],
        }
    )

    def _fake_load_exact_price_history(**kwargs):
        return "61241", "crsp_daily_root", "/tmp/crsp_daily_root", price_history

    monkeypatch.setattr(
        "src.replay_snapshot_enrichment._load_exact_price_history",
        _fake_load_exact_price_history,
    )

    snapshot = {
        "company_id": "0000002488",
        "as_of_time": "2024-09-02T00:00:00+00:00",
        "features": {
            "market.volatility_90d": {
                "name": "market.volatility_90d",
                "value": None,
                "support_mode": "unsupported",
                "component_breakdown": {"median_observation_gap_days": 31.0},
                "quality_flags": ["low_frequency_price_history"],
            },
            "market.drawdown_90d": {
                "name": "market.drawdown_90d",
                "value": None,
                "support_mode": "unsupported",
                "component_breakdown": {"median_observation_gap_days": 31.0},
                "quality_flags": ["low_frequency_price_history"],
            },
        },
    }

    enriched, changed, summary = enrich_snapshot_with_revenue_growth_inputs(
        snapshot,
        companyfacts_root=None,
        entity_identifier_path="/tmp/entity_identifier.parquet",
        crsp_daily_root="/tmp/crsp_daily_root",
    )

    assert changed is True
    assert enriched["features"]["market.volatility_90d"]["value"] is not None
    assert enriched["features"]["market.volatility_90d"]["support_mode"] == "exact"
    assert enriched["features"]["market.volatility_90d"]["fallback_used"] == "crsp_daily_root_price_history"
    assert enriched["features"]["market.drawdown_90d"]["value"] is not None
    assert enriched["features"]["market.drawdown_90d"]["support_mode"] == "exact"
    assert summary["price_history_permno"] == "61241"
