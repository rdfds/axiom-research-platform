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


