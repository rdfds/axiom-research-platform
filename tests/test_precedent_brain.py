from __future__ import annotations

from datetime import datetime, timedelta
import gzip
import json
import os
import tempfile

import numpy as np
import pandas as pd

import src.pipeline.precedent_brain as precedent_brain
from src.pipeline.latent_regime_model import fit_latent_regime_kmeans, raw_feature_matrix_from_compacts
from src.pipeline.historical_stores import build_historical_stores_from_outcomes
from src.pipeline.precedent_brain import (
    _apply_matching_feature_transforms,
    _weighted_distance_profile,
    _weighted_state_similarity_v2,
    augment_precedent_state_vector_columns,
    build_precedent_pack_v2,
    build_precedent_retrieval_index,
)


def _hist_df(n: int = 40) -> pd.DataFrame:
    rows = []
    t0 = datetime(2018, 1, 1)
    for i in range(n):
        rows.append(
            {
                "company_id": f"{1000 + (i % 8):06d}",
                "action_type": "buyback",
                "action_subtype": "buyback",
                "action_date": t0 + timedelta(days=30 * i),
                "source_event_id": f"evt_{i}",
                "action_size": 50.0 + i,
                "base_market_cap": 1000.0 + 10.0 * i,
                "base_margin": 0.10 + 0.001 * i,
                "base_net_debt": 200.0 + 2.0 * i,
                "base_leverage": 2.0 + 0.02 * i,
                "base_revenue_ttm": 500.0 + 5.0 * i,
                "base_roic": 0.08 + 0.0005 * i,
                "base_fcf_margin": 0.09 + 0.0005 * i,
                "base_sector": "TECH" if i % 3 else "INDUSTRIALS",
                "base_pe": 15.0 + 0.1 * i,
                "base_ev_ebitda": 8.0 + 0.05 * i,
                "leverage_delta": -0.05 + 0.002 * i,
                "fcf_margin_delta": 0.01 + 0.001 * i,
                "outcome_pe_6m": -0.05 + 0.01 * i,
                "outcome_pe_12m": -0.10 + 0.015 * i,
                "outcome_ev_ebitda_6m": -0.04 + 0.009 * i,
                "outcome_ev_ebitda_12m": -0.08 + 0.012 * i,
                "credit_spread_change_1m": -5.0 + 0.4 * i,
                "credit_spread_change_6m": -12.0 + 0.9 * i,
                "credit_spread_change_12m": -20.0 + 1.4 * i,
                "credit_spread_change_24m": -35.0 + 2.4 * i,
                "rating_migration_1m": float((i % 3) - 1),
                "rating_migration_6m": float((i % 5) - 2),
                "rating_migration_12m": float((i % 7) - 3),
                "rating_migration_24m": float((i % 9) - 4),
                "macro_hy_oas": 3.0 + 0.1 * (i % 10),
                "macro_vix": 15.0 + float(i % 12),
            }
        )
    # Inject explicit tails.
    rows[0]["outcome_pe_12m"] = -0.95
    rows[-1]["outcome_pe_12m"] = 1.10
    return pd.DataFrame(rows)


def _candidate_features() -> dict:
    return {
        "market_cap": 1500.0,
        "ebitda_margin": 0.13,
        "leverage_net_debt_ebitda": 2.4,
        "revenue_ttm": 650.0,
        "roic": 0.09,
        "fcf_margin": 0.12,
        "sector": "TECH",
    }


def _raw_feature_record(value: object, *, support_mode: str = "historical_outcome_fallback", quality_flags=()) -> dict:
    record = {
        "value": value,
        "support_mode": support_mode,
    }
    flags = [str(flag) for flag in quality_flags if flag]
    if flags:
        record["quality_flags"] = flags
    return record


def test_estimate_action_scale_supports_action_size_key() -> None:
    scale = precedent_brain._estimate_action_scale({"action_size": 300.0}, 1500.0)
    assert scale == 0.2


def test_candidate_market_cap_handles_canonical_and_historical_units() -> None:
    assert precedent_brain._candidate_market_cap({"scale.market_cap": {"value": 43_000_000_000.0}}) == 43_000_000_000.0
    assert precedent_brain._candidate_market_cap({"base_market_cap": 1200.0}) == 1_200_000_000.0


def test_retrieval_index_normalizes_historical_market_cap_for_action_scale() -> None:
    hist = pd.DataFrame(
        [
            {
                "company_id": "001000",
                "ticker": "TEST",
                "action_type": "new_debt_issuance",
                "action_subtype": "new_debt_issuance",
                "action_date": datetime(2024, 1, 1),
                "action_size": 200_000_000.0,
                "base_market_cap": 1000.0,
            }
        ]
    )

    idx = build_precedent_retrieval_index(hist)

    assert idx.action_scale_arr[0] == 0.2
    assert idx.market_cap_bucket_quantiles[0] == 1_000_000_000.0


def test_augment_precedent_state_vector_columns_can_skip_historical_taxonomy_lookup(monkeypatch):
    hist = pd.DataFrame(
        [
            {
                "company_id": "001000",
                "ticker": "TEST",
                "action_type": "new_debt_issuance",
                "action_subtype": "new_debt_issuance",
                "action_date": datetime(2024, 1, 1),
                "base_revenue_ttm": 500.0,
                "base_ebitda_ttm": 100.0,
                "base_ev_ebitda": 10.0,
                "base_total_debt": 200.0,
                "base_cash": 50.0,
            }
        ]
    )
    monkeypatch.setenv("PRECEDENT_DISABLE_HISTORICAL_TAXONOMY_LOOKUP", "1")

    out = augment_precedent_state_vector_columns(hist)

    assert out.loc[0, "state_vector_v1.valuation_multiple"] == 10.0
    assert str(out.loc[0, "sector"] or "") == ""
    monkeypatch.delenv("PRECEDENT_DISABLE_HISTORICAL_TAXONOMY_LOOKUP", raising=False)


def test_augment_precedent_state_vector_columns_can_use_refinitiv_taxonomy_when_snapshot_lookup_disabled(monkeypatch):
    hist = pd.DataFrame(
        [
            {
                "company_id": "001000",
                "ticker": "TEST",
                "action_type": "new_debt_issuance",
                "action_subtype": "new_debt_issuance",
                "action_date": datetime(2024, 1, 1),
                "base_revenue_ttm": 500.0,
                "base_ebitda_ttm": 100.0,
                "base_ev_ebitda": 10.0,
                "base_total_debt": 200.0,
                "base_cash": 50.0,
            }
        ]
    )
    monkeypatch.setenv("PRECEDENT_DISABLE_HISTORICAL_TAXONOMY_LOOKUP", "1")
    monkeypatch.setattr(
        precedent_brain,
        "_load_refinitiv_taxonomy_lookup",
        lambda: {"TEST": ("Information Technology", "Communications Equipment")},
    )

    out = augment_precedent_state_vector_columns(hist)

    assert str(out.loc[0, "sector"] or "") == "Information Technology"
    assert str(out.loc[0, "subsector"] or "") == "Communications Equipment"
    monkeypatch.delenv("PRECEDENT_DISABLE_HISTORICAL_TAXONOMY_LOOKUP", raising=False)


def test_load_sec_ticker_cik_lookup_can_fallback_to_company_tickers_json(monkeypatch, tmp_path):
    json_path = tmp_path / "company_tickers.json"
    json_path.write_text(
        json.dumps(
            {
                "0": {
                    "cik_str": 886128,
                    "ticker": "FCEL",
                    "title": "FuelCell Energy, Inc.",
                }
            }
        )
    )
    monkeypatch.setattr(precedent_brain, "_SEC_TICKER_CIK_PATH", tmp_path / "missing.parquet")
    monkeypatch.setattr(precedent_brain, "_SEC_COMPANY_TICKERS_JSON_PATH", json_path)
    precedent_brain._load_sec_ticker_cik_lookup.cache_clear()

    lookup = precedent_brain._load_sec_ticker_cik_lookup()

    assert lookup["FCEL"] == "0000886128"
    precedent_brain._load_sec_ticker_cik_lookup.cache_clear()


def _state_vector_candidate_features(**overrides) -> dict:
    base = {
        "market_cap": 1000.0,
        "sector": "CONSUMER_STAPLES",
        "state_vector_v1.size_log_revenue": 10.0,
        "state_vector_v1.profitability": 0.20,
        "state_vector_v1.growth": 0.05,
        "state_vector_v1.gross_obligation_burden": 1.50,
        "state_vector_v1.net_obligation_burden": 1.00,
        "state_vector_v1.liquidity_flexibility": 2.00,
        "state_vector_v1.interest_coverage": 10.00,
        "state_vector_v1.valuation_multiple": 12.00,
        "state_vector_v1.cash_generation": 0.04,
        "state_vector_v1.market_stress": 0.20,
        "state_vector_v1.market_access": 0.80,
        "state_vector_v1.rates_level": 4.25,
        "state_vector_v1.credit_spread": 3.00,
    }
    base.update(overrides)
    return base


def _state_vector_hist_row(
    *,
    company_id: str,
    action_type: str,
    action_subtype: str,
    offset_days: int,
    ticker: str,
    **overrides,
) -> dict:
    row = {
        "company_id": company_id,
        "action_type": action_type,
        "action_subtype": action_subtype,
        "action_date": datetime(2020, 1, 1) + timedelta(days=offset_days),
        "source_event_id": f"evt_{company_id}_{offset_days}",
        "action_size": 25.0,
        "base_market_cap": 1000.0,
        "base_sector": "CONSUMER_STAPLES",
        "ticker": ticker,
        "macro_hy_oas": 3.0,
        "macro_vix": 18.0,
        "outcome_pe_6m": 0.05,
        "outcome_pe_12m": 0.08,
        "outcome_ev_ebitda_6m": 0.04,
        "outcome_ev_ebitda_12m": 0.06,
        "credit_spread_change_1m": -2.0,
        "credit_spread_change_6m": -5.0,
        "credit_spread_change_12m": -8.0,
        "credit_spread_change_24m": -10.0,
        "rating_migration_1m": 0.0,
        "rating_migration_6m": 0.0,
        "rating_migration_12m": 0.0,
        "rating_migration_24m": 0.0,
        "state_vector_v1.size_log_revenue": 10.0,
        "state_vector_v1.profitability": 0.20,
        "state_vector_v1.growth": 0.05,
        "state_vector_v1.gross_obligation_burden": 1.50,
        "state_vector_v1.net_obligation_burden": 1.00,
        "state_vector_v1.liquidity_flexibility": 2.00,
        "state_vector_v1.interest_coverage": 10.00,
        "state_vector_v1.valuation_multiple": 12.00,
        "state_vector_v1.cash_generation": 0.04,
        "state_vector_v1.market_stress": 0.20,
        "state_vector_v1.market_access": 0.80,
        "state_vector_v1.rates_level": 4.25,
        "state_vector_v1.credit_spread": 3.00,
    }
    row.update(overrides)
    return row


