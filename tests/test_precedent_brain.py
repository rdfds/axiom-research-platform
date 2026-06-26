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


def test_similarity_retrieval_is_stable():
    df = _hist_df(40)
    pack1 = build_precedent_pack_v2(
        candidate_id="cand-1",
        run_id="run-1",
        company_id="001690",
        action_id="capital_return.open_market_buyback",
        action_subtype="open_market_buyback",
        action_params={"size_pct_market_cap": 0.05, "funding_mix": {"cash": 1.0}},
        candidate_features=_candidate_features(),
        candidate_regime={"credit_regime": "neutral", "risk_regime": "neutral", "vol_regime": "normal"},
        historical_df=df,
        top_k=20,
        min_k=10,
    )
    pack2 = build_precedent_pack_v2(
        candidate_id="cand-1",
        run_id="run-1",
        company_id="001690",
        action_id="capital_return.open_market_buyback",
        action_subtype="open_market_buyback",
        action_params={"size_pct_market_cap": 0.05, "funding_mix": {"cash": 1.0}},
        candidate_features=_candidate_features(),
        candidate_regime={"credit_regime": "neutral", "risk_regime": "neutral", "vol_regime": "normal"},
        historical_df=df,
        top_k=20,
        min_k=10,
    )
    ids1 = [c.precedent_id for c in pack1.retrieved_cohorts]
    ids2 = [c.precedent_id for c in pack2.retrieved_cohorts]
    assert ids1 == ids2
    scores1 = [round(s.score, 6) for s in pack1.similarity_scores]
    scores2 = [round(s.score, 6) for s in pack2.similarity_scores]
    assert scores1 == scores2


def test_outcome_distributions_and_regime_splits_present():
    pack = build_precedent_pack_v2(
        candidate_id="cand-2",
        run_id="run-2",
        company_id="001690",
        action_id="capital_return.open_market_buyback",
        action_subtype="open_market_buyback",
        action_params={"size_pct_market_cap": 0.05, "funding_mix": {"cash": 1.0}},
        candidate_features=_candidate_features(),
        candidate_regime={"credit_regime": "tight", "risk_regime": "risk_off", "vol_regime": "high"},
        historical_df=_hist_df(50),
        top_k=25,
        min_k=10,
    )
    h12 = pack.outcome_distributions.horizon_12m.valuation_multiple_change
    assert h12.sample_size > 0
    assert h12.p10 <= h12.p25 <= h12.median <= h12.p75 <= h12.p90
    assert pack.outcome_distributions.horizon_12m.credit_spread_change.sample_size > 0
    assert pack.outcome_distributions.horizon_12m.rating_migration.sample_size > 0
    assert len(pack.regime_splits) >= 1


def test_tail_detection_finds_extremes():
    pack = build_precedent_pack_v2(
        candidate_id="cand-3",
        run_id="run-3",
        company_id="001690",
        action_id="capital_return.open_market_buyback",
        action_subtype="open_market_buyback",
        action_params={"size_pct_market_cap": 0.05, "funding_mix": {"cash": 1.0}},
        candidate_features=_candidate_features(),
        candidate_regime={"credit_regime": "neutral", "risk_regime": "neutral", "vol_regime": "normal"},
        historical_df=_hist_df(60),
        top_k=30,
        min_k=10,
    )
    assert len(pack.tail_events) > 0
    explanations = [t.explanation for t in pack.tail_events]
    assert any("Bottom decile" in e for e in explanations)
    assert any("Top decile" in e for e in explanations)


def test_build_precedent_pack_v2_applies_second_stage_reranker(tmp_path):
    hist = pd.DataFrame(
        [
            _state_vector_hist_row(
                company_id="000101",
                action_type="capital_return.open_market_buyback",
                action_subtype="open_market_buyback",
                offset_days=30,
                ticker="VALA",
                **{
                    "sector": "TECH",
                    "state_vector_v1.size_log_revenue": 10.90,
                    "state_vector_v1.valuation_multiple": 58.0,
                    "state_vector_v1.cash_generation": 0.02,
                },
            ),
            _state_vector_hist_row(
                company_id="000102",
                action_type="capital_return.open_market_buyback",
                action_subtype="open_market_buyback",
                offset_days=60,
                ticker="SIZE",
                **{
                    "sector": "TECH",
                    "state_vector_v1.size_log_revenue": 10.00,
                    "state_vector_v1.valuation_multiple": 20.0,
                    "state_vector_v1.cash_generation": 0.02,
                },
            ),
        ]
    )
    candidate_features = _state_vector_candidate_features(
        sector="TECH",
        **{
            "state_vector_v1.size_log_revenue": 10.00,
            "state_vector_v1.valuation_multiple": 60.0,
            "state_vector_v1.cash_generation": 0.02,
        },
    )

    pack_base = build_precedent_pack_v2(
        candidate_id="cand-rerank-base",
        run_id="run-rerank-base",
        company_id="000999",
        action_id="capital_return.open_market_buyback",
        action_subtype="open_market_buyback",
        action_params={},
        candidate_features=candidate_features,
        candidate_regime={"credit_regime": "neutral", "risk_regime": "neutral", "vol_regime": "normal"},
        historical_df=hist,
        top_k=2,
        min_k=1,
    )
    base_scores = [score.score for score in pack_base.similarity_scores]

    payload_path = tmp_path / "precedent_distance_weights_v2.json"
    payload_path.write_text(
        json.dumps(
            {
                "scopes": {
                    "capital_return.open_market_buyback": {
                        "scope_key": "capital_return.open_market_buyback",
                        "use_in_runtime": True,
                        "default_enabled": True,
                        "second_stage_reranker": {
                            "feature_weights": {
                                "size_guardrail_similarity": 4.0,
                            },
                            "bias": 0.0,
                            "shortlist_size": 2,
                        },
                    }
                }
            }
        )
    )

    previous_path = os.environ.get("PRECEDENT_DISTANCE_V2_WEIGHTS_PATH")
    try:
        os.environ["PRECEDENT_DISTANCE_V2_WEIGHTS_PATH"] = str(payload_path)
        precedent_brain._PRECEDENT_DISTANCE_V2_WEIGHTS_CACHE.clear()
        pack_reranked = build_precedent_pack_v2(
            candidate_id="cand-rerank-live",
            run_id="run-rerank-live",
            company_id="000999",
            action_id="capital_return.open_market_buyback",
            action_subtype="open_market_buyback",
            action_params={},
            candidate_features=candidate_features,
            candidate_regime={"credit_regime": "neutral", "risk_regime": "neutral", "vol_regime": "normal"},
            historical_df=hist,
            top_k=2,
            min_k=1,
        )
    finally:
        if previous_path is None:
            os.environ.pop("PRECEDENT_DISTANCE_V2_WEIGHTS_PATH", None)
        else:
            os.environ["PRECEDENT_DISTANCE_V2_WEIGHTS_PATH"] = previous_path
        precedent_brain._PRECEDENT_DISTANCE_V2_WEIGHTS_CACHE.clear()

    reranked_scores = [score.score for score in pack_reranked.similarity_scores]
    assert pack_reranked.profiling["second_stage_reranker_applied"] is True
    assert reranked_scores != base_scores


def test_second_stage_reranker_feature_matrix_exposes_debt_compatibility_features():
    feature_names = list(precedent_brain._STATE_VECTOR_MATCHING_COLS)
    target_compact = {
        "state_vector_v1.size_log_revenue": 9.58,
        "state_vector_v1.profitability": 0.1188,
        "state_vector_v1.growth": 0.0633,
        "state_vector_v1.gross_obligation_burden": 2.4367,
        "state_vector_v1.net_obligation_burden": 1.6712,
        "state_vector_v1.liquidity_flexibility": 1.20,
        "state_vector_v1.interest_coverage": 2.7537,
        "state_vector_v1.valuation_multiple": 5.3863,
        "state_vector_v1.cash_generation": -0.0353,
        "state_vector_v1.market_stress": 0.1698,
        "state_vector_v1.market_access": 0.6293,
        "state_vector_v1.rates_level": 4.58,
        "state_vector_v1.credit_spread": 2.64,
    }
    good_match = dict(target_compact)
    good_match.update(
        {
            "state_vector_v1.profitability": 0.11,
            "state_vector_v1.cash_generation": -0.02,
            "state_vector_v1.market_access": 0.60,
            "state_vector_v1.market_stress": 0.19,
            "state_vector_v1.rates_level": 4.70,
            "state_vector_v1.credit_spread": 2.90,
        }
    )
    bad_match = dict(target_compact)
    bad_match.update(
        {
            "state_vector_v1.profitability": 0.22,
            "state_vector_v1.cash_generation": 0.05,
            "state_vector_v1.market_access": 0.88,
            "state_vector_v1.market_stress": 0.08,
            "state_vector_v1.rates_level": 0.13,
            "state_vector_v1.credit_spread": 5.96,
        }
    )
    candidate_vec = np.array([target_compact.get(name, np.nan) for name in feature_names], dtype=float)
    emb_raw = np.vstack(
        [
            np.array([good_match.get(name, np.nan) for name in feature_names], dtype=float),
            np.array([bad_match.get(name, np.nan) for name in feature_names], dtype=float),
        ]
    )

    payload = precedent_brain._second_stage_reranker_feature_matrix(
        emb_raw=emb_raw,
        candidate_vec_raw=candidate_vec,
        embedding_cols=feature_names,
        action_id="capital_structure.new_debt_issuance",
        action_subtype="new_debt_issuance",
        profile_version="weighted_distance_v2",
    )
    idx = {name: i for i, name in enumerate(payload["feature_names"])}
    matrix = np.asarray(payload["matrix"], dtype=float)

    assert matrix.shape[0] == 2
    assert matrix[0, idx["market_regime_similarity"]] > matrix[1, idx["market_regime_similarity"]]
    assert matrix[0, idx["borrower_quality_similarity"]] > matrix[1, idx["borrower_quality_similarity"]]
    assert matrix[0, idx["compatibility_penalty_factor"]] > matrix[1, idx["compatibility_penalty_factor"]]
    assert matrix[0, idx["debt_archetype_similarity"]] > matrix[1, idx["debt_archetype_similarity"]]
    assert matrix[0, idx["debt_archetype_gate"]] > matrix[1, idx["debt_archetype_gate"]]


def test_second_stage_reranker_feature_matrix_exposes_revolver_compatibility_features():
    feature_names = list(precedent_brain._STATE_VECTOR_MATCHING_COLS)
    target_compact = {
        "state_vector_v1.size_log_revenue": 9.58,
        "state_vector_v1.profitability": 0.09,
        "state_vector_v1.growth": 0.01,
        "state_vector_v1.gross_obligation_burden": 2.70,
        "state_vector_v1.net_obligation_burden": 1.95,
        "state_vector_v1.liquidity_flexibility": 0.52,
        "state_vector_v1.interest_coverage": 2.40,
        "state_vector_v1.valuation_multiple": 6.00,
        "state_vector_v1.cash_generation": -0.02,
        "state_vector_v1.market_stress": 0.27,
        "state_vector_v1.market_access": 0.58,
        "state_vector_v1.rates_level": 1.60,
        "state_vector_v1.credit_spread": 4.90,
    }
    stress_match = dict(target_compact)
    stress_match.update(
        {
            "state_vector_v1.profitability": 0.08,
            "state_vector_v1.cash_generation": -0.03,
            "state_vector_v1.liquidity_flexibility": 0.48,
            "state_vector_v1.market_stress": 0.28,
            "state_vector_v1.market_access": 0.56,
            "state_vector_v1.credit_spread": 5.05,
        }
    )
    routine_match = dict(target_compact)
    routine_match.update(
        {
            "state_vector_v1.profitability": 0.22,
            "state_vector_v1.cash_generation": 0.08,
            "state_vector_v1.gross_obligation_burden": 1.10,
            "state_vector_v1.net_obligation_burden": 0.60,
            "state_vector_v1.liquidity_flexibility": 3.10,
            "state_vector_v1.interest_coverage": 10.0,
            "state_vector_v1.market_stress": 0.16,
            "state_vector_v1.market_access": 0.84,
            "state_vector_v1.credit_spread": 2.70,
        }
    )
    candidate_vec = np.array([target_compact.get(name, np.nan) for name in feature_names], dtype=float)
    emb_raw = np.vstack(
        [
            np.array([stress_match.get(name, np.nan) for name in feature_names], dtype=float),
            np.array([routine_match.get(name, np.nan) for name in feature_names], dtype=float),
        ]
    )

    payload = precedent_brain._second_stage_reranker_feature_matrix(
        emb_raw=emb_raw,
        candidate_vec_raw=candidate_vec,
        embedding_cols=feature_names,
        action_id="capital_structure.revolver_draw_or_resize",
        action_subtype="revolver_draw_or_resize",
        profile_version="weighted_distance_v2",
    )
    idx = {name: i for i, name in enumerate(payload["feature_names"])}
    matrix = np.asarray(payload["matrix"], dtype=float)

    assert matrix.shape[0] == 2
    assert matrix[0, idx["stress_alignment_similarity"]] > matrix[1, idx["stress_alignment_similarity"]]
    assert matrix[0, idx["financing_pressure_similarity"]] > matrix[1, idx["financing_pressure_similarity"]]
    assert matrix[0, idx["compatibility_penalty_factor"]] > matrix[1, idx["compatibility_penalty_factor"]]
    assert matrix[0, idx["debt_archetype_similarity"]] > matrix[1, idx["debt_archetype_similarity"]]
    assert matrix[0, idx["debt_archetype_gate"]] > matrix[1, idx["debt_archetype_gate"]]


def test_build_precedent_pack_v2_preserves_debt_archetype_gate_through_reranking(tmp_path, monkeypatch):
    hist = pd.DataFrame(
        [
            _state_vector_hist_row(
                company_id="000701",
                action_type="new_debt_issuance",
                action_subtype="new_debt_issuance",
                offset_days=0,
                ticker="HEALTHY",
                **{
                    "normalized_action_family": "capital_structure",
                    "normalized_action_subfamily": "new_debt_issuance",
                    "normalized_action_id": "capital_structure.new_debt_issuance",
                    "base_sector": "Industrials",
                    "base_industry": "Commercial Services & Supplies",
                    "state_vector_v1.size_log_revenue": 9.58,
                    "state_vector_v1.profitability": 0.25,
                    "state_vector_v1.growth": 0.07,
                    "state_vector_v1.gross_obligation_burden": 1.0,
                    "state_vector_v1.net_obligation_burden": 0.5,
                    "state_vector_v1.liquidity_flexibility": 2.8,
                    "state_vector_v1.interest_coverage": 10.5,
                    "state_vector_v1.valuation_multiple": 14.0,
                    "state_vector_v1.cash_generation": 0.08,
                    "state_vector_v1.market_stress": 0.18,
                    "state_vector_v1.market_access": 0.86,
                    "state_vector_v1.rates_level": 4.58,
                    "state_vector_v1.credit_spread": 2.64,
                },
            ),
            _state_vector_hist_row(
                company_id="000702",
                action_type="new_debt_issuance",
                action_subtype="new_debt_issuance",
                offset_days=1,
                ticker="STRESSED",
                **{
                    "normalized_action_family": "capital_structure",
                    "normalized_action_subfamily": "new_debt_issuance",
                    "normalized_action_id": "capital_structure.new_debt_issuance",
                    "base_sector": "Industrials",
                    "base_industry": "Commercial Services & Supplies",
                    "state_vector_v1.size_log_revenue": 10.25,
                    "state_vector_v1.profitability": 0.10,
                    "state_vector_v1.growth": 0.03,
                    "state_vector_v1.gross_obligation_burden": 2.7,
                    "state_vector_v1.net_obligation_burden": 1.9,
                    "state_vector_v1.liquidity_flexibility": 1.6,
                    "state_vector_v1.interest_coverage": 2.4,
                    "state_vector_v1.valuation_multiple": 6.2,
                    "state_vector_v1.cash_generation": -0.03,
                    "state_vector_v1.market_stress": 0.18,
                    "state_vector_v1.market_access": 0.61,
                    "state_vector_v1.rates_level": 4.58,
                    "state_vector_v1.credit_spread": 2.64,
                },
            ),
        ]
    )
    candidate_features = {
        "operating.revenue_ttm_provider_direct": _raw_feature_record(3_802_000_000.0),
        "operating.ebitda_ltm_provider_direct": _raw_feature_record(451_600_000.0),
        "cash_flow.free_cash_flow_ttm": _raw_feature_record(-59_200_000.0),
        "capital_structure.total_debt_provider_direct": _raw_feature_record(1_100_400_000.0),
        "capital_structure.net_debt_normalized": _raw_feature_record(754_700_000.0),
        "liquidity.available_liquidity_normalized": _raw_feature_record(345_700_000.0),
        "capital_structure.current_debt_statement_direct": _raw_feature_record(800_000.0),
        "capital_structure.interest_expense_statement_direct": _raw_feature_record(164_000_000.0),
        "market.market_cap_provider_direct": _raw_feature_record(1_677_742_200.0),
        "market.ev_ebitda": _raw_feature_record(5.3863),
        "market.fcf_yield": _raw_feature_record(-0.0353),
        "market.credit_window_proxy": _raw_feature_record(0.8793),
        "market.equity_window_proxy": _raw_feature_record(0.2693),
        "market.credit_spread_level": _raw_feature_record(0.0121),
        "macro.fed_funds_effective": _raw_feature_record(4.58),
        "macro.hy_oas": _raw_feature_record(2.64),
        "taxonomy.sector": _raw_feature_record("Industrials"),
        "taxonomy.subsector": _raw_feature_record("Commercial Services & Supplies"),
    }

    payload_path = tmp_path / "precedent_distance_weights_v2.json"
    payload_path.write_text(
        json.dumps(
            {
                "scopes": {
                    "capital_structure.new_debt_issuance": {
                        "scope_key": "capital_structure.new_debt_issuance",
                        "use_in_runtime": True,
                        "default_enabled": True,
                        "second_stage_reranker": {
                            "feature_weights": {
                                "size_guardrail_similarity": 8.0,
                            },
                            "bias": 0.0,
                            "shortlist_size": 2,
                        },
                    }
                }
            }
        )
    )

    previous_path = os.environ.get("PRECEDENT_DISTANCE_V2_WEIGHTS_PATH")
    previous_version = os.environ.get("PRECEDENT_DISTANCE_PROFILE_VERSION")
    try:
        os.environ["PRECEDENT_DISTANCE_V2_WEIGHTS_PATH"] = str(payload_path)
        os.environ["PRECEDENT_DISTANCE_PROFILE_VERSION"] = "weighted_distance_v2"
        precedent_brain._PRECEDENT_DISTANCE_V2_WEIGHTS_CACHE.clear()
        pack = build_precedent_pack_v2(
            candidate_id="cand-debt-gate-reranker",
            run_id="run-debt-gate-reranker",
            company_id="001692",
            action_id="capital_structure.new_debt_issuance",
            action_subtype="new_debt_issuance",
            action_params={"amount_usd": 200_000_000.0},
            candidate_features=candidate_features,
            candidate_regime={"credit_regime": "neutral", "risk_regime": "neutral", "vol_regime": "normal"},
            historical_df=hist,
            top_k=2,
            min_k=1,
        )
    finally:
        if previous_path is None:
            os.environ.pop("PRECEDENT_DISTANCE_V2_WEIGHTS_PATH", None)
        else:
            os.environ["PRECEDENT_DISTANCE_V2_WEIGHTS_PATH"] = previous_path
        if previous_version is None:
            os.environ.pop("PRECEDENT_DISTANCE_PROFILE_VERSION", None)
        else:
            os.environ["PRECEDENT_DISTANCE_PROFILE_VERSION"] = previous_version
        precedent_brain._PRECEDENT_DISTANCE_V2_WEIGHTS_CACHE.clear()

    assert pack.profiling["second_stage_reranker_applied"] is True
    assert pack.retrieved_cohorts[0].company_id == "000702"


def test_build_precedent_pack_v2_routes_debt_support_lanes_before_healthy_context(tmp_path):
    hist = pd.DataFrame(
        [
            _state_vector_hist_row(
                company_id="000701",
                action_type="new_debt_issuance",
                action_subtype="new_debt_issuance",
                offset_days=0,
                ticker="HEALTHY",
                **{
                    "normalized_action_family": "capital_structure",
                    "normalized_action_subfamily": "new_debt_issuance",
                    "normalized_action_id": "capital_structure.new_debt_issuance",
                    "base_sector": "Industrials",
                    "base_industry": "Commercial Services & Supplies",
                    "state_vector_v1.size_log_revenue": 9.62,
                    "state_vector_v1.profitability": 0.22,
                    "state_vector_v1.growth": 0.07,
                    "state_vector_v1.gross_obligation_burden": 1.10,
                    "state_vector_v1.net_obligation_burden": 0.60,
                    "state_vector_v1.liquidity_flexibility": 3.10,
                    "state_vector_v1.interest_coverage": 10.0,
                    "state_vector_v1.valuation_multiple": 13.5,
                    "state_vector_v1.cash_generation": 0.08,
                    "state_vector_v1.market_stress": 0.16,
                    "state_vector_v1.market_access": 0.84,
                    "state_vector_v1.rates_level": 4.62,
                    "state_vector_v1.credit_spread": 2.70,
                },
            ),
            _state_vector_hist_row(
                company_id="000702",
                action_type="new_debt_issuance",
                action_subtype="new_debt_issuance",
                offset_days=1,
                ticker="PEERSTRESS",
                **{
                    "normalized_action_family": "capital_structure",
                    "normalized_action_subfamily": "new_debt_issuance",
                    "normalized_action_id": "capital_structure.new_debt_issuance",
                    "base_sector": "Industrials",
                    "base_industry": "Commercial Services & Supplies",
                    "state_vector_v1.size_log_revenue": 9.80,
                    "state_vector_v1.profitability": 0.10,
                    "state_vector_v1.growth": 0.02,
                    "state_vector_v1.gross_obligation_burden": 2.70,
                    "state_vector_v1.net_obligation_burden": 1.95,
                    "state_vector_v1.liquidity_flexibility": 1.50,
                    "state_vector_v1.interest_coverage": 2.40,
                    "state_vector_v1.valuation_multiple": 6.00,
                    "state_vector_v1.cash_generation": -0.02,
                    "state_vector_v1.market_stress": 0.18,
                    "state_vector_v1.market_access": 0.62,
                    "state_vector_v1.rates_level": 4.70,
                    "state_vector_v1.credit_spread": 2.90,
                },
            ),
            _state_vector_hist_row(
                company_id="001692",
                action_type="new_debt_issuance",
                action_subtype="new_debt_issuance",
                offset_days=2,
                ticker="SELFHIST",
                **{
                    "normalized_action_family": "capital_structure",
                    "normalized_action_subfamily": "new_debt_issuance",
                    "normalized_action_id": "capital_structure.new_debt_issuance",
                    "base_sector": "Industrials",
                    "base_industry": "Commercial Services & Supplies",
                    "state_vector_v1.size_log_revenue": 9.55,
                    "state_vector_v1.profitability": 0.09,
                    "state_vector_v1.growth": 0.01,
                    "state_vector_v1.gross_obligation_burden": 2.80,
                    "state_vector_v1.net_obligation_burden": 2.05,
                    "state_vector_v1.liquidity_flexibility": 1.30,
                    "state_vector_v1.interest_coverage": 2.20,
                    "state_vector_v1.valuation_multiple": 5.80,
                    "state_vector_v1.cash_generation": -0.03,
                    "state_vector_v1.market_stress": 0.19,
                    "state_vector_v1.market_access": 0.60,
                    "state_vector_v1.rates_level": 5.05,
                    "state_vector_v1.credit_spread": 3.05,
                },
            ),
        ]
    )
    candidate_features = {
        "operating.revenue_ttm_provider_direct": _raw_feature_record(3_802_000_000.0),
        "operating.ebitda_ltm_provider_direct": _raw_feature_record(451_600_000.0),
        "cash_flow.free_cash_flow_ttm": _raw_feature_record(-59_200_000.0),
        "capital_structure.total_debt_provider_direct": _raw_feature_record(1_100_400_000.0),
        "capital_structure.net_debt_normalized": _raw_feature_record(754_700_000.0),
        "liquidity.available_liquidity_normalized": _raw_feature_record(345_700_000.0),
        "capital_structure.current_debt_statement_direct": _raw_feature_record(800_000.0),
        "capital_structure.interest_expense_statement_direct": _raw_feature_record(164_000_000.0),
        "market.market_cap_provider_direct": _raw_feature_record(1_677_742_200.0),
        "market.ev_ebitda": _raw_feature_record(5.3863),
        "market.fcf_yield": _raw_feature_record(-0.0353),
        "market.credit_window_proxy": _raw_feature_record(0.8793),
        "market.equity_window_proxy": _raw_feature_record(0.2693),
        "market.credit_spread_level": _raw_feature_record(0.0121),
        "macro.fed_funds_effective": _raw_feature_record(4.58),
        "macro.hy_oas": _raw_feature_record(2.64),
        "taxonomy.sector": _raw_feature_record("Industrials"),
        "taxonomy.subsector": _raw_feature_record("Commercial Services & Supplies"),
    }

    payload_path = tmp_path / "precedent_distance_weights_v2.json"
    payload_path.write_text(
        json.dumps(
            {
                "scopes": {
                    "capital_structure.new_debt_issuance": {
                        "scope_key": "capital_structure.new_debt_issuance",
                        "use_in_runtime": True,
                        "default_enabled": True,
                        "second_stage_reranker": {
                            "feature_weights": {
                                "size_guardrail_similarity": 7.0,
                                "base_state_similarity": 0.5,
                            },
                            "bias": 0.0,
                            "shortlist_size": 3,
                        },
                    }
                }
            }
        )
    )

    previous_path = os.environ.get("PRECEDENT_DISTANCE_V2_WEIGHTS_PATH")
    previous_version = os.environ.get("PRECEDENT_DISTANCE_PROFILE_VERSION")
    try:
        os.environ["PRECEDENT_DISTANCE_V2_WEIGHTS_PATH"] = str(payload_path)
        os.environ["PRECEDENT_DISTANCE_PROFILE_VERSION"] = "weighted_distance_v2"
        precedent_brain._PRECEDENT_DISTANCE_V2_WEIGHTS_CACHE.clear()
        pack = build_precedent_pack_v2(
            candidate_id="cand-debt-routing",
            run_id="run-debt-routing",
            company_id="001692",
            action_id="capital_structure.new_debt_issuance",
            action_subtype="new_debt_issuance",
            action_params={"amount_usd": 200_000_000.0},
            candidate_features=candidate_features,
            candidate_regime={"credit_regime": "neutral", "risk_regime": "neutral", "vol_regime": "normal"},
            historical_df=hist,
            top_k=3,
            min_k=1,
        )
    finally:
        if previous_path is None:
            os.environ.pop("PRECEDENT_DISTANCE_V2_WEIGHTS_PATH", None)
        else:
            os.environ["PRECEDENT_DISTANCE_V2_WEIGHTS_PATH"] = previous_path
        if previous_version is None:
            os.environ.pop("PRECEDENT_DISTANCE_PROFILE_VERSION", None)
        else:
            os.environ["PRECEDENT_DISTANCE_PROFILE_VERSION"] = previous_version
        precedent_brain._PRECEDENT_DISTANCE_V2_WEIGHTS_CACHE.clear()

    ordered_company_ids = [case.company_id for case in pack.retrieved_cohorts]
    assert pack.profiling["debt_support_routing_applied"] is True
    assert ordered_company_ids[0] == "000702"
    assert ordered_company_ids[1] == "001692"
    assert "000701" not in ordered_company_ids[:2]


def test_build_precedent_pack_v2_routes_revolver_support_lanes_and_caps_company_repeats(tmp_path):
    hist = pd.DataFrame(
        [
            _state_vector_hist_row(
                company_id="000701",
                action_type="revolver_draw_or_resize",
                action_subtype="revolver_draw_or_resize",
                offset_days=0,
                ticker="ROUTINE",
                **{
                    "normalized_action_family": "capital_structure",
                    "normalized_action_subfamily": "revolver_draw_or_resize",
                    "normalized_action_id": "capital_structure.revolver_draw_or_resize",
                    "base_sector": "Industrials",
                    "base_industry": "Commercial Services & Supplies",
                    "state_vector_v1.size_log_revenue": 9.62,
                    "state_vector_v1.profitability": 0.22,
                    "state_vector_v1.growth": 0.07,
                    "state_vector_v1.gross_obligation_burden": 1.10,
                    "state_vector_v1.net_obligation_burden": 0.60,
                    "state_vector_v1.liquidity_flexibility": 3.10,
                    "state_vector_v1.interest_coverage": 10.0,
                    "state_vector_v1.valuation_multiple": 13.5,
                    "state_vector_v1.cash_generation": 0.08,
                    "state_vector_v1.market_stress": 0.16,
                    "state_vector_v1.market_access": 0.84,
                    "state_vector_v1.rates_level": 1.60,
                    "state_vector_v1.credit_spread": 2.70,
                },
            ),
            _state_vector_hist_row(
                company_id="000702",
                action_type="revolver_draw_or_resize",
                action_subtype="revolver_draw_or_resize",
                offset_days=1,
                ticker="PEERDRAW",
                **{
                    "normalized_action_family": "capital_structure",
                    "normalized_action_subfamily": "revolver_draw_or_resize",
                    "normalized_action_id": "capital_structure.revolver_draw_or_resize",
                    "base_sector": "Industrials",
                    "base_industry": "Commercial Services & Supplies",
                    "state_vector_v1.size_log_revenue": 9.80,
                    "state_vector_v1.profitability": 0.08,
                    "state_vector_v1.growth": 0.02,
                    "state_vector_v1.gross_obligation_burden": 2.70,
                    "state_vector_v1.net_obligation_burden": 1.95,
                    "state_vector_v1.liquidity_flexibility": 0.55,
                    "state_vector_v1.interest_coverage": 2.40,
                    "state_vector_v1.valuation_multiple": 6.00,
                    "state_vector_v1.cash_generation": -0.02,
                    "state_vector_v1.market_stress": 0.26,
                    "state_vector_v1.market_access": 0.58,
                    "state_vector_v1.rates_level": 1.55,
                    "state_vector_v1.credit_spread": 4.90,
                },
            ),
            _state_vector_hist_row(
                company_id="001692",
                action_type="revolver_draw_or_resize",
                action_subtype="revolver_draw_or_resize",
                offset_days=2,
                ticker="SELFDRAW1",
                **{
                    "normalized_action_family": "capital_structure",
                    "normalized_action_subfamily": "revolver_draw_or_resize",
                    "normalized_action_id": "capital_structure.revolver_draw_or_resize",
                    "base_sector": "Industrials",
                    "base_industry": "Commercial Services & Supplies",
                    "state_vector_v1.size_log_revenue": 9.55,
                    "state_vector_v1.profitability": 0.09,
                    "state_vector_v1.growth": 0.01,
                    "state_vector_v1.gross_obligation_burden": 2.80,
                    "state_vector_v1.net_obligation_burden": 2.05,
                    "state_vector_v1.liquidity_flexibility": 0.48,
                    "state_vector_v1.interest_coverage": 2.20,
                    "state_vector_v1.valuation_multiple": 5.80,
                    "state_vector_v1.cash_generation": -0.03,
                    "state_vector_v1.market_stress": 0.28,
                    "state_vector_v1.market_access": 0.56,
                    "state_vector_v1.rates_level": 1.70,
                    "state_vector_v1.credit_spread": 5.10,
                },
            ),
            _state_vector_hist_row(
                company_id="001692",
                action_type="revolver_draw_or_resize",
                action_subtype="revolver_draw_or_resize",
                offset_days=3,
                ticker="SELFDRAW2",
                **{
                    "normalized_action_family": "capital_structure",
                    "normalized_action_subfamily": "revolver_draw_or_resize",
                    "normalized_action_id": "capital_structure.revolver_draw_or_resize",
                    "base_sector": "Industrials",
                    "base_industry": "Commercial Services & Supplies",
                    "state_vector_v1.size_log_revenue": 9.58,
                    "state_vector_v1.profitability": 0.10,
                    "state_vector_v1.growth": 0.01,
                    "state_vector_v1.gross_obligation_burden": 2.75,
                    "state_vector_v1.net_obligation_burden": 2.00,
                    "state_vector_v1.liquidity_flexibility": 0.52,
                    "state_vector_v1.interest_coverage": 2.30,
                    "state_vector_v1.valuation_multiple": 5.90,
                    "state_vector_v1.cash_generation": -0.02,
                    "state_vector_v1.market_stress": 0.27,
                    "state_vector_v1.market_access": 0.57,
                    "state_vector_v1.rates_level": 1.68,
                    "state_vector_v1.credit_spread": 4.95,
                },
            ),
        ]
    )
    candidate_features = {
        "operating.revenue_ttm_provider_direct": _raw_feature_record(3_802_000_000.0),
        "operating.ebitda_ltm_provider_direct": _raw_feature_record(451_600_000.0),
        "cash_flow.free_cash_flow_ttm": _raw_feature_record(-59_200_000.0),
        "capital_structure.total_debt_provider_direct": _raw_feature_record(1_100_400_000.0),
        "capital_structure.net_debt_normalized": _raw_feature_record(754_700_000.0),
        "liquidity.available_liquidity_normalized": _raw_feature_record(145_700_000.0),
        "capital_structure.current_debt_statement_direct": _raw_feature_record(800_000_000.0),
        "capital_structure.interest_expense_statement_direct": _raw_feature_record(164_000_000.0),
        "market.market_cap_provider_direct": _raw_feature_record(1_677_742_200.0),
        "market.ev_ebitda": _raw_feature_record(5.3863),
        "market.fcf_yield": _raw_feature_record(-0.0353),
        "market.credit_window_proxy": _raw_feature_record(0.55),
        "market.equity_window_proxy": _raw_feature_record(0.18),
        "market.credit_spread_level": _raw_feature_record(0.049),
        "macro.fed_funds_effective": _raw_feature_record(1.60),
        "macro.hy_oas": _raw_feature_record(4.85),
        "taxonomy.sector": _raw_feature_record("Industrials"),
        "taxonomy.subsector": _raw_feature_record("Commercial Services & Supplies"),
    }

    previous_version = os.environ.get("PRECEDENT_DISTANCE_PROFILE_VERSION")
    try:
        os.environ["PRECEDENT_DISTANCE_PROFILE_VERSION"] = "weighted_distance_v2"
        precedent_brain._PRECEDENT_DISTANCE_V2_WEIGHTS_CACHE.clear()
        pack = build_precedent_pack_v2(
            candidate_id="cand-revolver-routing",
            run_id="run-revolver-routing",
            company_id="001692",
            action_id="capital_structure.revolver_draw_or_resize",
            action_subtype="revolver_draw_or_resize",
            action_params={"draw_amount_usd": 200_000_000.0},
            candidate_features=candidate_features,
            candidate_regime={"credit_regime": "neutral", "risk_regime": "neutral", "vol_regime": "normal"},
            historical_df=hist,
            top_k=3,
            min_k=1,
        )
    finally:
        if previous_version is None:
            os.environ.pop("PRECEDENT_DISTANCE_PROFILE_VERSION", None)
        else:
            os.environ["PRECEDENT_DISTANCE_PROFILE_VERSION"] = previous_version
        precedent_brain._PRECEDENT_DISTANCE_V2_WEIGHTS_CACHE.clear()

    ordered_company_ids = [case.company_id for case in pack.retrieved_cohorts]
    assert pack.profiling["debt_support_routing_applied"] is True
    assert pack.profiling["max_matches_per_company"] == 2
    assert ordered_company_ids[0] == "000702"
    assert ordered_company_ids.count("001692") <= 2


def test_build_precedent_pack_v2_applies_outcome_aware_reranker(tmp_path):
    hist = pd.DataFrame(
        [
            _state_vector_hist_row(
                company_id="000201",
                action_type="capital_return.open_market_buyback",
                action_subtype="open_market_buyback",
                offset_days=30,
                ticker="GOOD",
                **{
                    "sector": "TECH",
                    "state_vector_v1.size_log_revenue": 10.35,
                    "state_vector_v1.valuation_multiple": 48.0,
                    "state_vector_v1.cash_generation": 0.03,
                    "outcome_pe_6m": 0.45,
                    "outcome_pe_12m": 0.60,
                    "outcome_ev_ebitda_6m": 0.35,
                    "outcome_ev_ebitda_12m": 0.50,
                    "credit_spread_change_6m": -20.0,
                    "credit_spread_change_12m": -35.0,
                    "rating_migration_12m": 1.0,
                    "leverage_delta": -0.20,
                    "fcf_margin_delta": 0.03,
                },
            ),
            _state_vector_hist_row(
                company_id="000202",
                action_type="capital_return.open_market_buyback",
                action_subtype="open_market_buyback",
                offset_days=60,
                ticker="NEAR",
                **{
                    "sector": "TECH",
                    "state_vector_v1.size_log_revenue": 10.05,
                    "state_vector_v1.valuation_multiple": 50.0,
                    "state_vector_v1.cash_generation": 0.03,
                    "outcome_pe_6m": -0.10,
                    "outcome_pe_12m": -0.15,
                    "outcome_ev_ebitda_6m": -0.08,
                    "outcome_ev_ebitda_12m": -0.12,
                    "credit_spread_change_6m": 10.0,
                    "credit_spread_change_12m": 15.0,
                    "rating_migration_12m": -1.0,
                    "leverage_delta": 0.12,
                    "fcf_margin_delta": -0.01,
                },
            ),
        ]
    )
    candidate_features = _state_vector_candidate_features(
        sector="TECH",
        **{
            "state_vector_v1.size_log_revenue": 10.00,
            "state_vector_v1.valuation_multiple": 50.0,
            "state_vector_v1.cash_generation": 0.03,
        },
    )

    pack_base = build_precedent_pack_v2(
        candidate_id="cand-outcome-base",
        run_id="run-outcome-base",
        company_id="000999",
        action_id="capital_return.open_market_buyback",
        action_subtype="open_market_buyback",
        action_params={},
        candidate_features=candidate_features,
        candidate_regime={"credit_regime": "neutral", "risk_regime": "neutral", "vol_regime": "normal"},
        historical_df=hist,
        top_k=2,
        min_k=1,
    )
    base_ids = [score.precedent_id for score in pack_base.similarity_scores]

    payload_path = tmp_path / "precedent_distance_weights_v2.json"
    payload_path.write_text(
        json.dumps(
            {
                "scopes": {
                    "capital_return.open_market_buyback": {
                        "scope_key": "capital_return.open_market_buyback",
                        "use_in_runtime": True,
                        "default_enabled": True,
                        "outcome_aware_reranker": {
                            "feature_weights": {
                                "current_similarity_score": 0.5,
                                "outcome_equity_score": 2.5,
                                "outcome_valuation_score": 2.0,
                                "outcome_credit_score": 1.5,
                                "outcome_balance_sheet_score": 1.5,
                            },
                            "bias": 0.0,
                            "shortlist_size": 2,
                        },
                    }
                }
            }
        )
    )

    previous_path = os.environ.get("PRECEDENT_DISTANCE_V2_WEIGHTS_PATH")
    try:
        os.environ["PRECEDENT_DISTANCE_V2_WEIGHTS_PATH"] = str(payload_path)
        precedent_brain._PRECEDENT_DISTANCE_V2_WEIGHTS_CACHE.clear()
        pack_outcome = build_precedent_pack_v2(
            candidate_id="cand-outcome-live",
            run_id="run-outcome-live",
            company_id="000999",
            action_id="capital_return.open_market_buyback",
            action_subtype="open_market_buyback",
            action_params={},
            candidate_features=candidate_features,
            candidate_regime={"credit_regime": "neutral", "risk_regime": "neutral", "vol_regime": "normal"},
            historical_df=hist,
            top_k=2,
            min_k=1,
        )
    finally:
        if previous_path is None:
            os.environ.pop("PRECEDENT_DISTANCE_V2_WEIGHTS_PATH", None)
        else:
            os.environ["PRECEDENT_DISTANCE_V2_WEIGHTS_PATH"] = previous_path
        precedent_brain._PRECEDENT_DISTANCE_V2_WEIGHTS_CACHE.clear()

    outcome_ids = [score.precedent_id for score in pack_outcome.similarity_scores]
    assert pack_outcome.profiling["outcome_aware_reranker_applied"] is True
    assert outcome_ids != base_ids
    assert outcome_ids[0].startswith("000201::")


def test_build_precedent_pack_v2_applies_company_diversity_cap(tmp_path):
    hist = pd.DataFrame(
        [
            _state_vector_hist_row(
                company_id="000101",
                action_type="capital_return.open_market_buyback",
                action_subtype="open_market_buyback",
                offset_days=90,
                ticker="DUPA",
                **{
                    "sector": "TECH",
                    "state_vector_v1.size_log_revenue": 10.00,
                    "state_vector_v1.valuation_multiple": 60.0,
                    "state_vector_v1.cash_generation": 0.02,
                },
            ),
            _state_vector_hist_row(
                company_id="000101",
                action_type="capital_return.open_market_buyback",
                action_subtype="open_market_buyback",
                offset_days=60,
                ticker="DUPA",
                **{
                    "sector": "TECH",
                    "state_vector_v1.size_log_revenue": 10.02,
                    "state_vector_v1.valuation_multiple": 58.0,
                    "state_vector_v1.cash_generation": 0.02,
                },
            ),
            _state_vector_hist_row(
                company_id="000102",
                action_type="capital_return.open_market_buyback",
                action_subtype="open_market_buyback",
                offset_days=30,
                ticker="PEER",
                **{
                    "sector": "TECH",
                    "state_vector_v1.size_log_revenue": 10.05,
                    "state_vector_v1.valuation_multiple": 54.0,
                    "state_vector_v1.cash_generation": 0.02,
                },
            ),
        ]
    )
    candidate_features = _state_vector_candidate_features(
        sector="TECH",
        **{
            "state_vector_v1.size_log_revenue": 10.00,
            "state_vector_v1.valuation_multiple": 60.0,
            "state_vector_v1.cash_generation": 0.02,
        },
    )

    payload_path = tmp_path / "precedent_distance_weights_v2.json"
    payload_path.write_text(
        json.dumps(
            {
                "scopes": {
                    "capital_return.open_market_buyback": {
                        "scope_key": "capital_return.open_market_buyback",
                        "use_in_runtime": True,
                        "default_enabled": True,
                        "max_matches_per_company": 1,
                    }
                }
            }
        )
    )

    previous_path = os.environ.get("PRECEDENT_DISTANCE_V2_WEIGHTS_PATH")
    try:
        os.environ["PRECEDENT_DISTANCE_V2_WEIGHTS_PATH"] = str(payload_path)
        precedent_brain._PRECEDENT_DISTANCE_V2_WEIGHTS_CACHE.clear()
        pack = build_precedent_pack_v2(
            candidate_id="cand-diversity-live",
            run_id="run-diversity-live",
            company_id="000999",
            action_id="capital_return.open_market_buyback",
            action_subtype="open_market_buyback",
            action_params={},
            candidate_features=candidate_features,
            candidate_regime={"credit_regime": "neutral", "risk_regime": "neutral", "vol_regime": "normal"},
            historical_df=hist,
            top_k=2,
            min_k=1,
        )
    finally:
        if previous_path is None:
            os.environ.pop("PRECEDENT_DISTANCE_V2_WEIGHTS_PATH", None)
        else:
            os.environ["PRECEDENT_DISTANCE_V2_WEIGHTS_PATH"] = previous_path
        precedent_brain._PRECEDENT_DISTANCE_V2_WEIGHTS_CACHE.clear()

    retrieved_company_ids = [case.company_id for case in pack.retrieved_cohorts[:2]]
    assert retrieved_company_ids == ["000101", "000102"]
    assert pack.profiling["max_matches_per_company"] == 1
    assert pack.profiling["company_diversity_cap_applied"] is True
    assert pack.profiling["company_diversity_cap_relaxed"] is False


def test_mismatch_diagnostics_trigger_out_of_sample():
    features = _candidate_features()
    features["leverage_net_debt_ebitda"] = 8.0
    pack = build_precedent_pack_v2(
        candidate_id="cand-4",
        run_id="run-4",
        company_id="001690",
        action_id="capital_return.open_market_buyback",
        action_subtype="open_market_buyback",
        action_params={"size_pct_market_cap": 0.40, "funding_mix": {"cash": 1.0}},
        candidate_features=features,
        candidate_regime={"credit_regime": "tight", "risk_regime": "risk_off", "vol_regime": "high"},
        historical_df=_hist_df(45),
        top_k=20,
        min_k=10,
    )
    diag = pack.mismatch_diagnostics.to_dict() if hasattr(pack.mismatch_diagnostics, "to_dict") else pack.mismatch_diagnostics
    feature_names = [m["feature_name"] for m in diag.get("feature_mismatches", [])]
    assert "state_vector_v1.net_obligation_burden" in feature_names
    assert len(feature_names) >= 1


def test_step8_alias_fields_present_in_serialized_pack():
    pack = build_precedent_pack_v2(
        candidate_id="cand-5",
        run_id="run-5",
        company_id="001690",
        action_id="capital_return.open_market_buyback",
        action_subtype="open_market_buyback",
        action_params={"size_pct_market_cap": 0.05, "funding_mix": {"cash": 1.0}},
        candidate_features=_candidate_features(),
        candidate_regime={"credit_regime": "neutral", "risk_regime": "neutral", "vol_regime": "normal"},
        historical_df=_hist_df(30),
        top_k=20,
        min_k=10,
    )
    payload = pack.to_dict()
    assert "cohorts" in payload
    assert "distributions" in payload
    assert "tails" in payload
    assert "precedent_confidence" in payload
    assert "legacy_distributions" in payload
    assert isinstance(payload["legacy_distributions"], list)


def test_low_precedent_coverage_flag_when_below_minimum():
    pack = build_precedent_pack_v2(
        candidate_id="cand-6",
        run_id="run-6",
        company_id="001690",
        action_id="capital_return.open_market_buyback",
        action_subtype="open_market_buyback",
        action_params={"size_pct_market_cap": 0.05, "funding_mix": {"cash": 1.0}},
        candidate_features=_candidate_features(),
        candidate_regime={"credit_regime": "neutral", "risk_regime": "neutral", "vol_regime": "normal"},
        historical_df=_hist_df(7),
        top_k=30,
        min_k=10,
    )
    diag = pack.mismatch_diagnostics.to_dict() if hasattr(pack.mismatch_diagnostics, "to_dict") else pack.mismatch_diagnostics
    assert diag.get("low_precedent_coverage") is True


def test_sector_similarity_influences_scoring():
    hist = _hist_df(45)
    base = _candidate_features()
    match_pack = build_precedent_pack_v2(
        candidate_id="cand-7a",
        run_id="run-7",
        company_id="001690",
        action_id="capital_return.open_market_buyback",
        action_subtype="open_market_buyback",
        action_params={"size_pct_market_cap": 0.05, "funding_mix": {"cash": 1.0}},
        candidate_features=base,
        candidate_regime={"credit_regime": "neutral", "risk_regime": "neutral", "vol_regime": "normal"},
        historical_df=hist,
        top_k=20,
        min_k=10,
    )
    mismatch_features = dict(base)
    mismatch_features["sector"] = "UTILITIES"
    mismatch_pack = build_precedent_pack_v2(
        candidate_id="cand-7b",
        run_id="run-7",
        company_id="001690",
        action_id="capital_return.open_market_buyback",
        action_subtype="open_market_buyback",
        action_params={"size_pct_market_cap": 0.05, "funding_mix": {"cash": 1.0}},
        candidate_features=mismatch_features,
        candidate_regime={"credit_regime": "neutral", "risk_regime": "neutral", "vol_regime": "normal"},
        historical_df=hist,
        top_k=20,
        min_k=10,
    )
    s_match = np.mean([s.sector_similarity for s in match_pack.similarity_scores])
    s_mismatch = np.mean([s.sector_similarity for s in mismatch_pack.similarity_scores])
    assert s_match > s_mismatch


def test_retrieval_index_enriches_missing_historical_taxonomy_from_lookup(monkeypatch):
    monkeypatch.setattr(
        precedent_brain,
        "_historical_taxonomy_for_ticker",
        lambda ticker: (
            {
                "taxonomy.sector": "Consumer Staples",
                "taxonomy.subsector": "Household Products",
            }
            if str(ticker).upper() == "PG"
            else {}
        ),
    )
    hist = pd.DataFrame(
        [
            _state_vector_hist_row(
                company_id="000111",
                action_type="dividend_increase",
                action_subtype="dividend_increase",
                offset_days=0,
                ticker="PG",
                base_sector="",
                sector="",
                gics_sector="",
            )
        ]
    )
    idx = build_precedent_retrieval_index(hist)
    assert idx.sector_token_arr.tolist() == ["CONSUMER STAPLES"]
    assert idx.subsector_token_arr.tolist() == ["HOUSEHOLD PRODUCTS"]
    enriched = idx.df.iloc[0]
    assert str(enriched.get("taxonomy.sector")) == "Consumer Staples"
    assert str(enriched.get("taxonomy.subsector")) == "Household Products"


def test_snapshot_taxonomy_lookup_falls_back_to_catalog(monkeypatch, tmp_path):
    catalog_path = tmp_path / "catalog.jsonl.gz"
    rows = [
        {
            "company_id": "000123",
            "as_of_time": "2024-01-01T00:00:00+00:00",
            "features": {
                "taxonomy.sector": {"value": "Industrials", "support_mode": "heuristic", "confidence": 0.2},
                "taxonomy.subsector": {"value": "", "support_mode": "heuristic", "confidence": 0.2},
            },
        },
        {
            "company_id": "000123",
            "as_of_time": "2024-12-31T00:00:00+00:00",
            "features": {
                "taxonomy.sector": {"value": "Industrials", "support_mode": "exact", "confidence": 0.4},
                "taxonomy.subsector": {"value": "Machinery", "support_mode": "exact", "confidence": 0.4},
            },
        },
        {
            "company_id": "000456",
            "as_of_time": "2024-12-31T00:00:00+00:00",
            "features": {
                "taxonomy.sector": {"value": "Health Care", "support_mode": "exact", "confidence": 0.3},
                "taxonomy.subsector": {"value": "Biotechnology", "support_mode": "exact", "confidence": 0.3},
            },
        },
    ]
    with gzip.open(catalog_path, "wt") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    monkeypatch.setattr(precedent_brain, "_SNAPSHOT_TAXONOMY_LOOKUP_PATH", tmp_path / "missing.parquet")
    monkeypatch.setattr(precedent_brain, "_SNAPSHOT_TAXONOMY_CATALOG_FALLBACK_PATH", catalog_path)
    precedent_brain._load_snapshot_taxonomy_lookup.cache_clear()
    try:
        lookup = precedent_brain._load_snapshot_taxonomy_lookup()
    finally:
        precedent_brain._load_snapshot_taxonomy_lookup.cache_clear()
    assert lookup["000123"] == ("Industrials", "Machinery")
    assert lookup["000456"] == ("Health Care", "Biotechnology")


def test_historical_taxonomy_for_ticker_uses_sec_submission_snapshot_mapping(monkeypatch):
    monkeypatch.setattr(precedent_brain, "_load_refinitiv_taxonomy_lookup", lambda: {})
    monkeypatch.setattr(precedent_brain, "_load_sec_ticker_cik_lookup", lambda: {})
    monkeypatch.setattr(
        precedent_brain,
        "_load_sec_company_ticker_metadata_lookup",
        lambda: {
            "FCEL": {
                "ticker": "FCEL",
                "cik": "0000886128",
                "title": "FUELCELL ENERGY INC",
            }
        },
    )
    monkeypatch.setattr(
        precedent_brain,
        "_sec_submission_identity_for_cik",
        lambda cik: {
            "cik": "0000886128",
            "primary_ticker": "FCEL",
            "tickers": ["FCEL"],
            "sic": "3620",
            "sic_description": "Electrical Industrial Apparatus",
            "name": "FUELCELL ENERGY INC",
        }
        if str(cik) == "0000886128"
        else {},
    )
    monkeypatch.setattr(
        precedent_brain,
        "_snapshot_taxonomy_for_cik",
        lambda cik: ("Industrials", "Electrical Equipment") if str(cik) == "0000886128" else ("", ""),
    )
    precedent_brain._historical_taxonomy_for_ticker.cache_clear()
    try:
        assert precedent_brain._historical_taxonomy_for_ticker("FCEL", allow_sec_identity_heuristics=True) == {
            "taxonomy.sector": "Industrials",
            "taxonomy.subsector": "Electrical Equipment",
        }
    finally:
        precedent_brain._historical_taxonomy_for_ticker.cache_clear()


def test_historical_taxonomy_for_ticker_uses_sec_submission_identity_text(monkeypatch):
    monkeypatch.setattr(precedent_brain, "_load_refinitiv_taxonomy_lookup", lambda: {})
    monkeypatch.setattr(precedent_brain, "_load_sec_ticker_cik_lookup", lambda: {})
    monkeypatch.setattr(
        precedent_brain,
        "_load_sec_company_ticker_metadata_lookup",
        lambda: {
            "FCEL": {
                "ticker": "FCEL",
                "cik": "0000886128",
                "title": "FUELCELL ENERGY INC",
            },
        },
    )
    monkeypatch.setattr(precedent_brain, "_snapshot_taxonomy_for_cik", lambda cik: ("", ""))
    monkeypatch.setattr(
        precedent_brain,
        "_sec_submission_identity_for_cik",
        lambda cik: {
            "cik": "0000886128",
            "primary_ticker": "FCEL",
            "tickers": ["FCEL"],
            "sic": "3620",
            "sic_description": "Electrical Industrial Apparatus",
            "name": "FUELCELL ENERGY INC",
        }
        if str(cik) == "0000886128"
        else {},
    )
    precedent_brain._historical_taxonomy_for_ticker.cache_clear()
    try:
        assert precedent_brain._historical_taxonomy_for_ticker("FCEL", allow_sec_identity_heuristics=True) == {
            "taxonomy.sector": "Industrials",
            "taxonomy.subsector": "Electrical Equipment",
        }
    finally:
        precedent_brain._historical_taxonomy_for_ticker.cache_clear()


def test_historical_taxonomy_for_ticker_uses_sec_company_title_fallback(monkeypatch):
    monkeypatch.setattr(precedent_brain, "_load_refinitiv_taxonomy_lookup", lambda: {})
    monkeypatch.setattr(precedent_brain, "_load_sec_ticker_cik_lookup", lambda: {})
    monkeypatch.setattr(
        precedent_brain,
        "_load_sec_company_ticker_metadata_lookup",
        lambda: {
            "ENGN": {
                "ticker": "ENGN",
                "cik": "0001980845",
                "title": "enGene Holdings Inc.",
            }
        },
    )
    monkeypatch.setattr(precedent_brain, "_snapshot_taxonomy_for_cik", lambda cik: ("", ""))
    monkeypatch.setattr(precedent_brain, "_sec_submission_identity_for_cik", lambda cik: {})
    precedent_brain._historical_taxonomy_for_ticker.cache_clear()
    try:
        assert precedent_brain._historical_taxonomy_for_ticker("ENGN", allow_sec_identity_heuristics=True) == {
            "taxonomy.sector": "Health Care",
            "taxonomy.subsector": "Biotechnology",
        }
    finally:
        precedent_brain._historical_taxonomy_for_ticker.cache_clear()


def test_historical_taxonomy_for_ticker_skips_sec_identity_heuristics_by_default(monkeypatch):
    monkeypatch.setattr(precedent_brain, "_load_refinitiv_taxonomy_lookup", lambda: {})
    monkeypatch.setattr(precedent_brain, "_load_sec_ticker_cik_lookup", lambda: {})
    monkeypatch.setattr(
        precedent_brain,
        "_load_sec_company_ticker_metadata_lookup",
        lambda: {
            "ENGN": {
                "ticker": "ENGN",
                "cik": "0001980845",
                "title": "enGene Holdings Inc.",
            }
        },
    )
    monkeypatch.setattr(precedent_brain, "_snapshot_taxonomy_for_cik", lambda cik: ("", ""))
    monkeypatch.setattr(precedent_brain, "_sec_submission_identity_for_cik", lambda cik: {})
    precedent_brain._historical_taxonomy_for_ticker.cache_clear()
    try:
        assert precedent_brain._historical_taxonomy_for_ticker("ENGN") == {}
    finally:
        precedent_brain._historical_taxonomy_for_ticker.cache_clear()


def test_candidate_metric_dict_taxonomy_values_are_used_for_identity_matching():
    hist = pd.DataFrame(
        [
            _state_vector_hist_row(
                company_id="000211",
                action_type="dividend_increase",
                action_subtype="dividend_increase",
                offset_days=0,
                ticker="GOODIND",
                **{
                    "base_sector": "INDUSTRIALS",
                    "subsector": "MACHINERY",
                    "normalized_action_family": "capital_return",
                    "normalized_action_subfamily": "dividend_increase",
                    "normalized_action_id": "capital_return.dividend_increase",
                },
            ),
            _state_vector_hist_row(
                company_id="000212",
                action_type="dividend_increase",
                action_subtype="dividend_increase",
                offset_days=1,
                ticker="WRONGSEC",
                **{
                    "base_sector": "ENERGY",
                    "subsector": "INTEGRATED_OIL_GAS",
                    "normalized_action_family": "capital_return",
                    "normalized_action_subfamily": "dividend_increase",
                    "normalized_action_id": "capital_return.dividend_increase",
                },
            ),
        ]
    )
    payload = {
        "version": "precedent_distance_weights_v2",
        "scopes": {
            "capital_return.dividend_increase": {
                "scope_key": "capital_return.dividend_increase",
                "default_enabled": True,
                "use_in_runtime": True,
            }
        },
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/precedent_distance_weights_v2.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        old_path = os.environ.get("PRECEDENT_DISTANCE_V2_WEIGHTS_PATH")
        old_version = os.environ.get("PRECEDENT_DISTANCE_PROFILE_VERSION")
        os.environ["PRECEDENT_DISTANCE_V2_WEIGHTS_PATH"] = path
        os.environ.pop("PRECEDENT_DISTANCE_PROFILE_VERSION", None)
        try:
            pack = build_precedent_pack_v2(
                candidate_id="cand-taxonomy-dicts",
                run_id="run-taxonomy-dicts",
                company_id="001690",
                action_id="capital_return.dividend_increase",
                action_subtype="dividend_increase",
                action_params={},
                candidate_features=_state_vector_candidate_features(
                    **{
                        "taxonomy.sector": {"value": "Industrials"},
                        "taxonomy.subsector": {"value": "Machinery"},
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
                    }
                ),
                candidate_regime={"credit_regime": "neutral", "risk_regime": "neutral", "vol_regime": "normal"},
                historical_df=hist,
                top_k=2,
                min_k=1,
            )
        finally:
            if old_path is None:
                os.environ.pop("PRECEDENT_DISTANCE_V2_WEIGHTS_PATH", None)
            else:
                os.environ["PRECEDENT_DISTANCE_V2_WEIGHTS_PATH"] = old_path
            if old_version is None:
                os.environ.pop("PRECEDENT_DISTANCE_PROFILE_VERSION", None)
            else:
                os.environ["PRECEDENT_DISTANCE_PROFILE_VERSION"] = old_version
    assert pack.retrieved_cohorts[0].company_id == "000211"


def test_precedent_brain_accepts_explicit_historical_stores():
    stores = build_historical_stores_from_outcomes(_hist_df(36), dataset_version="test_v1")
    pack = build_precedent_pack_v2(
        candidate_id="cand-8",
        run_id="run-8",
        company_id="001690",
        action_id="capital_return.open_market_buyback",
        action_subtype="open_market_buyback",
        action_params={"size_pct_market_cap": 0.05, "funding_mix": {"cash": 1.0}},
        candidate_features=_candidate_features(),
        candidate_regime={"credit_regime": "neutral", "risk_regime": "neutral", "vol_regime": "normal"},
        historical_event_store=stores["historical_event_store"],
        historical_state_store=stores["historical_state_store"],
        historical_outcome_store=stores["historical_outcome_store"],
        regime_history=stores["regime_history"],
        top_k=20,
        min_k=10,
    )
    assert len(pack.retrieved_cohorts) >= 10


def test_key_state_features_surface_compact_state_vector_fields():
    pack = build_precedent_pack_v2(
        candidate_id="cand-8b",
        run_id="run-8b",
        company_id="001690",
        action_id="capital_return.open_market_buyback",
        action_subtype="open_market_buyback",
        action_params={"size_pct_market_cap": 0.05, "funding_mix": {"cash": 1.0}},
        candidate_features=_candidate_features(),
        candidate_regime={"credit_regime": "neutral", "risk_regime": "neutral", "vol_regime": "normal"},
        historical_df=_hist_df(36),
        top_k=20,
        min_k=10,
    )
    key_state = pack.retrieved_cohorts[0].key_state_features
    assert "state_vector_v1.size_log_revenue" in key_state
    assert "state_vector_v1.profitability" in key_state
    assert "state_vector_v1.net_obligation_burden" in key_state
    assert "state_vector_v1.cash_generation" in key_state
    assert "state_vector_v1.credit_spread" in key_state


def test_explicit_state_vector_history_overrides_legacy_matching_fields():
    hist = _hist_df(25).copy()
    hist["state_vector_v1.profitability"] = 0.33
    hist["state_vector_v1.net_obligation_burden"] = 1.75
    hist["state_vector_v1.size_log_revenue"] = np.log10(hist["base_revenue_ttm"].astype(float))
    pack = build_precedent_pack_v2(
        candidate_id="cand-8c",
        run_id="run-8c",
        company_id="001690",
        action_id="capital_return.open_market_buyback",
        action_subtype="open_market_buyback",
        action_params={"size_pct_market_cap": 0.05, "funding_mix": {"cash": 1.0}},
        candidate_features=_candidate_features(),
        candidate_regime={"credit_regime": "neutral", "risk_regime": "neutral", "vol_regime": "normal"},
        historical_df=hist,
        top_k=20,
        min_k=10,
    )
    key_state = pack.retrieved_cohorts[0].key_state_features
    assert key_state["state_vector_v1.profitability"] == 0.33
    assert key_state["state_vector_v1.net_obligation_burden"] == 1.75


def test_augment_precedent_state_vector_columns_prefers_raw_formula_inputs():
    hist = pd.DataFrame(
        [
            {
                "operating.revenue_ttm_provider_direct": 100.0,
                "operating.revenue_ttm_lag_1y": 80.0,
                "operating.ebitda_ltm_provider_direct": 20.0,
                "capital_structure.total_debt_provider_direct": 50.0,
                "capital_structure.net_debt_normalized": 30.0,
                "capital_structure.lease_liabilities_sec_exact": 5.0,
                "capital_structure.combined_retirement_liability": 10.0,
                "liquidity.cash_and_short_term_investments_provider_direct": 25.0,
                "capital_structure.current_debt_statement_direct": 10.0,
                "capital_structure.interest_expense_statement_direct": 4.0,
                "market.market_cap_provider_direct": 200.0,
                "cash_flow.free_cash_flow_ttm": 12.0,
                "market.volatility_90d": 0.30,
                "market.drawdown_90d": -0.20,
                "market.credit_window_proxy": 0.70,
                "market.equity_window_proxy": 0.80,
                "market.credit_spread_level": 0.02,
                "macro.fed_funds_effective": 4.50,
                "macro.hy_oas": 3.25,
            }
        ]
    )

    augmented = augment_precedent_state_vector_columns(hist)
    row = augmented.iloc[0]

    assert np.isclose(row["state_vector_v1.size_log_revenue"], np.log10(100.0))
    assert np.isclose(row["state_vector_v1.profitability"], 0.20)
    assert np.isclose(row["state_vector_v1.growth"], 0.25)
    assert np.isclose(row["state_vector_v1.gross_obligation_burden"], 3.25)
    assert np.isclose(row["state_vector_v1.net_obligation_burden"], 2.0)
    assert np.isclose(row["state_vector_v1.liquidity_flexibility"], 2.5)
    assert np.isclose(row["state_vector_v1.interest_coverage"], 5.0)
    assert np.isclose(row["state_vector_v1.valuation_multiple"], 11.5)
    assert np.isclose(row["state_vector_v1.cash_generation"], 0.06)
    assert np.isclose(row["state_vector_v1.rates_level"], 4.50)
    assert np.isclose(row["state_vector_v1.credit_spread"], 3.25)


def test_augment_precedent_state_vector_columns_uses_richer_contract_baseline_fields():
    hist = pd.DataFrame(
        [
            {
                "base_revenue_ttm": 100.0,
                "base_revenue_ttm_lag_1y": 80.0,
                "base_ebitda_ttm": 20.0,
                "base_margin": 0.20,
                "base_total_debt": 50.0,
                "base_net_debt": 30.0,
                "base_available_liquidity": 25.0,
                "base_current_debt": 10.0,
                "base_interest_expense": 4.0,
                "base_market_cap": 200.0,
                "base_fcf_yield": 0.06,
                "macro_fed_funds_effective": 4.50,
                "macro_hy_oas": 3.25,
            }
        ]
    )

    augmented = augment_precedent_state_vector_columns(hist)
    row = augmented.iloc[0]

    assert np.isclose(row["state_vector_v1.size_log_revenue"], np.log10(100.0 * 1_000_000.0))
    assert np.isclose(row["state_vector_v1.profitability"], 0.20)
    assert np.isclose(row["state_vector_v1.growth"], 0.25)
    assert np.isclose(row["state_vector_v1.gross_obligation_burden"], 2.5)
    assert np.isclose(row["state_vector_v1.net_obligation_burden"], 1.5)
    assert np.isclose(row["state_vector_v1.liquidity_flexibility"], 2.5)
    assert np.isclose(row["state_vector_v1.interest_coverage"], 5.0)
    assert np.isclose(row["state_vector_v1.valuation_multiple"], 11.25)
    assert np.isclose(row["state_vector_v1.cash_generation"], 0.06)
    assert np.isclose(row["state_vector_v1.rates_level"], 4.50)
    assert np.isclose(row["state_vector_v1.credit_spread"], 3.25)


def test_augment_precedent_state_vector_columns_scales_historical_revenue_to_dollars_for_size():
    hist = pd.DataFrame([{"base_revenue_ttm": 86469.0}])

    augmented = augment_precedent_state_vector_columns(hist)
    row = augmented.iloc[0]

    assert np.isclose(row["state_vector_v1.size_log_revenue"], np.log10(86469.0 * 1_000_000.0))


def test_augment_precedent_state_vector_columns_uses_historical_market_stress_and_access_inputs():
    hist = pd.DataFrame(
        [
            {
                "base_volatility_30d": 0.24,
                "base_volatility_90d": 0.30,
                "base_drawdown_90d": -0.20,
                "base_momentum_60d": 0.10,
                "base_credit_spread_level": 0.025,
                "base_ev_ebitda": 12.0,
            }
        ]
    )

    augmented = augment_precedent_state_vector_columns(hist)
    row = augmented.iloc[0]

    expected_market_stress = (0.30 * 0.6) + (0.20 * 0.4)
    expected_equity_window = np.mean([1.0 - (0.24 / 0.8), (0.10 + 0.2) / 0.4, 12.0 / 20.0])
    expected_credit_window = np.mean([1.0 - (0.025 / 0.10), 1.0 - (0.24 / 1.0)])
    expected_spread_access = 1.0 - (0.025 / 0.08)
    expected_market_access = np.average(
        [expected_credit_window, expected_equity_window, expected_spread_access],
        weights=[0.4, 0.4, 0.2],
    )

    assert np.isclose(row["state_vector_v1.market_stress"], expected_market_stress)
    assert np.isclose(row["state_vector_v1.market_access"], expected_market_access)


def test_augment_precedent_state_vector_columns_falls_back_to_macro_vix_for_market_stress():
    hist = pd.DataFrame([{"macro_vix": 24.0}])

    augmented = augment_precedent_state_vector_columns(hist)
    row = augmented.iloc[0]

    assert np.isclose(row["state_vector_v1.market_stress"], 24.0 / 80.0)


def test_build_historical_stores_preserves_richer_contract_baseline_fields():
    hist = _hist_df(5).copy()
    hist["base_revenue_ttm_lag_1y"] = 400.0
    hist["base_revenue_growth_yoy"] = 0.25
    hist["base_ebitda_ttm"] = 100.0
    hist["base_cash"] = 40.0
    hist["base_total_debt"] = 80.0
    hist["base_current_debt"] = 15.0
    hist["base_available_liquidity"] = 40.0
    hist["base_interest_expense"] = 5.0
    hist["base_fcf_yield"] = 0.04
    hist["macro_fed_funds_effective"] = 4.25
    hist["macro_real_gdp_growth_yoy"] = 0.02

    stores = build_historical_stores_from_outcomes(hist, dataset_version="test_rich_contract_v1")
    snapshots = stores["historical_state_store"].snapshots

    for col in (
        "base_revenue_ttm_lag_1y",
        "base_revenue_growth_yoy",
        "base_ebitda_ttm",
        "base_cash",
        "base_total_debt",
        "base_current_debt",
        "base_available_liquidity",
        "base_interest_expense",
        "base_fcf_yield",
        "macro_fed_funds_effective",
        "macro_real_gdp_growth_yoy",
    ):
        assert col in snapshots.columns


def test_retrieval_index_path_matches_direct_path():
    hist = _hist_df(40)
    idx = build_precedent_retrieval_index(hist)
    kwargs = dict(
        candidate_id="cand-9",
        run_id="run-9",
        company_id="001690",
        action_id="capital_return.open_market_buyback",
        action_subtype="open_market_buyback",
        action_params={"size_pct_market_cap": 0.05, "funding_mix": {"cash": 1.0}},
        candidate_features=_candidate_features(),
        candidate_regime={"credit_regime": "neutral", "risk_regime": "neutral", "vol_regime": "normal"},
        top_k=20,
        min_k=10,
    )
    pack_direct = build_precedent_pack_v2(historical_df=hist, **kwargs)
    pack_index = build_precedent_pack_v2(retrieval_index=idx, **kwargs)
    ids_direct = [x.precedent_id for x in pack_direct.retrieved_cohorts]
    ids_index = [x.precedent_id for x in pack_index.retrieved_cohorts]
    assert ids_direct == ids_index
    assert [x.regime_label for x in pack_direct.regime_splits] == [x.regime_label for x in pack_index.regime_splits]
    assert [
        (x.follow_on_action_id, round(float(x.frequency), 6), round(float(x.median_time_to_follow_on or 0.0), 6))
        for x in pack_direct.second_order_effects
    ] == [
        (x.follow_on_action_id, round(float(x.frequency), 6), round(float(x.median_time_to_follow_on or 0.0), 6))
        for x in pack_index.second_order_effects
    ]


def test_retrieval_index_path_handles_utc_action_dates():
    hist = _hist_df(40)
    hist["action_date"] = pd.to_datetime(hist["action_date"], utc=True)
    idx = build_precedent_retrieval_index(hist)
    pack = build_precedent_pack_v2(
        retrieval_index=idx,
        candidate_id="cand-utc",
        run_id="run-utc",
        company_id="001690",
        action_id="capital_return.open_market_buyback",
        action_subtype="open_market_buyback",
        action_params={"size_pct_market_cap": 0.05, "funding_mix": {"cash": 1.0}},
        candidate_features=_candidate_features(),
        candidate_regime={"credit_regime": "neutral", "risk_regime": "neutral", "vol_regime": "normal"},
        top_k=20,
        min_k=10,
    )
    assert pack.retrieved_cohorts


def test_weighted_coverage_gate_downranks_null_heavy_matches():
    hist = pd.DataFrame(
        [
            _state_vector_hist_row(company_id="000101", action_type="dividend_increase", action_subtype="dividend_increase", offset_days=0, ticker="GOOD1"),
            _state_vector_hist_row(
                company_id="000102",
                action_type="dividend_increase",
                action_subtype="dividend_increase",
                offset_days=1,
                ticker="THIN",
                **{
                    "state_vector_v1.size_log_revenue": np.nan,
                    "state_vector_v1.profitability": np.nan,
                    "state_vector_v1.growth": np.nan,
                    "state_vector_v1.liquidity_flexibility": np.nan,
                    "state_vector_v1.interest_coverage": np.nan,
                    "state_vector_v1.valuation_multiple": np.nan,
                    "state_vector_v1.cash_generation": np.nan,
                    "state_vector_v1.net_obligation_burden": np.nan,
                    "state_vector_v1.market_access": np.nan,
                },
            ),
            _state_vector_hist_row(company_id="000103", action_type="dividend_increase", action_subtype="dividend_increase", offset_days=2, ticker="GOOD2"),
        ]
    )
    pack = build_precedent_pack_v2(
        candidate_id="cand-weighted-coverage",
        run_id="run-weighted-coverage",
        company_id="001690",
        action_id="capital_return.dividend_increase",
        action_subtype="dividend_increase",
        action_params={},
        candidate_features=_state_vector_candidate_features(),
        candidate_regime={"credit_regime": "neutral", "risk_regime": "neutral", "vol_regime": "normal"},
        historical_df=hist,
        top_k=3,
        min_k=2,
    )
    company_ids = [case.company_id for case in pack.retrieved_cohorts]
    diag = pack.mismatch_diagnostics
    assert "000102" not in company_ids
    assert diag.get("weighted_coverage_gate_applied") is True


def test_size_guardrail_filters_absurd_size_mismatches():
    hist = pd.DataFrame(
        [
            _state_vector_hist_row(company_id="000201", action_type="buyback", action_subtype="buyback", offset_days=0, ticker="GOOD1"),
            _state_vector_hist_row(
                company_id="000202",
                action_type="buyback",
                action_subtype="buyback",
                offset_days=1,
                ticker="HUGE",
                **{"state_vector_v1.size_log_revenue": 12.8},
            ),
            _state_vector_hist_row(company_id="000203", action_type="buyback", action_subtype="buyback", offset_days=2, ticker="GOOD2", **{"state_vector_v1.size_log_revenue": 10.2}),
        ]
    )
    pack = build_precedent_pack_v2(
        candidate_id="cand-size-guardrail",
        run_id="run-size-guardrail",
        company_id="001690",
        action_id="capital_return.open_market_buyback",
        action_subtype="open_market_buyback",
        action_params={},
        candidate_features=_state_vector_candidate_features(),
        candidate_regime={"credit_regime": "neutral", "risk_regime": "neutral", "vol_regime": "normal"},
        historical_df=hist,
        top_k=3,
        min_k=2,
    )
    company_ids = [case.company_id for case in pack.retrieved_cohorts]
    diag = pack.mismatch_diagnostics
    assert "000202" not in company_ids
    assert diag.get("size_guardrail_applied") is True


def test_dividend_matching_no_longer_lets_liquidity_tails_overpower_valuation():
    dividend_hist = pd.DataFrame(
        [
            _state_vector_hist_row(
                company_id="000301",
                action_type="dividend_increase",
                action_subtype="dividend_increase",
                offset_days=0,
                ticker="LIQ",
                **{
                    "state_vector_v1.net_obligation_burden": 1.10,
                    "state_vector_v1.liquidity_flexibility": 2.10,
                    "state_vector_v1.interest_coverage": 11.0,
                    "state_vector_v1.valuation_multiple": 24.0,
                    "state_vector_v1.cash_generation": 0.045,
                },
            ),
            _state_vector_hist_row(
                company_id="000302",
                action_type="dividend_increase",
                action_subtype="dividend_increase",
                offset_days=1,
                ticker="VAL",
                **{
                    "state_vector_v1.net_obligation_burden": 1.05,
                    "state_vector_v1.liquidity_flexibility": 0.80,
                    "state_vector_v1.interest_coverage": 8.0,
                    "state_vector_v1.valuation_multiple": 12.05,
                    "state_vector_v1.cash_generation": 0.01,
                },
            ),
        ]
    )
    buyback_hist = dividend_hist.copy()
    buyback_hist["action_type"] = "buyback"
    buyback_hist["action_subtype"] = "buyback"

    dividend_pack = build_precedent_pack_v2(
        candidate_id="cand-dividend-weights",
        run_id="run-dividend-weights",
        company_id="001690",
        action_id="capital_return.dividend_increase",
        action_subtype="dividend_increase",
        action_params={},
        candidate_features=_state_vector_candidate_features(),
        candidate_regime={"credit_regime": "neutral", "risk_regime": "neutral", "vol_regime": "normal"},
        historical_df=dividend_hist,
        top_k=2,
        min_k=1,
    )
    buyback_pack = build_precedent_pack_v2(
        candidate_id="cand-buyback-weights",
        run_id="run-buyback-weights",
        company_id="001690",
        action_id="capital_return.open_market_buyback",
        action_subtype="open_market_buyback",
        action_params={},
        candidate_features=_state_vector_candidate_features(),
        candidate_regime={"credit_regime": "neutral", "risk_regime": "neutral", "vol_regime": "normal"},
        historical_df=buyback_hist,
        top_k=2,
        min_k=1,
    )

    assert dividend_pack.retrieved_cohorts[0].company_id == "000302"
    assert buyback_pack.retrieved_cohorts[0].company_id == "000302"


def test_weighted_distance_profile_uses_learned_scope_override():
    payload = {
        "version": "precedent_distance_weights_v1",
        "scopes": {
            "capital_return": {
                "weights": {
                    "state_vector_v1.valuation_multiple": 2.75,
                    "state_vector_v1.cash_generation": 1.80,
                },
                "use_in_runtime": True,
                "holdout_pair_correlation": 0.42,
                "holdout_prior_pair_correlation": 0.31,
                "n_pairs": 1234,
            }
        },
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/precedent_distance_weights_v1.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        old = os.environ.get("PRECEDENT_DISTANCE_WEIGHTS_PATH")
        old_version = os.environ.get("PRECEDENT_DISTANCE_PROFILE_VERSION")
        os.environ["PRECEDENT_DISTANCE_WEIGHTS_PATH"] = path
        os.environ["PRECEDENT_DISTANCE_PROFILE_VERSION"] = "weighted_distance_v1"
        try:
            profile = _weighted_distance_profile("capital_return.open_market_buyback", "open_market_buyback")
        finally:
            if old is None:
                os.environ.pop("PRECEDENT_DISTANCE_WEIGHTS_PATH", None)
            else:
                os.environ["PRECEDENT_DISTANCE_WEIGHTS_PATH"] = old
            if old_version is None:
                os.environ.pop("PRECEDENT_DISTANCE_PROFILE_VERSION", None)
            else:
                os.environ["PRECEDENT_DISTANCE_PROFILE_VERSION"] = old_version
    assert profile["weight_scope"] == "capital_return"
    assert np.isclose(profile["weights"]["state_vector_v1.valuation_multiple"], 2.75)
    assert np.isclose(profile["weights"]["state_vector_v1.cash_generation"], 1.80)
    assert np.isclose(profile["learned_holdout_pair_correlation"], 0.42)


def test_weighted_distance_profile_v2_uses_runtime_scope_override():
    payload = {
        "version": "precedent_distance_weights_v2",
        "scopes": {
            "capital_structure": {
                "scope_key": "capital_structure",
                "group_weights": {
                    "capital_structure": 1.8,
                    "valuation": 0.7,
                },
                "feature_relative_weights": {
                    "state_vector_v1.valuation_multiple": 1.5,
                },
                "penalties": {
                    "regime_rate_penalty_weight": 0.7,
                },
                "use_in_runtime": True,
            }
        },
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/precedent_distance_weights_v2.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        old_path = os.environ.get("PRECEDENT_DISTANCE_V2_WEIGHTS_PATH")
        old_version = os.environ.get("PRECEDENT_DISTANCE_PROFILE_VERSION")
        os.environ["PRECEDENT_DISTANCE_V2_WEIGHTS_PATH"] = path
        os.environ["PRECEDENT_DISTANCE_PROFILE_VERSION"] = "weighted_distance_v2"
        try:
            profile = _weighted_distance_profile("capital_structure.new_debt_issuance", "new_debt_issuance")
        finally:
            if old_path is None:
                os.environ.pop("PRECEDENT_DISTANCE_V2_WEIGHTS_PATH", None)
            else:
                os.environ["PRECEDENT_DISTANCE_V2_WEIGHTS_PATH"] = old_path
            if old_version is None:
                os.environ.pop("PRECEDENT_DISTANCE_PROFILE_VERSION", None)
            else:
                os.environ["PRECEDENT_DISTANCE_PROFILE_VERSION"] = old_version
    assert profile["version"] == "weighted_distance_v2"
    assert profile["weight_scope"] == "capital_structure"
    assert profile["group_weights"]["capital_structure"] > profile["group_weights"]["valuation"]
    assert profile["regime_rate_penalty_weight"] == 0.7


def test_weighted_distance_profile_v2_identity_transform_mode_skips_default_transforms():
    payload = {
        "version": "precedent_distance_weights_v2",
        "scopes": {
            "capital_return.open_market_buyback": {
                "scope_key": "capital_return.open_market_buyback",
                "feature_transform_mode": "identity",
                "feature_transforms": {
                    "state_vector_v1.cash_generation": {"kind": "signed_asinh", "scale": 0.05},
                },
                "use_in_runtime": True,
            }
        },
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/precedent_distance_weights_v2.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        old_path = os.environ.get("PRECEDENT_DISTANCE_V2_WEIGHTS_PATH")
        old_version = os.environ.get("PRECEDENT_DISTANCE_PROFILE_VERSION")
        os.environ["PRECEDENT_DISTANCE_V2_WEIGHTS_PATH"] = path
        os.environ["PRECEDENT_DISTANCE_PROFILE_VERSION"] = "weighted_distance_v2"
        try:
            profile = _weighted_distance_profile("capital_return.open_market_buyback", "open_market_buyback")
        finally:
            if old_path is None:
                os.environ.pop("PRECEDENT_DISTANCE_V2_WEIGHTS_PATH", None)
            else:
                os.environ["PRECEDENT_DISTANCE_V2_WEIGHTS_PATH"] = old_path
            if old_version is None:
                os.environ.pop("PRECEDENT_DISTANCE_PROFILE_VERSION", None)
            else:
                os.environ["PRECEDENT_DISTANCE_PROFILE_VERSION"] = old_version
    assert profile["feature_transform_mode"] == "identity"
    assert profile["feature_transforms"] == {
        "state_vector_v1.cash_generation": {"kind": "signed_asinh", "scale": 0.05}
    }
    assert "state_vector_v1.valuation_multiple" not in profile["feature_transforms"]


def test_weighted_distance_profile_v2_can_be_default_enabled_for_scope():
    payload = {
        "version": "precedent_distance_weights_v2",
        "scopes": {
            "capital_structure": {
                "scope_key": "capital_structure",
                "default_enabled": True,
                "group_weights": {
                    "capital_structure": 1.8,
                    "valuation": 0.7,
                },
                "feature_relative_weights": {
                    "state_vector_v1.profitability": 1.4,
                    "state_vector_v1.cash_generation": 0.8,
                },
                "use_in_runtime": True,
            }
        },
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/precedent_distance_weights_v2.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        old_path = os.environ.get("PRECEDENT_DISTANCE_V2_WEIGHTS_PATH")
        old_version = os.environ.get("PRECEDENT_DISTANCE_PROFILE_VERSION")
        os.environ["PRECEDENT_DISTANCE_V2_WEIGHTS_PATH"] = path
        os.environ.pop("PRECEDENT_DISTANCE_PROFILE_VERSION", None)
        try:
            cap_profile = _weighted_distance_profile("capital_structure.new_debt_issuance", "new_debt_issuance")
            buyback_profile = _weighted_distance_profile("capital_return.open_market_buyback", "open_market_buyback")
        finally:
            if old_path is None:
                os.environ.pop("PRECEDENT_DISTANCE_V2_WEIGHTS_PATH", None)
            else:
                os.environ["PRECEDENT_DISTANCE_V2_WEIGHTS_PATH"] = old_path
            if old_version is None:
                os.environ.pop("PRECEDENT_DISTANCE_PROFILE_VERSION", None)
            else:
                os.environ["PRECEDENT_DISTANCE_PROFILE_VERSION"] = old_version
    assert cap_profile["version"] == "weighted_distance_v2"
    assert cap_profile["weight_scope"] == "capital_structure"
    assert np.isclose(cap_profile["feature_relative_weights"]["state_vector_v1.profitability"], 1.4)
    assert np.isclose(cap_profile["feature_relative_weights"]["state_vector_v1.cash_generation"], 0.8)
    assert buyback_profile["version"] == "weighted_distance_v1"


def test_weighted_distance_v2_penalizes_large_regime_mismatch():
    hist = pd.DataFrame(
        [
            _state_vector_hist_row(
                company_id="000401",
                action_type="new_debt_issuance",
                action_subtype="new_debt_issuance",
                offset_days=0,
                ticker="BADREGIME",
                **{
                    "normalized_action_family": "capital_structure",
                    "normalized_action_subfamily": "new_debt_issuance",
                    "normalized_action_id": "capital_structure.new_debt_issuance",
                    "state_vector_v1.valuation_multiple": 12.0,
                    "state_vector_v1.rates_level": 0.20,
                    "state_vector_v1.credit_spread": 6.20,
                },
            ),
            _state_vector_hist_row(
                company_id="000402",
                action_type="new_debt_issuance",
                action_subtype="new_debt_issuance",
                offset_days=1,
                ticker="GOODREGIME",
                **{
                    "normalized_action_family": "capital_structure",
                    "normalized_action_subfamily": "new_debt_issuance",
                    "normalized_action_id": "capital_structure.new_debt_issuance",
                    "state_vector_v1.valuation_multiple": 13.2,
                    "state_vector_v1.rates_level": 4.20,
                    "state_vector_v1.credit_spread": 3.05,
                },
            ),
        ]
    )
    payload = {
        "version": "precedent_distance_weights_v2",
        "scopes": {
            "capital_structure": {
                "scope_key": "capital_structure",
                "group_weights": {
                    "identity": 1.0,
                    "capital_structure": 1.4,
                    "liquidity": 1.2,
                    "valuation": 1.1,
                    "market": 0.9,
                    "macro_regime": 1.5,
                },
                "feature_relative_weights": {
                    "state_vector_v1.valuation_multiple": 1.4,
                },
                "penalties": {
                    "regime_rate_gap_threshold": 0.5,
                    "regime_rate_penalty_weight": 1.2,
                    "regime_credit_gap_threshold": 0.75,
                    "regime_credit_penalty_weight": 1.0,
                },
                "use_in_runtime": True,
            }
        },
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/precedent_distance_weights_v2.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        old_path = os.environ.get("PRECEDENT_DISTANCE_V2_WEIGHTS_PATH")
        old_version = os.environ.get("PRECEDENT_DISTANCE_PROFILE_VERSION")
        os.environ["PRECEDENT_DISTANCE_V2_WEIGHTS_PATH"] = path
        os.environ["PRECEDENT_DISTANCE_PROFILE_VERSION"] = "weighted_distance_v2"
        try:
            pack = build_precedent_pack_v2(
                candidate_id="cand-v2-regime",
                run_id="run-v2-regime",
                company_id="001690",
                action_id="capital_structure.new_debt_issuance",
                action_subtype="new_debt_issuance",
                action_params={},
                candidate_features=_state_vector_candidate_features(
                    **{
                        "state_vector_v1.rates_level": 4.25,
                        "state_vector_v1.credit_spread": 3.00,
                    }
                ),
                candidate_regime={"credit_regime": "neutral", "risk_regime": "neutral", "vol_regime": "normal"},
                historical_df=hist,
                top_k=2,
                min_k=1,
            )
        finally:
            if old_path is None:
                os.environ.pop("PRECEDENT_DISTANCE_V2_WEIGHTS_PATH", None)
            else:
                os.environ["PRECEDENT_DISTANCE_V2_WEIGHTS_PATH"] = old_path
            if old_version is None:
                os.environ.pop("PRECEDENT_DISTANCE_PROFILE_VERSION", None)
            else:
                os.environ["PRECEDENT_DISTANCE_PROFILE_VERSION"] = old_version
    assert pack.retrieved_cohorts[0].company_id == "000402"
    diag = pack.mismatch_diagnostics
    assert diag.get("state_distance_version") == "weighted_distance_v2"


def test_candidate_state_feature_weight_multipliers_downweight_current_debt_liquidity_proxy():
    candidate_features = {
        "operating.revenue_ttm_provider_direct": _raw_feature_record(3_802_000_000.0),
        "operating.ebitda_ltm_provider_direct": _raw_feature_record(451_600_000.0),
        "cash_flow.free_cash_flow_ttm": _raw_feature_record(-59_200_000.0),
        "capital_structure.total_debt_provider_direct": _raw_feature_record(1_100_400_000.0),
        "capital_structure.net_debt_normalized": _raw_feature_record(754_700_000.0),
        "liquidity.available_liquidity_normalized": _raw_feature_record(345_700_000.0),
        "capital_structure.current_debt_statement_direct": _raw_feature_record(800_000.0),
        "capital_structure.interest_expense_statement_direct": _raw_feature_record(164_000_000.0),
        "market.market_cap_provider_direct": _raw_feature_record(1_677_742_200.0),
        "market.ev_ebitda": _raw_feature_record(5.3863),
        "market.fcf_yield": _raw_feature_record(-0.0353),
        "market.credit_window_proxy": _raw_feature_record(0.8793),
        "market.equity_window_proxy": _raw_feature_record(0.2693),
        "market.credit_spread_level": _raw_feature_record(0.0121),
        "macro.fed_funds_effective": _raw_feature_record(4.58),
        "macro.hy_oas": _raw_feature_record(2.64),
        "taxonomy.sector": _raw_feature_record("Industrials"),
        "taxonomy.subsector": _raw_feature_record("Commercial Services & Supplies"),
    }

    multipliers = precedent_brain._candidate_state_feature_weight_multipliers(
        candidate_features,
        action_id="capital_structure.new_debt_issuance",
        action_subtype="new_debt_issuance",
    )

    liquidity_multiplier = float(multipliers["state_vector_v1.liquidity_flexibility"])
    rates_multiplier = float(multipliers["state_vector_v1.rates_level"])
    credit_multiplier = float(multipliers["state_vector_v1.credit_spread"])

    assert liquidity_multiplier <= 0.15
    assert rates_multiplier > liquidity_multiplier
    assert credit_multiplier > liquidity_multiplier


def test_weighted_distance_profile_v2_strengthens_capital_structure_regime_features(monkeypatch):
    monkeypatch.setenv("PRECEDENT_DISTANCE_PROFILE_VERSION", "weighted_distance_v2")
    profile = _weighted_distance_profile("capital_structure.new_debt_issuance", "new_debt_issuance")

    assert profile["version"] == "weighted_distance_v2"
    assert "state_vector_v1.rates_level" in profile["critical_features"]
    assert "state_vector_v1.credit_spread" in profile["critical_features"]
    assert profile["regime_rate_gap_threshold"] <= 0.75
    assert profile["regime_credit_gap_threshold"] <= 0.90
    assert profile["feature_relative_weights"]["state_vector_v1.credit_spread"] > profile["feature_relative_weights"]["state_vector_v1.liquidity_flexibility"]


def test_weighted_distance_v2_prefers_credit_regime_match_when_candidate_liquidity_is_proxy(monkeypatch):
    hist = pd.DataFrame(
        [
            _state_vector_hist_row(
                company_id="000501",
                action_type="new_debt_issuance",
                action_subtype="new_debt_issuance",
                offset_days=0,
                ticker="BADREGIME",
                **{
                    "normalized_action_family": "capital_structure",
                    "normalized_action_subfamily": "new_debt_issuance",
                    "normalized_action_id": "capital_structure.new_debt_issuance",
                    "base_sector": "Industrials",
                    "base_industry": "Commercial Services & Supplies",
                    "state_vector_v1.profitability": 0.118,
                    "state_vector_v1.growth": 0.01,
                    "state_vector_v1.gross_obligation_burden": 2.43,
                    "state_vector_v1.net_obligation_burden": 1.67,
                    "state_vector_v1.liquidity_flexibility": 15.0,
                    "state_vector_v1.interest_coverage": 2.76,
                    "state_vector_v1.valuation_multiple": 5.35,
                    "state_vector_v1.cash_generation": -0.016,
                    "state_vector_v1.market_stress": 0.17,
                    "state_vector_v1.market_access": 0.27,
                    "state_vector_v1.rates_level": 0.13,
                    "state_vector_v1.credit_spread": 5.96,
                },
            ),
            _state_vector_hist_row(
                company_id="000502",
                action_type="new_debt_issuance",
                action_subtype="new_debt_issuance",
                offset_days=1,
                ticker="GOODREGIME",
                **{
                    "normalized_action_family": "capital_structure",
                    "normalized_action_subfamily": "new_debt_issuance",
                    "normalized_action_id": "capital_structure.new_debt_issuance",
                    "base_sector": "Industrials",
                    "base_industry": "Commercial Services & Supplies",
                    "state_vector_v1.profitability": 0.105,
                    "state_vector_v1.growth": -0.01,
                    "state_vector_v1.gross_obligation_burden": 2.70,
                    "state_vector_v1.net_obligation_burden": 1.85,
                    "state_vector_v1.liquidity_flexibility": 3.0,
                    "state_vector_v1.interest_coverage": 2.55,
                    "state_vector_v1.valuation_multiple": 6.10,
                    "state_vector_v1.cash_generation": -0.030,
                    "state_vector_v1.market_stress": 0.19,
                    "state_vector_v1.market_access": 0.26,
                    "state_vector_v1.rates_level": 4.58,
                    "state_vector_v1.credit_spread": 2.64,
                },
            ),
        ]
    )
    candidate_features = {
        "operating.revenue_ttm_provider_direct": _raw_feature_record(3_802_000_000.0),
        "operating.ebitda_ltm_provider_direct": _raw_feature_record(451_600_000.0),
        "cash_flow.free_cash_flow_ttm": _raw_feature_record(-59_200_000.0),
        "capital_structure.total_debt_provider_direct": _raw_feature_record(1_100_400_000.0),
        "capital_structure.net_debt_normalized": _raw_feature_record(754_700_000.0),
        "liquidity.available_liquidity_normalized": _raw_feature_record(345_700_000.0),
        "capital_structure.current_debt_statement_direct": _raw_feature_record(800_000.0),
        "capital_structure.interest_expense_statement_direct": _raw_feature_record(164_000_000.0),
        "market.market_cap_provider_direct": _raw_feature_record(1_677_742_200.0),
        "market.ev_ebitda": _raw_feature_record(5.3863),
        "market.fcf_yield": _raw_feature_record(-0.0353),
        "market.credit_window_proxy": _raw_feature_record(0.8793),
        "market.equity_window_proxy": _raw_feature_record(0.2693),
        "market.credit_spread_level": _raw_feature_record(0.0121),
        "macro.fed_funds_effective": _raw_feature_record(4.58),
        "macro.hy_oas": _raw_feature_record(2.64),
        "taxonomy.sector": _raw_feature_record("Industrials"),
        "taxonomy.subsector": _raw_feature_record("Commercial Services & Supplies"),
    }

    monkeypatch.setenv("PRECEDENT_DISTANCE_PROFILE_VERSION", "weighted_distance_v2")
    pack = build_precedent_pack_v2(
        candidate_id="cand-liquidity-proxy",
        run_id="run-liquidity-proxy",
        company_id="001690",
        action_id="capital_structure.new_debt_issuance",
        action_subtype="new_debt_issuance",
        action_params={"amount_usd": 200_000_000.0},
        candidate_features=candidate_features,
        candidate_regime={"credit_regime": "neutral", "risk_regime": "neutral", "vol_regime": "normal"},
        historical_df=hist,
        top_k=2,
        min_k=1,
    )

    assert pack.retrieved_cohorts[0].company_id == "000502"


def test_weighted_distance_v2_prefers_stressed_borrower_in_same_financing_environment(monkeypatch):
    hist = pd.DataFrame(
        [
            _state_vector_hist_row(
                company_id="000601",
                action_type="new_debt_issuance",
                action_subtype="new_debt_issuance",
                offset_days=0,
                ticker="BADREGIME",
                **{
                    "normalized_action_family": "capital_structure",
                    "normalized_action_subfamily": "new_debt_issuance",
                    "normalized_action_id": "capital_structure.new_debt_issuance",
                    "base_sector": "Industrials",
                    "base_industry": "Commercial Services & Supplies",
                    "state_vector_v1.profitability": 0.11,
                    "state_vector_v1.growth": 0.09,
                    "state_vector_v1.gross_obligation_burden": 2.5,
                    "state_vector_v1.net_obligation_burden": 1.8,
                    "state_vector_v1.liquidity_flexibility": 1.9,
                    "state_vector_v1.interest_coverage": 2.8,
                    "state_vector_v1.valuation_multiple": 5.8,
                    "state_vector_v1.cash_generation": -0.02,
                    "state_vector_v1.market_stress": 0.17,
                    "state_vector_v1.market_access": 0.62,
                    "state_vector_v1.rates_level": 0.13,
                    "state_vector_v1.credit_spread": 5.96,
                },
            ),
            _state_vector_hist_row(
                company_id="000602",
                action_type="new_debt_issuance",
                action_subtype="new_debt_issuance",
                offset_days=1,
                ticker="HEALTHY",
                **{
                    "normalized_action_family": "capital_structure",
                    "normalized_action_subfamily": "new_debt_issuance",
                    "normalized_action_id": "capital_structure.new_debt_issuance",
                    "base_sector": "Industrials",
                    "base_industry": "Commercial Services & Supplies",
                    "state_vector_v1.profitability": 0.26,
                    "state_vector_v1.growth": 0.06,
                    "state_vector_v1.gross_obligation_burden": 1.0,
                    "state_vector_v1.net_obligation_burden": 0.4,
                    "state_vector_v1.liquidity_flexibility": 2.5,
                    "state_vector_v1.interest_coverage": 11.0,
                    "state_vector_v1.valuation_multiple": 14.0,
                    "state_vector_v1.cash_generation": 0.09,
                    "state_vector_v1.market_stress": 0.18,
                    "state_vector_v1.market_access": 0.86,
                    "state_vector_v1.rates_level": 4.58,
                    "state_vector_v1.credit_spread": 2.64,
                },
            ),
            _state_vector_hist_row(
                company_id="000603",
                action_type="new_debt_issuance",
                action_subtype="new_debt_issuance",
                offset_days=2,
                ticker="STRESSED",
                **{
                    "normalized_action_family": "capital_structure",
                    "normalized_action_subfamily": "new_debt_issuance",
                    "normalized_action_id": "capital_structure.new_debt_issuance",
                    "base_sector": "Industrials",
                    "base_industry": "Commercial Services & Supplies",
                    "state_vector_v1.profitability": 0.10,
                    "state_vector_v1.growth": 0.05,
                    "state_vector_v1.gross_obligation_burden": 2.7,
                    "state_vector_v1.net_obligation_burden": 1.9,
                    "state_vector_v1.liquidity_flexibility": 1.6,
                    "state_vector_v1.interest_coverage": 2.4,
                    "state_vector_v1.valuation_multiple": 6.2,
                    "state_vector_v1.cash_generation": -0.03,
                    "state_vector_v1.market_stress": 0.18,
                    "state_vector_v1.market_access": 0.61,
                    "state_vector_v1.rates_level": 4.58,
                    "state_vector_v1.credit_spread": 2.64,
                },
            ),
        ]
    )
    candidate_features = {
        "operating.revenue_ttm_provider_direct": _raw_feature_record(3_802_000_000.0),
        "operating.ebitda_ltm_provider_direct": _raw_feature_record(451_600_000.0),
        "cash_flow.free_cash_flow_ttm": _raw_feature_record(-59_200_000.0),
        "capital_structure.total_debt_provider_direct": _raw_feature_record(1_100_400_000.0),
        "capital_structure.net_debt_normalized": _raw_feature_record(754_700_000.0),
        "liquidity.available_liquidity_normalized": _raw_feature_record(345_700_000.0),
        "capital_structure.current_debt_statement_direct": _raw_feature_record(800_000.0),
        "capital_structure.interest_expense_statement_direct": _raw_feature_record(164_000_000.0),
        "market.market_cap_provider_direct": _raw_feature_record(1_677_742_200.0),
        "market.ev_ebitda": _raw_feature_record(5.3863),
        "market.fcf_yield": _raw_feature_record(-0.0353),
        "market.credit_window_proxy": _raw_feature_record(0.8793),
        "market.equity_window_proxy": _raw_feature_record(0.2693),
        "market.credit_spread_level": _raw_feature_record(0.0121),
        "macro.fed_funds_effective": _raw_feature_record(4.58),
        "macro.hy_oas": _raw_feature_record(2.64),
        "taxonomy.sector": _raw_feature_record("Industrials"),
        "taxonomy.subsector": _raw_feature_record("Commercial Services & Supplies"),
    }

    monkeypatch.setenv("PRECEDENT_DISTANCE_PROFILE_VERSION", "weighted_distance_v2")
    pack = build_precedent_pack_v2(
        candidate_id="cand-stressed-align",
        run_id="run-stressed-align",
        company_id="001691",
        action_id="capital_structure.new_debt_issuance",
        action_subtype="new_debt_issuance",
        action_params={"amount_usd": 200_000_000.0},
        candidate_features=candidate_features,
        candidate_regime={"credit_regime": "neutral", "risk_regime": "neutral", "vol_regime": "normal"},
        historical_df=hist,
        top_k=3,
        min_k=1,
    )

    assert pack.retrieved_cohorts[0].company_id == "000603"


def test_narrative_mismatch_triggers_with_real_text():
    hist = _hist_df(35).copy()
    hist["headline"] = "Board approves share repurchase authorization"
    hist["text"] = "capital return buyback repurchase dividend shareholder payout"
    features = _candidate_features()
    features["narrative_text"] = "transformational integration expansion platform acquisition synergy pipeline"
    pack = build_precedent_pack_v2(
        candidate_id="cand-10",
        run_id="run-10",
        company_id="001690",
        action_id="capital_return.open_market_buyback",
        action_subtype="open_market_buyback",
        action_params={"size_pct_market_cap": 0.05, "funding_mix": {"cash": 1.0}},
        candidate_features=features,
        candidate_regime={"credit_regime": "neutral", "risk_regime": "neutral", "vol_regime": "normal"},
        historical_df=hist,
        top_k=20,
        min_k=10,
    )
    diag = pack.mismatch_diagnostics.to_dict() if hasattr(pack.mismatch_diagnostics, "to_dict") else pack.mismatch_diagnostics
    assert diag.get("narrative_real_text_rows", 0) >= 5
    assert diag.get("narrative_mismatch") is True


def test_retrieval_tier_metadata_is_present():
    pack = build_precedent_pack_v2(
        candidate_id="cand-11",
        run_id="run-11",
        company_id="001690",
        action_id="capital_return.open_market_buyback",
        action_subtype="open_market_buyback",
        action_params={"size_pct_market_cap": 0.05, "funding_mix": {"cash": 1.0}},
        candidate_features=_candidate_features(),
        candidate_regime={"credit_regime": "tight", "risk_regime": "risk_off", "vol_regime": "high"},
        historical_df=_hist_df(50),
        top_k=20,
        min_k=10,
    )
    diag = pack.mismatch_diagnostics.to_dict() if hasattr(pack.mismatch_diagnostics, "to_dict") else pack.mismatch_diagnostics
    assert diag.get("retrieval_tier") in {"exact", "sibling_type", "global"}
    assert isinstance(diag.get("regime_prefilter_applied"), bool)
    assert isinstance(diag.get("sector_prefilter_applied"), bool)


def test_tier_confidence_discount_lowers_non_exact_confidence():
    hist = _hist_df(60).copy()
    hist["action_type"] = "capital_return"
    hist["action_subtype"] = ["open_market_buyback" if i < 40 else "dividend_cut" for i in range(len(hist))]
    hist["action_id"] = ["capital_return." + s for s in hist["action_subtype"].astype(str).tolist()]

    common_kwargs = dict(
        candidate_id="cand-tier",
        run_id="run-tier",
        company_id="001690",
        action_params={"size_pct_market_cap": 0.05, "funding_mix": {"cash": 1.0}},
        candidate_features=_candidate_features(),
        candidate_regime={"credit_regime": "neutral", "risk_regime": "neutral", "vol_regime": "normal"},
        historical_df=hist,
        top_k=20,
        min_k=10,
    )
    pack_exact = build_precedent_pack_v2(
        action_id="capital_return.open_market_buyback",
        action_subtype="open_market_buyback",
        **common_kwargs,
    )
    pack_sibling = build_precedent_pack_v2(
        action_id="capital_return.tender_offer_buyback",
        action_subtype="tender_offer_buyback",
        **common_kwargs,
    )
    diag_exact = (
        pack_exact.mismatch_diagnostics.to_dict()
        if hasattr(pack_exact.mismatch_diagnostics, "to_dict")
        else pack_exact.mismatch_diagnostics
    )
    diag_sibling = (
        pack_sibling.mismatch_diagnostics.to_dict()
        if hasattr(pack_sibling.mismatch_diagnostics, "to_dict")
        else pack_sibling.mismatch_diagnostics
    )
    assert diag_exact.get("retrieval_tier") == "exact"
    assert diag_sibling.get("retrieval_tier") in {"sibling_type", "global"}
    assert float(diag_exact.get("tier_confidence_discount", 1.0)) >= 0.99
    assert float(diag_sibling.get("tier_confidence_discount", 1.0)) < 1.0
    exact_pre = float(diag_exact.get("confidence_pre_tier_discount", 0.0))
    exact_disc = float(diag_exact.get("tier_confidence_discount", 1.0))
    sibling_pre = float(diag_sibling.get("confidence_pre_tier_discount", 0.0))
    sibling_disc = float(diag_sibling.get("tier_confidence_discount", 1.0))
    assert abs(pack_exact.calibration_confidence - (exact_pre * exact_disc)) < 1e-9
    assert abs(pack_sibling.calibration_confidence - (sibling_pre * sibling_disc)) < 1e-9


def test_refinancing_infers_term_loan_family_for_exact_matching():
    effective_subtype = precedent_brain._effective_action_subtype(
        "capital_structure.refinancing",
        None,
        {"instrument_type": "term_loan"},
    )
    assert effective_subtype == "refinancing_term_loan_family"
    resolved_subtype = precedent_brain._resolved_normalized_subfamily(
        "capital_structure",
        "refinancing",
        "loan_refinancing",
        "Term Loan B",
    )
    assert resolved_subtype == "refinancing_term_loan_family"
    assert precedent_brain._historical_action_family(
        action_type="loan_refinancing",
        action_subtype="Revolver/Line >= 1 Yr.",
    ) == "capital_structure.refinancing_revolver_family"
    family_weights = precedent_brain._candidate_action_family_weights(
        "capital_structure.refinancing",
        effective_subtype,
    )
    assert family_weights[0][0] == "capital_structure.refinancing_term_loan_family"
    exact_keys = precedent_brain._candidate_exact_action_keys(
        "capital_structure.refinancing",
        effective_subtype,
        "refinancing",
        "refinancing",
    )
    assert exact_keys == ("refinancing_term_loan_family",)
    assert precedent_brain._candidate_action_id_keys(
        "capital_structure.refinancing",
        effective_subtype,
    ) == ()


def test_hard_sector_market_cap_prefilter_applies_when_enough_depth():
    hist = _hist_df(60).copy()
    hist["action_type"] = "capital_return"
    hist["action_subtype"] = "open_market_buyback"
    hist["action_id"] = "capital_return.open_market_buyback"
    hist.loc[:29, "base_sector"] = "TECH"
    hist.loc[30:, "base_sector"] = "UTILITIES"
    hist.loc[:29, "base_market_cap"] = np.linspace(900.0, 1800.0, 30)
    hist.loc[30:, "base_market_cap"] = np.linspace(5.0e8, 1.8e9, 30)

    features = _candidate_features()
    features["sector"] = "TECH"
    features["market_cap"] = 1200.0

    pack = build_precedent_pack_v2(
        candidate_id="cand-hard-filter",
        run_id="run-hard-filter",
        company_id="001690",
        action_id="capital_return.open_market_buyback",
        action_subtype="open_market_buyback",
        action_params={"size_pct_market_cap": 0.05},
        candidate_features=features,
        candidate_regime={"credit_regime": "neutral", "risk_regime": "neutral", "vol_regime": "normal"},
        historical_df=hist,
        top_k=20,
        min_k=10,
    )
    diag = pack.mismatch_diagnostics.to_dict() if hasattr(pack.mismatch_diagnostics, "to_dict") else pack.mismatch_diagnostics
    assert diag.get("hard_prefilter_applied") is True
    assert diag.get("hard_prefilter_relaxed") is False
    assert diag.get("market_cap_prefilter_applied") is True
    assert all((c.key_state_features.get("base_sector") or "").upper() == "TECH" for c in pack.retrieved_cohorts)
    assert all(float(c.key_state_features.get("base_market_cap") or 0.0) < 1.0e7 for c in pack.retrieved_cohorts)


def test_hard_prefilter_relaxes_when_bucket_and_sector_too_sparse():
    hist = _hist_df(30).copy()
    hist["action_type"] = "capital_return"
    hist["action_subtype"] = "open_market_buyback"
    hist["action_id"] = "capital_return.open_market_buyback"
    hist["base_sector"] = "TECH"
    hist.loc[:4, "base_market_cap"] = np.linspace(950.0, 1500.0, 5)
    hist.loc[5:, "base_market_cap"] = np.linspace(3.0e8, 1.2e9, 25)

    features = _candidate_features()
    features["sector"] = "TECH"
    features["market_cap"] = 1200.0

    pack = build_precedent_pack_v2(
        candidate_id="cand-hard-relax",
        run_id="run-hard-relax",
        company_id="001690",
        action_id="capital_return.open_market_buyback",
        action_subtype="open_market_buyback",
        action_params={"size_pct_market_cap": 0.05},
        candidate_features=features,
        candidate_regime={"credit_regime": "neutral", "risk_regime": "neutral", "vol_regime": "normal"},
        historical_df=hist,
        top_k=20,
        min_k=10,
    )
    diag = pack.mismatch_diagnostics.to_dict() if hasattr(pack.mismatch_diagnostics, "to_dict") else pack.mismatch_diagnostics
    assert diag.get("hard_prefilter_applied") is False
    assert diag.get("hard_prefilter_relaxed") is True
    assert any(float(c.key_state_features.get("base_market_cap") or 0.0) > 1.0e8 for c in pack.retrieved_cohorts)


def test_weighted_distance_v2_feature_transforms_compress_extreme_safety_tails():
    old_version = os.environ.get("PRECEDENT_DISTANCE_PROFILE_VERSION")
    os.environ["PRECEDENT_DISTANCE_PROFILE_VERSION"] = "weighted_distance_v2"
    try:
        profile = _weighted_distance_profile("capital_return.dividend_increase", "dividend_increase")
    finally:
        if old_version is None:
            os.environ.pop("PRECEDENT_DISTANCE_PROFILE_VERSION", None)
        else:
            os.environ["PRECEDENT_DISTANCE_PROFILE_VERSION"] = old_version

    embedding_cols = (
        "state_vector_v1.liquidity_flexibility",
        "state_vector_v1.interest_coverage",
        "state_vector_v1.valuation_multiple",
    )
    emb_raw = np.array(
        [
            [6.0, 20.0, 12.0],
            [150.0, 114.0, 4.0],
        ],
        dtype=float,
    )
    candidate_vec_raw = np.array([10.0, 30.0, 12.0], dtype=float)
    transformed_emb, transformed_candidate, transformed_features = _apply_matching_feature_transforms(
        emb_raw,
        candidate_vec_raw,
        embedding_cols,
        profile,
    )

    raw_liquidity_ratio = abs(emb_raw[1, 0] - candidate_vec_raw[0]) / abs(emb_raw[0, 0] - candidate_vec_raw[0])
    transformed_liquidity_ratio = abs(transformed_emb[1, 0] - transformed_candidate[0]) / abs(
        transformed_emb[0, 0] - transformed_candidate[0]
    )
    raw_coverage_ratio = abs(emb_raw[1, 1] - candidate_vec_raw[1]) / abs(emb_raw[0, 1] - candidate_vec_raw[1])
    transformed_coverage_ratio = abs(transformed_emb[1, 1] - transformed_candidate[1]) / abs(
        transformed_emb[0, 1] - transformed_candidate[1]
    )

    assert "state_vector_v1.liquidity_flexibility" in transformed_features
    assert "state_vector_v1.interest_coverage" in transformed_features
    assert transformed_liquidity_ratio < raw_liquidity_ratio
    assert transformed_coverage_ratio < raw_coverage_ratio


def test_weighted_distance_v2_latent_regime_penalty_prefers_same_buyback_regime():
    feature_names = [
        "state_vector_v1.growth",
        "state_vector_v1.valuation_multiple",
        "state_vector_v1.cash_generation",
    ]
    compact_rows = [
        {
            "state_vector_v1.growth": 0.18,
            "state_vector_v1.valuation_multiple": 48.0,
            "state_vector_v1.cash_generation": 0.01,
        },
        {
            "state_vector_v1.growth": 0.16,
            "state_vector_v1.valuation_multiple": 42.0,
            "state_vector_v1.cash_generation": 0.015,
        },
        {
            "state_vector_v1.growth": 0.02,
            "state_vector_v1.valuation_multiple": 13.0,
            "state_vector_v1.cash_generation": 0.06,
        },
        {
            "state_vector_v1.growth": 0.00,
            "state_vector_v1.valuation_multiple": 9.0,
            "state_vector_v1.cash_generation": 0.07,
        },
    ]
    latent_model = fit_latent_regime_kmeans(
        raw_feature_matrix_from_compacts(compact_rows, feature_names=feature_names),
        feature_names=feature_names,
        n_clusters=2,
        seed=7,
        max_iter=20,
    )
    payload = {
        "version": "precedent_distance_weights_v2",
        "scopes": {
            "capital_return.open_market_buyback": {
                "scope_key": "capital_return.open_market_buyback",
                "default_enabled": True,
                "use_in_runtime": True,
                "latent_regime_model": latent_model,
                "latent_regime_penalty_weight": 2.0,
            }
        },
    }
    embedding_cols = tuple(feature_names)
    emb_raw = np.array(
        [
            [0.17, 44.0, 0.012],
            [0.01, 11.0, 0.065],
        ],
        dtype=float,
    )
    candidate_vec_raw = np.array([0.19, 50.0, 0.010], dtype=float)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/precedent_distance_weights_v2.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        old_path = os.environ.get("PRECEDENT_DISTANCE_V2_WEIGHTS_PATH")
        old_version = os.environ.get("PRECEDENT_DISTANCE_PROFILE_VERSION")
        os.environ["PRECEDENT_DISTANCE_V2_WEIGHTS_PATH"] = path
        os.environ["PRECEDENT_DISTANCE_PROFILE_VERSION"] = "weighted_distance_v2"
        try:
            result = _weighted_state_similarity_v2(
                emb_raw=emb_raw,
                candidate_vec_raw=candidate_vec_raw,
                embedding_cols=embedding_cols,
                action_id="capital_return.open_market_buyback",
                action_subtype="open_market_buyback",
            )
        finally:
            if old_path is None:
                os.environ.pop("PRECEDENT_DISTANCE_V2_WEIGHTS_PATH", None)
            else:
                os.environ["PRECEDENT_DISTANCE_V2_WEIGHTS_PATH"] = old_path
            if old_version is None:
                os.environ.pop("PRECEDENT_DISTANCE_PROFILE_VERSION", None)
            else:
                os.environ["PRECEDENT_DISTANCE_PROFILE_VERSION"] = old_version

    assert result["latent_regime_penalty_factor"][0] > result["latent_regime_penalty_factor"][1]


def test_weighted_distance_v2_target_regime_mixture_can_shift_buyback_weights_by_target():
    feature_names = [
        "state_vector_v1.growth",
        "state_vector_v1.valuation_multiple",
        "state_vector_v1.cash_generation",
    ]
    compact_rows = [
        {
            "state_vector_v1.growth": 0.18,
            "state_vector_v1.valuation_multiple": 48.0,
            "state_vector_v1.cash_generation": 0.01,
        },
        {
            "state_vector_v1.growth": 0.16,
            "state_vector_v1.valuation_multiple": 42.0,
            "state_vector_v1.cash_generation": 0.015,
        },
        {
            "state_vector_v1.growth": 0.00,
            "state_vector_v1.valuation_multiple": 11.0,
            "state_vector_v1.cash_generation": 0.06,
        },
        {
            "state_vector_v1.growth": -0.02,
            "state_vector_v1.valuation_multiple": 9.0,
            "state_vector_v1.cash_generation": 0.07,
        },
    ]
    model = fit_latent_regime_kmeans(
        raw_feature_matrix_from_compacts(compact_rows, feature_names=feature_names),
        feature_names=feature_names,
        n_clusters=2,
        seed=7,
        max_iter=20,
    )
    payload = {
        "version": "precedent_distance_weights_v2",
        "scopes": {
            "capital_return.open_market_buyback": {
                "scope_key": "capital_return.open_market_buyback",
                "default_enabled": True,
                "use_in_runtime": True,
                "target_regime_mixture": {
                    "model": model,
                    "regimes": [
                        {
                            "cluster": 0,
                            "feature_relative_weights": {
                                "state_vector_v1.growth": 0.6,
                                "state_vector_v1.valuation_multiple": 2.5,
                                "state_vector_v1.cash_generation": 0.4,
                            },
                            "interaction_terms": [],
                        },
                        {
                            "cluster": 1,
                            "feature_relative_weights": {
                                "state_vector_v1.growth": 0.4,
                                "state_vector_v1.valuation_multiple": 0.6,
                                "state_vector_v1.cash_generation": 2.2,
                            },
                            "interaction_terms": [],
                        },
                    ],
                },
            }
        },
    }
    embedding_cols = tuple(feature_names)
    emb_raw = np.array(
        [
            [0.15, 44.0, 0.015],
            [0.02, 12.0, 0.06],
        ],
        dtype=float,
    )
    candidate_vec_raw = np.array([0.19, 50.0, 0.010], dtype=float)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/precedent_distance_weights_v2.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        old_path = os.environ.get("PRECEDENT_DISTANCE_V2_WEIGHTS_PATH")
        old_version = os.environ.get("PRECEDENT_DISTANCE_PROFILE_VERSION")
        os.environ["PRECEDENT_DISTANCE_V2_WEIGHTS_PATH"] = path
        os.environ["PRECEDENT_DISTANCE_PROFILE_VERSION"] = "weighted_distance_v2"
        try:
            result = _weighted_state_similarity_v2(
                emb_raw=emb_raw,
                candidate_vec_raw=candidate_vec_raw,
                embedding_cols=embedding_cols,
                action_id="capital_return.open_market_buyback",
                action_subtype="open_market_buyback",
            )
        finally:
            if old_path is None:
                os.environ.pop("PRECEDENT_DISTANCE_V2_WEIGHTS_PATH", None)
            else:
                os.environ["PRECEDENT_DISTANCE_V2_WEIGHTS_PATH"] = old_path
            if old_version is None:
                os.environ.pop("PRECEDENT_DISTANCE_PROFILE_VERSION", None)
            else:
                os.environ["PRECEDENT_DISTANCE_PROFILE_VERSION"] = old_version

    assert result["state_similarity"][0] > result["state_similarity"][1]
    assert result["target_regime_membership"].size == 2


def test_weighted_distance_v2_prefers_same_industry_over_cross_sector_safety_proxy():
    hist = pd.DataFrame(
        [
            _state_vector_hist_row(
                company_id="000501",
                action_type="dividend_increase",
                action_subtype="dividend_increase",
                offset_days=0,
                ticker="GOODIND",
                **{
                    "base_sector": "INDUSTRIALS",
                    "subsector": "MACHINERY",
                    "normalized_action_family": "capital_return",
                    "normalized_action_subfamily": "dividend_increase",
                    "normalized_action_id": "capital_return.dividend_increase",
                    "state_vector_v1.size_log_revenue": 9.8,
                    "state_vector_v1.profitability": 0.24,
                    "state_vector_v1.growth": 0.02,
                    "state_vector_v1.gross_obligation_burden": 1.15,
                    "state_vector_v1.net_obligation_burden": 0.15,
                    "state_vector_v1.liquidity_flexibility": 12.0,
                    "state_vector_v1.interest_coverage": 22.0,
                    "state_vector_v1.valuation_multiple": 11.8,
                    "state_vector_v1.cash_generation": 0.05,
                    "state_vector_v1.market_stress": 0.24,
                    "state_vector_v1.market_access": 0.79,
                },
            ),
            _state_vector_hist_row(
                company_id="000502",
                action_type="dividend_increase",
                action_subtype="dividend_increase",
                offset_days=1,
                ticker="WRONGSEC",
                **{
                    "base_sector": "ENERGY",
                    "subsector": "INTEGRATED_OIL_GAS",
                    "normalized_action_family": "capital_return",
                    "normalized_action_subfamily": "dividend_increase",
                    "normalized_action_id": "capital_return.dividend_increase",
                    "state_vector_v1.size_log_revenue": 9.9,
                    "state_vector_v1.profitability": 0.09,
                    "state_vector_v1.growth": 0.01,
                    "state_vector_v1.gross_obligation_burden": 0.85,
                    "state_vector_v1.net_obligation_burden": 0.05,
                    "state_vector_v1.liquidity_flexibility": 145.0,
                    "state_vector_v1.interest_coverage": 118.0,
                    "state_vector_v1.valuation_multiple": 4.2,
                    "state_vector_v1.cash_generation": 0.14,
                    "state_vector_v1.market_stress": 0.40,
                    "state_vector_v1.market_access": 0.76,
                },
            ),
        ]
    )
    payload = {
        "version": "precedent_distance_weights_v2",
        "scopes": {
            "capital_return.dividend_increase": {
                "scope_key": "capital_return.dividend_increase",
                "default_enabled": True,
                "use_in_runtime": True,
            }
        },
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/precedent_distance_weights_v2.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        old_path = os.environ.get("PRECEDENT_DISTANCE_V2_WEIGHTS_PATH")
        old_version = os.environ.get("PRECEDENT_DISTANCE_PROFILE_VERSION")
        os.environ["PRECEDENT_DISTANCE_V2_WEIGHTS_PATH"] = path
        os.environ.pop("PRECEDENT_DISTANCE_PROFILE_VERSION", None)
        try:
            pack = build_precedent_pack_v2(
                candidate_id="cand-v2-industry",
                run_id="run-v2-industry",
                company_id="001690",
                action_id="capital_return.dividend_increase",
                action_subtype="dividend_increase",
                action_params={},
                candidate_features=_state_vector_candidate_features(
                    **{
                        "taxonomy.sector": "Industrials",
                        "taxonomy.subsector": "Machinery",
                        "sector": "INDUSTRIALS",
                        "state_vector_v1.size_log_revenue": 9.75,
                        "state_vector_v1.profitability": 0.28,
                        "state_vector_v1.growth": 0.01,
                        "state_vector_v1.gross_obligation_burden": 0.95,
                        "state_vector_v1.net_obligation_burden": 0.03,
                        "state_vector_v1.liquidity_flexibility": 148.0,
                        "state_vector_v1.interest_coverage": 114.0,
                        "state_vector_v1.valuation_multiple": 12.4,
                        "state_vector_v1.cash_generation": 0.08,
                        "state_vector_v1.market_stress": 0.25,
                        "state_vector_v1.market_access": 0.79,
                    }
                ),
                candidate_regime={"credit_regime": "neutral", "risk_regime": "neutral", "vol_regime": "normal"},
                historical_df=hist,
                top_k=2,
                min_k=1,
            )
        finally:
            if old_path is None:
                os.environ.pop("PRECEDENT_DISTANCE_V2_WEIGHTS_PATH", None)
            else:
                os.environ["PRECEDENT_DISTANCE_V2_WEIGHTS_PATH"] = old_path
            if old_version is None:
                os.environ.pop("PRECEDENT_DISTANCE_PROFILE_VERSION", None)
            else:
                os.environ["PRECEDENT_DISTANCE_PROFILE_VERSION"] = old_version
    assert pack.retrieved_cohorts[0].company_id == "000501"


def test_weighted_distance_v2_identity_prefilter_drops_known_cross_sector_rows_when_same_sector_depth_exists():
    hist_rows = []
    for i in range(5):
        hist_rows.append(
            _state_vector_hist_row(
                company_id=f"00060{i}",
                action_type="dividend_increase",
                action_subtype="dividend_increase",
                offset_days=i,
                ticker=f"GOOD{i}",
                **{
                    "base_sector": "INDUSTRIALS",
                    "subsector": "MACHINERY",
                    "normalized_action_family": "capital_return",
                    "normalized_action_subfamily": "dividend_increase",
                    "normalized_action_id": "capital_return.dividend_increase",
                    "state_vector_v1.size_log_revenue": 9.8 + 0.01 * i,
                    "state_vector_v1.profitability": 0.24 + 0.002 * i,
                    "state_vector_v1.growth": 0.01,
                    "state_vector_v1.gross_obligation_burden": 1.05,
                    "state_vector_v1.net_obligation_burden": 0.12,
                    "state_vector_v1.liquidity_flexibility": 9.0 + i,
                    "state_vector_v1.interest_coverage": 18.0 + i,
                    "state_vector_v1.valuation_multiple": 11.7 + 0.1 * i,
                    "state_vector_v1.cash_generation": 0.05,
                    "state_vector_v1.market_stress": 0.24,
                    "state_vector_v1.market_access": 0.79,
                },
            )
        )
    hist_rows.append(
        _state_vector_hist_row(
            company_id="000699",
            action_type="dividend_increase",
            action_subtype="dividend_increase",
            offset_days=20,
            ticker="WRONGSEC",
            **{
                "base_sector": "ENERGY",
                "subsector": "INTEGRATED_OIL_GAS",
                "normalized_action_family": "capital_return",
                "normalized_action_subfamily": "dividend_increase",
                "normalized_action_id": "capital_return.dividend_increase",
                "state_vector_v1.size_log_revenue": 9.9,
                "state_vector_v1.profitability": 0.09,
                "state_vector_v1.growth": 0.01,
                "state_vector_v1.gross_obligation_burden": 0.85,
                "state_vector_v1.net_obligation_burden": 0.05,
                "state_vector_v1.liquidity_flexibility": 145.0,
                "state_vector_v1.interest_coverage": 118.0,
                "state_vector_v1.valuation_multiple": 4.2,
                "state_vector_v1.cash_generation": 0.14,
                "state_vector_v1.market_stress": 0.40,
                "state_vector_v1.market_access": 0.76,
            },
        )
    )
    hist = pd.DataFrame(hist_rows)
    payload = {
        "version": "precedent_distance_weights_v2",
        "scopes": {
            "capital_return.dividend_increase": {
                "scope_key": "capital_return.dividend_increase",
                "default_enabled": True,
                "use_in_runtime": True,
            }
        },
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/precedent_distance_weights_v2.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        old_path = os.environ.get("PRECEDENT_DISTANCE_V2_WEIGHTS_PATH")
        old_version = os.environ.get("PRECEDENT_DISTANCE_PROFILE_VERSION")
        os.environ["PRECEDENT_DISTANCE_V2_WEIGHTS_PATH"] = path
        os.environ.pop("PRECEDENT_DISTANCE_PROFILE_VERSION", None)
        try:
            pack = build_precedent_pack_v2(
                candidate_id="cand-v2-identity-gate",
                run_id="run-v2-identity-gate",
                company_id="001690",
                action_id="capital_return.dividend_increase",
                action_subtype="dividend_increase",
                action_params={},
                candidate_features=_state_vector_candidate_features(
                    **{
                        "taxonomy.sector": "Industrials",
                        "taxonomy.subsector": "Machinery",
                        "sector": "INDUSTRIALS",
                        "state_vector_v1.size_log_revenue": 9.75,
                        "state_vector_v1.profitability": 0.28,
                        "state_vector_v1.growth": 0.01,
                        "state_vector_v1.gross_obligation_burden": 0.95,
                        "state_vector_v1.net_obligation_burden": 0.03,
                        "state_vector_v1.liquidity_flexibility": 148.0,
                        "state_vector_v1.interest_coverage": 114.0,
                        "state_vector_v1.valuation_multiple": 12.4,
                        "state_vector_v1.cash_generation": 0.08,
                        "state_vector_v1.market_stress": 0.25,
                        "state_vector_v1.market_access": 0.79,
                    }
                ),
                candidate_regime={"credit_regime": "neutral", "risk_regime": "neutral", "vol_regime": "normal"},
                historical_df=hist,
                top_k=5,
                min_k=10,
            )
        finally:
            if old_path is None:
                os.environ.pop("PRECEDENT_DISTANCE_V2_WEIGHTS_PATH", None)
            else:
                os.environ["PRECEDENT_DISTANCE_V2_WEIGHTS_PATH"] = old_path
            if old_version is None:
                os.environ.pop("PRECEDENT_DISTANCE_PROFILE_VERSION", None)
            else:
                os.environ["PRECEDENT_DISTANCE_PROFILE_VERSION"] = old_version
    diag = pack.mismatch_diagnostics.to_dict() if hasattr(pack.mismatch_diagnostics, "to_dict") else pack.mismatch_diagnostics
    assert diag.get("identity_prefilter_applied") is True
    assert diag.get("identity_prefilter_mode") == "subsector"
    assert all(case.company_id != "000699" for case in pack.retrieved_cohorts[:5])


def test_regime_thresholds_handles_missing_macro_columns():
    frame = pd.DataFrame(
        {
            "ticker": ["IFF", "DD"],
            "company_id": ["006078", "004060"],
            "action_date": pd.to_datetime(["2020-05-15", "2020-04-16"], utc=True),
            "normalized_action_id": ["capital_structure.refinancing", "capital_structure.refinancing"],
            "base_market_cap": [10.0, 20.0],
            "base_revenue_ttm": [1.0, 2.0],
            "base_total_debt": [3.0, 4.0],
        }
    )
    thresholds = precedent_brain._regime_thresholds(frame)

    assert thresholds == {
        "hy_q25": 0.0,
        "hy_q75": 0.0,
        "vix_q25": 0.0,
        "vix_q75": 0.0,
    }
    index = build_precedent_retrieval_index(frame)
    assert len(index.df) == 2
