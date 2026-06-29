from __future__ import annotations

import json

import numpy as np
import pandas as pd

import src.pipeline.precedent_quality_learning as precedent_quality_learning
from src.pipeline.latent_regime_model import fit_latent_regime_kmeans, raw_feature_matrix_from_compacts
from src.pipeline.precedent_quality_learning import (
    _feature_advantage_from_compacts,
    _outcome_aware_reranker_prior,
    _second_stage_reranker_prior,
    build_pairwise_matrix,
    build_outcome_aware_reranker_matrix,
    build_scope_payload_with_pairwise_weights,
    build_second_stage_reranker_matrix,
    learn_feature_transforms_from_pairwise_supervision,
    load_feature_transform_prior,
    load_feature_weight_prior,
    search_target_regime_mixture_from_supervision,
)


def test_feature_advantage_from_compacts_respects_transform_spec():
    row = {
        "target_compact": {"state_vector_v1.valuation_multiple": 60.0},
        "positive_compact": {"state_vector_v1.valuation_multiple": 38.0},
        "negative_compact": {"state_vector_v1.valuation_multiple": 22.0},
    }

    raw_advantage = _feature_advantage_from_compacts(row, "state_vector_v1.valuation_multiple", {})
    clipped_advantage = _feature_advantage_from_compacts(
        row,
        "state_vector_v1.valuation_multiple",
        {"kind": "signed_log1p_cap", "cap": 25.0},
    )

    assert raw_advantage is not None
    assert clipped_advantage is not None
    assert clipped_advantage < raw_advantage


def test_build_scope_payload_with_pairwise_weights_persists_feature_transforms(tmp_path):
    base_payload = {
        "scopes": {
            "capital_return.open_market_buyback": {
                "feature_relative_weights": {
                    "state_vector_v1.valuation_multiple": 1.0,
                }
            }
        }
    }
    base_path = tmp_path / "base.json"
    base_path.write_text(json.dumps(base_payload))

    payload = build_scope_payload_with_pairwise_weights(
        base_path,
        scope_key="capital_return.open_market_buyback",
        learned_weights={
            "state_vector_v1.valuation_multiple": 1.5,
            "pairwise_interaction::state_vector_v1.growth::state_vector_v1.valuation_multiple": 0.8,
            "latent_regime::similarity": 0.7,
        },
        learned_feature_transforms={
            "state_vector_v1.valuation_multiple": {"kind": "signed_asinh", "scale": 40.0}
        },
        latent_regime_model={
            "version": "latent_regime_kmeans_soft_v1",
            "feature_names": ["state_vector_v1.growth", "state_vector_v1.valuation_multiple"],
            "n_clusters": 2,
            "medians": [0.0, 20.0],
            "scales": [0.1, 10.0],
            "centroids": [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]],
            "temperature": 1.0,
        },
    )

    scope = payload["scopes"]["capital_return.open_market_buyback"]
    assert scope["use_in_runtime"] is True
    assert scope["default_enabled"] is True
    assert scope["feature_relative_weights"]["state_vector_v1.valuation_multiple"] == 1.5
    assert scope["feature_transforms"]["state_vector_v1.valuation_multiple"] == {
        "kind": "signed_asinh",
        "scale": 40.0,
    }
    assert scope["interaction_terms"] == [
        {
            "features": [
                "state_vector_v1.growth",
                "state_vector_v1.valuation_multiple",
            ],
            "weight": 0.8,
        }
    ]
    assert scope["latent_regime_penalty_weight"] == 0.7
    assert scope["latent_regime_model"]["n_clusters"] == 2


def test_load_feature_transform_prior_respects_identity_mode(tmp_path):
    base_payload = {
        "scopes": {
            "capital_return.open_market_buyback": {
                "feature_transform_mode": "identity",
                "feature_transforms": {
                    "state_vector_v1.cash_generation": {"kind": "signed_asinh", "scale": 0.05}
                },
            }
        }
    }
    base_path = tmp_path / "base.json"
    base_path.write_text(json.dumps(base_payload))

    transforms = load_feature_transform_prior(
        base_path,
        scope_key="capital_return.open_market_buyback",
        feature_names=[
            "state_vector_v1.valuation_multiple",
            "state_vector_v1.cash_generation",
        ],
    )

    assert transforms["state_vector_v1.valuation_multiple"] == {}
    assert transforms["state_vector_v1.cash_generation"] == {
        "kind": "signed_asinh",
        "scale": 0.05,
    }


def test_build_scope_payload_with_pairwise_weights_persists_identity_transform_mode(tmp_path):
    base_payload = {
        "scopes": {
            "capital_return.open_market_buyback": {
                "feature_relative_weights": {
                    "state_vector_v1.valuation_multiple": 1.0,
                },
                "feature_transforms": {
                    "state_vector_v1.valuation_multiple": {"kind": "signed_log1p_cap", "cap": 25.0}
                },
            }
        }
    }
    base_path = tmp_path / "base.json"
    base_path.write_text(json.dumps(base_payload))

    payload = build_scope_payload_with_pairwise_weights(
        base_path,
        scope_key="capital_return.open_market_buyback",
        learned_weights={"state_vector_v1.valuation_multiple": 1.2},
        feature_transform_mode="identity",
    )

    scope = payload["scopes"]["capital_return.open_market_buyback"]
    assert scope["feature_transform_mode"] == "identity"
    assert "feature_transforms" not in scope


def test_build_scope_payload_with_pairwise_weights_persists_second_stage_reranker(tmp_path):
    base_payload = {
        "scopes": {
            "capital_return.open_market_buyback": {
                "feature_relative_weights": {
                    "state_vector_v1.valuation_multiple": 1.0,
                }
            }
        }
    }
    base_path = tmp_path / "base.json"
    base_path.write_text(json.dumps(base_payload))

    payload = build_scope_payload_with_pairwise_weights(
        base_path,
        scope_key="capital_return.open_market_buyback",
        learned_weights={},
        second_stage_reranker={
            "feature_weights": {
                "base_state_similarity": 1.4,
                "unweighted_state_similarity": 1.2,
            },
            "bias": 0.25,
            "shortlist_size": 90,
        },
    )

    scope = payload["scopes"]["capital_return.open_market_buyback"]
    assert scope["second_stage_reranker"]["feature_weights"]["base_state_similarity"] == 1.4
    assert scope["second_stage_reranker"]["feature_weights"]["unweighted_state_similarity"] == 1.2
    assert scope["second_stage_reranker"]["bias"] == 0.25
    assert scope["second_stage_reranker"]["shortlist_size"] == 90


def test_build_scope_payload_with_pairwise_weights_persists_outcome_aware_reranker(tmp_path):
    base_payload = {
        "scopes": {
            "capital_return.open_market_buyback": {
                "feature_relative_weights": {
                    "state_vector_v1.valuation_multiple": 1.0,
                }
            }
        }
    }
    base_path = tmp_path / "base.json"
    base_path.write_text(json.dumps(base_payload))

    payload = build_scope_payload_with_pairwise_weights(
        base_path,
        scope_key="capital_return.open_market_buyback",
        learned_weights={},
        outcome_aware_reranker={
            "feature_weights": {
                "current_similarity_score": 1.3,
                "outcome_valuation_score": 0.9,
            },
            "bias": 0.1,
            "shortlist_size": 50,
        },
    )

    scope = payload["scopes"]["capital_return.open_market_buyback"]
    assert scope["outcome_aware_reranker"]["feature_weights"]["current_similarity_score"] == 1.3
    assert scope["outcome_aware_reranker"]["feature_weights"]["outcome_valuation_score"] == 0.9
    assert scope["outcome_aware_reranker"]["bias"] == 0.1
    assert scope["outcome_aware_reranker"]["shortlist_size"] == 50


def test_build_second_stage_reranker_matrix_uses_same_action_pairs_only():
    df = pd.DataFrame(
        [
            {
                "company_id": "000001",
                "as_of_time": "2024-01-01T00:00:00+00:00",
                "anchor_action_id": "capital_return.open_market_buyback",
                "competitor_action_id": "capital_return.open_market_buyback",
                "target_compact": {
                    "state_vector_v1.valuation_multiple": 40.0,
                    "state_vector_v1.cash_generation": 0.02,
                },
                "positive_compact": {
                    "state_vector_v1.valuation_multiple": 35.0,
                    "state_vector_v1.cash_generation": 0.021,
                },
                "target_sector": "TECH",
                "target_subsector": "SEMI",
                "positive_sector": "TECH",
                "positive_subsector": "SEMI",
                "negative_sector": "TECH",
                "negative_subsector": "HARDWARE",
                "target_action_scale": 0.20,
                "positive_action_scale": 0.21,
                "negative_action_scale": 0.80,
                "negative_compact": {
                    "state_vector_v1.valuation_multiple": 10.0,
                    "state_vector_v1.cash_generation": 0.07,
                },
            },
            {
                "company_id": "000002",
                "as_of_time": "2024-01-02T00:00:00+00:00",
                "anchor_action_id": "capital_return.open_market_buyback",
                "competitor_action_id": "capital_return.dividend_increase",
                "target_compact": {
                    "state_vector_v1.valuation_multiple": 12.0,
                    "state_vector_v1.cash_generation": 0.04,
                },
                "positive_compact": {
                    "state_vector_v1.valuation_multiple": 11.0,
                    "state_vector_v1.cash_generation": 0.039,
                },
                "target_sector": "INDUSTRIALS",
                "target_subsector": "ELEC",
                "positive_sector": "INDUSTRIALS",
                "positive_subsector": "ELEC",
                "negative_sector": "TECH",
                "negative_subsector": "SEMI",
                "target_action_scale": 0.05,
                "positive_action_scale": 0.08,
                "negative_action_scale": 0.30,
                "negative_compact": {
                    "state_vector_v1.valuation_multiple": 8.0,
                    "state_vector_v1.cash_generation": 0.06,
                },
            },
        ]
    )

    matrix = build_second_stage_reranker_matrix(df, same_action_only=True)

    assert matrix["pair_count"] == 1
    assert len(matrix["selected_features"]) >= 4
    assert "parameter_similarity" in matrix["selected_features"]
    assert "sector_similarity" in matrix["selected_features"]
    assert "regime_similarity" in matrix["selected_features"]
    assert matrix["X"].shape[0] == 2


def test_build_second_stage_reranker_matrix_uses_pair_metadata_for_sector_and_scale():
    df = pd.DataFrame(
        [
            {
                "company_id": "000001",
                "as_of_time": "2024-01-01T00:00:00+00:00",
                "anchor_action_id": "capital_structure.new_debt_issuance",
                "competitor_action_id": "capital_structure.new_debt_issuance",
                "target_compact": {
                    "state_vector_v1.valuation_multiple": 20.0,
                    "state_vector_v1.cash_generation": 0.02,
                },
                "positive_compact": {
                    "state_vector_v1.valuation_multiple": 19.0,
                    "state_vector_v1.cash_generation": 0.021,
                },
                "negative_compact": {
                    "state_vector_v1.valuation_multiple": 18.0,
                    "state_vector_v1.cash_generation": 0.022,
                },
                "target_sector": "TECH",
                "target_subsector": "SEMI",
                "positive_sector": "TECH",
                "positive_subsector": "SEMI",
                "negative_sector": "TECH",
                "negative_subsector": "HARDWARE",
                "target_action_scale": 0.20,
                "positive_action_scale": 0.21,
                "negative_action_scale": 0.80,
            },
            {
                "company_id": "000002",
                "as_of_time": "2024-01-02T00:00:00+00:00",
                "anchor_action_id": "capital_structure.new_debt_issuance",
                "competitor_action_id": "capital_structure.new_debt_issuance",
                "target_compact": {
                    "state_vector_v1.valuation_multiple": 10.0,
                    "state_vector_v1.cash_generation": 0.03,
                },
                "positive_compact": {
                    "state_vector_v1.valuation_multiple": 9.5,
                    "state_vector_v1.cash_generation": 0.031,
                },
                "negative_compact": {
                    "state_vector_v1.valuation_multiple": 9.0,
                    "state_vector_v1.cash_generation": 0.032,
                },
                "target_sector": "INDUSTRIALS",
                "target_subsector": "ELEC",
                "positive_sector": "INDUSTRIALS",
                "positive_subsector": "ELEC",
                "negative_sector": "TECH",
                "negative_subsector": "SEMI",
                "target_action_scale": 0.05,
                "positive_action_scale": 0.08,
                "negative_action_scale": 0.30,
            },
        ]
    )

    matrix = build_second_stage_reranker_matrix(df, same_action_only=True)
    feature_idx = {name: idx for idx, name in enumerate(matrix["selected_features"])}

    assert matrix["pair_count"] == 2
    assert np.abs(matrix["X"][:, feature_idx["sector_similarity"]]).sum() > 0.0
    assert np.abs(matrix["X"][:, feature_idx["parameter_similarity"]]).sum() > 0.0


def test_build_outcome_aware_reranker_matrix_uses_outcome_lookup():
    pairs = pd.DataFrame(
        [
            {
                "company_id": "000001",
                "as_of_time": "2024-01-01T00:00:00+00:00",
                "anchor_action_id": "capital_return.open_market_buyback",
                "competitor_action_id": "capital_return.open_market_buyback",
                "positive_precedent_id": "000001::2024-03-31T00:00:00::0",
                "negative_precedent_id": "000002::2024-03-31T00:00:00::0",
                "positive_similarity_score": 0.82,
                "negative_similarity_score": 0.61,
            },
        ]
    )
    outcomes = pd.DataFrame(
        [
            {
                "company_id": "000001",
                "action_date": "2024-03-31T00:00:00",
                "normalized_action_id": "capital_return.open_market_buyback",
                "outcome_pe_6m": 0.30,
                "outcome_pe_12m": 0.40,
                "outcome_ev_ebitda_6m": 0.25,
                "outcome_ev_ebitda_12m": 0.35,
                "credit_spread_change_1m": -15.0,
                "credit_spread_change_6m": -25.0,
                "credit_spread_change_12m": -30.0,
                "credit_spread_change_24m": -40.0,
                "rating_migration_1m": 1.0,
                "rating_migration_6m": 1.0,
                "rating_migration_12m": 1.0,
                "rating_migration_24m": 1.0,
                "leverage_delta": -0.20,
                "fcf_margin_delta": 0.03,
            },
            {
                "company_id": "000002",
                "action_date": "2024-03-31T00:00:00",
                "normalized_action_id": "capital_return.open_market_buyback",
                "outcome_pe_6m": -0.10,
                "outcome_pe_12m": -0.20,
                "outcome_ev_ebitda_6m": -0.05,
                "outcome_ev_ebitda_12m": -0.15,
                "credit_spread_change_1m": 10.0,
                "credit_spread_change_6m": 20.0,
                "credit_spread_change_12m": 25.0,
                "credit_spread_change_24m": 35.0,
                "rating_migration_1m": -1.0,
                "rating_migration_6m": -1.0,
                "rating_migration_12m": -1.0,
                "rating_migration_24m": -1.0,
                "leverage_delta": 0.15,
                "fcf_margin_delta": -0.02,
            },
        ]
    )

    matrix = build_outcome_aware_reranker_matrix(pairs, outcomes_df=outcomes, same_action_only=True)

    assert matrix["pair_count"] == 1
    assert "outcome_valuation_score" in matrix["selected_features"]
    assert "outcome_credit_score" in matrix["selected_features"]
    assert matrix["feature_coverage"]["current_similarity_score"] == 1
    assert matrix["feature_coverage"]["outcome_valuation_score"] == 1
    assert matrix["feature_coverage"]["outcome_credit_score"] == 1
    assert matrix["X"].shape == (2, len(matrix["selected_features"]))


def test_outcome_aware_reranker_prior_keeps_similarity_as_baseline():
    prior = _outcome_aware_reranker_prior(
        [
            "current_similarity_score",
            "outcome_equity_score",
            "outcome_valuation_score",
            "outcome_credit_score",
        ]
    )

    assert prior.tolist() == [1.0, 0.0, 0.0, 0.0]


