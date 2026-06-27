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


