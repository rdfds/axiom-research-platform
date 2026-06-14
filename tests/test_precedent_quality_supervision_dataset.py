from __future__ import annotations

import json

from src.model_feature_bundle import _STATE_VECTOR_V1_FEATURES
from src.pipeline.latent_regime_model import fit_latent_regime_kmeans, latent_regime_memberships
from scripts.build_precedent_quality_supervision_dataset import (
    _PAIRWISE_FEATURE_GAP_SUMMARY_FEATURES,
    _build_same_action_analog_positive_source,
    _debt_issuance_archetype_profile,
    _enrich_match_compact,
    _infer_target_taxonomy_from_same_action_universe,
    _load_snapshot_row,
    _normalize_as_of_time,
    _outcome_row_action_params,
    _pair_rows_for_case,
    _rank_hard_negative_matches,
    _rank_same_action_hard_confusers,
    _resolve_teacher_recipe,
    _target_context_from_anchor_outcome,
    _target_context_from_same_action_universe,
)


def test_pairwise_feature_gap_summary_tracks_all_compact_features():
    assert tuple(_PAIRWISE_FEATURE_GAP_SUMMARY_FEATURES) == tuple(_STATE_VECTOR_V1_FEATURES)


def test_outcome_row_action_params_carries_refinancing_subtype_context():
    params = _outcome_row_action_params(
        {
            "action_size": 250_000_000.0,
            "raw_action_subtype": "Term Loan B",
        }
    )
    assert params["amount_usd"] == 250_000_000.0
    assert params["action_size"] == 250_000_000.0
    assert params["source_action_subtype"] == "Term Loan B"


def test_target_context_from_anchor_outcome_prefers_requested_refinancing_subtype():
    case = {
        "company_id": "1001",
        "source_company_id": "1001",
        "anchor_action_id": "capital_structure.refinancing",
        "anchor_action_date": "2020-01-15T00:00:00Z",
        "anchor_action_subtype": "Term Loan B",
    }
    anchor_outcomes_lookup = {
        ("1001", "capital_structure.refinancing"): [
            {
                "company_id": "1001",
                "action_date": "2020-01-15T00:00:00Z",
                "raw_action_subtype": "Revolver/Line >= 1 Yr.",
                "action_subtype": "Revolver/Line >= 1 Yr.",
                "action_size": 600_000_000.0,
                "state_vector_v1.size_log_revenue": 7.0,
                "state_vector_v1.net_obligation_burden": 1.0,
                "base_sector": "Industrials",
                "base_industry": "Machinery",
            },
            {
                "company_id": "1001",
                "action_date": "2020-01-15T00:00:00Z",
                "raw_action_subtype": "Term Loan B",
                "action_subtype": "Term Loan B",
                "action_size": 450_000_000.0,
                "state_vector_v1.size_log_revenue": 9.5,
                "state_vector_v1.net_obligation_burden": 2.5,
                "base_sector": "Industrials",
                "base_industry": "Machinery",
            },
        ]
    }

    context = _target_context_from_anchor_outcome(case, anchor_outcomes_lookup=anchor_outcomes_lookup)

    assert context is not None
    assert context["target_action_params"]["source_action_subtype"] == "Term Loan B"
    assert context["target_compact"]["state_vector_v1.size_log_revenue"] == 9.5
    assert context["target_compact"]["state_vector_v1.net_obligation_burden"] == 2.5


def test_resolve_teacher_recipe_same_action_best_analog_standardizes_flags():
    config = _resolve_teacher_recipe(
        teacher_recipe="same_action_best_analog",
        positive_source_mode="include_retrieved",
        include_within_action_hard_negatives=False,
        include_same_action_positive_ordering=False,
        actual_anchor_within_action_negative_source="retrieved_pool",
        always_include_actual_anchor_positive=True,
        same_family_negatives_only_if_available=True,
        hard_negative_taxonomy_mode="prefer_same_subsector_then_sector",
    )

    assert config["teacher_recipe"] == "same_action_best_analog"
    assert config["positive_source_mode"] == "analog_consensus_same_action_universe"
    assert config["include_within_action_hard_negatives"] is True
    assert config["include_same_action_positive_ordering"] is True
    assert config["actual_anchor_within_action_negative_source"] == "same_action_universe"
    assert config["always_include_actual_anchor_positive"] is False
    assert config["same_family_negatives_only_if_available"] is False
    assert config["hard_negative_taxonomy_mode"] == "prefer_same_subsector_then_sector"


