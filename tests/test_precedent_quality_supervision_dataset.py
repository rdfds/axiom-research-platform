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


