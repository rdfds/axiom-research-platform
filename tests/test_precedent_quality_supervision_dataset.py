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


def test_resolve_teacher_recipe_same_action_regime_best_analog_standardizes_flags():
    config = _resolve_teacher_recipe(
        teacher_recipe="same_action_regime_best_analog",
        positive_source_mode="include_retrieved",
        include_within_action_hard_negatives=False,
        include_same_action_positive_ordering=False,
        actual_anchor_within_action_negative_source="retrieved_pool",
        always_include_actual_anchor_positive=True,
        same_family_negatives_only_if_available=True,
        hard_negative_taxonomy_mode="none",
    )

    assert config["teacher_recipe"] == "same_action_regime_best_analog"
    assert config["positive_source_mode"] == "analog_regime_consensus_same_action_universe"
    assert config["include_within_action_hard_negatives"] is True
    assert config["include_same_action_positive_ordering"] is True
    assert config["actual_anchor_within_action_negative_source"] == "same_action_universe"
    assert config["always_include_actual_anchor_positive"] is False
    assert config["same_family_negatives_only_if_available"] is False


def test_load_snapshot_row_supports_modern_snapshot_store_layout(tmp_path):
    snapshot_root = tmp_path / "snapshot_cache" / "keyed"
    modern_path = snapshot_root / "company_id=0000001800" / "snapshot_as_of=20240902T000000Z.json"
    modern_path.parent.mkdir(parents=True)
    payload = {"company_id": "0000001800", "as_of_time": "2024-09-02T00:00:00+00:00"}
    modern_path.write_text(json.dumps(payload))

    loaded = _load_snapshot_row(snapshot_root, company_id="0000001800", as_of_time="2024-09-02T00:00:00+00:00")
    assert loaded == payload


def test_load_snapshot_row_prefers_snapshot_catalog_when_available(tmp_path):
    snapshot_root = tmp_path / "snapshot_cache" / "keyed"
    legacy_path = snapshot_root / "as_of_date=2024-09-02" / "company_id=0000001800.json"
    legacy_path.parent.mkdir(parents=True)
    legacy_path.write_text(json.dumps({"company_id": "0000001800", "as_of_time": "2024-09-02T00:00:00+00:00", "source": "cache"}))

    catalog_path = tmp_path / "snapshot_catalog.jsonl.gz"
    catalog_payload = {
        "company_id": "0000001800",
        "as_of_time": "2024-09-02T00:00:00+00:00",
        "source": "catalog",
    }
    import gzip

    with gzip.open(catalog_path, "wt") as handle:
        handle.write(json.dumps(catalog_payload) + "\n")

    loaded = _load_snapshot_row(
        snapshot_root,
        company_id="0000001800",
        as_of_time="2024-09-02T00:00:00Z",
        snapshot_catalog_path=catalog_path,
    )
    assert loaded == catalog_payload
    assert _normalize_as_of_time("2024-09-02T00:00:00Z") == _normalize_as_of_time(
        "2024-09-02T00:00:00+00:00"
    )


def test_rank_hard_negative_matches_prefers_same_subsector_then_sector():
    matches = [
        {
            "precedent_id": "other-sector",
            "similarity_score": 0.95,
            "key_state_features": {
                "sector": "Consumer Discretionary",
                "subsector": "Retail",
                "state_vector_v1.net_obligation_burden": 0.5,
                "state_vector_v1.liquidity_flexibility": 2.0,
                "state_vector_v1.interest_coverage": 8.0,
            },
        },
        {
            "precedent_id": "same-sector",
            "similarity_score": 0.80,
            "key_state_features": {
                "sector": "Industrials",
                "subsector": "Electrical",
                "state_vector_v1.net_obligation_burden": 0.5,
                "state_vector_v1.liquidity_flexibility": 2.0,
                "state_vector_v1.interest_coverage": 8.0,
            },
        },
        {
            "precedent_id": "same-subsector",
            "similarity_score": 0.70,
            "key_state_features": {
                "sector": "Industrials",
                "subsector": "Machinery",
                "state_vector_v1.net_obligation_burden": 0.5,
                "state_vector_v1.liquidity_flexibility": 2.0,
                "state_vector_v1.interest_coverage": 8.0,
            },
        },
    ]

    ranked = _rank_hard_negative_matches(
        matches,
        target_compact={
            "state_vector_v1.net_obligation_burden": 0.4,
            "state_vector_v1.liquidity_flexibility": 1.9,
            "state_vector_v1.interest_coverage": 7.5,
        },
        target_sector="Industrials",
        target_subsector="Machinery",
        taxonomy_mode="prefer_same_subsector_then_sector",
    )

    assert [row["precedent_id"] for row in ranked] == [
        "same-subsector",
        "same-sector",
        "other-sector",
    ]


def test_rank_hard_negative_matches_breaks_ties_with_safety_distance():
    matches = [
        {
            "precedent_id": "farther",
            "similarity_score": 0.95,
            "key_state_features": {
                "sector": "Industrials",
                "subsector": "Machinery",
                "state_vector_v1.net_obligation_burden": 2.0,
                "state_vector_v1.liquidity_flexibility": 8.0,
                "state_vector_v1.interest_coverage": 20.0,
            },
        },
        {
            "precedent_id": "closer",
            "similarity_score": 0.70,
            "key_state_features": {
                "sector": "Industrials",
                "subsector": "Machinery",
                "state_vector_v1.net_obligation_burden": 0.45,
                "state_vector_v1.liquidity_flexibility": 2.1,
                "state_vector_v1.interest_coverage": 8.2,
            },
        },
    ]

    ranked = _rank_hard_negative_matches(
        matches,
        target_compact={
            "state_vector_v1.net_obligation_burden": 0.4,
            "state_vector_v1.liquidity_flexibility": 2.0,
            "state_vector_v1.interest_coverage": 8.0,
        },
        target_sector="Industrials",
        target_subsector="Machinery",
        taxonomy_mode="prefer_same_subsector_then_sector",
    )

    assert [row["precedent_id"] for row in ranked] == ["closer", "farther"]


def test_enrich_match_compact_backfills_missing_state_vector_fields_from_outcomes_lookup():
    match = {
        "company_id": "001078",
        "action_id": "capital_return.open_market_buyback",
        "decision_time": "2024-01-11 00:00:00",
        "key_state_features": {
            "state_vector_v1.growth": None,
            "state_vector_v1.cash_generation": 0.05,
            "sector": "Health Care",
        },
    }
    lookup = {
        (
            "001078",
            "capital_return.open_market_buyback",
            "2024-01-11T00:00:00+00:00",
        ): {
            "state_vector_v1.growth": 0.12,
            "state_vector_v1.cash_generation": 0.08,
            "state_vector_v1.market_stress": 0.22,
            "subsector": "Health Care Equipment & Supplies",
        }
    }

    enriched = _enrich_match_compact(match, precedent_outcomes_lookup=lookup)

    assert enriched["state_vector_v1.growth"] == 0.12
    assert enriched["state_vector_v1.cash_generation"] == 0.05
    assert enriched["state_vector_v1.market_stress"] == 0.22
    assert enriched["subsector"] == "Health Care Equipment & Supplies"


def test_target_context_from_anchor_outcome_uses_anchor_row_state_features():
    case = {
        "company_id": "0000002488",
        "source_company_id": "0000002488",
        "anchor_action_id": "capital_structure.new_debt_issuance",
        "anchor_action_date": "2024-09-02T00:00:00+00:00",
    }
    lookup = {
        ("0000002488", "capital_structure.new_debt_issuance"): [
            {
                "company_id": "0000002488",
                "normalized_action_id": "capital_structure.new_debt_issuance",
                "action_date": "2024-09-02T00:00:00+00:00",
                "state_vector_v1.valuation_multiple": 11.5,
                "state_vector_v1.net_obligation_burden": 0.8,
                "action_size": 250.0,
                "base_market_cap": 1000.0,
                "sector": "Information Technology",
                "subsector": "Semiconductors",
            }
        ]
    }

    context = _target_context_from_anchor_outcome(case, anchor_outcomes_lookup=lookup)

    assert context is not None
    assert context["target_source"] == "anchor_outcome_fallback"
    assert context["target_compact"]["state_vector_v1.valuation_multiple"] == 11.5
    assert context["target_compact"]["state_vector_v1.net_obligation_burden"] == 0.8
    assert context["target_taxonomy"] == {
        "sector": "Information Technology",
        "subsector": "Semiconductors",
    }
    assert context["target_action_params"] == {
        "amount_usd": 250.0,
        "action_size": 250.0,
    }
    assert context["target_market_cap"] == 1000.0


def test_infer_target_taxonomy_from_same_action_universe_uses_company_history():
    action_id = "capital_structure.equity_issuance"

    taxonomy = _infer_target_taxonomy_from_same_action_universe(
        {
            "company_id": "025430",
            "source_company_id": "025430",
            "anchor_action_id": action_id,
            "ticker": "FCEL",
        },
        same_action_universe_lookup={
            action_id: {
                "rows": [
                    {
                        "company_id": "025430",
                        "ticker": "FCEL",
                        "sector": "Industrials",
                        "subsector": "Electrical Equipment",
                    },
                    {
                        "company_id": "025430",
                        "ticker": "FCEL",
                        "taxonomy.sector": "Industrials",
                        "taxonomy.subsector": "Electrical Equipment",
                    },
                    {
                        "company_id": "999999",
                        "ticker": "OTHER",
                        "sector": "Health Care",
                        "subsector": "Biotechnology",
                    },
                ]
            }
        },
    )

    assert taxonomy == {
        "sector": "Industrials",
        "subsector": "Electrical Equipment",
    }


def test_target_context_from_same_action_universe_uses_exact_company_history():
    context = _target_context_from_same_action_universe(
        {
            "source_company_id": "162233",
            "company_id": "162233",
            "ticker": "DGLY",
            "anchor_action_id": "capital_structure.equity_issuance",
            "anchor_action_date": "2024-12-30",
        },
        same_action_universe_lookup={
            "capital_structure.equity_issuance": {
                "rows": [
                    {
                        "company_id": "162233",
                        "ticker": "DGLY",
                        "action_date": "2024-12-31T00:00:00Z",
                        "taxonomy.sector": "Information Technology",
                        "taxonomy.subsector": "Communications Equipment",
                        "state_vector_v1.size_log_revenue": 1.25,
                        "state_vector_v1.cash_generation": -0.5,
                        "state_vector_v1.market_access": -1.2,
                        "base_market_cap": 42.0,
                        "action_size": 10.0,
                    },
                    {
                        "company_id": "162233",
                        "ticker": "DGLY",
                        "action_date": "2023-12-31T00:00:00Z",
                        "taxonomy.sector": "Health Care",
                        "taxonomy.subsector": "Biotechnology",
                        "state_vector_v1.size_log_revenue": 9.99,
                        "base_market_cap": 99.0,
                        "action_size": 5.0,
                    },
                ]
            }
        },
    )

    assert context is not None
    assert context["target_source"] == "same_action_universe_fallback"
    assert context["target_taxonomy"] == {
        "sector": "Information Technology",
        "subsector": "Communications Equipment",
    }
    assert context["target_compact"]["state_vector_v1.size_log_revenue"] == 1.25
    assert context["target_market_cap"] == 42.0
    assert context["target_action_params"]["action_size"] == 10.0


