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


def test_infer_target_taxonomy_from_same_action_universe_uses_nearest_taxonomy_neighbors():
    action_id = "mna.platform_acquisition"
    target_compact = {
        "state_vector_v1.size_log_revenue": 6.0,
        "state_vector_v1.profitability": -0.4,
        "state_vector_v1.growth": 1.2,
        "state_vector_v1.valuation_multiple": -8.0,
    }
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
                        "company_id": "100001",
                        "ticker": "PEER1",
                        "taxonomy.sector": "Industrials",
                        "taxonomy.subsector": "Electrical Equipment",
                        "state_vector_v1.size_log_revenue": 6.1,
                        "state_vector_v1.profitability": -0.5,
                        "state_vector_v1.growth": 1.1,
                        "state_vector_v1.valuation_multiple": -7.5,
                    },
                    {
                        "company_id": "100002",
                        "ticker": "PEER2",
                        "taxonomy.sector": "Industrials",
                        "taxonomy.subsector": "Electrical Equipment",
                        "state_vector_v1.size_log_revenue": 5.9,
                        "state_vector_v1.profitability": -0.3,
                        "state_vector_v1.growth": 1.0,
                        "state_vector_v1.valuation_multiple": -8.4,
                    },
                    {
                        "company_id": "100003",
                        "ticker": "OTHER",
                        "taxonomy.sector": "Health Care",
                        "taxonomy.subsector": "Biotechnology",
                        "state_vector_v1.size_log_revenue": 9.0,
                        "state_vector_v1.profitability": 0.6,
                        "state_vector_v1.growth": -0.2,
                        "state_vector_v1.valuation_multiple": 12.0,
                    },
                ],
                "feature_scales": {
                    "state_vector_v1.size_log_revenue": 1.0,
                    "state_vector_v1.profitability": 1.0,
                    "state_vector_v1.growth": 1.0,
                    "state_vector_v1.valuation_multiple": 1.0,
                },
            }
        },
        target_compact=target_compact,
    )

    assert taxonomy == {
        "sector": "Industrials",
        "subsector": "Electrical Equipment",
    }


def test_infer_target_taxonomy_from_same_action_universe_ignores_nan_features():
    action_id = "mna.platform_acquisition"
    target_compact = {
        "state_vector_v1.size_log_revenue": 6.0,
        "state_vector_v1.profitability": float("nan"),
        "state_vector_v1.growth": 1.2,
        "state_vector_v1.valuation_multiple": -8.0,
        "state_vector_v1.cash_generation": -0.6,
    }
    taxonomy = _infer_target_taxonomy_from_same_action_universe(
        {
            "company_id": "162233",
            "source_company_id": "162233",
            "anchor_action_id": action_id,
            "ticker": "DGLY",
        },
        same_action_universe_lookup={
            action_id: {
                "rows": [
                    {
                        "company_id": "100001",
                        "ticker": "PEER1",
                        "taxonomy.sector": "Industrials",
                        "taxonomy.subsector": "Electrical Equipment",
                        "state_vector_v1.size_log_revenue": 6.1,
                        "state_vector_v1.profitability": -0.5,
                        "state_vector_v1.growth": 1.1,
                        "state_vector_v1.valuation_multiple": -7.5,
                        "state_vector_v1.cash_generation": -0.5,
                    },
                    {
                        "company_id": "100002",
                        "ticker": "OTHER",
                        "taxonomy.sector": "Health Care",
                        "taxonomy.subsector": "Biotechnology",
                        "state_vector_v1.size_log_revenue": 9.0,
                        "state_vector_v1.profitability": 0.6,
                        "state_vector_v1.growth": -0.2,
                        "state_vector_v1.valuation_multiple": 12.0,
                        "state_vector_v1.cash_generation": 0.4,
                    },
                ],
                "feature_scales": {
                    "state_vector_v1.size_log_revenue": 1.0,
                    "state_vector_v1.profitability": 1.0,
                    "state_vector_v1.growth": 1.0,
                    "state_vector_v1.valuation_multiple": 1.0,
                    "state_vector_v1.cash_generation": 1.0,
                },
            }
        },
        target_compact=target_compact,
    )

    assert taxonomy == {
        "sector": "Industrials",
        "subsector": "Electrical Equipment",
    }


def test_infer_target_taxonomy_from_same_action_universe_prefers_direct_historical_ticker_taxonomy(monkeypatch):
    action_id = "capital_structure.equity_issuance"
    monkeypatch.setattr(
        "scripts.build_precedent_quality_supervision_dataset._direct_historical_ticker_taxonomy",
        lambda ticker: {
            "sector": "Industrials",
            "subsector": "Electrical Equipment",
        }
        if str(ticker).upper() == "FCEL"
        else {},
    )

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
                        "company_id": "100001",
                        "ticker": "PEER1",
                        "taxonomy.sector": "Health Care",
                        "taxonomy.subsector": "Biotechnology",
                        "state_vector_v1.size_log_revenue": 6.1,
                        "state_vector_v1.profitability": -0.5,
                        "state_vector_v1.growth": 1.1,
                        "state_vector_v1.valuation_multiple": -7.5,
                    }
                ],
                "feature_scales": {
                    "state_vector_v1.size_log_revenue": 1.0,
                    "state_vector_v1.profitability": 1.0,
                    "state_vector_v1.growth": 1.0,
                    "state_vector_v1.valuation_multiple": 1.0,
                },
            }
        },
        target_compact={
            "state_vector_v1.size_log_revenue": 6.0,
            "state_vector_v1.profitability": -0.4,
            "state_vector_v1.growth": 1.2,
            "state_vector_v1.valuation_multiple": -8.0,
        },
    )

    assert taxonomy == {
        "sector": "Industrials",
        "subsector": "Electrical Equipment",
    }


def test_infer_target_taxonomy_from_same_action_universe_skips_knn_for_equity_issuance(monkeypatch):
    action_id = "capital_structure.equity_issuance"
    monkeypatch.setattr(
        "scripts.build_precedent_quality_supervision_dataset._direct_historical_ticker_taxonomy",
        lambda ticker: {},
    )

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
                        "company_id": "100001",
                        "ticker": "PEER1",
                        "taxonomy.sector": "Health Care",
                        "taxonomy.subsector": "Biotechnology",
                        "state_vector_v1.size_log_revenue": 6.1,
                        "state_vector_v1.profitability": -0.5,
                        "state_vector_v1.growth": 1.1,
                        "state_vector_v1.valuation_multiple": -7.5,
                    },
                    {
                        "company_id": "100002",
                        "ticker": "PEER2",
                        "taxonomy.sector": "Health Care",
                        "taxonomy.subsector": "Biotechnology",
                        "state_vector_v1.size_log_revenue": 5.9,
                        "state_vector_v1.profitability": -0.3,
                        "state_vector_v1.growth": 1.0,
                        "state_vector_v1.valuation_multiple": -8.4,
                    },
                ],
                "feature_scales": {
                    "state_vector_v1.size_log_revenue": 1.0,
                    "state_vector_v1.profitability": 1.0,
                    "state_vector_v1.growth": 1.0,
                    "state_vector_v1.valuation_multiple": 1.0,
                },
            }
        },
        target_compact={
            "state_vector_v1.size_log_revenue": 6.0,
            "state_vector_v1.profitability": -0.4,
            "state_vector_v1.growth": 1.2,
            "state_vector_v1.valuation_multiple": -8.0,
        },
    )

    assert taxonomy == {}


def test_same_action_analog_positive_source_uses_action_scale_when_state_is_tied():
    action_id = "capital_structure.new_debt_issuance"
    target_compact = {feature: 0.0 for feature in _STATE_VECTOR_V1_FEATURES}
    row_common = {
        "normalized_action_id": action_id,
        "action_date": "2024-01-15T00:00:00+00:00",
        "taxonomy.sector": "Information Technology",
        "taxonomy.subsector": "Semiconductors",
    }
    rows = [
        {
            **row_common,
            "company_id": "1111111111",
            "ticker": "CLOSE",
            "action_size": 100.0,
            "base_market_cap": 1000.0,
            **target_compact,
        },
        {
            **row_common,
            "company_id": "2222222222",
            "ticker": "FAR",
            "action_size": 10.0,
            "base_market_cap": 1000.0,
            **target_compact,
        },
    ]

    source = _build_same_action_analog_positive_source(
        case={
            "company_id": "0000002488",
            "source_company_id": "0000002488",
            "anchor_action_id": action_id,
            "anchor_action_date": "2024-09-02T00:00:00+00:00",
            "as_of_time": "2024-09-02T00:00:00+00:00",
        },
        target_compact=target_compact,
        target_taxonomy={"sector": "Information Technology", "subsector": "Semiconductors"},
        target_action_params={"amount_usd": 100.0, "action_size": 100.0},
        target_market_cap=1000.0,
        top_k=2,
        positive_limit_per_source=1,
        negative_limit_per_competitor=1,
        same_action_universe_lookup={
            action_id: {
                "rows": rows,
                "feature_scales": {feature: 1.0 for feature in _STATE_VECTOR_V1_FEATURES},
            }
        },
        regime_aware=False,
    )

    assert source is not None
    assert source["matches"][0]["ticker"] == "CLOSE"


def test_same_action_analog_positive_source_prioritizes_debt_borrower_profile():
    action_id = "capital_structure.new_debt_issuance"
    target_compact = {feature: 0.0 for feature in _STATE_VECTOR_V1_FEATURES}
    target_compact.update(
        {
            "state_vector_v1.profitability": 0.12,
            "state_vector_v1.cash_generation": -0.03,
            "state_vector_v1.gross_obligation_burden": 2.4,
            "state_vector_v1.net_obligation_burden": 1.7,
            "state_vector_v1.interest_coverage": 2.8,
            "state_vector_v1.valuation_multiple": 5.4,
            "state_vector_v1.market_access": 0.63,
            "state_vector_v1.market_stress": 0.17,
            "state_vector_v1.rates_level": 4.58,
            "state_vector_v1.credit_spread": 2.64,
            "state_vector_v1.growth": 0.06,
            "state_vector_v1.liquidity_flexibility": 2.0,
        }
    )
    row_common = {
        "normalized_action_id": action_id,
        "action_date": "2024-01-15T00:00:00+00:00",
        "taxonomy.sector": "Industrials",
        "taxonomy.subsector": "Commercial Services & Supplies",
        "action_size": 100.0,
        "base_market_cap": 1000.0,
        "state_vector_v1.size_log_revenue": 9.6,
        "state_vector_v1.liquidity_flexibility": 2.0,
    }
    rows = [
        {
            **row_common,
            "company_id": "1111111111",
            "ticker": "DISTRESS",
            **target_compact,
        },
        {
            **row_common,
            "company_id": "2222222222",
            "ticker": "HEALTHY",
            **{
                **target_compact,
                "state_vector_v1.profitability": 0.28,
                "state_vector_v1.cash_generation": 0.10,
                "state_vector_v1.gross_obligation_burden": 0.8,
                "state_vector_v1.net_obligation_burden": 0.3,
                "state_vector_v1.interest_coverage": 18.0,
                "state_vector_v1.valuation_multiple": 13.0,
                "state_vector_v1.market_access": 0.88,
                "state_vector_v1.market_stress": 0.10,
                "state_vector_v1.growth": 0.06,
            },
        },
    ]

    source = _build_same_action_analog_positive_source(
        case={
            "company_id": "0000002488",
            "source_company_id": "0000002488",
            "anchor_action_id": action_id,
            "anchor_action_date": "2024-09-02T00:00:00+00:00",
            "as_of_time": "2024-09-02T00:00:00+00:00",
        },
        target_compact=target_compact,
        target_taxonomy={"sector": "Industrials", "subsector": "Commercial Services & Supplies"},
        target_action_params={"amount_usd": 100.0, "action_size": 100.0},
        target_market_cap=1000.0,
        top_k=2,
        positive_limit_per_source=1,
        negative_limit_per_competitor=1,
        same_action_universe_lookup={
            action_id: {
                "rows": rows,
                "feature_scales": {feature: 1.0 for feature in _STATE_VECTOR_V1_FEATURES},
            }
        },
        regime_aware=False,
    )

    assert source is not None
    assert source["matches"][0]["company_id"] == "1111111111"


def test_debt_issuance_archetype_profile_separates_distressed_refinancing_and_opportunistic():
    distressed = _debt_issuance_archetype_profile(
        compact_features={
            "state_vector_v1.profitability": 0.08,
            "state_vector_v1.cash_generation": -0.05,
            "state_vector_v1.gross_obligation_burden": 2.9,
            "state_vector_v1.net_obligation_burden": 2.1,
            "state_vector_v1.interest_coverage": 1.7,
            "state_vector_v1.liquidity_flexibility": 0.8,
            "state_vector_v1.market_access": 0.48,
            "state_vector_v1.market_stress": 0.24,
            "state_vector_v1.credit_spread": 4.4,
            "state_vector_v1.valuation_multiple": 4.5,
        },
        action_scale=0.10,
    )
    refinancing = _debt_issuance_archetype_profile(
        compact_features={
            "state_vector_v1.profitability": 0.17,
            "state_vector_v1.cash_generation": 0.01,
            "state_vector_v1.gross_obligation_burden": 2.4,
            "state_vector_v1.net_obligation_burden": 1.9,
            "state_vector_v1.interest_coverage": 4.2,
            "state_vector_v1.liquidity_flexibility": 0.9,
            "state_vector_v1.market_access": 0.73,
            "state_vector_v1.market_stress": 0.16,
            "state_vector_v1.credit_spread": 2.9,
            "state_vector_v1.valuation_multiple": 8.0,
        },
        action_scale=0.22,
    )
    opportunistic = _debt_issuance_archetype_profile(
        compact_features={
            "state_vector_v1.profitability": 0.28,
            "state_vector_v1.cash_generation": 0.07,
            "state_vector_v1.gross_obligation_burden": 1.1,
            "state_vector_v1.net_obligation_burden": 0.4,
            "state_vector_v1.interest_coverage": 10.0,
            "state_vector_v1.liquidity_flexibility": 3.2,
            "state_vector_v1.market_access": 0.91,
            "state_vector_v1.market_stress": 0.09,
            "state_vector_v1.credit_spread": 2.2,
            "state_vector_v1.valuation_multiple": 14.0,
        },
        action_scale=0.06,
    )

    assert distressed["label"] == "distressed_borrower"
    assert refinancing["label"] == "refinancing_pressure"
    assert opportunistic["label"] == "opportunistic_issuer"


def test_same_action_analog_positive_source_prefers_refinancing_pressure_over_opportunistic():
    action_id = "capital_structure.new_debt_issuance"
    target_compact = {feature: 0.0 for feature in _STATE_VECTOR_V1_FEATURES}
    target_compact.update(
        {
            "state_vector_v1.profitability": 0.16,
            "state_vector_v1.cash_generation": 0.01,
            "state_vector_v1.gross_obligation_burden": 2.5,
            "state_vector_v1.net_obligation_burden": 2.0,
            "state_vector_v1.interest_coverage": 4.0,
            "state_vector_v1.valuation_multiple": 8.5,
            "state_vector_v1.market_access": 0.74,
            "state_vector_v1.market_stress": 0.15,
            "state_vector_v1.rates_level": 4.58,
            "state_vector_v1.credit_spread": 2.64,
            "state_vector_v1.growth": 0.04,
            "state_vector_v1.liquidity_flexibility": 0.8,
        }
    )
    row_common = {
        "normalized_action_id": action_id,
        "action_date": "2024-01-15T00:00:00+00:00",
        "taxonomy.sector": "Industrials",
        "taxonomy.subsector": "Commercial Services & Supplies",
        "state_vector_v1.size_log_revenue": 9.4,
    }
    rows = [
        {
            **row_common,
            "company_id": "1111111111",
            "ticker": "REFI",
            "action_size": 220.0,
            "base_market_cap": 1000.0,
            **target_compact,
        },
        {
            **row_common,
            "company_id": "2222222222",
            "ticker": "OPPORTUNISTIC",
            "action_size": 60.0,
            "base_market_cap": 1000.0,
            **{
                **target_compact,
                "state_vector_v1.profitability": 0.29,
                "state_vector_v1.cash_generation": 0.08,
                "state_vector_v1.gross_obligation_burden": 0.9,
                "state_vector_v1.net_obligation_burden": 0.2,
                "state_vector_v1.interest_coverage": 12.0,
                "state_vector_v1.market_access": 0.91,
                "state_vector_v1.market_stress": 0.08,
                "state_vector_v1.credit_spread": 2.1,
                "state_vector_v1.liquidity_flexibility": 3.5,
                "state_vector_v1.valuation_multiple": 14.0,
            },
        },
    ]

    source = _build_same_action_analog_positive_source(
        case={
            "company_id": "0000002488",
            "source_company_id": "0000002488",
            "anchor_action_id": action_id,
            "anchor_action_date": "2024-09-02T00:00:00+00:00",
            "as_of_time": "2024-09-02T00:00:00+00:00",
        },
        target_compact=target_compact,
        target_taxonomy={"sector": "Industrials", "subsector": "Commercial Services & Supplies"},
        target_action_params={"amount_usd": 220.0, "action_size": 220.0},
        target_market_cap=1000.0,
        top_k=2,
        positive_limit_per_source=1,
        negative_limit_per_competitor=1,
        same_action_universe_lookup={
            action_id: {
                "rows": rows,
                "feature_scales": {feature: 1.0 for feature in _STATE_VECTOR_V1_FEATURES},
            }
        },
        regime_aware=False,
    )

    assert source is not None
    assert source["matches"][0]["ticker"] == "REFI"


def test_same_action_analog_positive_source_excludes_same_company_same_action_date():
    action_id = "capital_structure.new_debt_issuance"
    target_compact = {feature: 0.0 for feature in _STATE_VECTOR_V1_FEATURES}
    target_compact.update({"state_vector_v1.profitability": 0.12, "state_vector_v1.credit_spread": 2.64})
    rows = [
        {
            "company_id": "0000002488",
            "ticker": "SELF",
            "normalized_action_id": action_id,
            "action_date": "2024-09-02T00:00:00+00:00",
            "taxonomy.sector": "Industrials",
            "taxonomy.subsector": "Commercial Services & Supplies",
            "action_size": 100.0,
            "base_market_cap": 1000.0,
            **target_compact,
        },
        {
            "company_id": "1111111111",
            "ticker": "OTHER",
            "normalized_action_id": action_id,
            "action_date": "2024-08-15T00:00:00+00:00",
            "taxonomy.sector": "Industrials",
            "taxonomy.subsector": "Commercial Services & Supplies",
            "action_size": 95.0,
            "base_market_cap": 1000.0,
            **target_compact,
        },
    ]

    source = _build_same_action_analog_positive_source(
        case={
            "company_id": "0000002488",
            "source_company_id": "0000002488",
            "anchor_action_id": action_id,
            "anchor_action_date": "2024-09-02T00:00:00+00:00",
            "as_of_time": "2024-09-02T00:00:00+00:00",
        },
        target_compact=target_compact,
        target_taxonomy={"sector": "Industrials", "subsector": "Commercial Services & Supplies"},
        target_action_params={"amount_usd": 100.0, "action_size": 100.0},
        target_market_cap=1000.0,
        top_k=2,
        positive_limit_per_source=1,
        negative_limit_per_competitor=1,
        same_action_universe_lookup={
            action_id: {
                "rows": rows,
                "feature_scales": {feature: 1.0 for feature in _STATE_VECTOR_V1_FEATURES},
            }
        },
        regime_aware=False,
    )

    assert source is not None
    assert source["matches"][0]["ticker"] == "OTHER"


def test_rank_same_action_hard_confusers_prefers_wrong_archetype_in_same_regime():
    target_compact = {
        feature: 0.0 for feature in _STATE_VECTOR_V1_FEATURES
    }
    target_compact.update(
        {
            "state_vector_v1.profitability": 0.12,
            "state_vector_v1.cash_generation": -0.03,
            "state_vector_v1.gross_obligation_burden": 2.4,
            "state_vector_v1.net_obligation_burden": 1.7,
            "state_vector_v1.interest_coverage": 2.8,
            "state_vector_v1.valuation_multiple": 5.4,
            "state_vector_v1.market_access": 0.63,
            "state_vector_v1.market_stress": 0.17,
            "state_vector_v1.rates_level": 4.58,
            "state_vector_v1.credit_spread": 2.64,
        }
    )
    matches = [
        {
            "ticker": "HEALTHY",
            "action_scale": 0.1,
            "analog_distance": 0.2,
            "similarity_score": 0.9,
            "key_state_features": {
                **target_compact,
                "sector": "Industrials",
                "subsector": "Commercial Services & Supplies",
                "state_vector_v1.profitability": 0.30,
                "state_vector_v1.cash_generation": 0.12,
                "state_vector_v1.gross_obligation_burden": 0.8,
                "state_vector_v1.net_obligation_burden": 0.4,
                "state_vector_v1.interest_coverage": 12.0,
                "state_vector_v1.market_access": 0.9,
            },
        },
        {
            "ticker": "DISTRESS",
            "action_scale": 0.1,
            "analog_distance": 0.25,
            "similarity_score": 0.85,
            "key_state_features": {
                **target_compact,
                "sector": "Industrials",
                "subsector": "Commercial Services & Supplies",
            },
        },
    ]

    ranked = _rank_same_action_hard_confusers(
        matches,
        action_id="capital_structure.new_debt_issuance",
        target_compact=target_compact,
        target_sector="Industrials",
        target_subsector="Commercial Services & Supplies",
        target_action_scale=0.1,
    )

    assert ranked[0]["ticker"] == "HEALTHY"


def test_action_specific_same_action_distance_revolver_prefers_liquidity_stress_peer():
    target_compact = {
        feature: 0.0 for feature in _STATE_VECTOR_V1_FEATURES
    }
    target_compact.update(
        {
            "state_vector_v1.profitability": 0.06,
            "state_vector_v1.cash_generation": -0.02,
            "state_vector_v1.gross_obligation_burden": 2.8,
            "state_vector_v1.net_obligation_burden": 2.1,
            "state_vector_v1.liquidity_flexibility": 0.35,
            "state_vector_v1.interest_coverage": 2.4,
            "state_vector_v1.market_access": 0.56,
            "state_vector_v1.market_stress": 0.31,
            "state_vector_v1.credit_spread": 4.9,
            "state_vector_v1.rates_level": 1.20,
            "state_vector_v1.valuation_multiple": 6.5,
        }
    )
    stressed_peer = {
        **target_compact,
        "state_vector_v1.liquidity_flexibility": 0.50,
        "state_vector_v1.market_stress": 0.28,
        "state_vector_v1.credit_spread": 4.4,
    }
    routine_self_history = {
        **target_compact,
        "state_vector_v1.profitability": 0.18,
        "state_vector_v1.cash_generation": 0.08,
        "state_vector_v1.gross_obligation_burden": 1.2,
        "state_vector_v1.net_obligation_burden": 0.8,
        "state_vector_v1.liquidity_flexibility": 3.2,
        "state_vector_v1.interest_coverage": 7.5,
        "state_vector_v1.market_access": 0.86,
        "state_vector_v1.market_stress": 0.12,
        "state_vector_v1.credit_spread": 2.2,
        "state_vector_v1.valuation_multiple": 12.0,
    }

    stressed_distance = _action_specific_same_action_distance(
        action_id="capital_structure.revolver_draw_or_resize",
        target_compact=target_compact,
        candidate_features=stressed_peer,
        feature_scales={feature: 1.0 for feature in _STATE_VECTOR_V1_FEATURES},
        target_action_scale=0.22,
        candidate_action_scale=0.20,
    )
    routine_distance = _action_specific_same_action_distance(
        action_id="capital_structure.revolver_draw_or_resize",
        target_compact=target_compact,
        candidate_features=routine_self_history,
        feature_scales={feature: 1.0 for feature in _STATE_VECTOR_V1_FEATURES},
        target_action_scale=0.22,
        candidate_action_scale=0.22,
    )

    assert stressed_distance < routine_distance


def test_same_action_analog_positive_source_revolver_prefers_cross_company_and_caps_self_history():
    action_id = "capital_structure.revolver_draw_or_resize"
    target_compact = {feature: 0.0 for feature in _STATE_VECTOR_V1_FEATURES}
    target_compact.update(
        {
            "state_vector_v1.profitability": 0.05,
            "state_vector_v1.cash_generation": -0.01,
            "state_vector_v1.gross_obligation_burden": 2.9,
            "state_vector_v1.net_obligation_burden": 2.2,
            "state_vector_v1.liquidity_flexibility": 0.40,
            "state_vector_v1.interest_coverage": 2.5,
            "state_vector_v1.market_access": 0.55,
            "state_vector_v1.market_stress": 0.30,
            "state_vector_v1.credit_spread": 4.8,
        }
    )
    rows = [
        {
            "company_id": "0000002488",
            "ticker": "SELF1",
            "normalized_action_id": action_id,
            "action_date": "2024-08-01T00:00:00+00:00",
            "taxonomy.sector": "Consumer Discretionary",
            "taxonomy.subsector": "Specialty Retail",
            "action_size": 200.0,
            "base_market_cap": 1000.0,
            **target_compact,
        },
        {
            "company_id": "0000002488",
            "ticker": "SELF2",
            "normalized_action_id": action_id,
            "action_date": "2024-07-01T00:00:00+00:00",
            "taxonomy.sector": "Consumer Discretionary",
            "taxonomy.subsector": "Specialty Retail",
            "action_size": 195.0,
            "base_market_cap": 1000.0,
            **target_compact,
        },
        {
            "company_id": "1111111111",
            "ticker": "PEER1",
            "normalized_action_id": action_id,
            "action_date": "2024-07-15T00:00:00+00:00",
            "taxonomy.sector": "Consumer Discretionary",
            "taxonomy.subsector": "Specialty Retail",
            "action_size": 205.0,
            "base_market_cap": 1000.0,
            **target_compact,
            "state_vector_v1.market_stress": 0.28,
        },
        {
            "company_id": "2222222222",
            "ticker": "PEER2",
            "normalized_action_id": action_id,
            "action_date": "2024-06-20T00:00:00+00:00",
            "taxonomy.sector": "Consumer Discretionary",
            "taxonomy.subsector": "Specialty Retail",
            "action_size": 180.0,
            "base_market_cap": 1000.0,
            **target_compact,
            "state_vector_v1.market_stress": 0.32,
        },
    ]

    source = _build_same_action_analog_positive_source(
        case={
            "company_id": "0000002488",
            "source_company_id": "0000002488",
            "anchor_action_id": action_id,
            "anchor_action_date": "2024-09-02T00:00:00+00:00",
            "as_of_time": "2024-09-02T00:00:00+00:00",
        },
        target_compact=target_compact,
        target_taxonomy={"sector": "Consumer Discretionary", "subsector": "Specialty Retail"},
        target_action_params={"draw_amount_usd": 200.0, "action_size": 200.0},
        target_market_cap=1000.0,
        top_k=3,
        positive_limit_per_source=3,
        negative_limit_per_competitor=2,
        same_action_universe_lookup={
            action_id: {
                "rows": rows,
                "feature_scales": {feature: 1.0 for feature in _STATE_VECTOR_V1_FEATURES},
            }
        },
        regime_aware=False,
    )

    assert source is not None
    assert source["matches"][0]["ticker"] == "PEER1"
    assert [match["company_id"] for match in source["full_matches"]].count("0000002488") <= 1


def test_same_action_analog_positive_source_revolver_regime_aware_prefers_stress_regime_peers():
    action_id = "capital_structure.revolver_draw_or_resize"
    target_compact = {feature: 0.0 for feature in _STATE_VECTOR_V1_FEATURES}
    target_compact.update(
        {
            "state_vector_v1.profitability": 0.05,
            "state_vector_v1.cash_generation": -0.01,
            "state_vector_v1.growth": 0.00,
            "state_vector_v1.gross_obligation_burden": 2.9,
            "state_vector_v1.net_obligation_burden": 2.2,
            "state_vector_v1.liquidity_flexibility": 0.40,
            "state_vector_v1.interest_coverage": 2.5,
            "state_vector_v1.market_access": 0.55,
            "state_vector_v1.market_stress": 0.30,
            "state_vector_v1.credit_spread": 4.8,
            "state_vector_v1.rates_level": 1.20,
            "state_vector_v1.valuation_multiple": 6.5,
        }
    )
    rows = [
        {
            "company_id": "0000002488",
            "ticker": "SELF_ROUTINE",
            "normalized_action_id": action_id,
            "action_date": "2024-08-01T00:00:00+00:00",
            "taxonomy.sector": "Consumer Discretionary",
            "taxonomy.subsector": "Specialty Retail",
            "action_size": 200.0,
            "base_market_cap": 1000.0,
            **target_compact,
            "state_vector_v1.profitability": 0.18,
            "state_vector_v1.cash_generation": 0.08,
            "state_vector_v1.gross_obligation_burden": 1.2,
            "state_vector_v1.net_obligation_burden": 0.8,
            "state_vector_v1.liquidity_flexibility": 3.2,
            "state_vector_v1.interest_coverage": 7.5,
            "state_vector_v1.market_access": 0.86,
            "state_vector_v1.market_stress": 0.12,
            "state_vector_v1.credit_spread": 2.2,
            "state_vector_v1.valuation_multiple": 12.0,
        },
        {
            "company_id": "1111111111",
            "ticker": "PEER_STRESS",
            "normalized_action_id": action_id,
            "action_date": "2024-07-15T00:00:00+00:00",
            "taxonomy.sector": "Consumer Discretionary",
            "taxonomy.subsector": "Specialty Retail",
            "action_size": 205.0,
            "base_market_cap": 1000.0,
            **target_compact,
            "state_vector_v1.liquidity_flexibility": 0.50,
            "state_vector_v1.market_stress": 0.28,
            "state_vector_v1.credit_spread": 4.4,
        },
        {
            "company_id": "2222222222",
            "ticker": "PEER_BUFFER",
            "normalized_action_id": action_id,
            "action_date": "2024-06-20T00:00:00+00:00",
            "taxonomy.sector": "Consumer Discretionary",
            "taxonomy.subsector": "Specialty Retail",
            "action_size": 180.0,
            "base_market_cap": 1000.0,
            **target_compact,
            "state_vector_v1.profitability": 0.15,
            "state_vector_v1.cash_generation": 0.05,
            "state_vector_v1.liquidity_flexibility": 2.2,
            "state_vector_v1.interest_coverage": 6.5,
            "state_vector_v1.market_access": 0.82,
            "state_vector_v1.market_stress": 0.14,
            "state_vector_v1.credit_spread": 2.6,
        },
    ]

    source = _build_same_action_analog_positive_source(
        case={
            "company_id": "0000002488",
            "source_company_id": "0000002488",
            "anchor_action_id": action_id,
            "anchor_action_date": "2024-09-02T00:00:00+00:00",
            "as_of_time": "2024-09-02T00:00:00+00:00",
        },
        target_compact=target_compact,
        target_taxonomy={"sector": "Consumer Discretionary", "subsector": "Specialty Retail"},
        target_action_params={"draw_amount_usd": 200.0, "action_size": 200.0},
        target_market_cap=1000.0,
        top_k=2,
        positive_limit_per_source=2,
        negative_limit_per_competitor=2,
        same_action_universe_lookup={
            action_id: {
                "rows": rows,
                "feature_scales": {feature: 1.0 for feature in _STATE_VECTOR_V1_FEATURES},
            }
        },
        regime_aware=True,
    )

    assert source is not None
    assert source["matches"][0]["ticker"] == "PEER_STRESS"
    assert all(match["ticker"] != "SELF_ROUTINE" for match in source["matches"])


def test_pair_rows_include_actual_anchor_positive_against_same_action_retrieved_negatives():
    case = {
        "company_id": "0000002488",
        "as_of_time": "2024-09-02T00:00:00+00:00",
        "anchor_action_id": "capital_return.open_market_buyback",
        "anchor_action_family": "capital_return",
        "source_company_id": "0000002488",
        "anchor_action_date": "2024-09-02T00:00:00+00:00",
    }
    precedent_index = {
        "candidate_rows": [
            {
                "candidate_id": "anchor-1",
                "action_id": "capital_return.open_market_buyback",
                "precedent_confidence": 0.9,
            }
        ]
    }
    precedent_matches = {
        "results": [
            {
                "candidate": {"candidate_id": "anchor-1"},
                "precedent_pack": {
                    "matches": [
                        {
                            "precedent_id": "p1",
                            "company_id": "111111",
                            "action_id": "capital_return.open_market_buyback",
                            "decision_time": "2023-01-01T00:00:00+00:00",
                            "similarity_score": 0.91,
                            "key_state_features": {"state_vector_v1.valuation_multiple": 20.0},
                        },
                        {
                            "precedent_id": "p2",
                            "company_id": "222222",
                            "action_id": "capital_return.open_market_buyback",
                            "decision_time": "2022-01-01T00:00:00+00:00",
                            "similarity_score": 0.82,
                            "key_state_features": {"state_vector_v1.valuation_multiple": 12.0},
                        },
                    ]
                },
            }
        ]
    }
    anchor_outcomes_lookup = {
        ("0000002488", "capital_return.open_market_buyback"): [
            {"company_id": "0000002488", "action_date": "2024-09-02T00:00:00+00:00"}
        ]
    }

    rows = _pair_rows_for_case(
        case=case,
        precedent_index=precedent_index,
        precedent_matches=precedent_matches,
        target_compact={"state_vector_v1.valuation_multiple": 25.0},
        target_taxonomy={"sector": "Information Technology", "subsector": "Semiconductors"},
        target_source="anchor_outcome_fallback",
        top_k=2,
        anchor_outcomes_lookup=anchor_outcomes_lookup,
        precedent_outcomes_lookup={},
        positive_limit_per_source=1,
        negative_limit_per_competitor=0,
        same_family_negatives_only_if_available=False,
        always_include_actual_anchor_positive=True,
        include_within_action_hard_negatives=True,
        positive_source_mode="include_retrieved",
        hard_negative_taxonomy_mode="none",
    )

    actual_rows = [row for row in rows if row["pair_source"] == "actual_anchor_outcome_vs_same_action_retrieved"]
    assert len(actual_rows) == 2
    assert {row["target_source"] for row in actual_rows} == {"anchor_outcome_fallback"}
    assert {row["negative_precedent_id"] for row in actual_rows} == {"p1", "p2"}
    assert {row["negative_source"] for row in actual_rows} == {"same_action_retrieved_pool"}


def test_pair_rows_fall_back_to_retrieved_same_action_negatives_when_actual_anchor_missing():
    case = {
        "company_id": "0000002488",
        "as_of_time": "2024-09-02T00:00:00+00:00",
        "anchor_action_id": "capital_return.open_market_buyback",
        "anchor_action_family": "capital_return",
    }
    precedent_index = {
        "candidate_rows": [
            {
                "candidate_id": "anchor-1",
                "action_id": "capital_return.open_market_buyback",
                "precedent_confidence": 0.9,
            }
        ]
    }
    precedent_matches = {
        "results": [
            {
                "candidate": {"candidate_id": "anchor-1"},
                "precedent_pack": {
                    "matches": [
                        {
                            "precedent_id": "p1",
                            "company_id": "111111",
                            "action_id": "capital_return.open_market_buyback",
                            "decision_time": "2023-01-01T00:00:00+00:00",
                            "similarity_score": 0.91,
                            "key_state_features": {"state_vector_v1.valuation_multiple": 20.0},
                        },
                        {
                            "precedent_id": "p2",
                            "company_id": "222222",
                            "action_id": "capital_return.open_market_buyback",
                            "decision_time": "2022-01-01T00:00:00+00:00",
                            "similarity_score": 0.82,
                            "key_state_features": {"state_vector_v1.valuation_multiple": 12.0},
                        },
                        {
                            "precedent_id": "p3",
                            "company_id": "333333",
                            "action_id": "capital_return.open_market_buyback",
                            "decision_time": "2021-01-01T00:00:00+00:00",
                            "similarity_score": 0.77,
                            "key_state_features": {"state_vector_v1.valuation_multiple": 8.0},
                        },
                    ]
                },
            }
        ]
    }

    rows = _pair_rows_for_case(
        case=case,
        precedent_index=precedent_index,
        precedent_matches=precedent_matches,
        target_compact={"state_vector_v1.valuation_multiple": 25.0},
        target_taxonomy={"sector": "Information Technology", "subsector": "Semiconductors"},
        top_k=3,
        anchor_outcomes_lookup={},
        precedent_outcomes_lookup={},
        positive_limit_per_source=1,
        negative_limit_per_competitor=0,
        same_family_negatives_only_if_available=False,
        always_include_actual_anchor_positive=False,
        include_within_action_hard_negatives=True,
        positive_source_mode="include_retrieved",
        hard_negative_taxonomy_mode="none",
    )

    retrieved_rows = [row for row in rows if row["pair_source"] == "retrieved_anchor_action_vs_same_action_retrieved"]
    assert len(retrieved_rows) == 2
    assert {row["positive_precedent_id"] for row in retrieved_rows} == {"p1"}
    assert {row["negative_precedent_id"] for row in retrieved_rows} == {"p2", "p3"}


def test_pair_rows_prefer_actual_anchor_source_when_requested():
    case = {
        "company_id": "0000002488",
        "as_of_time": "2024-09-02T00:00:00+00:00",
        "anchor_action_id": "capital_return.open_market_buyback",
        "anchor_action_family": "capital_return",
    }
    precedent_index = {
        "candidate_rows": [
            {
                "candidate_id": "anchor-1",
                "action_id": "capital_return.open_market_buyback",
                "precedent_confidence": 0.9,
            }
        ]
    }
    precedent_matches = {
        "results": [
            {
                "candidate": {"candidate_id": "anchor-1"},
                "precedent_pack": {
                    "matches": [
                        {
                            "precedent_id": "p1",
                            "company_id": "111111",
                            "action_id": "capital_return.open_market_buyback",
                            "decision_time": "2023-01-01T00:00:00+00:00",
                            "similarity_score": 0.91,
                            "key_state_features": {"state_vector_v1.valuation_multiple": 20.0},
                        },
                        {
                            "precedent_id": "p2",
                            "company_id": "222222",
                            "action_id": "capital_return.open_market_buyback",
                            "decision_time": "2022-01-01T00:00:00+00:00",
                            "similarity_score": 0.82,
                            "key_state_features": {"state_vector_v1.valuation_multiple": 12.0},
                        },
                    ]
                },
            }
        ]
    }
    anchor_outcomes_lookup = {
        ("0000002488", "capital_return.open_market_buyback"): [
            {"company_id": "0000002488", "action_date": "2024-09-02T00:00:00+00:00"}
        ]
    }

    rows = _pair_rows_for_case(
        case=case,
        precedent_index=precedent_index,
        precedent_matches=precedent_matches,
        target_compact={"state_vector_v1.valuation_multiple": 25.0},
        target_taxonomy={"sector": "Information Technology", "subsector": "Semiconductors"},
        top_k=2,
        anchor_outcomes_lookup=anchor_outcomes_lookup,
        precedent_outcomes_lookup={},
        positive_limit_per_source=1,
        negative_limit_per_competitor=0,
        same_family_negatives_only_if_available=False,
        always_include_actual_anchor_positive=True,
        include_within_action_hard_negatives=True,
        positive_source_mode="actual_anchor_preferred",
        hard_negative_taxonomy_mode="none",
    )

    assert rows
    assert {row["positive_source"] for row in rows} == {"actual_anchor_outcome"}


def test_pair_rows_can_use_same_action_analog_consensus_teacher():
    case = {
        "company_id": "0000002488",
        "source_company_id": "0000002488",
        "as_of_time": "2024-09-02T00:00:00+00:00",
        "anchor_action_id": "capital_return.open_market_buyback",
        "anchor_action_family": "capital_return",
        "anchor_action_date": "2024-09-02T00:00:00+00:00",
    }
    precedent_index = {
        "candidate_rows": [
            {
                "candidate_id": "anchor_retrieved",
                "action_id": "capital_return.open_market_buyback",
                "precedent_confidence": 0.9,
            }
        ]
    }
    precedent_matches = {
        "results": [
            {
                "candidate_id": "anchor_retrieved",
                "precedent_pack": {
                    "matches": [
                        {
                            "precedent_id": "999999::2024-02-01::0",
                            "company_id": "999999",
                            "action_id": "capital_return.open_market_buyback",
                            "decision_time": "2024-02-01T00:00:00+00:00",
                            "similarity_score": 0.91,
                            "state_vector_v1.valuation_multiple": 14.0,
                            "state_vector_v1.growth": 0.01,
                            "state_vector_v1.cash_generation": 0.030,
                        }
                    ]
                },
            }
        ]
    }
    same_action_universe_lookup = {
        "capital_return.open_market_buyback": {
            "feature_scales": {
                "state_vector_v1.valuation_multiple": 10.0,
                "state_vector_v1.growth": 0.10,
                "state_vector_v1.cash_generation": 0.05,
            },
            "rows": [
                {
                    "company_id": "111111",
                    "normalized_action_id": "capital_return.open_market_buyback",
                    "action_date": "2024-06-01T00:00:00+00:00",
                    "taxonomy.sector": "Information Technology",
                    "taxonomy.subsector": "Semiconductors",
                    "state_vector_v1.valuation_multiple": 58.0,
                    "state_vector_v1.growth": 0.05,
                    "state_vector_v1.cash_generation": 0.020,
                },
                {
                    "company_id": "222222",
                    "normalized_action_id": "capital_return.open_market_buyback",
                    "action_date": "2024-05-01T00:00:00+00:00",
                    "taxonomy.sector": "Information Technology",
                    "taxonomy.subsector": "Semiconductors",
                    "state_vector_v1.valuation_multiple": 40.0,
                    "state_vector_v1.growth": 0.03,
                    "state_vector_v1.cash_generation": 0.018,
                },
                {
                    "company_id": "333333",
                    "normalized_action_id": "capital_return.open_market_buyback",
                    "action_date": "2024-04-01T00:00:00+00:00",
                    "taxonomy.sector": "Information Technology",
                    "taxonomy.subsector": "Semiconductors",
                    "state_vector_v1.valuation_multiple": 24.0,
                    "state_vector_v1.growth": -0.02,
                    "state_vector_v1.cash_generation": 0.010,
                },
                {
                    "company_id": "444444",
                    "normalized_action_id": "capital_return.open_market_buyback",
                    "action_date": "2024-03-01T00:00:00+00:00",
                    "taxonomy.sector": "Information Technology",
                    "taxonomy.subsector": "Software",
                    "state_vector_v1.valuation_multiple": 59.0,
                    "state_vector_v1.growth": 0.06,
                    "state_vector_v1.cash_generation": 0.021,
                },
            ],
        }
    }

    rows = _pair_rows_for_case(
        case=case,
        precedent_index=precedent_index,
        precedent_matches=precedent_matches,
        target_compact={
            "state_vector_v1.valuation_multiple": 60.0,
            "state_vector_v1.growth": 0.06,
            "state_vector_v1.cash_generation": 0.021,
        },
        target_taxonomy={"sector": "Information Technology", "subsector": "Semiconductors"},
        top_k=2,
        anchor_outcomes_lookup={},
        precedent_outcomes_lookup={},
        positive_limit_per_source=1,
        negative_limit_per_competitor=0,
        same_family_negatives_only_if_available=False,
        always_include_actual_anchor_positive=False,
        include_within_action_hard_negatives=True,
        positive_source_mode="analog_consensus_same_action_universe",
        hard_negative_taxonomy_mode="none",
        same_action_universe_lookup=same_action_universe_lookup,
    )

    assert rows
    assert {row["positive_source"] for row in rows} == {"analog_consensus_same_action_universe"}
    assert {row["pair_source"] for row in rows} == {"analog_consensus_same_action_universe_vs_same_action_confusers"}
    assert {row["positive_precedent_company_id"] for row in rows} == {"111111"}
    assert {row["negative_precedent_company_id"] for row in rows} >= {"222222", "333333"}


def test_same_action_analog_confusers_do_not_reintroduce_retrieved_anchor_noise():
    case = {
        "company_id": "0000002488",
        "source_company_id": "0000002488",
        "as_of_time": "2024-09-02T00:00:00+00:00",
        "anchor_action_id": "capital_return.open_market_buyback",
        "anchor_action_family": "capital_return",
        "anchor_action_date": "2024-09-02T00:00:00+00:00",
    }
    precedent_index = {
        "candidate_rows": [
            {
                "candidate_id": "anchor_retrieved",
                "action_id": "capital_return.open_market_buyback",
                "precedent_confidence": 0.9,
            }
        ]
    }
    precedent_matches = {
        "results": [
            {
                "candidate_id": "anchor_retrieved",
                "precedent_pack": {
                    "matches": [
                        {
                            "precedent_id": "999999::2024-02-01::0",
                            "company_id": "999999",
                            "action_id": "capital_return.open_market_buyback",
                            "decision_time": "2024-02-01T00:00:00+00:00",
                            "similarity_score": 0.99,
                            "key_state_features": {
                                "sector": "Consumer Discretionary",
                                "subsector": "Retail",
                            },
                        }
                    ]
                },
            }
        ]
    }
    same_action_universe_lookup = {
        "capital_return.open_market_buyback": {
            "feature_scales": {
                "state_vector_v1.valuation_multiple": 10.0,
                "state_vector_v1.growth": 0.10,
                "state_vector_v1.cash_generation": 0.05,
            },
            "rows": [
                {
                    "company_id": "111111",
                    "normalized_action_id": "capital_return.open_market_buyback",
                    "action_date": "2024-06-01T00:00:00+00:00",
                    "taxonomy.sector": "Information Technology",
                    "taxonomy.subsector": "Semiconductors",
                    "state_vector_v1.valuation_multiple": 58.0,
                    "state_vector_v1.growth": 0.05,
                    "state_vector_v1.cash_generation": 0.020,
                },
                {
                    "company_id": "222222",
                    "normalized_action_id": "capital_return.open_market_buyback",
                    "action_date": "2024-05-01T00:00:00+00:00",
                    "taxonomy.sector": "Information Technology",
                    "taxonomy.subsector": "Semiconductors",
                    "state_vector_v1.valuation_multiple": 40.0,
                    "state_vector_v1.growth": 0.03,
                    "state_vector_v1.cash_generation": 0.018,
                },
                {
                    "company_id": "333333",
                    "normalized_action_id": "capital_return.open_market_buyback",
                    "action_date": "2024-04-01T00:00:00+00:00",
                    "taxonomy.sector": "Information Technology",
                    "taxonomy.subsector": "Semiconductors",
                    "state_vector_v1.valuation_multiple": 24.0,
                    "state_vector_v1.growth": -0.02,
                    "state_vector_v1.cash_generation": 0.010,
                },
                {
                    "company_id": "444444",
                    "normalized_action_id": "capital_return.open_market_buyback",
                    "action_date": "2024-03-01T00:00:00+00:00",
                    "taxonomy.sector": "Information Technology",
                    "taxonomy.subsector": "Hardware",
                    "state_vector_v1.valuation_multiple": 20.0,
                    "state_vector_v1.growth": -0.03,
                    "state_vector_v1.cash_generation": 0.011,
                },
            ],
        }
    }

    rows = _pair_rows_for_case(
        case=case,
        precedent_index=precedent_index,
        precedent_matches=precedent_matches,
        target_compact={
            "state_vector_v1.valuation_multiple": 60.0,
            "state_vector_v1.growth": 0.06,
            "state_vector_v1.cash_generation": 0.021,
        },
        target_taxonomy={"sector": "Information Technology", "subsector": "Semiconductors"},
        top_k=2,
        anchor_outcomes_lookup={},
        precedent_outcomes_lookup={},
        positive_limit_per_source=1,
        negative_limit_per_competitor=0,
        same_family_negatives_only_if_available=False,
        always_include_actual_anchor_positive=False,
        include_within_action_hard_negatives=True,
        positive_source_mode="analog_consensus_same_action_universe",
        hard_negative_taxonomy_mode="none",
        same_action_universe_lookup=same_action_universe_lookup,
    )

    assert rows
    assert "999999" not in {row["negative_precedent_company_id"] for row in rows}


def test_pair_rows_can_add_same_action_rank_ordering_pairs_for_analog_teacher():
    case = {
        "company_id": "0000002488",
        "source_company_id": "0000002488",
        "as_of_time": "2024-09-02T00:00:00+00:00",
        "anchor_action_id": "capital_return.open_market_buyback",
        "anchor_action_family": "capital_return",
        "anchor_action_date": "2024-09-02T00:00:00+00:00",
    }
    same_action_universe_lookup = {
        "capital_return.open_market_buyback": {
            "feature_scales": {
                "state_vector_v1.valuation_multiple": 10.0,
                "state_vector_v1.growth": 0.10,
                "state_vector_v1.cash_generation": 0.05,
            },
            "rows": [
                {
                    "company_id": "111111",
                    "normalized_action_id": "capital_return.open_market_buyback",
                    "action_date": "2024-06-01T00:00:00+00:00",
                    "taxonomy.sector": "Information Technology",
                    "taxonomy.subsector": "Semiconductors",
                    "state_vector_v1.valuation_multiple": 58.0,
                    "state_vector_v1.growth": 0.05,
                    "state_vector_v1.cash_generation": 0.020,
                },
                {
                    "company_id": "222222",
                    "normalized_action_id": "capital_return.open_market_buyback",
                    "action_date": "2024-05-01T00:00:00+00:00",
                    "taxonomy.sector": "Information Technology",
                    "taxonomy.subsector": "Semiconductors",
                    "state_vector_v1.valuation_multiple": 40.0,
                    "state_vector_v1.growth": 0.03,
                    "state_vector_v1.cash_generation": 0.018,
                },
                {
                    "company_id": "333333",
                    "normalized_action_id": "capital_return.open_market_buyback",
                    "action_date": "2024-04-01T00:00:00+00:00",
                    "taxonomy.sector": "Information Technology",
                    "taxonomy.subsector": "Semiconductors",
                    "state_vector_v1.valuation_multiple": 24.0,
                    "state_vector_v1.growth": -0.02,
                    "state_vector_v1.cash_generation": 0.010,
                },
            ],
        }
    }

    rows = _pair_rows_for_case(
        case=case,
        precedent_index={"candidate_rows": []},
        precedent_matches={"results": []},
        target_compact={
            "state_vector_v1.valuation_multiple": 60.0,
            "state_vector_v1.growth": 0.06,
            "state_vector_v1.cash_generation": 0.021,
        },
        target_taxonomy={"sector": "Information Technology", "subsector": "Semiconductors"},
        top_k=3,
        anchor_outcomes_lookup={},
        precedent_outcomes_lookup={},
        positive_limit_per_source=1,
        negative_limit_per_competitor=0,
        same_family_negatives_only_if_available=False,
        always_include_actual_anchor_positive=False,
        include_within_action_hard_negatives=False,
        include_same_action_positive_ordering=True,
        positive_source_mode="analog_consensus_same_action_universe",
        hard_negative_taxonomy_mode="none",
        same_action_universe_lookup=same_action_universe_lookup,
    )

    ordering_rows = [row for row in rows if row["pair_source"] == "analog_consensus_same_action_universe_rank_ordering"]
    assert ordering_rows
    assert {row["positive_precedent_company_id"] for row in ordering_rows} >= {"111111", "222222"}
    assert {row["negative_precedent_company_id"] for row in ordering_rows} >= {"222222", "333333"}


def test_same_action_rank_ordering_uses_local_window():
    case = {
        "company_id": "0000002488",
        "source_company_id": "0000002488",
        "as_of_time": "2024-09-02T00:00:00+00:00",
        "anchor_action_id": "capital_return.open_market_buyback",
        "anchor_action_family": "capital_return",
        "anchor_action_date": "2024-09-02T00:00:00+00:00",
    }
    same_action_universe_lookup = {
        "capital_return.open_market_buyback": {
            "feature_scales": {
                "state_vector_v1.valuation_multiple": 10.0,
                "state_vector_v1.growth": 0.10,
                "state_vector_v1.cash_generation": 0.05,
            },
            "rows": [
                {
                    "company_id": company_id,
                    "normalized_action_id": "capital_return.open_market_buyback",
                    "action_date": action_date,
                    "taxonomy.sector": "Information Technology",
                    "taxonomy.subsector": "Semiconductors",
                    "state_vector_v1.valuation_multiple": valuation,
                    "state_vector_v1.growth": growth,
                    "state_vector_v1.cash_generation": cash_generation,
                }
                for company_id, action_date, valuation, growth, cash_generation in [
                    ("111111", "2024-06-01T00:00:00+00:00", 58.0, 0.050, 0.020),
                    ("222222", "2024-05-01T00:00:00+00:00", 52.0, 0.045, 0.019),
                    ("333333", "2024-04-01T00:00:00+00:00", 45.0, 0.040, 0.018),
                    ("444444", "2024-03-01T00:00:00+00:00", 30.0, 0.010, 0.014),
                    ("555555", "2024-02-01T00:00:00+00:00", 18.0, -0.020, 0.010),
                ]
            ],
        }
    }

    rows = _pair_rows_for_case(
        case=case,
        precedent_index={"candidate_rows": []},
        precedent_matches={"results": []},
        target_compact={
            "state_vector_v1.valuation_multiple": 60.0,
            "state_vector_v1.growth": 0.06,
            "state_vector_v1.cash_generation": 0.021,
        },
        target_taxonomy={"sector": "Information Technology", "subsector": "Semiconductors"},
        top_k=4,
        anchor_outcomes_lookup={},
        precedent_outcomes_lookup={},
        positive_limit_per_source=1,
        negative_limit_per_competitor=0,
        same_family_negatives_only_if_available=False,
        always_include_actual_anchor_positive=False,
        include_within_action_hard_negatives=False,
        include_same_action_positive_ordering=True,
        positive_source_mode="analog_consensus_same_action_universe",
        hard_negative_taxonomy_mode="none",
        same_action_universe_lookup=same_action_universe_lookup,
    )

    ordering_rows = [row for row in rows if row["pair_source"] == "analog_consensus_same_action_universe_rank_ordering"]
    assert ordering_rows
    top_vs_far_tail = [
        row
        for row in ordering_rows
        if row["positive_precedent_company_id"] == "111111" and row["negative_precedent_company_id"] == "555555"
    ]
    assert top_vs_far_tail == []


def test_pair_rows_can_use_same_action_regime_analog_consensus_teacher():
    case = {
        "company_id": "0000002488",
        "source_company_id": "0000002488",
        "as_of_time": "2024-09-02T00:00:00+00:00",
        "anchor_action_id": "capital_return.open_market_buyback",
        "anchor_action_family": "capital_return",
        "anchor_action_date": "2024-09-02T00:00:00+00:00",
    }
    precedent_index = {"candidate_rows": []}
    precedent_matches = {"results": []}
    rows_payload = [
        {
            "company_id": "111111",
            "normalized_action_id": "capital_return.open_market_buyback",
            "action_date": "2024-06-01T00:00:00+00:00",
            "taxonomy.sector": "Information Technology",
            "taxonomy.subsector": "Semiconductors",
            "state_vector_v1.valuation_multiple": 58.0,
            "state_vector_v1.growth": 0.14,
            "state_vector_v1.cash_generation": 0.012,
        },
        {
            "company_id": "222222",
            "normalized_action_id": "capital_return.open_market_buyback",
            "action_date": "2024-05-01T00:00:00+00:00",
            "taxonomy.sector": "Information Technology",
            "taxonomy.subsector": "Semiconductors",
            "state_vector_v1.valuation_multiple": 46.0,
            "state_vector_v1.growth": 0.06,
            "state_vector_v1.cash_generation": 0.020,
        },
        {
            "company_id": "333333",
            "normalized_action_id": "capital_return.open_market_buyback",
            "action_date": "2024-04-01T00:00:00+00:00",
            "taxonomy.sector": "Information Technology",
            "taxonomy.subsector": "Semiconductors",
            "state_vector_v1.valuation_multiple": 18.0,
            "state_vector_v1.growth": 0.02,
            "state_vector_v1.cash_generation": 0.060,
        },
        {
            "company_id": "444444",
            "normalized_action_id": "capital_return.open_market_buyback",
            "action_date": "2024-03-01T00:00:00+00:00",
            "taxonomy.sector": "Information Technology",
            "taxonomy.subsector": "Semiconductors",
            "state_vector_v1.valuation_multiple": 16.0,
            "state_vector_v1.growth": 0.01,
            "state_vector_v1.cash_generation": 0.055,
        },
    ]
    feature_matrix = []
    for row in rows_payload:
        feature_matrix.append(
            [
                float(row.get(feature)) if row.get(feature) is not None else float("nan")
                for feature in _STATE_VECTOR_V1_FEATURES
            ]
        )
    import numpy as np

    feature_matrix_np = np.asarray(feature_matrix, dtype=float)
    regime_model = fit_latent_regime_kmeans(
        feature_matrix_np,
        feature_names=_STATE_VECTOR_V1_FEATURES,
        n_clusters=2,
        seed=7,
        max_iter=50,
    )
    regime_memberships = latent_regime_memberships(feature_matrix_np, regime_model)

    same_action_universe_lookup = {
        "capital_return.open_market_buyback": {
            "feature_scales": {
                "state_vector_v1.valuation_multiple": 10.0,
                "state_vector_v1.growth": 0.10,
                "state_vector_v1.cash_generation": 0.05,
            },
            "rows": rows_payload,
            "feature_matrix": feature_matrix_np,
            "company_id_arr": np.asarray([row["company_id"] for row in rows_payload], dtype=object),
            "action_time_arr": np.asarray(
                [
                    np.datetime64("2024-06-01T00:00:00"),
                    np.datetime64("2024-05-01T00:00:00"),
                    np.datetime64("2024-04-01T00:00:00"),
                    np.datetime64("2024-03-01T00:00:00"),
                ]
            ),
            "sector_arr": np.asarray(["Information Technology"] * 4, dtype=object),
            "subsector_arr": np.asarray(["Semiconductors"] * 4, dtype=object),
            "latent_regime_model": regime_model,
            "latent_regime_memberships": regime_memberships,
        }
    }

    rows = _pair_rows_for_case(
        case=case,
        precedent_index=precedent_index,
        precedent_matches=precedent_matches,
        target_compact={
            "state_vector_v1.valuation_multiple": 60.0,
            "state_vector_v1.growth": 0.05,
            "state_vector_v1.cash_generation": 0.020,
        },
        target_taxonomy={"sector": "Information Technology", "subsector": "Semiconductors"},
        top_k=2,
        anchor_outcomes_lookup={},
        precedent_outcomes_lookup={},
        positive_limit_per_source=1,
        negative_limit_per_competitor=0,
        same_family_negatives_only_if_available=False,
        always_include_actual_anchor_positive=False,
        include_within_action_hard_negatives=True,
        positive_source_mode="analog_regime_consensus_same_action_universe",
        hard_negative_taxonomy_mode="none",
        same_action_universe_lookup=same_action_universe_lookup,
    )

    assert rows
    assert {row["positive_source"] for row in rows} == {"analog_regime_consensus_same_action_universe"}
    assert {row["pair_source"] for row in rows} == {"analog_regime_consensus_same_action_universe_vs_same_action_confusers"}
    assert {row["positive_precedent_company_id"] for row in rows} == {"111111"}


def test_actual_anchor_teacher_can_mine_within_action_negatives_from_same_action_universe():
    case = {
        "company_id": "0000002488",
        "source_company_id": "0000002488",
        "as_of_time": "2024-09-02T00:00:00+00:00",
        "anchor_action_id": "capital_return.open_market_buyback",
        "anchor_action_family": "capital_return",
        "anchor_action_date": "2024-12-31T00:00:00+00:00",
    }
    precedent_index = {"candidate_rows": []}
    precedent_matches = {"results": []}
    anchor_outcomes_lookup = {
        ("0000002488", "capital_return.open_market_buyback"): [
            {
                "company_id": "0000002488",
                "action_date": "2024-12-31T00:00:00+00:00",
                "state_vector_v1.valuation_multiple": 36.0,
            }
        ]
    }
    same_action_universe_lookup = {
        "capital_return.open_market_buyback": {
            "feature_scales": {
                "state_vector_v1.valuation_multiple": 10.0,
                "state_vector_v1.growth": 0.10,
                "state_vector_v1.cash_generation": 0.05,
            },
            "rows": [
                {
                    "company_id": "111111",
                    "normalized_action_id": "capital_return.open_market_buyback",
                    "action_date": "2024-06-01T00:00:00+00:00",
                    "taxonomy.sector": "Information Technology",
                    "taxonomy.subsector": "Semiconductors",
                    "state_vector_v1.valuation_multiple": 58.0,
                    "state_vector_v1.growth": 0.05,
                    "state_vector_v1.cash_generation": 0.020,
                },
                {
                    "company_id": "222222",
                    "normalized_action_id": "capital_return.open_market_buyback",
                    "action_date": "2024-05-01T00:00:00+00:00",
                    "taxonomy.sector": "Information Technology",
                    "taxonomy.subsector": "Semiconductors",
                    "state_vector_v1.valuation_multiple": 40.0,
                    "state_vector_v1.growth": 0.03,
                    "state_vector_v1.cash_generation": 0.018,
                },
                {
                    "company_id": "333333",
                    "normalized_action_id": "capital_return.open_market_buyback",
                    "action_date": "2024-04-01T00:00:00+00:00",
                    "taxonomy.sector": "Information Technology",
                    "taxonomy.subsector": "Semiconductors",
                    "state_vector_v1.valuation_multiple": 24.0,
                    "state_vector_v1.growth": -0.02,
                    "state_vector_v1.cash_generation": 0.010,
                },
            ],
        }
    }

    rows = _pair_rows_for_case(
        case=case,
        precedent_index=precedent_index,
        precedent_matches=precedent_matches,
        target_compact={
            "state_vector_v1.valuation_multiple": 60.0,
            "state_vector_v1.growth": 0.06,
            "state_vector_v1.cash_generation": 0.021,
        },
        target_taxonomy={"sector": "Information Technology", "subsector": "Semiconductors"},
        top_k=2,
        anchor_outcomes_lookup=anchor_outcomes_lookup,
        precedent_outcomes_lookup={},
        positive_limit_per_source=1,
        negative_limit_per_competitor=2,
        same_family_negatives_only_if_available=False,
        always_include_actual_anchor_positive=True,
        include_within_action_hard_negatives=True,
        actual_anchor_within_action_negative_source="same_action_universe",
        positive_source_mode="actual_anchor_preferred",
        hard_negative_taxonomy_mode="none",
        same_action_universe_lookup=same_action_universe_lookup,
    )

    assert rows
    assert {row["positive_source"] for row in rows} == {"actual_anchor_outcome"}
    assert {row["pair_source"] for row in rows} == {"actual_anchor_outcome_vs_same_action_universe"}
    assert {row["negative_source"] for row in rows} == {"same_action_universe"}
    assert {row["negative_precedent_company_id"] for row in rows} == {"111111", "222222", "333333"}
