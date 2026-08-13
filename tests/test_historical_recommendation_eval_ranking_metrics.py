from __future__ import annotations

from src.historical_recommendation_eval import (
    _aggregate_historical_cases,
    _score_precedent_ranking,
)


def test_score_precedent_ranking_prefers_anchor_action_and_family():
    precedent_index = {
        "candidate_rows": [
            {
                "action_id": "capital_structure.equity_issuance",
                "precedent_confidence": 0.82,
            },
            {
                "action_id": "capital_structure.refinancing",
                "precedent_confidence": 0.65,
            },
            {
                "action_id": "capital_return.open_market_buyback",
                "precedent_confidence": 0.40,
            },
        ]
    }
    ranking = _score_precedent_ranking(
        precedent_index=precedent_index,
        anchor_action_id="capital_structure.equity_issuance",
        anchor_action_family="capital_structure",
        anchor_action_support={"support_mode": "exact_supported"},
    )
    assert ranking["reason"] == "ok"
    assert ranking["anchor_action_precedent_rank"] == 1
    assert ranking["anchor_action_precedent_mrr"] == 1.0
    assert ranking["anchor_action_precedent_top1"] is True
    assert ranking["anchor_family_precedent_rank"] == 1
    assert ranking["anchor_support_adjusted_precedent_rank"] == 1
    assert ranking["anchor_action_precedent_margin"] == 0.17


def test_aggregate_historical_cases_exposes_precedent_ranking_metrics():
    cases = [
        {
            "anchor_action_family": "capital_structure",
            "top_action_ids": ["capital_structure.equity_issuance"],
            "historical_alignment": {
                "score": 1.0,
                "primary_exact_match": True,
                "primary_family_match": True,
                "primary_support_adjusted_match": True,
                "any_exact_match": True,
                "any_family_match": True,
                "any_support_adjusted_match": True,
            },
            "precedent_ranking": {
                "reason": "ok",
                "anchor_action_precedent_top1": True,
                "anchor_action_precedent_mrr": 1.0,
                "anchor_action_precedent_margin": 0.20,
                "anchor_family_precedent_top1": True,
                "anchor_family_precedent_mrr": 1.0,
                "anchor_family_precedent_margin": 0.20,
                "anchor_support_adjusted_precedent_top1": True,
                "anchor_support_adjusted_precedent_mrr": 1.0,
                "anchor_support_adjusted_precedent_margin": 0.20,
            },
        },
        {
            "anchor_action_family": "capital_structure",
            "top_action_ids": ["capital_structure.refinancing"],
            "historical_alignment": {
                "score": 0.6,
                "primary_exact_match": False,
                "primary_family_match": True,
                "primary_support_adjusted_match": True,
                "any_exact_match": False,
                "any_family_match": True,
                "any_support_adjusted_match": True,
            },
            "precedent_ranking": {
                "reason": "ok",
                "anchor_action_precedent_top1": False,
                "anchor_action_precedent_mrr": 0.5,
                "anchor_action_precedent_margin": -0.05,
                "anchor_family_precedent_top1": True,
                "anchor_family_precedent_mrr": 1.0,
                "anchor_family_precedent_margin": 0.10,
                "anchor_support_adjusted_precedent_top1": False,
                "anchor_support_adjusted_precedent_mrr": 0.5,
                "anchor_support_adjusted_precedent_margin": -0.05,
            },
        },
    ]
    aggregate = _aggregate_historical_cases(cases)
    assert aggregate["precedent_ranking_case_count"] == 2
    assert aggregate["anchor_action_precedent_top1_rate"] == 0.5
    assert aggregate["anchor_action_precedent_mrr_mean"] == 0.75
    assert aggregate["anchor_action_precedent_margin_mean"] == 0.075
    assert aggregate["anchor_action_precedent_margin_min"] == -0.05
    assert aggregate["anchor_action_precedent_negative_margin_case_count"] == 1
    assert aggregate["anchor_family_precedent_top1_rate"] == 1.0
    assert aggregate["anchor_family_precedent_mrr_mean"] == 1.0
    assert aggregate["anchor_family_precedent_margin_min"] == 0.1
    assert aggregate["anchor_family_precedent_negative_margin_case_count"] == 0
    assert aggregate["anchor_support_adjusted_precedent_top1_rate"] == 0.5
    assert aggregate["anchor_support_adjusted_precedent_mrr_mean"] == 0.75
    assert aggregate["anchor_support_adjusted_precedent_margin_min"] == -0.05
    assert aggregate["anchor_support_adjusted_precedent_negative_margin_case_count"] == 1
