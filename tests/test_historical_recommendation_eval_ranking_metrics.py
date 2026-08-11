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


