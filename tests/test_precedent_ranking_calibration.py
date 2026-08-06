from src.pipeline.precedent_brain import _compute_calibration_confidence
from src.pipeline.precedent_index import _query_rank_score, build_precedent_index


def test_compute_calibration_confidence_rewards_strong_sibling_matches():
    weak = _compute_calibration_confidence(
        retrieval_tier="sibling_type",
        exact_match_count=1,
        exact_support_min=6,
        cohort_size=10,
        base_similarity=0.44,
        top_similarity_mean=0.47,
        top_similarity_p25=0.41,
        top_action_match_score=0.70,
        mismatch_count=1,
        regime_mismatch=False,
        parameter_mismatch=False,
        narrative_mismatch=False,
    )
    strong = _compute_calibration_confidence(
        retrieval_tier="sibling_type",
        exact_match_count=5,
        exact_support_min=6,
        cohort_size=18,
        base_similarity=0.62,
        top_similarity_mean=0.73,
        top_similarity_p25=0.66,
        top_action_match_score=0.94,
        mismatch_count=0,
        regime_mismatch=False,
        parameter_mismatch=False,
        narrative_mismatch=False,
    )

    assert strong["exact_support_ratio"] > weak["exact_support_ratio"]
    assert strong["similarity_signal"] > weak["similarity_signal"]
    assert strong["calibration_confidence"] > weak["calibration_confidence"]


def test_query_rank_score_prefers_exact_supported_rows():
    exact = _query_rank_score(
        precedent_confidence=0.58,
        sample_size=18,
        out_of_sample_flag=False,
        source="overall",
        retrieval_tier="exact",
        low_precedent_coverage=False,
        exact_support_ratio=1.0,
        top_similarity_mean=0.74,
        top_action_match_score=0.95,
    )
    weak_global = _query_rank_score(
        precedent_confidence=0.58,
        sample_size=18,
        out_of_sample_flag=True,
        source="overall",
        retrieval_tier="global",
        low_precedent_coverage=True,
        exact_support_ratio=0.0,
        top_similarity_mean=0.46,
        top_action_match_score=0.68,
    )

    assert exact > weak_global

def test_build_precedent_index_carries_diagnostics_into_rows():
    index = build_precedent_index(
        run_id="r1",
        precedent_matches=[
            {
                "candidate": {
                    "candidate_id": "c1",
                    "action_type": "mna",
                    "action_subtype": "platform_acquisition",
                    "action_id": "mna.platform_acquisition",
                },
                "precedent_pack": {
                    "precedent_confidence": 0.61,
                    "mismatch_diagnostics": {
                        "out_of_sample_flag": False,
                        "low_precedent_coverage": False,
                        "retrieval_tier": "sibling_type",
                        "exact_support_ratio": 0.83,
                        "top_similarity_mean": 0.71,
                        "top_action_match_score": 0.92,
                    },
                    "outcome_distributions": {
                        "horizon_12m": {
                            "equity_return_vs_sector": {
                                "mean": 1.0,
                                "median": 1.0,
                                "p10": 0.0,
                                "p25": 0.5,
                                "p75": 1.5,
                                "p90": 2.0,
                                "sample_size": 14,
                            }
                        }
                    },
                    "regime_splits": [],
                    "retrieved_cohorts": [],
                },
            }
        ],
    )

    assert index["candidate_rows"][0]["retrieval_tier"] == "sibling_type"
    assert index["candidate_rows"][0]["exact_support_ratio"] == 0.83
    row = next(r for r in index["distribution_rows"] if r["metric"] == "equity_return_vs_sector" and r["sample_size"] == 14)
    assert row["retrieval_tier"] == "sibling_type"
    assert row["top_similarity_mean"] == 0.71
    assert row["top_action_match_score"] == 0.92
    assert row["query_score"] > 0.0
