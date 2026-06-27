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


