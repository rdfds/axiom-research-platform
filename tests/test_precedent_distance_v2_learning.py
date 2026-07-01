from __future__ import annotations

import copy

from src.pipeline.precedent_distance_v2_learning import (
    build_precedent_distance_v2_payload,
    coordinate_search_scope_configuration,
    default_scope_configuration,
    extract_report_aggregate,
    is_better_report_aggregate,
)


def test_is_better_report_aggregate_respects_lexicographic_priority():
    objective = {
        "objective_order": [
            {"metric": "error_rate", "direction": "minimize"},
            {"metric": "coverage_skip_rate", "direction": "minimize"},
            {"metric": "mean_alignment_score", "direction": "maximize"},
        ]
    }
    incumbent = {
        "error_rate": 0.00,
        "coverage_skip_rate": 0.10,
        "mean_alignment_score": 0.70,
    }
    candidate = {
        "error_rate": 0.02,
        "coverage_skip_rate": 0.00,
        "mean_alignment_score": 0.95,
    }
    assert is_better_report_aggregate(candidate, incumbent, objective) is False


