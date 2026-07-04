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


def test_coordinate_search_scope_configuration_improves_toward_preferred_weight():
    objective = {
        "objective_order": [
            {"metric": "error_rate", "direction": "minimize"},
            {"metric": "coverage_skip_rate", "direction": "minimize"},
            {"metric": "mean_alignment_score", "direction": "maximize"},
        ],
        "search_space": {
            "group_weights": ["valuation"],
            "within_group_relative_weights": {"min": 0.5, "max": 2.0},
        },
    }

    def evaluate_scope_config(scope_config):
        valuation_weight = float(scope_config["group_weights"]["valuation"])
        alignment = max(0.0, 1.0 - abs(valuation_weight - 1.4))
        return {
            "aggregate": {
                "error_rate": 0.0,
                "coverage_skip_rate": 0.0,
                "mean_alignment_score": alignment,
            }
        }

    search = coordinate_search_scope_configuration(
        scope_key="capital_structure",
        objective_config=objective,
        evaluate_scope_config=evaluate_scope_config,
        max_rounds=1,
    )
    assert float(search["best_config"]["group_weights"]["valuation"]) == 1.4
    assert float(search["best_aggregate"]["mean_alignment_score"]) == 1.0


def test_coordinate_search_scope_configuration_respects_custom_group_grid_values():
    objective = {
        "objective_order": [
            {"metric": "mean_alignment_score", "direction": "maximize"},
        ],
        "search_space": {
            "group_weights": ["valuation"],
            "group_weight_grid_values": [0.7, 1.3],
            "optimize_feature_relative_weights": False,
            "optimize_gates": False,
            "optimize_penalties": False,
            "optimize_blend_weights": False,
        },
    }

    seen_weights = []

    def evaluate_scope_config(scope_config):
        valuation_weight = float(scope_config["group_weights"]["valuation"])
        seen_weights.append(valuation_weight)
        alignment = 1.0 if abs(valuation_weight - 1.3) <= 1e-12 else 0.0
        return {"aggregate": {"mean_alignment_score": alignment}}

    search = coordinate_search_scope_configuration(
        scope_key="capital_structure",
        objective_config=objective,
        evaluate_scope_config=evaluate_scope_config,
        max_rounds=1,
    )
    assert sorted(set(seen_weights)) == [0.7, 0.88, 1.3]
    assert float(search["best_config"]["group_weights"]["valuation"]) == 1.3


