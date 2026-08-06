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


def test_coordinate_search_scope_configuration_can_target_specific_feature_weights_only():
    objective = {
        "objective_order": [
            {"metric": "mean_alignment_score", "direction": "maximize"},
        ],
        "search_space": {
            "optimize_group_weights": False,
            "optimize_feature_relative_weights": True,
            "feature_relative_weight_features": [
                "state_vector_v1.size_log_revenue",
                "state_vector_v1.profitability",
            ],
            "within_group_relative_weights": {"min": 0.8, "max": 1.2},
            "optimize_gates": False,
            "optimize_penalties": False,
            "optimize_blend_weights": False,
        },
    }

    seen_feature_weights = []
    seen_group_weights = []

    def evaluate_scope_config(scope_config):
        seen_feature_weights.append(
            (
                float(scope_config["feature_relative_weights"]["state_vector_v1.size_log_revenue"]),
                float(scope_config["feature_relative_weights"]["state_vector_v1.profitability"]),
            )
        )
        seen_group_weights.append(float(scope_config["group_weights"]["identity"]))
        profitability_weight = float(scope_config["feature_relative_weights"]["state_vector_v1.profitability"])
        alignment = 1.0 if abs(profitability_weight - 1.2) <= 1e-12 else 0.0
        return {"aggregate": {"mean_alignment_score": alignment}}

    search = coordinate_search_scope_configuration(
        scope_key="capital_structure",
        objective_config=objective,
        evaluate_scope_config=evaluate_scope_config,
        max_rounds=1,
    )
    assert float(search["best_config"]["feature_relative_weights"]["state_vector_v1.profitability"]) == 1.2
    assert sorted(set(seen_group_weights)) == [1.05]
    assert all(weight_pair[0] in {0.8, 1.0, 1.2} for weight_pair in seen_feature_weights)
    assert all(weight_pair[1] in {0.8, 1.0, 1.2} for weight_pair in seen_feature_weights)


def test_coordinate_search_scope_configuration_respects_custom_feature_grid_values():
    objective = {
        "objective_order": [
            {"metric": "mean_alignment_score", "direction": "maximize"},
        ],
        "search_space": {
            "optimize_group_weights": False,
            "optimize_feature_relative_weights": True,
            "feature_relative_weight_features": ["state_vector_v1.profitability"],
            "feature_relative_weight_grid_values": [0.9, 1.4],
            "within_group_relative_weights": {"min": 0.8, "max": 1.5},
            "optimize_gates": False,
            "optimize_penalties": False,
            "optimize_blend_weights": False,
        },
    }

    seen_profitability_weights = []

    def evaluate_scope_config(scope_config):
        profitability_weight = float(scope_config["feature_relative_weights"]["state_vector_v1.profitability"])
        seen_profitability_weights.append(profitability_weight)
        alignment = 1.0 if abs(profitability_weight - 1.4) <= 1e-12 else 0.0
        return {"aggregate": {"mean_alignment_score": alignment}}

    search = coordinate_search_scope_configuration(
        scope_key="capital_structure",
        objective_config=objective,
        evaluate_scope_config=evaluate_scope_config,
        max_rounds=1,
    )
    assert sorted(set(seen_profitability_weights)) == [0.9, 1.15, 1.4]
    assert float(search["best_config"]["feature_relative_weights"]["state_vector_v1.profitability"]) == 1.4


def test_build_precedent_distance_v2_payload_embeds_scope_configs():
    scope = default_scope_configuration("capital_structure")
    scope["metrics"] = {"mean_alignment_score": 0.8}
    payload = build_precedent_distance_v2_payload(
        scopes={"capital_structure": copy.deepcopy(scope)},
        objective_config={"objective_order": []},
        benchmark_key="capstructure",
    )
    assert payload["version"] == "precedent_distance_weights_v2"
    assert payload["scopes"]["capital_structure"]["metrics"]["mean_alignment_score"] == 0.8
    assert payload["benchmark_key"] == "capstructure"


def test_default_scope_configuration_applies_buyback_exact_defaults():
    scope = default_scope_configuration("capital_return.open_market_buyback")
    assert float(scope["feature_relative_weights"]["state_vector_v1.valuation_multiple"]) == 1.35
    assert float(scope["feature_relative_weights"]["state_vector_v1.cash_generation"]) == 1.2
    assert float(scope["gates"]["max_size_gap"]) == 1.15


def test_extract_report_aggregate_preserves_requested_counts():
    report = {
        "aggregate": {"error_rate": 0.0},
        "runs_analyzed": 12,
        "supported_case_count": 10,
        "case_count_requested": 10,
    }
    aggregate = extract_report_aggregate(report)
    assert aggregate["runs_analyzed"] == 12
    assert aggregate["supported_case_count"] == 10
    assert aggregate["case_count_requested"] == 10
