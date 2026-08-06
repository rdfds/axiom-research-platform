from __future__ import annotations

from src.action_ontology import build_default_action_schema_registry


def test_registry_contains_all_required_actions():
    registry = build_default_action_schema_registry()
    action_ids = {a["action_id"] for a in registry.actions}

    expected = {
        "capital_return.open_market_buyback",
        "capital_return.accelerated_share_repurchase",
        "capital_return.tender_offer_buyback",
        "capital_return.dividend_increase",
        "capital_return.dividend_cut",
        "capital_return.special_dividend",
        "capital_return.dividend_initiate",
        "capital_structure.new_debt_issuance",
        "capital_structure.refinancing",
        "capital_structure.tender_offer_debt",
        "capital_structure.exchange_offer",
        "capital_structure.liability_management_exercise",
        "capital_structure.revolver_draw_or_resize",
        "capital_structure.equity_issuance",
        "capital_structure.convertible_issuance",
        "capital_structure.preferred_issuance",
        "mna.tuck_in_acquisition",
        "mna.platform_acquisition",
        "mna.transformational_acquisition",
        "mna.go_private_lbo",
        "mna.minority_investment",
        "portfolio.divestiture_full",
        "portfolio.divestiture_partial",
        "portfolio.asset_sale",
        "portfolio.spin_off",
        "portfolio.carve_out_ipo",
        "portfolio.joint_venture",
        "restructuring.cost_program",
        "restructuring.workforce_reduction",
        "restructuring.footprint_optimization",
        "restructuring.working_capital_program",
        "restructuring.asset_impairment_or_write_down",
        "restructuring.chapter_pathway",
        "restructuring.out_of_court_restructuring",
        "governance.board_refresh",
        "governance.activist_settlement",
        "governance.poison_pill_or_defensive_action",
        "governance.ceo_transition",
        "governance.capital_allocation_policy_reset",
        "governance.stock_split",
    }
    assert expected.issubset(action_ids)
    assert len(action_ids) == 40


def test_registry_schema_and_integrity_validation_pass():
    registry = build_default_action_schema_registry()
    assert registry.validate_schema() == []
    assert registry.validate_registry_integrity() == []


def test_query_methods():
    registry = build_default_action_schema_registry()
    action = registry.get_action("capital_return.open_market_buyback")
    assert action is not None
    assert action["action_type"] == "capital_return"

    by_type = registry.generate_actions_under_type("capital_structure")
    assert len(by_type) == 9

    channels = registry.fetch_mechanism_channels("capital_return.open_market_buyback")
    assert len(channels) >= 1

    edges = registry.fetch_dependency_graph_edges("capital_return.open_market_buyback")
    assert len(edges) >= 1

    planner_edges = registry.fetch_planner_dependency_edges("capital_return.open_market_buyback")
    assert len(planner_edges) >= 1

    lead_dist = registry.fetch_planner_lead_time_distribution("capital_return.open_market_buyback")
    assert lead_dist["median_days"] == 30
    assert lead_dist["p25_days"] == 16
    assert lead_dist["p75_days"] == 60
    assert lead_dist["source"] == "schema_prior_interpolated"


def test_planner_dependency_edge_mapping_normalizes_rule_types():
    registry = build_default_action_schema_registry()
    edges = registry.fetch_planner_dependency_edges("capital_return.open_market_buyback")
    edge_by_target = {edge["target_action"]: edge for edge in edges}

    assert edge_by_target["mna.transformational_acquisition"]["relationship_type"] == "conflicts"
    assert edge_by_target["mna.transformational_acquisition"]["original_rule_type"] == "conflicts_with"
    assert edge_by_target["capital_structure.refinancing"]["relationship_type"] == "recommended_after"
    assert edge_by_target["capital_structure.refinancing"]["original_rule_type"] == "preferred_after"


def test_planner_lead_time_distribution_interpolates_schema_prior():
    registry = build_default_action_schema_registry()
    dist = registry.fetch_planner_lead_time_distribution("mna.platform_acquisition")

    assert dist == {
        "action_id": "mna.platform_acquisition",
        "minimum_days": 45,
        "mean_days": 192.5,
        "median_days": 180,
        "p25_days": 112,
        "p75_days": 272,
        "p90_days": 365,
        "conditional_adjustments": [],
        "source": "schema_prior_interpolated",
    }


def test_candidate_validator_catches_missing_required_params():
    registry = build_default_action_schema_registry()
    result = registry.validate_candidate(
        {
            "action_id": "capital_return.open_market_buyback",
            "parameters": {
                "funding_mix": {"cash": 0.5, "debt": 0.5, "equity": 0.0},
            },
            "available_features": [
                "liquidity.available_for_actions",
                "capital_structure.net_leverage",
                "market.market_cap",
            ],
            "available_evidence_classes": ["financial_disclosure", "market_signal"],
        }
    )
    assert result.valid is False
    assert any(e.startswith("missing_required_param:size_pct_market_cap") for e in result.errors)


def test_candidate_validator_funding_mix_rule():
    registry = build_default_action_schema_registry()
    result = registry.validate_candidate(
        {
            "action_id": "capital_return.open_market_buyback",
            "parameters": {
                "size_pct_market_cap": 0.1,
                "funding_mix": {"cash": 0.6, "debt": 0.6, "equity": 0.0},
            },
            "available_features": [
                "liquidity.available_for_actions",
                "capital_structure.net_leverage",
                "market.market_cap",
            ],
            "available_evidence_classes": ["financial_disclosure", "market_signal"],
        }
    )
    assert result.valid is False
    assert "funding_mix_sum_not_one:funding_mix" in result.errors


def test_candidate_validator_segment_reference():
    registry = build_default_action_schema_registry()
    result = registry.validate_candidate(
        {
            "action_id": "portfolio.spin_off",
            "parameters": {
                "segment_reference": "SEG_UNKNOWN",
            },
            "known_segments": ["SEG_A", "SEG_B"],
            "available_features": ["strategic.constraint_set", "market.equity_window_proxy"],
            "available_evidence_classes": ["segment_disclosure", "management_statement", "financial_disclosure"],
        }
    )
    assert result.valid is False
    assert any("segment_reference_not_found:segment_reference:SEG_UNKNOWN" == e for e in result.errors)
