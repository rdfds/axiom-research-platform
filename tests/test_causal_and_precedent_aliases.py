from src.action_ontology import build_default_action_schema_registry
from src.causal_impact_model import action_id_to_outcomes_action_type, action_subtype_to_outcomes_subtype
from src.pipeline.precedent_brain import (
    _candidate_action_family_scale_weights,
    _candidate_action_family_weights,
)


def test_special_dividend_uses_dividend_regular_causal_fallback():
    alias = action_id_to_outcomes_action_type("capital_return.special_dividend")
    subtype = action_subtype_to_outcomes_subtype(
        action_id="capital_return.special_dividend",
        action_type="capital_return",
        action_subtype="special_dividend",
    )
    assert alias == "dividend_regular"
    assert subtype == "regular"


def test_revolver_uses_revolver_subtype_causal_fallback():
    alias = action_id_to_outcomes_action_type("capital_structure.revolver_draw_or_resize")
    subtype = action_subtype_to_outcomes_subtype(
        action_id="capital_structure.revolver_draw_or_resize",
        action_type="capital_structure",
        action_subtype="revolver_draw_or_resize",
    )
    assert alias == "loan_issuance"
    assert subtype == "revolver_line_1_yr"


def test_dividend_initiate_and_lbo_aliases_are_wired():
    dividend_alias = action_id_to_outcomes_action_type("capital_return.dividend_initiate")
    dividend_subtype = action_subtype_to_outcomes_subtype(
        action_id="capital_return.dividend_initiate",
        action_type="capital_return",
        action_subtype="dividend_initiate",
    )
    lbo_alias = action_id_to_outcomes_action_type("mna.go_private_lbo")
    lbo_subtype = action_subtype_to_outcomes_subtype(
        action_id="mna.go_private_lbo",
        action_type="mna",
        action_subtype="go_private_lbo",
    )
    assert dividend_alias == "dividend_regular"
    assert dividend_subtype == "regular"
    assert lbo_alias == "acquisition"
    assert lbo_subtype == "acquisition_lbo"


def test_transformational_acquisition_prefers_large_platform_precedents():
    family_weights = _candidate_action_family_weights("mna.transformational_acquisition", "")
    scale_weights = _candidate_action_family_scale_weights(
        action_id="mna.transformational_acquisition",
        action_subtype="transformational_acquisition",
        action_params={"target_size_pct_ev": 0.4},
        candidate_features={"market_cap": 10_000_000_000.0},
    )
    assert family_weights[0][0] == "mna.platform_merger"
    assert scale_weights[0][0] == "mna.platform_merger.scale_large"


def test_dividend_initiate_and_lbo_precedent_families():
    dividend_weights = _candidate_action_family_weights("capital_return.dividend_initiate", "")
    lbo_weights = _candidate_action_family_weights("mna.go_private_lbo", "")
    lbo_scale = _candidate_action_family_scale_weights(
        action_id="mna.go_private_lbo",
        action_subtype="go_private_lbo",
        action_params={"target_size_pct_ev": 0.9},
        candidate_features={"market_cap": 10_000_000_000.0},
    )
    assert dividend_weights[0][0] == "capital_return.dividend_initiate"
    assert lbo_weights[0][0] == "mna.platform_lbo"
    assert lbo_scale[0][0] == "mna.platform_lbo.scale_large"


def test_asset_sale_and_revolver_pick_family_scale_keys():
    asset_weights = _candidate_action_family_scale_weights(
        action_id="portfolio.asset_sale",
        action_subtype="asset_sale",
        action_params={"estimated_proceeds_usd": 600_000_000.0},
        candidate_features={"market_cap": 4_000_000_000.0},
    )
    revolver_weights = _candidate_action_family_scale_weights(
        action_id="capital_structure.revolver_draw_or_resize",
        action_subtype="revolver_draw_or_resize",
        action_params={"draw_amount_usd": 25_000_000.0},
        candidate_features={"market_cap": 5_000_000_000.0},
    )
    assert asset_weights[0][0] == "portfolio.divestiture.scale_medium"
    assert revolver_weights[0][0] == "capital_structure.revolver.amount_small"


def test_registry_contains_new_actions():
    registry = build_default_action_schema_registry(version="v1.0")
    assert registry.get_action("capital_return.dividend_initiate") is not None
    assert registry.get_action("mna.go_private_lbo") is not None
