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


