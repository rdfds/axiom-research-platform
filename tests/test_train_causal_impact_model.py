from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.train_causal_impact_model import (
    _build_targets,
    _cell_scope_mask,
    _capital_phase1_defaults,
    _parse_objective_allowlist,
    _resolve_outcomes_path,
    _resolve_dr_control_scope,
    _validate_action_allowlist_coverage,
    _with_action_cells,
)


def test_parse_objective_allowlist_filters_unknown_values():
    out = _parse_objective_allowlist(
        "value_creation, risk_reduction, growth_v2, optionality_v2, unknown, value_creation"
    )
    assert out == ["value_creation", "risk_reduction", "growth_v2", "optionality_v2"]


def test_capital_phase1_defaults_collect_enabled_and_weak_prior_actions():
    actions, objectives = _capital_phase1_defaults(
        {
            "actions": {
                "capital_return.open_market_buyback": {
                    "status": "enabled",
                    "objective_allowlist": ["value_creation", "risk_reduction"],
                },
                "capital_return.special_dividend": {
                    "status": "weak_prior_only",
                    "objective_allowlist": ["value_creation"],
                },
                "governance.stock_split": {
                    "status": "blocked",
                    "objective_allowlist": ["value_creation"],
                },
            }
        }
    )
    assert actions == [
        "capital_return.open_market_buyback",
        "capital_return.special_dividend",
    ]
    assert objectives == ["value_creation", "risk_reduction"]


def test_with_action_cells_prefers_normalized_action_columns():
    df = pd.DataFrame(
        {
            "action_type": ["dividend"],
            "action_subtype": ["regular"],
            "normalized_action_family": ["capital_structure"],
            "normalized_action_subfamily": ["refinancing"],
            "normalized_action_id": ["capital_structure.refinancing"],
        }
    )
    out = _with_action_cells(df, cell_level="action_subtype")
    assert out.loc[0, "action_type_key"] == "capital_structure"
    assert out.loc[0, "action_subtype_key"] == "refinancing"
    assert out.loc[0, "action_id_key"] == "capital_structure.refinancing"
    assert out.loc[0, "action_cell"] == "capital_structure::refinancing"


