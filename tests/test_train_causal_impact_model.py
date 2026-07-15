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


