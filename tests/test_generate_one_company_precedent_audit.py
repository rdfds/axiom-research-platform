from __future__ import annotations

import importlib.util
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


MODULE_PATH = Path("./scripts/generate_one_company_precedent_audit.py")
SPEC = importlib.util.spec_from_file_location("generate_one_company_precedent_audit", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_confidence_lines_include_coverage_and_action_score() -> None:
    lines = audit._confidence_lines(
        "Learned",
        {
            "calibration_confidence": 0.61,
            "confidence_label": "medium",
            "retrieval_tier": "exact",
            "out_of_sample_flag": False,
            "exact_match_count": 42,
            "minimum_exact_support": 5,
            "top_similarity_mean": 0.88,
            "top_weighted_feature_coverage": 0.97,
            "top_critical_feature_coverage": 0.91,
            "top_action_match_score": 1.0,
        },
    )
    joined = "\n".join(lines)
    assert "top-support critical coverage" in joined
    assert "top-support action match score" in joined


def test_action_params_from_outcome_row_carries_refinancing_family_hints() -> None:
    params = audit._action_params_from_outcome_row(
        {
            "normalized_action_id": "capital_structure.refinancing",
            "raw_action_subtype": "Revolver/Line >= 1 Yr.",
            "action_size": 180_000_000.0,
        }
    )
    assert params["amount_usd"] == 180_000_000.0
    assert params["source_action_subtype"] == "Revolver/Line >= 1 Yr."
    assert params["instrument_type"] == "revolver"


def test_match_explanation_lines_call_out_closest_and_gaps() -> None:
    target_values = {key: 0.0 for key in audit._STATE_VECTOR_V1_FEATURES}
    target_values["state_vector_v1.valuation_multiple"] = 60.0
    target_values["state_vector_v1.growth"] = 0.15
    target_values["state_vector_v1.market_stress"] = 0.40
    match_row = pd.Series(
        {
            "state_vector_v1.valuation_multiple": 58.0,
            "state_vector_v1.growth": 0.12,
            "state_vector_v1.market_stress": 0.05,
        }
    )
    feature_scales = {key: 1.0 for key in audit._STATE_VECTOR_V1_FEATURES}
    lines = audit._match_explanation_lines(
        action_id="capital_return.open_market_buyback",
        target_values=target_values,
        match_row=match_row,
        feature_scales=feature_scales,
    )
    joined = "\n".join(lines)
    assert "Why it matched" in joined
    assert "Main gaps" in joined
    assert "valuation multiple" in joined
    assert "market stress" in joined


