from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def test_company_state_showcase_has_provenance_rich_features() -> None:
    sample = _load("examples/company_state_snapshot/company_state_hd.sample.json")
    assert sample["ticker"] == "HD"
    assert len(sample["features"]) >= 10
    for feature in sample["features"].values():
        assert feature["as_of_time"]
        assert feature["computed_at"]
        assert feature["provenance"]
        assert "confidence" in feature


def test_precedent_showcase_exposes_retrieval_evidence() -> None:
    sample = _load("examples/precedent_retrieval/precedent_retrieval.sample.json")
    assert sample["retrieved_candidate_rows_in_sample"] >= 6
    first = sample["results"][0]
    assert first["action_id"]
    assert first["precedent_confidence"] is not None
    assert first["top_matches"]
    assert "mismatch_diagnostics" in first
    assert "outcome_distribution_12m_or_nearest" in first


def test_cfo_decision_surface_showcase_has_decision_and_dossier_layers() -> None:
    sample = _load("examples/cfo_decision_surface/cfo_decision_surface_hd.sample.json")
    surface = sample["home_depot_decision_surface"]
    assert surface["mna_decision_summary"]["authoritative_recommendation"]["label"]
    assert surface["capital_allocation_frontier"]["points"]
    assert surface["defensible_model_layers"]

    dossier = sample["board_ready_dossier_excerpt"]
    assert dossier["recommendation_thesis"]["decision_confidence"]
    assert dossier["sizing_guidance"]["parameter_optimization"]
    assert dossier["supporting_evidence"]
    assert dossier["recommendation_contract"]["primary_action_is_clean_recommendation"] is True
